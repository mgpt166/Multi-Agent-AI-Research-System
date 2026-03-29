"""
app/graph/runner.py
===================
Entry points for starting and resuming LangGraph research jobs.

This module is the bridge between the FastAPI layer and the LangGraph graph.
It is always called from a background thread (via ThreadPoolExecutor in routes.py)
because graph.invoke() is synchronous and blocking.

Functions:
    run_research_job()      Start a brand-new research job from an initial state.
                            Runs until the graph hits the HITL interrupt or completes.

    resume_research_job()   Resume a job paused at the HITL interrupt by injecting
                            the human's decision via LangGraph's Command(resume=...).

    cancel_hitl_timeout()   Cancel a pending auto-approve timer (called on manual approval).

Threading / HITL pattern:
    1. run_research_job() calls graph.invoke(initial_state, config)
    2. The graph runs plan_research → await_human_approval, where interrupt() fires
    3. graph.invoke() returns (the interrupt unwinds the call stack)
    4. Job status is set to 'awaiting_approval' in the job store
    5. A threading.Timer is started — if no human approval arrives within
       HITL_TIMEOUT_SECONDS (default 180), the job is auto-approved.
    6. Human POSTs to /approve/{job_id} → resume_research_job() is called,
       which cancels the timer and resumes the graph.

Config:
    thread_id = job_id is used as the LangGraph thread identifier so the
    checkpointer can find the saved state when resuming.
    HITL_TIMEOUT_SECONDS: env var controlling auto-approve delay (default 180).
"""

from __future__ import annotations
import logging
import threading
from datetime import datetime
from langgraph.types import Command

from app.config import (
    HITL_TIMEOUT_SECONDS, MAX_ITERATIONS,
    MAX_COST_PER_QUERY, COST_WARNING_THRESHOLD,
    GROQ_MODEL, GROQ_SUB_AGENT_MODEL,
)
from app.graph.graph import research_graph, checkpointer
from app.graph.state import ResearchState
from app.utils.job_store import job_store
from app.utils.cost_tracker import CostTracker
from app.utils.tracer import tracer_registry
from app.utils.metrics import metrics
from app.utils.groq_retry import init_job_stats, get_job_stats, clear_job_stats

_logger = logging.getLogger(__name__)

# Seconds to wait for human approval before auto-approving (configurable via env)
_HITL_TIMEOUT = HITL_TIMEOUT_SECONDS

# Tracks pending timers per job so they can be cancelled on manual approval
_hitl_timers: dict[str, threading.Timer] = {}


def _record_completion(job_id: str, cost_tracker: CostTracker, job: dict) -> None:
    """Update metrics and tracer when a job finishes successfully."""
    token_usage = job.get("token_usage", {})
    total_tokens = token_usage.get("total", sum(
        v for k, v in token_usage.items() if k != "total"
    ))
    duration = job.get("duration_seconds", 0)

    # Populate cost tracker from accumulated token_usage so the breakdown is
    # visible in the Observability UI. Tokens are counted as input-only because
    # agents store combined counts — cost is slightly underestimated (~10-15%).
    _token_model_map = {
        "lead":      ("lead_researcher", GROQ_MODEL),
        "sub_agents": ("sub_agents",     GROQ_SUB_AGENT_MODEL),
        "citation":  ("citation_agent",  GROQ_MODEL),
    }
    for key, tokens in token_usage.items():
        if key == "total" or not isinstance(tokens, (int, float)) or tokens == 0:
            continue
        agent_name, model = _token_model_map.get(key, (key, GROQ_MODEL))
        cost_tracker.add_usage(int(tokens), 0, agent_name, model=model)

    cost_summary = cost_tracker.get_summary()

    if cost_tracker.is_budget_exceeded():
        metrics.record_budget_exceeded()

    metrics.record_request_complete(
        duration_seconds=float(duration),
        cost=cost_summary["total"],
        tokens=int(total_tokens),
    )

    tracer = tracer_registry.get(job_id)
    if tracer:
        tracer.log_event("job_complete", f"duration={duration}s tokens={total_tokens}")
        job_store.update_job(job_id, token_usage={**token_usage, "cost_summary": cost_summary})
        # Persist observability data to SQLite so the UI can display it after restart
        spans_only = [e for e in tracer.get_timeline() if e.get("type") == "span"]
        call_stats = get_job_stats(job_id)
        job_store.save_job_trace(
            job_id=job_id,
            query=job.get("query", ""),
            trace_summary=tracer.get_summary(),
            timeline=spans_only,
            cost_summary=cost_summary,
            call_stats=call_stats,
        )
        clear_job_stats(job_id)
        tracer_registry.remove(job_id)


def _make_config(job_id: str) -> dict:
    """Build the LangGraph config dict that identifies this job's checkpoint thread."""
    return {"configurable": {"thread_id": job_id}}


def _schedule_hitl_timeout(job_id: str) -> None:
    """
    Start a background timer that auto-approves the job if no human decision arrives.

    The timer fires after HITL_TIMEOUT_SECONDS. If the job is still in
    'awaiting_approval' at that point, it is automatically approved so the
    pipeline continues rather than hanging forever.

    The timer is stored in _hitl_timers so cancel_hitl_timeout() can stop it
    if the human approves first.
    """
    def _auto_approve():
        _hitl_timers.pop(job_id, None)
        job = job_store.get_job(job_id)
        if not job or job.get("status") != "awaiting_approval":
            # Human already acted — nothing to do
            return
        _logger.warning(
            "HITL timeout reached for job %s — auto-approving after %ds",
            job_id, _HITL_TIMEOUT,
        )
        resume_research_job(
            job_id,
            {"decision": "approved", "feedback": f"Auto-approved: no human response within {_HITL_TIMEOUT}s"},
        )

    timer = threading.Timer(_HITL_TIMEOUT, _auto_approve)
    timer.daemon = True   # Don't block process shutdown
    _hitl_timers[job_id] = timer
    timer.start()
    _logger.info("HITL auto-approve scheduled in %ds for job %s", _HITL_TIMEOUT, job_id)


def cancel_hitl_timeout(job_id: str) -> None:
    """
    Cancel a pending HITL auto-approve timer.

    Called by resume_research_job() when the human approves before the timeout
    fires, so the timer doesn't trigger a duplicate resume.
    """
    timer = _hitl_timers.pop(job_id, None)
    if timer:
        timer.cancel()
        _logger.info("HITL timeout cancelled for job %s (human approved)", job_id)


def run_research_job(
    job_id: str,
    query: str,
    depth: str = "moderate",
    output_folder: str | None = None,
    max_iterations: int | None = None,
) -> None:
    """
    Start a new research job from scratch.

    Builds the initial ResearchState, invokes the graph, and runs until either:
        - The HITL interrupt fires (graph pauses at await_human_approval), or
        - The graph reaches END (complete or cancelled)

    On error, the job is marked 'failed' in the store and the exception is re-raised
    so the ThreadPoolExecutor can log it.

    Args:
        job_id:         UUID of the pre-registered job in the job store.
        query:          Research question to investigate.
        depth:          Research depth (simple | moderate | deep).
        output_folder:  Optional custom output directory for the .docx report.
        max_iterations: Optional override for max research rounds.
    """
    job_store.update_job(job_id, status="planning", phase="planning")
    metrics.record_request_start()
    init_job_stats(job_id)

    # Create per-job cost tracker and request tracer
    cost_tracker = CostTracker(
        max_budget=MAX_COST_PER_QUERY,
        warning_threshold=COST_WARNING_THRESHOLD,
    )
    tracer = tracer_registry.create(job_id, query)
    tracer.log_event("job_start", f"Research job started: {query[:80]}")

    # Build a fully-initialised state so no node ever gets a KeyError
    initial_state: ResearchState = {
        "job_id": job_id,
        "query": query,
        "depth": depth,
        "output_folder": output_folder,
        "max_iterations": max_iterations or MAX_ITERATIONS,
        "start_time": datetime.utcnow().isoformat(),
        # HITL-1 state
        "hitl_status": "awaiting_approval",
        "hitl_feedback": "",
        "hitl_round": 0,
        # Planning outputs
        "research_plan": None,
        "sub_agent_tasks": [],
        # Research execution
        "sub_agent_results": [],
        "accumulated_findings": "",
        "source_map": {},
        "iteration_count": 0,
        "sufficiency_signal": "sufficient",
        # Synthesis review
        "synthesized_narrative": "",
        "synthesis_review_count": 0,
        "synthesis_review_signal": "approved",
        "synthesis_rework_instructions": [],
        # Citation outputs
        "annotated_narrative": "",
        "citation_map": {},
        "bibliography": [],
        # Document output
        "document_path": None,
        # Metrics
        "token_usage": {},
        "error_log": [],
        "final_response": None,
    }

    config = _make_config(job_id)

    try:
        span_id = tracer.start_span("pipeline_run")
        # Blocking call — runs until HITL interrupt or graph END
        research_graph.invoke(initial_state, config)
        tracer.end_span(span_id)

        # If the graph paused at HITL, start the auto-approve countdown
        job = job_store.get_job(job_id)
        if job and job.get("status") == "awaiting_approval":
            _schedule_hitl_timeout(job_id)
        elif job and job.get("status") == "complete":
            _record_completion(job_id, cost_tracker, job)

    except Exception as exc:
        metrics.record_request_failed()
        tracer.log_event("error", str(exc)[:200])
        tracer_registry.remove(job_id)
        clear_job_stats(job_id)
        job_store.update_job(job_id, status="failed", error=str(exc))
        raise


def resume_research_job(job_id: str, decision_data: dict) -> None:
    """
    Resume a job that is paused at the HITL interrupt.

    Injects the human's decision back into the graph via LangGraph's
    Command(resume=...) mechanism. The graph continues from the
    await_human_approval node using the saved checkpoint state.

    Args:
        job_id:         UUID of the job currently in 'awaiting_approval' status.
        decision_data:  Dict with keys:
                            decision: "approved" | "refine"
                            feedback: str (required if decision == "refine")

    Raises:
        Exception: Re-raised after marking the job 'failed' in the store.
    """
    # Cancel the auto-approve timer if the human is acting before it fires
    cancel_hitl_timeout(job_id)

    config = _make_config(job_id)
    job_store.update_job(job_id, status="running", phase="running")

    tracer = tracer_registry.get(job_id)
    cost_tracker = CostTracker(
        max_budget=MAX_COST_PER_QUERY,
        warning_threshold=COST_WARNING_THRESHOLD,
    )

    try:
        if tracer:
            tracer.log_event("hitl_resume", f"Decision: {decision_data.get('decision')}")
        # Command(resume=...) passes the decision data to the interrupt() call site
        research_graph.invoke(Command(resume=decision_data), config)

        job = job_store.get_job(job_id)
        if job and job.get("status") == "complete":
            _record_completion(job_id, cost_tracker, job)

    except Exception as exc:
        metrics.record_request_failed()
        if tracer:
            tracer.log_event("error", str(exc)[:200])
        tracer_registry.remove(job_id)
        job_store.update_job(job_id, status="failed", error=str(exc))
        raise
