"""
app/graph/nodes.py
==================
All LangGraph node functions and conditional routing functions for the research pipeline.

Each node function receives the full ResearchState and returns a partial dict
of state updates. LangGraph merges the returned dict into state before calling
the next node.

Node execution order (happy path):
    plan_research → await_human_approval → spawn_subagents → collect_results
    → evaluate_sufficiency → synthesize → review_synthesis → cite
    → generate_document → save_to_project → respond

Routing functions (conditional edges):
    _route_after_hitl()         After HITL: approved→research, refine→replan, rejected→cancel
    _route_after_sufficiency()  After evaluation: needs_more→more research, else→synthesize
    _route_after_review()       After self-review: needs_rework→rework, else→cite

Agent singletons:
    Agents are instantiated once at module load and reused across all node calls.
    This avoids re-initialising the Anthropic client on every invocation.

Input:  ResearchState TypedDict (full shared state)
Output: dict — partial state update (only changed fields)
"""
from __future__ import annotations
import os
from datetime import datetime
from langgraph.types import interrupt

from app.config import (
    MAX_SUBAGENTS, MAX_ITERATIONS, MAX_TOOL_ROUNDS,
    MAX_HITL_REFINE_ROUNDS, MAX_SYNTHESIS_REVIEW_ROUNDS,
    SUBAGENT_TIMEOUT_SECONDS, OUTPUT_DIR,
)
from app.graph.state import ResearchState
from app.agents.lead_researcher import LeadResearcher
from app.agents.sub_agent import ResearchSubAgent
from app.agents.citation_agent import CitationAgent
from app.agents.document_generator import DocumentGenerator
from app.utils.job_store import job_store
from app.utils.tracer import tracer_registry
from app.utils.groq_retry import set_trace_context, clear_trace_context

_lead = LeadResearcher()
_sub_agent = ResearchSubAgent()
_citation = CitationAgent()
_doc_gen = DocumentGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: plan_research
# ─────────────────────────────────────────────────────────────────────────────
def plan_research(state: ResearchState) -> dict:
    """LeadResearcher analyses the query, decomposes into sub-topics, writes task descriptions."""
    job_store.update_job(state["job_id"], status="planning", phase="planning")
    job_store.emit_event(state["job_id"], "🔄 Planning research...")
    tracer = tracer_registry.get(state["job_id"])
    span_id = tracer.start_span("plan_research") if tracer else None

    set_trace_context(state["job_id"], "plan_research")
    try:
        plan_result = _lead.plan(
            query=state["query"],
            depth=state["depth"],
            hitl_feedback=state.get("hitl_feedback", ""),
            hitl_round=state.get("hitl_round", 0),
        )
    finally:
        clear_trace_context()

    n_agents = len(plan_result.get("tasks", []))
    if tracer and span_id:
        tracer.end_span(span_id, {"tasks": n_agents})
    job_store.emit_event(state["job_id"], f"📋 Plan ready — {n_agents} sub-agent(s) assigned")
    return {
        "research_plan": plan_result["plan"],
        "sub_agent_tasks": plan_result["tasks"],
        "token_usage": _merge_tokens(state.get("token_usage", {}), plan_result.get("token_usage", {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: await_human_approval  (HITL interrupt)
# ─────────────────────────────────────────────────────────────────────────────
def await_human_approval(state: ResearchState) -> dict:
    """
    Pauses the graph and surfaces the research plan to the human.
    Resumes when human POSTs to /approve/{job_id}.
    """
    plan = state["research_plan"]
    job_store.update_job(
        state["job_id"],
        status="awaiting_approval",
        phase="awaiting_approval",
        research_plan=plan,
        hitl_round=state.get("hitl_round", 0),
    )
    job_store.emit_event(state["job_id"], "👤 Awaiting human approval of research plan...")

    # Graph pauses here. The interrupt value is what /status returns as hitl.research_plan.
    human_decision: dict = interrupt({
        "research_plan": plan,
        "hitl_round": state.get("hitl_round", 0),
    })

    decision = human_decision.get("decision", "approved")
    job_store.emit_event(state["job_id"], f"✅ Human decision: {decision}")
    feedback = human_decision.get("feedback", "")

    if decision == "approved":
        return {"hitl_status": "approved", "hitl_feedback": ""}
    elif decision == "rejected":
        return {"hitl_status": "rejected"}
    else:  # refine
        new_round = state.get("hitl_round", 0) + 1
        job_store.update_job(state["job_id"], hitl_round=new_round)
        return {
            "hitl_status": "refining",
            "hitl_feedback": feedback,
            "hitl_round": new_round,
        }


def _route_after_hitl(state: ResearchState) -> str:
    """Conditional edge after HITL approval node."""
    status = state.get("hitl_status", "approved")
    if status == "approved":
        return "spawn_subagents"
    elif status == "rejected":
        return "respond"
    else:  # refining
        if state.get("hitl_round", 0) >= MAX_HITL_REFINE_ROUNDS:
            # Too many refines — force proceed
            return "spawn_subagents"
        return "plan_research"


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: spawn_subagents
# ─────────────────────────────────────────────────────────────────────────────
def spawn_subagents(state: ResearchState) -> dict:
    """
    Dispatches 1-N sub-agents in parallel (using Python threads for MVP).
    Each sub-agent receives its task and uses Claude's web_search + web_fetch tools.
    """
    import concurrent.futures

    tasks = state.get("sub_agent_tasks", [])
    n = min(len(tasks), MAX_SUBAGENTS)
    tasks = tasks[:n]

    # Safety guard — should never happen after lead_researcher fallbacks, but prevents crash
    if n == 0:
        tasks = [{
            "id": 1,
            "title": state["query"][:60],
            "objective": f"Research: {state['query']}",
            "scope": "Comprehensive research on the topic",
            "search_strategy": [state["query"], f"{state['query']} overview", f"{state['query']} 2025"],
            "output_format": (
                'Return a JSON with: summary (str), key_facts (list[str]), '
                'sources (list[{url,title,date}]), confidence (0-1), coverage_gaps (list[str])'
            ),
            "stopping_criteria": "Stop after 8-12 web searches or 5+ distinct authoritative sources.",
        }]
        n = 1
        job_store.emit_event(state["job_id"], "⚠️ No tasks from plan — using fallback single task")

    job_store.update_job(
        state["job_id"],
        status="running",
        phase="researching",
        sub_agents_active=n,
        sub_agent_count=n,
    )
    job_store.emit_event(state["job_id"], f"🚀 Spawning {n} sub-agent(s) in parallel...")
    tracer = tracer_registry.get(state["job_id"])
    span_id = tracer.start_span("spawn_subagents", {"agent_count": n}) if tracer else None

    results = []
    token_totals: dict = {}

    timeout_secs = SUBAGENT_TIMEOUT_SECONDS

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(_sub_agent.execute, dict(task, job_id=state["job_id"])): task
            for task in tasks
        }
        try:
            completed = concurrent.futures.as_completed(futures, timeout=timeout_secs)
            for fut in completed:
                task = futures[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    token_totals = _merge_tokens(token_totals, result.get("token_usage", {}))
                except Exception as exc:
                    results.append({
                        "task_id": task.get("id", 0),
                        "summary": f"[Sub-agent failed: {exc}]",
                        "key_facts": [],
                        "sources": [],
                        "confidence": 0.0,
                        "coverage_gaps": [str(exc)],
                        "tool_call_count": 0,
                        "token_usage": {},
                    })
        except concurrent.futures.TimeoutError:
            # Collect whatever finished; stub out the rest so synthesis can proceed
            for fut, task in futures.items():
                if fut.done():
                    try:
                        result = fut.result()
                        if result not in results:
                            results.append(result)
                            token_totals = _merge_tokens(token_totals, result.get("token_usage", {}))
                    except Exception as exc:
                        results.append({
                            "task_id": task.get("id", 0),
                            "summary": f"[Sub-agent failed: {exc}]",
                            "key_facts": [], "sources": [],
                            "confidence": 0.0, "coverage_gaps": [str(exc)],
                            "tool_call_count": 0, "token_usage": {},
                        })
                else:
                    fut.cancel()
                    results.append({
                        "task_id": task.get("id", 0),
                        "title": task.get("title", ""),
                        "summary": f"[Sub-agent timed out after {timeout_secs}s]",
                        "key_facts": [], "sources": [],
                        "confidence": 0.0,
                        "coverage_gaps": [f"Task timed out after {timeout_secs}s"],
                        "tool_call_count": 0, "token_usage": {},
                    })

    job_store.update_job(state["job_id"], sub_agents_active=0)
    if tracer and span_id:
        tracer.end_span(span_id, {"results": len(results)})
    job_store.emit_event(state["job_id"], f"✅ All sub-agents done — {len(results)} result(s) collected")

    # Build source_map from all results
    source_map = dict(state.get("source_map", {}))
    for r in results:
        for src in r.get("sources", []):
            url = src.get("url", "")
            if url and url not in source_map:
                source_map[url] = src

    return {
        "sub_agent_results": results,
        "source_map": source_map,
        "token_usage": _merge_tokens(state.get("token_usage", {}), token_totals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: collect_results
# ─────────────────────────────────────────────────────────────────────────────
def collect_results(state: ResearchState) -> dict:
    """Aggregates sub-agent results into accumulated_findings. Updates iteration count."""
    results = state.get("sub_agent_results", [])
    prev_findings = state.get("accumulated_findings", "")
    new_chunks = []
    for r in results:
        if r.get("summary"):
            new_chunks.append(f"### Sub-topic: {r.get('title', 'Research')}\n{r['summary']}")

    combined = (prev_findings + "\n\n" + "\n\n".join(new_chunks)).strip()
    new_count = state.get("iteration_count", 0) + 1
    job_store.update_job(state["job_id"], iteration_count=new_count)
    return {
        "accumulated_findings": combined,
        "iteration_count": new_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: evaluate_sufficiency
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_sufficiency(state: ResearchState) -> dict:
    """LeadResearcher evaluates whether findings are sufficient or more research is needed."""
    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    current = state.get("iteration_count", 1)

    if current >= max_iter:
        job_store.emit_event(state["job_id"], "🔍 Sufficiency check: force_stop")
        return {"sufficiency_signal": "force_stop"}

    signal = _lead.evaluate_sufficiency(
        query=state["query"],
        findings=state.get("accumulated_findings", ""),
        sub_agent_results=state.get("sub_agent_results", []),
        iteration_count=current,
        max_iterations=max_iter,
    )
    job_store.emit_event(state["job_id"], f"🔍 Sufficiency check: {signal}")
    return {"sufficiency_signal": signal}


def _route_after_sufficiency(state: ResearchState) -> str:
    sig = state.get("sufficiency_signal", "sufficient")
    if sig == "needs_more":
        return "spawn_subagents"
    return "synthesize"


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: synthesize
# ─────────────────────────────────────────────────────────────────────────────
def synthesize(state: ResearchState) -> dict:
    """LeadResearcher synthesizes all sub-agent findings into a coherent narrative."""
    job_store.update_job(state["job_id"], phase="synthesizing")
    job_store.emit_event(state["job_id"], "📝 Synthesizing all findings...")
    tracer = tracer_registry.get(state["job_id"])
    span_id = tracer.start_span("synthesize") if tracer else None

    rework_instructions = state.get("synthesis_rework_instructions", [])

    set_trace_context(state["job_id"], "synthesize")
    try:
        result = _lead.synthesize(
            query=state["query"],
            findings=state.get("accumulated_findings", ""),
            sub_agent_results=state.get("sub_agent_results", []),
            sub_agent_tasks=state.get("sub_agent_tasks", []),
            source_map=state.get("source_map", {}),
            rework_instructions=rework_instructions,
        )
    finally:
        clear_trace_context()

    if tracer and span_id:
        tracer.end_span(span_id)
    job_store.emit_event(state["job_id"], "✅ Synthesis complete")
    return {
        "synthesized_narrative": result["narrative"],
        "token_usage": _merge_tokens(state.get("token_usage", {}), result.get("token_usage", {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: review_synthesis  (LeadResearcher self-review, no human)
# ─────────────────────────────────────────────────────────────────────────────
def review_synthesis(state: ResearchState) -> dict:
    """
    LeadResearcher checks synthesis against task assignment map.
    Max 3 rounds. After 3, force_proceed regardless of quality.
    """
    round_num = state.get("synthesis_review_count", 0)
    max_rounds = MAX_SYNTHESIS_REVIEW_ROUNDS

    if round_num >= max_rounds:
        return {
            "synthesis_review_signal": "force_proceed",
            "synthesis_review_count": round_num,
        }

    job_store.emit_event(state["job_id"], f"🔎 Reviewing synthesis (round {round_num + 1})...")

    set_trace_context(state["job_id"], "review_synthesis")
    try:
        result = _lead.review_synthesis(
            query=state["query"],
            narrative=state.get("synthesized_narrative", ""),
            sub_agent_tasks=state.get("sub_agent_tasks", []),
            sub_agent_results=state.get("sub_agent_results", []),
        )
    finally:
        clear_trace_context()

    signal = result["signal"]
    new_count = round_num + 1
    job_store.update_job(state["job_id"], synthesis_review_count=new_count)
    job_store.emit_event(state["job_id"], f"📊 Review result: {signal}")

    return {
        "synthesis_review_signal": signal,
        "synthesis_rework_instructions": result.get("rework_instructions", []),
        "synthesis_review_count": new_count,
        "token_usage": _merge_tokens(state.get("token_usage", {}), result.get("token_usage", {})),
    }


def _route_after_review(state: ResearchState) -> str:
    sig = state.get("synthesis_review_signal", "approved")
    if sig == "needs_rework" and state.get("synthesis_review_count", 0) < MAX_SYNTHESIS_REVIEW_ROUNDS:
        return "targeted_rework"
    return "cite"


# ─────────────────────────────────────────────────────────────────────────────
# Node 8: targeted_rework
# ─────────────────────────────────────────────────────────────────────────────
def targeted_rework(state: ResearchState) -> dict:
    """
    Sends targeted rework instructions to specific sub-agents.
    Sub-agents refine using already-gathered sources (minimal new searches).
    Results replace or supplement existing sub_agent_results.
    """
    instructions = state.get("synthesis_rework_instructions", [])
    if not instructions:
        return {}

    import concurrent.futures
    rework_tasks = []
    # build rework_tasks list first so we can log count
    for instr in instructions:
        task_id = instr.get("task_id")
        original_task = next(
            (t for t in state.get("sub_agent_tasks", []) if t.get("id") == task_id),
            None,
        )
        if original_task:
            rework_task = dict(original_task)
            rework_task["rework_instruction"] = instr.get("instruction", "")
            rework_task["existing_sources"] = [
                s for r in state.get("sub_agent_results", [])
                if r.get("task_id") == task_id
                for s in r.get("sources", [])
            ]
            rework_tasks.append(rework_task)

    job_store.emit_event(state["job_id"], f"🔧 Running targeted rework on {len(rework_tasks)} task(s)...")
    new_results = []
    token_totals: dict = {}
    rework_timeout = SUBAGENT_TIMEOUT_SECONDS

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rework_tasks) or 1) as pool:
        futures = {pool.submit(_sub_agent.rework, t): t for t in rework_tasks}
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=rework_timeout):
                task = futures[fut]
                try:
                    res = fut.result()
                    new_results.append(res)
                    token_totals = _merge_tokens(token_totals, res.get("token_usage", {}))
                except Exception as exc:
                    new_results.append({
                        "task_id": task.get("id", 0),
                        "title": task.get("title", "Rework"),
                        "summary": f"[Rework failed: {exc}]",
                        "key_facts": [], "sources": [],
                        "confidence": 0.0,
                        "coverage_gaps": [f"Rework task failed: {exc}"],
                        "token_usage": {},
                    })
        except concurrent.futures.TimeoutError:
            for fut, task in futures.items():
                if not fut.done():
                    fut.cancel()
                    new_results.append({
                        "task_id": task.get("id", 0),
                        "title": task.get("title", "Rework"),
                        "summary": f"[Rework timed out after {rework_timeout}s]",
                        "key_facts": [], "sources": [],
                        "confidence": 0.0,
                        "coverage_gaps": [f"Rework timed out after {rework_timeout}s"],
                        "token_usage": {},
                    })

    # Merge new sources into source_map
    source_map = dict(state.get("source_map", {}))
    for r in new_results:
        for src in r.get("sources", []):
            url = src.get("url", "")
            if url:
                source_map[url] = src

    # Update accumulated_findings with reworked content
    prev = state.get("accumulated_findings", "")
    rework_chunks = [f"### Reworked: {r.get('title', '')}\n{r['summary']}" for r in new_results if r.get("summary")]
    combined = (prev + "\n\n" + "\n\n".join(rework_chunks)).strip() if rework_chunks else prev

    return {
        "sub_agent_results": new_results,
        "accumulated_findings": combined,
        "source_map": source_map,
        "token_usage": _merge_tokens(state.get("token_usage", {}), token_totals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 9: cite
# ─────────────────────────────────────────────────────────────────────────────
def cite(state: ResearchState) -> dict:
    """CitationAgent maps every factual claim to a source. Produces annotated narrative + bibliography."""
    job_store.update_job(state["job_id"], phase="citing")
    job_store.emit_event(state["job_id"], "📚 Adding citations...")
    tracer = tracer_registry.get(state["job_id"])
    span_id = tracer.start_span("cite") if tracer else None

    set_trace_context(state["job_id"], "cite")
    try:
        result = _citation.annotate(
            narrative=state.get("synthesized_narrative", ""),
            source_map=state.get("source_map", {}),
        )
    finally:
        clear_trace_context()

    bibliography = result.get("bibliography", [])
    if tracer and span_id:
        tracer.end_span(span_id, {"citations": len(bibliography)})
    job_store.update_job(state["job_id"], citation_count=len(bibliography))
    job_store.emit_event(state["job_id"], f"✅ Citations done — {len(bibliography)} source(s)")
    return {
        "annotated_narrative": result["annotated_narrative"],
        "citation_map": result["citation_map"],
        "bibliography": bibliography,
        "token_usage": _merge_tokens(state.get("token_usage", {}), result.get("token_usage", {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 10: generate_document
# ─────────────────────────────────────────────────────────────────────────────
def generate_document(state: ResearchState) -> dict:
    """DocumentGeneratorAgent produces the .docx report."""
    job_store.update_job(state["job_id"], phase="generating")
    job_store.emit_event(state["job_id"], "📄 Generating report document...")

    output_dir = state.get("output_folder") or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(state["query"])[:50]
    doc_path = os.path.join(output_dir, f"{ts}_{slug}.docx")

    _doc_gen.generate(
        query=state["query"],
        annotated_narrative=state.get("annotated_narrative", ""),
        bibliography=state.get("bibliography", []),
        synthesis_review_count=state.get("synthesis_review_count", 0),
        synthesis_review_signal=state.get("synthesis_review_signal", "approved"),
        metadata={
            "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "sources": len(state.get("source_map", {})),
            "depth": state.get("depth", "moderate"),
            "iterations": state.get("iteration_count", 1),
        },
        output_path=doc_path,
    )

    job_store.emit_event(state["job_id"], f"✅ Report saved: {doc_path}")
    return {"document_path": doc_path}


# ─────────────────────────────────────────────────────────────────────────────
# Node 11: save_to_project  (no-op in MVP — document_generator already saves)
# ─────────────────────────────────────────────────────────────────────────────
def save_to_project(state: ResearchState) -> dict:
    job_store.update_job(
        state["job_id"],
        document_path=state.get("document_path"),
        source_count=len(state.get("source_map", {})),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 12: respond
# ─────────────────────────────────────────────────────────────────────────────
def respond(state: ResearchState) -> dict:
    """Assemble final API response payload and mark job complete (or cancelled)."""
    from datetime import datetime as dt

    if state.get("hitl_status") == "rejected":
        job_store.update_job(state["job_id"], status="cancelled", phase="cancelled")
        job_store.emit_event(state["job_id"], "🚫 Research cancelled by user")
        return {"final_response": {"status": "cancelled"}}

    narrative = state.get("annotated_narrative", state.get("synthesized_narrative", ""))
    snippet = " ".join(narrative.split()[:80]) + ("…" if len(narrative.split()) > 80 else "")

    end = dt.utcnow()
    start_str = state.get("start_time", end.isoformat())
    try:
        start = dt.fromisoformat(start_str)
        duration = int((end - start).total_seconds())
    except Exception:
        duration = 0

    token_usage = state.get("token_usage", {})
    total_tokens = sum(token_usage.values())

    job_store.update_job(
        state["job_id"],
        status="complete",
        phase="complete",
        document_path=state.get("document_path"),
        summary_snippet=snippet,
        duration_seconds=duration,
        token_usage={"total": total_tokens, **token_usage},
    )
    job_store.emit_event(state["job_id"], "🎉 Research complete!")

    return {
        "final_response": {
            "status": "complete",
            "document_path": state.get("document_path"),
            "summary_snippet": snippet,
            "duration_seconds": duration,
            "token_usage": token_usage,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _merge_tokens(base: dict, addition: dict) -> dict:
    merged = dict(base)
    for k, v in addition.items():
        merged[k] = merged.get(k, 0) + v
    return merged


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^\w]+", "_", text.lower()).strip("_")
