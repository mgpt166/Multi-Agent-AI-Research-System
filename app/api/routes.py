"""
app/api/routes.py
=================
FastAPI route handlers for all public API endpoints.

This module is the HTTP boundary of the system. It receives requests,
validates them via Pydantic schemas, dispatches work to background threads
(because LangGraph runs synchronously), and returns structured responses.

Endpoints:
    POST /research              Submit a new research query. Returns 202 immediately;
                                work runs in background. Poll /status for progress.

    POST /approve/{job_id}      Submit human decision (approved / refine / rejected)
                                on the research plan. Resumes or cancels the graph.

    GET  /status/{job_id}       Returns current job status. Response shape varies:
                                - awaiting_approval: includes full research plan (hitl block)
                                - running: includes live progress (progress block)
                                - complete: includes result metadata and download URL

    GET  /download/{job_id}     Streams the finished .docx report as a file download.

    GET  /jobs                  Returns a list of all jobs in the in-memory store.

Threading model:
    LangGraph's graph.invoke() is synchronous and blocking. To avoid blocking
    FastAPI's async event loop, all graph calls are dispatched to a
    ThreadPoolExecutor via asyncio.get_running_loop().run_in_executor().
    FastAPI's BackgroundTasks handles the fire-and-forget lifecycle.
"""

from __future__ import annotations
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.config import MAX_CONCURRENT_JOBS, MAX_HITL_REFINE_ROUNDS
from app.api.schemas import (
    ResearchRequest, ApprovalRequest,
    ResearchResponse, ApprovalResponse, JobStatusResponse,
    JobStatus, HITLDecision
)
from app.utils.job_store import job_store
from app.graph.runner import run_research_job, resume_research_job

router = APIRouter()

# Thread pool for running synchronous LangGraph calls without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)


# ── Background Task Helpers ───────────────────────────────────────────────────

async def _run_in_background(job_id: str, query: str, depth: str, output_folder, max_iterations):
    """Runs run_research_job() in a thread so it doesn't block FastAPI's event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, run_research_job, job_id, query, depth, output_folder, max_iterations)


async def _resume_in_background(job_id: str, decision_data: dict):
    """Runs resume_research_job() in a thread after a human HITL decision is received."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, resume_research_job, job_id, decision_data)


# ── Route Handlers ────────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchResponse, status_code=202)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submit a new research query.

    Creates a job, starts the LangGraph pipeline in the background, and returns
    immediately with a job_id. The graph will pause at the HITL checkpoint once
    planning is complete — poll /status/{job_id} to detect when approval is needed.

    Args:
        request: ResearchRequest containing query, depth, and optional overrides.

    Returns:
        ResearchResponse (202): job_id, initial status, and poll URL.
    """
    # Register the job in the store before dispatching so /status is immediately valid
    job_id = job_store.create_job(
        query=request.query,
        depth=request.depth.value,
        output_folder=request.output_folder,
        max_iterations=request.max_iterations,
    )

    # Dispatch the graph to a background thread — it will run until the HITL interrupt
    background_tasks.add_task(
        _run_in_background,
        job_id, request.query, request.depth.value,
        request.output_folder, request.max_iterations,
    )

    return ResearchResponse(
        job_id=job_id,
        status=JobStatus.queued,
        poll_url=f"/status/{job_id}",
    )


@router.post("/approve/{job_id}", response_model=ApprovalResponse)
async def approve_research(job_id: str, request: ApprovalRequest, background_tasks: BackgroundTasks):
    """
    Submit the human decision on the research plan.

    Valid only when job status is 'awaiting_approval'. Three outcomes:
        approved  — graph resumes, sub-agents start searching
        refine    — feedback is sent to LeadResearcher, graph replans (max 3 rounds)
        rejected  — job is cancelled, graph is not resumed

    Args:
        job_id:  ID of the job paused at the HITL checkpoint.
        request: ApprovalRequest with decision and optional feedback.

    Returns:
        ApprovalResponse: confirmed decision and new job status.

    Raises:
        404: Job not found.
        409: Job is not currently awaiting approval.
        400: 'refine' decision submitted without feedback.
    """
    # Validate job exists and is in the correct state
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.awaiting_approval.value:
        raise HTTPException(status_code=409, detail=f"Job is not awaiting approval (status: {job['status']})")
    if request.decision == HITLDecision.refine and not request.feedback:
        raise HTTPException(status_code=400, detail="feedback is required when decision is 'refine'")

    # Rejected: cancel the job without resuming the graph
    if request.decision == HITLDecision.rejected:
        job_store.update_job(job_id, status="cancelled")
        return ApprovalResponse(
            job_id=job_id,
            decision="rejected",
            status=JobStatus.cancelled,
            hitl_round=job.get("hitl_round", 0),
        )

    # Approved or refine: pass decision data back into the graph via Command(resume=...)
    decision_data = {"decision": request.decision, "feedback": request.feedback or ""}
    background_tasks.add_task(_resume_in_background, job_id, decision_data)
    job_store.update_job(job_id, status="running")

    return ApprovalResponse(
        job_id=job_id,
        decision=request.decision,
        status=JobStatus.running,
        hitl_round=job.get("hitl_round", 0),
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """
    Get the current status of a research job.

    The response shape changes depending on the job's current status:
        - awaiting_approval: includes the full research plan in the 'hitl' block
        - running/planning:  includes live progress in the 'progress' block
        - complete:          includes result metadata and download URL in 'result'
        - failed:            includes error message in 'error'

    Args:
        job_id: ID returned from POST /research.

    Returns:
        JobStatusResponse: current status with conditionally populated blocks.

    Raises:
        404: Job not found.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatusResponse(
        job_id=job_id,
        status=JobStatus(job["status"]),
        error=job.get("error"),
    )

    # Populate HITL block when the graph is paused waiting for human approval
    if job["status"] == "awaiting_approval" and job.get("research_plan"):
        from app.api.schemas import HITLInfo, ResearchPlanPayload, SubTopicInfo
        plan = job["research_plan"]
        sub_topics = [SubTopicInfo(**t) for t in plan.get("sub_topics", [])]
        response.hitl = HITLInfo(
            research_plan=ResearchPlanPayload(
                interpreted_goal=plan.get("interpreted_goal", ""),
                sub_topics=sub_topics,
                sub_agent_count=plan.get("sub_agent_count", 1),
                depth=plan.get("depth", "moderate"),
                estimated_tokens=plan.get("estimated_tokens", 0),
                assumptions=plan.get("assumptions", []),
            ),
            approve_url=f"/approve/{job_id}",
            hitl_round=job.get("hitl_round", 0),
            max_refine_rounds=MAX_HITL_REFINE_ROUNDS,
        )

    # Populate progress block while the job is actively running
    if job["status"] in ("running", "planning", "awaiting_approval"):
        from app.api.schemas import ProgressInfo
        response.progress = ProgressInfo(
            phase=job.get("phase", "planning"),
            iterations_completed=job.get("iteration_count", 0),
            sub_agents_active=job.get("sub_agents_active", 0),
            synthesis_review_count=job.get("synthesis_review_count", 0),
        )

    # Populate result block once the report is fully generated
    if job["status"] == "complete" and job.get("document_path"):
        from app.api.schemas import ResultInfo
        response.result = ResultInfo(
            file_path=job["document_path"],
            download_url=f"/download/{job_id}",
            metadata={
                "query": job["query"],
                "duration_seconds": job.get("duration_seconds", 0),
                "token_usage": job.get("token_usage", {}),
                "source_count": job.get("source_count", 0),
                "citation_count": job.get("citation_count", 0),
                "sub_agent_count": job.get("sub_agent_count", 0),
                "iterations": job.get("iteration_count", 0),
            },
            summary_snippet=job.get("summary_snippet", ""),
        )

    return response


@router.get("/download/{job_id}")
async def download_report(job_id: str):
    """
    Download the finished .docx research report.

    Streams the file directly as a binary response. Only available once
    the job status is 'complete'.

    Args:
        job_id: ID of the completed job.

    Returns:
        FileResponse: .docx binary stream with appropriate content-type header.

    Raises:
        404: Job not found or report file missing from disk.
        409: Job not yet complete.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Report not ready yet")

    doc_path = job.get("document_path")
    if not doc_path or not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        doc_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="report.docx",
    )


@router.get("/jobs")
async def list_jobs():
    """
    List all research jobs in the in-memory store.

    Returns all jobs regardless of status. Useful for dashboards and debugging.
    Note: this store is reset on server restart — use a persistent store in production.

    Returns:
        list[dict]: All job records with their current state.
    """
    return job_store.list_jobs()
