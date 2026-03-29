"""
app/api/schemas.py
==================
Pydantic models for all API request and response payloads.

Defines the contract between the HTTP layer and the rest of the system.
All incoming data is validated against these models before reaching route handlers.
All outgoing data is serialised through these models.

Request models:
    ResearchRequest     — body for POST /research
    ApprovalRequest     — body for POST /approve/{job_id}

Response models:
    ResearchResponse    — 202 reply from POST /research
    ApprovalResponse    — reply from POST /approve/{job_id}
    JobStatusResponse   — reply from GET /status/{job_id} (varies by status)
    ResultInfo          — nested inside JobStatusResponse when job is complete
    ProgressInfo        — nested inside JobStatusResponse while job is running
    HITLInfo            — nested inside JobStatusResponse while awaiting approval

Enums:
    ResearchDepth       — simple | moderate | deep
    HITLDecision        — approved | rejected | refine
    JobStatus           — all possible job lifecycle states
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class ResearchDepth(str, Enum):
    """Controls how many sub-agents are spawned: simple=1, moderate=2, deep=3."""
    simple = "simple"
    moderate = "moderate"
    deep = "deep"


class HITLDecision(str, Enum):
    """Human decision at the HITL checkpoint after plan_research."""
    approved = "approved"   # proceed to sub-agent research
    rejected = "rejected"   # cancel the job entirely
    refine = "refine"       # send feedback back to LeadResearcher for replanning


class JobStatus(str, Enum):
    """All valid states a research job can be in during its lifecycle."""
    queued = "queued"                       # created, not yet started
    planning = "planning"                   # LeadResearcher decomposing query
    awaiting_approval = "awaiting_approval" # paused at HITL interrupt
    running = "running"                     # sub-agents active or synthesizing
    complete = "complete"                   # report ready for download
    failed = "failed"                       # unrecoverable error
    cancelled = "cancelled"                 # human rejected the plan


# ── Request Models ────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """
    Body for POST /research.

    Fields:
        query           Research question to investigate (5–2000 chars).
        depth           Controls number of sub-agents and search breadth.
        output_folder   Override the default ./output directory for this job.
        max_iterations  Override max research rounds (1–5). Defaults to env var.
    """
    query: str = Field(..., min_length=5, max_length=2000)
    depth: ResearchDepth = ResearchDepth.moderate
    output_folder: Optional[str] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=5)


class ApprovalRequest(BaseModel):
    """
    Body for POST /approve/{job_id}.

    Fields:
        decision    Human verdict on the research plan.
        feedback    Required when decision == 'refine'. Sent to LeadResearcher
                    as instructions for revising the plan.
    """
    decision: HITLDecision
    feedback: Optional[str] = Field(None, description="Required when decision == refine")

    # Store enum values as plain strings so comparisons work after deserialisation
    model_config = {"use_enum_values": True}


# ── Nested Response Models ────────────────────────────────────────────────────

class SubTopicInfo(BaseModel):
    """One sub-topic entry inside a ResearchPlanPayload."""
    id: int
    title: str
    scope: str
    assigned_to: str


class ResearchPlanPayload(BaseModel):
    """
    The full research plan produced by LeadResearcher, surfaced to the human
    during the HITL checkpoint so they can decide to approve, refine, or reject.
    """
    interpreted_goal: str
    sub_topics: list[SubTopicInfo]
    sub_agent_count: int
    depth: str
    estimated_tokens: int
    assumptions: list[str]


class HITLInfo(BaseModel):
    """
    Included in JobStatusResponse when status == 'awaiting_approval'.
    Contains the full research plan and the URL to submit a decision.
    """
    research_plan: ResearchPlanPayload
    approve_url: str
    hitl_round: int
    max_refine_rounds: int


class ProgressInfo(BaseModel):
    """
    Included in JobStatusResponse while the job is active (queued/planning/running).
    Gives the caller a live view of pipeline progress without exposing internal state.
    """
    phase: str
    iterations_completed: int
    sub_agents_active: int
    synthesis_review_count: int


class TokenUsage(BaseModel):
    """Breakdown of tokens consumed across agents for cost tracking."""
    total: int = 0
    lead: int = 0
    sub_agents: int = 0
    citation: int = 0
    document: int = 0


class ResultInfo(BaseModel):
    """
    Included in JobStatusResponse when status == 'complete'.
    Contains the download URL and report metadata.
    """
    file_path: str
    download_url: str
    metadata: dict[str, Any]
    summary_snippet: str


# ── Top-level Response Models ─────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """
    Response for GET /status/{job_id}.

    The optional fields are populated based on current status:
        hitl        — present when status == awaiting_approval
        progress    — present when status in (queued, planning, running, awaiting_approval)
        result      — present when status == complete
        error       — present when status == failed
    """
    job_id: str
    status: JobStatus
    hitl: Optional[HITLInfo] = None
    progress: Optional[ProgressInfo] = None
    result: Optional[ResultInfo] = None
    error: Optional[str] = None


class ResearchResponse(BaseModel):
    """Response for POST /research (202 Accepted). Use poll_url to track progress."""
    job_id: str
    status: JobStatus
    poll_url: str


class ApprovalResponse(BaseModel):
    """Response for POST /approve/{job_id}. Confirms the decision and new job status."""
    job_id: str
    decision: str
    status: JobStatus
    hitl_round: int
