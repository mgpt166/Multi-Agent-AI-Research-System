"""
app/graph/state.py
==================
Defines ResearchState — the shared state TypedDict that travels through the
entire LangGraph pipeline from the first node to the last.

Every node function receives the full state and returns a partial dict of
fields it wants to update. LangGraph merges the returned dict into the state
using each field's reducer (default: replace; Annotated with operator.add: append).

Field groups:
    Job metadata        — immutable inputs set at job start (job_id, query, depth)
    HITL-1 (approval)   — tracks the human plan-review checkpoint state
    Planning            — the research plan and sub-agent task briefs from LeadResearcher
    Research execution  — sub-agent results, aggregated findings, iteration tracking
    Synthesis review    — automated self-review loop state (count, signal, rework instructions)
    Citation            — annotated narrative, citation map, bibliography
    Document            — output file path
    Metrics             — token usage, error log, timing
    Final response      — assembled at the respond node for job completion

Reducer notes:
    sub_agent_results   — Annotated with operator.add so parallel sub-agent results
                          from spawn_subagents fan-in by appending, not overwriting.
    error_log           — Same pattern: errors from any node accumulate without loss.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Annotated
import operator


class ResearchState(TypedDict):
    # ── Job metadata ──────────────────────────────────────────────
    job_id: str
    query: str
    depth: str                        # simple | moderate | deep
    output_folder: Optional[str]
    max_iterations: int
    start_time: str

    # ── HITL-1 (Plan Review) ──────────────────────────────────────
    hitl_status: str                  # awaiting_approval | approved | rejected | refining
    hitl_feedback: str
    hitl_round: int

    # ── Planning ──────────────────────────────────────────────────
    research_plan: Optional[dict]     # full plan from LeadResearcher
    sub_agent_tasks: list[dict]       # task descriptions per sub-agent

    # ── Research execution ────────────────────────────────────────
    sub_agent_results: Annotated[list[dict], operator.add]
    accumulated_findings: str
    source_map: dict                  # url -> {title, date, relevance_score}
    iteration_count: int
    sufficiency_signal: str           # needs_more | sufficient | force_stop

    # ── Synthesis review (automated) ─────────────────────────────
    synthesized_narrative: str
    synthesis_review_count: int
    synthesis_review_signal: str      # approved | needs_rework | force_proceed
    synthesis_rework_instructions: list[dict]

    # ── Citation ─────────────────────────────────────────────────
    annotated_narrative: str
    citation_map: dict
    bibliography: list[dict]

    # ── Document ─────────────────────────────────────────────────
    document_path: Optional[str]

    # ── Metrics ──────────────────────────────────────────────────
    token_usage: dict
    error_log: Annotated[list[dict], operator.add]

    # ── Final response ───────────────────────────────────────────
    final_response: Optional[dict]
