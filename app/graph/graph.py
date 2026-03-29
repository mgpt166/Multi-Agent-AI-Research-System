"""
app/graph/graph.py
==================
Assembles and compiles the LangGraph StateGraph for the full research pipeline.

This module is the authoritative definition of the pipeline topology —
which nodes exist, in what order they execute, and what conditional
routing logic governs transitions between them.

Graph topology:
    START
      → plan_research
      → await_human_approval          [HITL interrupt]
          ├── approved  → spawn_subagents
          ├── refine    → plan_research (feedback loop, max 3 rounds)
          └── rejected  → respond
      → spawn_subagents
      → collect_results
      → evaluate_sufficiency
          ├── needs_more  → spawn_subagents (research loop, max MAX_ITERATIONS)
          └── sufficient  → synthesize
      → synthesize
      → review_synthesis
          ├── needs_rework → targeted_rework → synthesize (review loop, max 3 rounds)
          └── approved     → cite
      → cite
      → generate_document
      → save_to_project
      → respond
      → END

Checkpointing:
    MemorySaver persists the full graph state so that when the HITL interrupt
    fires and the HTTP request returns, the state is not lost. The graph is
    resumed later via Command(resume=decision_data) with the same thread_id.
    For production, swap MemorySaver for PostgresSaver.

Exports:
    research_graph  — compiled graph instance (singleton) used by runner.py
    checkpointer    — exposed so runner.py can use the same checkpointer instance
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import ResearchState
from app.graph.nodes import (
    plan_research,
    await_human_approval, _route_after_hitl,
    spawn_subagents,
    collect_results,
    evaluate_sufficiency, _route_after_sufficiency,
    synthesize,
    review_synthesis, _route_after_review,
    targeted_rework,
    cite,
    generate_document,
    save_to_project,
    respond,
)

# In-memory checkpointer — persists graph state across HITL interrupts within a session
checkpointer = MemorySaver()


def build_graph() -> StateGraph:
    """
    Construct and compile the full research pipeline as a LangGraph StateGraph.

    Returns:
        Compiled StateGraph with MemorySaver checkpointer attached.
    """
    builder = StateGraph(ResearchState)

    # ── Register all pipeline nodes ────────────────────────────────
    builder.add_node("plan_research", plan_research)
    builder.add_node("await_human_approval", await_human_approval)
    builder.add_node("spawn_subagents", spawn_subagents)
    builder.add_node("collect_results", collect_results)
    builder.add_node("evaluate_sufficiency", evaluate_sufficiency)
    builder.add_node("synthesize", synthesize)
    builder.add_node("review_synthesis", review_synthesis)
    builder.add_node("targeted_rework", targeted_rework)
    builder.add_node("cite", cite)
    builder.add_node("generate_document", generate_document)
    builder.add_node("save_to_project", save_to_project)
    builder.add_node("respond", respond)

    # ── Linear entry: planning always runs first ───────────────────
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "await_human_approval")

    # ── HITL branch: approved → research, refine → replan, rejected → cancel
    builder.add_conditional_edges(
        "await_human_approval",
        _route_after_hitl,
        {
            "plan_research": "plan_research",
            "spawn_subagents": "spawn_subagents",
            "respond": "respond",
        },
    )

    # ── Research loop: collect → evaluate → [more research or synthesize]
    builder.add_edge("spawn_subagents", "collect_results")
    builder.add_edge("collect_results", "evaluate_sufficiency")
    builder.add_conditional_edges(
        "evaluate_sufficiency",
        _route_after_sufficiency,
        {
            "spawn_subagents": "spawn_subagents",
            "synthesize": "synthesize",
        },
    )

    # ── Synthesis review loop: synthesize → review → [rework or cite]
    builder.add_edge("synthesize", "review_synthesis")
    builder.add_conditional_edges(
        "review_synthesis",
        _route_after_review,
        {
            "targeted_rework": "targeted_rework",
            "cite": "cite",
        },
    )
    builder.add_edge("targeted_rework", "synthesize")

    # ── Linear exit: citation → document → save → respond → done
    builder.add_edge("cite", "generate_document")
    builder.add_edge("generate_document", "save_to_project")
    builder.add_edge("save_to_project", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)


# Compiled graph singleton — imported by runner.py to invoke and resume jobs
research_graph = build_graph()
