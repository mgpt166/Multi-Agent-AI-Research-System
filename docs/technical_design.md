# Technical Design Document
## Multi-Agent AI Research System

| Field | Value |
|-------|-------|
| **Project** | Multi-Agent AI Research System |
| **Version** | 0.2.0 (Phase 1+2+3) |
| **Date** | 2026-03-29 |
| **Status** | Active |
| **HLD Reference** | [IDEAS.md](../IDEAS.md) — high-level architecture, vision, and design rationale |

> This document is a developer quick-reference for implementation details. For "why was it built this way" decisions, read IDEAS.md. For "how to run it", read README.md.

---

## Table of Contents

1. [Tech Stack & Dependencies](#1-tech-stack--dependencies)
2. [Project Structure](#2-project-structure)
3. [State Schema](#3-state-schema)
4. [Agent & Node Definitions](#4-agent--node-definitions)
5. [Graph / Workflow Architecture](#5-graph--workflow-architecture)
6. [Data Flow Walkthrough](#6-data-flow-walkthrough)
7. [Tools & External Integrations](#7-tools--external-integrations)
8. [Configuration & Environment](#8-configuration--environment)
9. [Entry Points & How to Run](#9-entry-points--how-to-run)
10. [Error Handling & Observability](#10-error-handling--observability)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Quick Reference Cheat Sheet](#12-quick-reference-cheat-sheet)
13. [Web UI Design](#13-web-ui-design)
14. [Job Persistence (SQLite)](#14-job-persistence-sqlite)
15. [Real-time Progress (Gradio Polling)](#15-real-time-progress-gradio-polling)

---

## 1. Tech Stack & Dependencies

### Language & Runtime
- **Python 3.11+** — required for `TypedDict` with `Annotated` reducers, `match` syntax compatibility, and modern `asyncio` patterns

### Core Frameworks

| Framework | Version | Why it's used |
|-----------|---------|---------------|
| **LangGraph** | `>=0.2.50` | Provides `StateGraph` with built-in `interrupt()` for HITL checkpointing, `MemorySaver` for state persistence across HTTP requests, and conditional edge routing. Chosen over a plain Python loop because HITL requires persisting mid-execution state — LangGraph handles this natively. |
| **FastAPI** | `>=0.115.0` | Async HTTP framework. Used for the API layer. `BackgroundTasks` handles fire-and-forget graph execution without blocking the event loop. Pydantic v2 integration provides automatic request/response validation. |
| **Groq SDK** | `>=0.13.0` | Python SDK for Groq's OpenAI-compatible inference API. Used for all LLM calls via `groq_chat()` wrapper in `app/utils/groq_retry.py`, which adds rate-limit retry, per-call throttling, and LLM call stats tracking. |

### LLM Models

| Model | Model ID | Used by | Why |
|-------|----------|---------|-----|
| Llama 3.3 70B Versatile | `llama-3.3-70b-versatile` | LeadResearcher, CitationAgent | Best reasoning available on Groq free tier. Used for planning, synthesis, self-review, and citation. |
> **Decision Log**: All agents use `llama-3.3-70b-versatile`. The 8b model was tested for sub-agents but caused hallucinated URLs and fabricated sources. Both models are configurable via `GROQ_MODEL` and `GROQ_SUB_AGENT_MODEL` env vars.

### Supporting Libraries

| Library | Version | Why it's used |
|---------|---------|---------------|
| **tavily-python** | `>=0.3.0` | Default search provider. AI-optimised search API, returns clean content for LLMs. No HTML parsing needed. Free tier: 1000 searches/month. |
| **python-docx** | `>=1.1.2` | Generates `.docx` Word reports. No LLM needed — pure deterministic formatter. Chosen over PDF because `.docx` is editable post-generation. |
| **uvicorn[standard]** | `>=0.32.0` | ASGI server for FastAPI. `[standard]` includes `watchfiles` for `--reload` in dev. |
| **pydantic** | `>=2.9.0` | Request/response schema validation. v2 used for performance and `model_config` syntax. |
| **python-dotenv** | `>=1.0.0` | Loads `.env` into `os.environ` at startup before any agent initialises its API client. |

### Search Provider Abstraction

The system uses a provider-neutral search layer (`app/tools/`) so the web search backend can be swapped by changing one environment variable. The sub-agent runs a **manual tool-use loop** — our code executes searches and returns results as `tool_result` blocks — which means both the search provider and the LLM provider are independently swappable.

```
SEARCH_PROVIDER=tavily  →  TavilySearchProvider
SEARCH_PROVIDER=bing    →  BingSearchProvider     (future)
SEARCH_PROVIDER=brave   →  BraveSearchProvider    (future)
```

### Supporting Libraries (Phase 2 additions)

| Library / Module | Version | Why it's used |
|-----------------|---------|---------------|
| **sqlite3** | built-in | Replaces in-memory `JobStore` dict with a durable SQLite database. No install required — part of Python's standard library. Database file: `data/jobs.db`. |
| **gradio** | `>=4.0.0` | Provides the browser-based Web UI. Mounted directly onto the FastAPI app via `gr.mount_gradio_app(app, demo, path="/ui")`. Serves its own assets — no static files folder needed. Chosen for its professional built-in components (`gr.Textbox`, `gr.Dropdown`, `gr.Dataframe`, `gr.File`, `gr.Timer`), native FastAPI integration, and built-in polling support that replaces a custom SSE endpoint. |

### Supporting Libraries (Phase 3 additions)

| Library / Module | Version | Why it's used |
|-----------------|---------|---------------|
| **groq** | `>=0.13.0` | Groq SDK. Provides the `Groq` client used by all agents. All calls route through `app/utils/groq_retry.py` which wraps the SDK with retry, throttling, and call-stats tracking. |

### Infrastructure (MVP)
- **No database** — in-memory `JobStore` (dict + `threading.RLock`)
- **No Redis / queue** — `ThreadPoolExecutor` for background job execution
- **No Docker** — runs directly with `python -m app.main`
- **No vector DB** — web search only, no document embedding

⚠️ All in-memory state is lost on server restart. See [Section 11](#11-key-design-decisions) for production upgrade path. See [Section 14](#14-job-persistence-sqlite) for the Phase 2 SQLite upgrade.

> **Note on Phase 2 SSE removal:** The original design included a custom SSE endpoint (`app/api/sse.py`) and per-job `queue.Queue` infrastructure for real-time progress. This was replaced by Gradio's built-in polling (`gr.Timer`) which achieves the same result with zero additional server infrastructure. See [Section 15](#15-real-time-progress-gradio-polling).

---

## 2. Project Structure

```
Multi-Agent-AI-Research-System/
│
├── README.md                        # User-facing: setup, quick start, API reference
├── IDEAS.md                         # Architecture source of truth (HLD)
├── docs/
│   └── technical_design.md          # ← You are here
├── requirements.txt                 # Python package dependencies
├── .env.example                     # Environment variable template
│
├── data/                            # Persistent data (Phase 2 — git-ignored except .gitkeep)
│   └── jobs.db                      # SQLite database — durable job records
│
├── output/                          # Generated .docx reports (git-ignored except .gitkeep)
│   └── {YYYYMMDD_HHMMSS}_{slug}/
│       └── report.docx
│
└── app/
    ├── __init__.py
    ├── main.py                      # FastAPI app factory + uvicorn entrypoint
    │
    ├── api/
    │   ├── __init__.py
    │   ├── schemas.py               # Pydantic request/response models + enums
    │   └── routes.py                # HTTP endpoint handlers (POST/GET/DELETE)
    │
    ├── db/                          # Phase 2: SQLite persistence layer
    │   ├── __init__.py
    │   ├── database.py              # Connection factory, schema init (CREATE TABLE IF NOT EXISTS)
    │   └── models.py                # Row-to-dict helpers, typed job record accessors
    │
    ├── graph/
    │   ├── __init__.py
    │   ├── state.py                 # ResearchState TypedDict — shared pipeline state
    │   ├── graph.py                 # StateGraph assembly — nodes, edges, compile
    │   ├── nodes.py                 # All 12 node functions + 3 routing functions
    │   └── runner.py                # run_research_job() + resume_research_job()
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── lead_researcher.py       # Opus: plan / evaluate / synthesize / review
    │   ├── sub_agent.py             # 70b: web search agentic loop
    │   ├── citation_agent.py        # [CITE: url] → [N] + bibliography
    │   └── document_generator.py   # python-docx .docx report writer
    │
    ├── ui/
    │   └── gradio_app.py            # Phase 2: Gradio UI — four tabs, mounted onto FastAPI at /ui
    │
    └── utils/
        ├── __init__.py
        ├── job_store.py             # Job state interface — wraps SQLite DB (Phase 2+)
        ├── groq_retry.py            # Phase 3: Groq call wrapper — retry, throttle, LLM call stats
        ├── tracer.py                # Phase 3: Per-job request tracer (spans, events, timeline)
        ├── cost_tracker.py          # Phase 3: Per-job cost tracker (token → USD by model)
        └── metrics.py               # Phase 3: Process-level metrics (uptime, request counts)
```

**Phase 2 new endpoints (additions to existing 5):**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/ui` (Gradio mount) | Gradio Web UI — mounted via `gr.mount_gradio_app(app, demo, path="/ui")` |

---

## 3. State Schema

**File:** [app/graph/state.py](../app/graph/state.py)

`ResearchState` is a `TypedDict` that acts as the shared whiteboard for the entire pipeline. Every node reads from it and returns a partial dict of updates.

```python
class ResearchState(TypedDict):
    # ── Job Metadata (set at start, never mutated) ─────────────────────────
    job_id: str
    query: str
    depth: str                         # "simple" | "moderate" | "deep"
    output_folder: Optional[str]
    max_iterations: int
    start_time: str                    # ISO format UTC timestamp

    # ── HITL-1 (Plan Review Checkpoint) ────────────────────────────────────
    hitl_status: str                   # "awaiting_approval"|"approved"|"rejected"|"refining"
    hitl_feedback: str                 # Human's refinement text, injected into next plan()
    hitl_round: int                    # Increments on each refine; capped at MAX_HITL_REFINE_ROUNDS

    # ── Planning ───────────────────────────────────────────────────────────
    research_plan: Optional[dict]      # Full plan JSON from LeadResearcher.plan()
    sub_agent_tasks: list[dict]        # Task briefs for each sub-agent

    # ── Research Execution ─────────────────────────────────────────────────
    sub_agent_results: Annotated[list[dict], operator.add]  # Fan-in reducer: APPENDS
    accumulated_findings: str          # Concatenated sub-agent summaries across iterations
    source_map: dict                   # {url: {title, date, relevance_score}}
    iteration_count: int               # Research rounds completed
    sufficiency_signal: str            # "needs_more" | "sufficient" | "force_stop"

    # ── Synthesis Review (Automated) ───────────────────────────────────────
    synthesized_narrative: str         # Output of LeadResearcher.synthesize()
    synthesis_review_count: int        # Self-review rounds (0–3)
    synthesis_review_signal: str       # "approved" | "needs_rework" | "force_proceed"
    synthesis_rework_instructions: list[dict]  # [{task_id, instruction}]

    # ── Citation ───────────────────────────────────────────────────────────
    annotated_narrative: str           # synthesized_narrative with [N] replacing [CITE: url]
    citation_map: dict                 # {url: citation_number}
    bibliography: list[dict]           # [{number, url, title, date}]

    # ── Document ───────────────────────────────────────────────────────────
    document_path: Optional[str]       # Absolute path to saved .docx file

    # ── Metrics ────────────────────────────────────────────────────────────
    token_usage: dict                  # Accumulated {"lead": N, "sub_agents": N}
    error_log: Annotated[list[dict], operator.add]  # Accumulates across all nodes
    final_response: Optional[dict]     # Assembled at respond node
```

### State Field Ownership

| Field | Written by | Read by |
|-------|-----------|---------|
| `job_id`, `query`, `depth` | `runner.py` (init) | All nodes |
| `research_plan`, `sub_agent_tasks` | `plan_research` | `await_human_approval`, `synthesize`, `review_synthesis` |
| `hitl_status`, `hitl_feedback`, `hitl_round` | `await_human_approval` | `_route_after_hitl`, `plan_research` |
| `sub_agent_results` | `spawn_subagents`, `targeted_rework` (appended) | `collect_results`, `evaluate_sufficiency`, `synthesize`, `review_synthesis` |
| `accumulated_findings` | `collect_results`, `targeted_rework` | `evaluate_sufficiency`, `synthesize` |
| `source_map` | `spawn_subagents`, `targeted_rework` | `synthesize`, `cite`, `generate_document` |
| `synthesized_narrative` | `synthesize` | `review_synthesis`, `cite` |
| `synthesis_review_signal`, `synthesis_rework_instructions` | `review_synthesis` | `_route_after_review`, `targeted_rework`, `synthesize` |
| `annotated_narrative`, `bibliography` | `cite` | `generate_document` |
| `document_path` | `generate_document` | `save_to_project`, `respond` |
| `token_usage` | Every agent node (merged) | `respond` |

### Reducer Note
`sub_agent_results` and `error_log` use `Annotated[list, operator.add]`. This means when `spawn_subagents` returns `{"sub_agent_results": [result1, result2]}`, LangGraph **appends** to the existing list rather than replacing it. This is essential for the parallel fan-in pattern — without it, only one sub-agent's result would survive.

---

## 4. Agent & Node Definitions

### Node 1: `plan_research`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/lead_researcher.py](../app/agents/lead_researcher.py)

| | |
|-|-|
| **Responsibility** | LeadResearcher decomposes the query into sub-topics and writes task briefs |
| **Reads** | `query`, `depth`, `hitl_feedback`, `hitl_round` |
| **Writes** | `research_plan`, `sub_agent_tasks`, `token_usage` |
| **Model** | Claude Opus 4.6 with `thinking={"type": "adaptive"}` |
| **Error handling** | `_parse_json()` falls back to brace extraction if Claude wraps JSON in markdown |

**Key logic:**
```python
def plan_research(state: ResearchState) -> dict:
    plan_result = _lead.plan(
        query=state["query"],
        depth=state["depth"],
        hitl_feedback=state.get("hitl_feedback", ""),
        hitl_round=state.get("hitl_round", 0),
    )
    return {
        "research_plan": plan_result["plan"],
        "sub_agent_tasks": plan_result["tasks"],
        "token_usage": _merge_tokens(state.get("token_usage", {}), plan_result.get("token_usage", {})),
    }
```

---

### Node 2: `await_human_approval` (HITL Interrupt)
**File:** [app/graph/nodes.py](../app/graph/nodes.py)

| | |
|-|-|
| **Responsibility** | Pause the graph; surface the research plan to the human via API |
| **Reads** | `research_plan`, `hitl_round` |
| **Writes** | `hitl_status`, `hitl_feedback`, `hitl_round` |
| **Tools** | LangGraph `interrupt()` — serialises state to MemorySaver |
| **Error handling** | Defaults to `"approved"` if decision missing from resume payload |

**Key logic:**
```python
def await_human_approval(state: ResearchState) -> dict:
    # Graph freezes here; resumes when POST /approve sends Command(resume=decision_data)
    human_decision: dict = interrupt({
        "research_plan": plan,
        "hitl_round": state.get("hitl_round", 0),
    })
    decision = human_decision.get("decision", "approved")
    # Returns hitl_status which _route_after_hitl reads to decide next node
```

**Routing function:**
```python
def _route_after_hitl(state) -> str:
    status = state.get("hitl_status", "approved")
    if status == "approved":   return "spawn_subagents"
    elif status == "rejected": return "respond"
    else:  # refining
        if state.get("hitl_round", 0) >= MAX_HITL_REFINE_ROUNDS:
            return "spawn_subagents"   # force-proceed after 3 refines
        return "plan_research"
```

---

### Node 3: `spawn_subagents`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/sub_agent.py](../app/agents/sub_agent.py)

| | |
|-|-|
| **Responsibility** | Dispatch 1–N sub-agents in parallel; collect structured findings |
| **Reads** | `sub_agent_tasks`, `source_map` |
| **Writes** | `sub_agent_results` (appended), `source_map`, `token_usage` |
| **Tools** | `web_search_20260209`, `web_fetch_20260209` (via sub-agents) |
| **Parallelism** | `ThreadPoolExecutor` — one thread per sub-agent, `timeout=120s` |
| **Error handling** | Failed sub-agent futures produce a stub result with `confidence=0.0` — pipeline continues |

**Key logic:**
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
    futures = {pool.submit(_sub_agent.execute, task): task for task in tasks}
    for fut in concurrent.futures.as_completed(futures, timeout=120):
        try:
            results.append(fut.result())
        except Exception as exc:
            results.append({"summary": f"[Sub-agent failed: {exc}]", "confidence": 0.0, ...})
```

---

### Node 4: `collect_results`
**File:** [app/graph/nodes.py](../app/graph/nodes.py)

| | |
|-|-|
| **Responsibility** | Aggregate all sub-agent summaries into `accumulated_findings`; increment iteration count |
| **Reads** | `sub_agent_results`, `accumulated_findings` |
| **Writes** | `accumulated_findings`, `iteration_count` |
| **Tools** | None |

Concatenates each sub-agent's `summary` field under a `### Sub-topic:` heading. Appends to any previous iteration's findings so the research loop builds incrementally.

---

### Node 5: `evaluate_sufficiency`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/lead_researcher.py](../app/agents/lead_researcher.py)

| | |
|-|-|
| **Responsibility** | LeadResearcher decides if findings are adequate or need another research round |
| **Reads** | `query`, `accumulated_findings`, `sub_agent_results`, `iteration_count`, `max_iterations` |
| **Writes** | `sufficiency_signal` |
| **Model** | Claude Opus 4.6 (no thinking — short, low-stakes call, `max_tokens=50`) |
| **Force-stop** | Returns `"force_stop"` immediately if `iteration_count >= max_iterations` |

**Routing function:**
```python
def _route_after_sufficiency(state) -> str:
    sig = state.get("sufficiency_signal", "sufficient")
    if sig == "needs_more": return "spawn_subagents"
    return "synthesize"   # covers "sufficient" and "force_stop"
```

---

### Node 6: `synthesize`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/lead_researcher.py](../app/agents/lead_researcher.py)

| | |
|-|-|
| **Responsibility** | LeadResearcher weaves all findings into a structured narrative with `[CITE: url]` markers |
| **Reads** | `query`, `accumulated_findings`, `sub_agent_results`, `sub_agent_tasks`, `source_map`, `synthesis_rework_instructions` |
| **Writes** | `synthesized_narrative`, `token_usage` |
| **Model** | Claude Opus 4.6 with `thinking={"type": "adaptive"}`, `max_tokens=8000` |

Rework instructions (if present from a prior `review_synthesis` cycle) are injected as a `## REWORK INSTRUCTIONS` section in the prompt, directing Opus to address specific gaps.

---

### Node 7: `review_synthesis`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/lead_researcher.py](../app/agents/lead_researcher.py)

| | |
|-|-|
| **Responsibility** | Self-review synthesis against task assignment map; issue targeted rework instructions if needed |
| **Reads** | `query`, `synthesized_narrative`, `sub_agent_tasks`, `sub_agent_results`, `synthesis_review_count` |
| **Writes** | `synthesis_review_signal`, `synthesis_rework_instructions`, `synthesis_review_count`, `token_usage` |
| **Model** | Claude Opus 4.6 (no thinking), `max_tokens=1000` |
| **Hard cap** | If `synthesis_review_count >= MAX_SYNTHESIS_REVIEW_ROUNDS`, returns `"force_proceed"` without calling LLM |

**Routing function:**
```python
def _route_after_review(state) -> str:
    sig = state.get("synthesis_review_signal", "approved")
    if sig == "needs_rework" and state.get("synthesis_review_count", 0) < MAX_ROUNDS:
        return "targeted_rework"
    return "cite"   # approved OR force_proceed OR cap reached
```

---

### Node 8: `targeted_rework`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/sub_agent.py](../app/agents/sub_agent.py)

| | |
|-|-|
| **Responsibility** | Re-run specific sub-agents with gap-filling instructions; minimal new searches |
| **Reads** | `synthesis_rework_instructions`, `sub_agent_tasks`, `sub_agent_results`, `source_map` |
| **Writes** | `sub_agent_results` (appended), `accumulated_findings`, `source_map`, `token_usage` |
| **Tools** | Same web tools as `spawn_subagents`; stopping criteria reduced to 3–5 additional searches |
| **Parallelism** | `ThreadPoolExecutor`, `timeout=90s` |

Builds a modified task dict injecting `rework_instruction` and hints about already-gathered sources so the sub-agent focuses on gaps, not re-research.

---

### Node 9: `cite`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/citation_agent.py](../app/agents/citation_agent.py)

| | |
|-|-|
| **Responsibility** | Convert `[CITE: url]` markers to `[N]` numbered citations; build bibliography |
| **Reads** | `synthesized_narrative`, `source_map` |
| **Writes** | `annotated_narrative`, `citation_map`, `bibliography`, `token_usage` |
| **LLM fallback** | Claude Sonnet 4.6 only if regex pass finds < 3 citations |

---

### Node 10: `generate_document`
**File:** [app/graph/nodes.py](../app/graph/nodes.py) → calls [app/agents/document_generator.py](../app/agents/document_generator.py)

| | |
|-|-|
| **Responsibility** | Produce the `.docx` report from annotated narrative + bibliography |
| **Reads** | `query`, `annotated_narrative`, `bibliography`, `synthesis_review_signal`, `synthesis_review_count`, `source_map`, `depth`, `iteration_count` |
| **Writes** | `document_path` |
| **LLM** | None — deterministic python-docx formatter |
| **Output path** | `{OUTPUT_DIR}/{YYYYMMDD_HHMMSS}_{query_slug}/report.docx` |

---

### Node 11: `save_to_project`
**File:** [app/graph/nodes.py](../app/graph/nodes.py)

Thin node. Updates `job_store` with `document_path` and `source_count`. Returns empty dict (no state changes).

---

### Node 12: `respond`
**File:** [app/graph/nodes.py](../app/graph/nodes.py)

Assembles the `final_response` dict and marks the job `complete` (or `cancelled` if rejected). Calculates `duration_seconds` from `start_time`. Updates `job_store` with all completion metadata.

---

## 5. Graph / Workflow Architecture

**File:** [app/graph/graph.py](../app/graph/graph.py)

### Mermaid Diagram

```mermaid
graph TD
    START([START]) --> plan_research

    plan_research --> await_human_approval

    await_human_approval -->|approved| spawn_subagents
    await_human_approval -->|refine, round lt 3| plan_research
    await_human_approval -->|refine, round ge 3| spawn_subagents
    await_human_approval -->|rejected| respond

    spawn_subagents --> collect_results
    collect_results --> evaluate_sufficiency

    evaluate_sufficiency -->|needs_more| spawn_subagents
    evaluate_sufficiency -->|sufficient / force_stop| synthesize

    synthesize --> review_synthesis

    review_synthesis -->|needs_rework, count lt 3| targeted_rework
    review_synthesis -->|approved / force_proceed| cite

    targeted_rework --> synthesize

    cite --> generate_document
    generate_document --> save_to_project
    save_to_project --> respond
    respond --> END([END])
```

### Graph Builder Code (Annotated)

```python
# app/graph/graph.py

checkpointer = MemorySaver()   # persists state across HITL interrupt

def build_graph():
    builder = StateGraph(ResearchState)

    # Register all 12 nodes
    builder.add_node("plan_research", plan_research)
    builder.add_node("await_human_approval", await_human_approval)
    # ... (all 12 nodes)

    # Linear entry
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "await_human_approval")

    # HITL branch — routing function returns "spawn_subagents" | "plan_research" | "respond"
    builder.add_conditional_edges("await_human_approval", _route_after_hitl, {...})

    # Research loop — routing returns "spawn_subagents" | "synthesize"
    builder.add_edge("spawn_subagents", "collect_results")
    builder.add_edge("collect_results", "evaluate_sufficiency")
    builder.add_conditional_edges("evaluate_sufficiency", _route_after_sufficiency, {...})

    # Synthesis review loop — routing returns "targeted_rework" | "cite"
    builder.add_edge("synthesize", "review_synthesis")
    builder.add_conditional_edges("review_synthesis", _route_after_review, {...})
    builder.add_edge("targeted_rework", "synthesize")   # loop back

    # Linear exit
    builder.add_edge("cite", "generate_document")
    builder.add_edge("generate_document", "save_to_project")
    builder.add_edge("save_to_project", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)

research_graph = build_graph()  # singleton used by runner.py
```

### Loop Caps (Preventing Runaway Execution)

| Loop | Cap | Enforced in |
|------|-----|-------------|
| HITL refine | `MAX_HITL_REFINE_ROUNDS` (default 3) | `_route_after_hitl` |
| Research iterations | `MAX_ITERATIONS` (default 3) | `evaluate_sufficiency` node |
| Synthesis review | `MAX_SYNTHESIS_REVIEW_ROUNDS` (default 1) | `review_synthesis` node |

---

## 6. Data Flow Walkthrough

### Scenario A: Happy Path (Approved, Sufficient on First Round)

**What the user does:** Sends a research question. Reviews and approves the plan. Downloads the finished report.

---

#### STEP 1 — User submits the research query

```
User → POST /research
Body: {"query": "Impact of AI on healthcare 2025", "depth": "moderate"}
```

**How `routes.py` and `runner.py` connect — in plain English:**

Think of `routes.py` as the **reception desk** and `runner.py` as the **back office**.

When a request arrives:
1. The reception desk (`routes.py`) takes the request, registers the job, and immediately tells the user *"we got it, here's your job ID"* — it does NOT wait for the research to finish
2. It hands the actual work to a **background thread** (like passing a ticket to the back office)
3. That background thread calls `runner.py`, which starts the LangGraph pipeline
4. The user polls `/status` separately to check progress

This separation exists because research takes 3–8 minutes. If `routes.py` waited for it to finish, the HTTP request would time out. So the API responds instantly (202) and the work happens independently in the background.

```
routes.py                          background thread
─────────────────────────────      ──────────────────────────────────────
start_research()                   _run_in_background()
  │                                  │
  ├─ job_store.create_job()          └─ runner.run_research_job()
  │   → job registered in memory           │
  │                                        └─ research_graph.invoke()
  ├─ background_tasks.add_task() ──────────►     (LangGraph pipeline starts)
  │   → hands work to thread pool
  │
  └─ return 202 immediately
      → user gets job_id NOW
        (research still running in background)
```

| What happens | File | Function called |
|---|---|---|
| FastAPI receives and validates the request body | `app/api/routes.py` | `start_research()` |
| New job registered: status=`"queued"`, all fields initialised | `app/utils/job_store.py` | `job_store.create_job()` |
| Work handed to background thread (non-blocking) | `app/api/routes.py` | `background_tasks.add_task(_run_in_background, ...)` |
| HTTP 202 returned to user **immediately** — research not started yet | `app/api/routes.py` | returns `ResearchResponse` |
| Background thread executes `_run_in_background()` which directly calls `run_research_job()` in `runner.py` via `run_in_executor` | `app/api/routes.py` → `app/graph/runner.py` | `_run_in_background()` calls `run_research_job()` |
| Runner builds the initial state and starts the LangGraph pipeline | `app/graph/runner.py` | `run_research_job()` → `research_graph.invoke(initial_state, config)` |

**User gets back instantly:**
```json
{"job_id": "abc-123", "status": "queued", "poll_url": "/status/abc-123"}
```
> The research is now running in the background. The user polls `/status/abc-123` every few seconds to check progress.

---

#### STEP 2 — LeadResearcher plans the research

```
Graph node: plan_research   (running in background thread)
```

| What happens | File | Function called |
|---|---|---|
| Node reads `query` and `depth` from state | `app/graph/nodes.py` | `plan_research()` |
| Calls Claude Opus with adaptive thinking | `app/agents/lead_researcher.py` | `LeadResearcher.plan()` |
| Opus decomposes query into 2 sub-topics (moderate depth = 2 agents) | Anthropic API | `claude-opus-4-6` |
| JSON plan parsed from Opus response | `app/agents/lead_researcher.py` | `_parse_json()` |
| State updated: `research_plan`, `sub_agent_tasks` written | `app/graph/nodes.py` | `plan_research()` returns dict |

---

#### STEP 3 — Graph pauses for human approval (HITL)

```
Graph node: await_human_approval   ← graph FREEZES here
```

| What happens | File | Function called |
|---|---|---|
| Node calls `interrupt()` — graph execution suspends | `app/graph/nodes.py` | `await_human_approval()` |
| Full graph state is saved to memory | `app/graph/graph.py` | `MemorySaver` checkpointer |
| Job status updated to `"awaiting_approval"` | `app/utils/job_store.py` | `job_store.update_job()` |

**User polls and sees the plan:**
```
GET /status/abc-123
→ routes.py: get_status()
→ job_store.get_job()
→ Returns: {"status": "awaiting_approval", "hitl": {"research_plan": {...}}}
```

The `hitl.research_plan` block shows the user exactly what Opus understood and how it plans to research it — before any expensive work begins.

---

#### STEP 4 — User approves the plan

```
User → POST /approve/abc-123
Body: {"decision": "approved"}
```

| What happens | File | Function called |
|---|---|---|
| FastAPI validates the decision | `app/api/routes.py` | `approve_research()` |
| Graph resume dispatched to background thread | `app/api/routes.py` | `_resume_in_background()` |
| Graph resumes from saved checkpoint with decision injected | `app/graph/runner.py` | `resume_research_job()` → `graph.invoke(Command(resume=...))` |
| `await_human_approval` node reads `"approved"` decision | `app/graph/nodes.py` | `await_human_approval()` resumes |
| Routing function sends flow to `spawn_subagents` | `app/graph/nodes.py` | `_route_after_hitl()` returns `"spawn_subagents"` |

---

#### STEP 5 — Sub-agents search the web in parallel

```
Graph node: spawn_subagents
```

| What happens | File | Function called |
|---|---|---|
| Node reads the 2 task briefs from state | `app/graph/nodes.py` | `spawn_subagents()` |
| 2 threads launched simultaneously (one per sub-topic) | `app/graph/nodes.py` | `ThreadPoolExecutor` |
| **Thread 1:** Sub-agent searches for Sub-topic 1 | `app/agents/sub_agent.py` | `ResearchSubAgent.execute(task_1)` |
| **Thread 2:** Sub-agent searches for Sub-topic 2 | `app/agents/sub_agent.py` | `ResearchSubAgent.execute(task_2)` |
| Claude Sonnet runs tool-use loop: requests searches | Anthropic API | `claude-sonnet-4-6` |
| Our code executes each search via Tavily | `app/tools/tavily_provider.py` | `TavilySearchProvider.search()` |
| Search results returned to Claude as `tool_result` blocks | `app/agents/sub_agent.py` | `_execute_tool()` |
| Loop repeats until Claude calls `end_turn` | `app/agents/sub_agent.py` | agentic loop in `execute()` |
| Both threads complete; results fan-in to state | `app/graph/state.py` | `operator.add` reducer appends both results |

---

#### STEP 6 — Results collected and evaluated

```
Graph nodes: collect_results → evaluate_sufficiency
```

| What happens | File | Function called |
|---|---|---|
| Sub-agent summaries merged into `accumulated_findings` | `app/graph/nodes.py` | `collect_results()` |
| Iteration count incremented to 1 | `app/utils/job_store.py` | `job_store.update_job(iteration_count=1)` |
| Opus reviews findings: are they sufficient? | `app/agents/lead_researcher.py` | `LeadResearcher.evaluate_sufficiency()` |
| Average confidence 0.75 → returns `"sufficient"` | Anthropic API | `claude-opus-4-6` (50 tokens, no thinking) |
| Routing function sends flow to `synthesize` | `app/graph/nodes.py` | `_route_after_sufficiency()` returns `"synthesize"` |

---

#### STEP 7 — LeadResearcher synthesises findings

```
Graph node: synthesize
```

| What happens | File | Function called |
|---|---|---|
| Node reads all findings, source map, task list | `app/graph/nodes.py` | `synthesize()` |
| Opus with adaptive thinking writes the full narrative | `app/agents/lead_researcher.py` | `LeadResearcher.synthesize()` |
| Every factual claim tagged `[CITE: https://...]` inline | Anthropic API | `claude-opus-4-6` (up to 8000 tokens) |
| `synthesized_narrative` written to state | `app/graph/nodes.py` | returns `{"synthesized_narrative": "..."}` |

---

#### STEP 8 — LeadResearcher self-reviews the synthesis

```
Graph node: review_synthesis
```

| What happens | File | Function called |
|---|---|---|
| Opus checks: did every sub-topic get covered? | `app/agents/lead_researcher.py` | `LeadResearcher.review_synthesis()` |
| Returns `"approved"` — no gaps found | Anthropic API | `claude-opus-4-6` |
| Routing function sends flow to `cite` | `app/graph/nodes.py` | `_route_after_review()` returns `"cite"` |

---

#### STEP 9 — Citations added

```
Graph node: cite
```

| What happens | File | Function called |
|---|---|---|
| Regex scans narrative for `[CITE: url]` markers | `app/agents/citation_agent.py` | `CitationAgent.annotate()` |
| Each URL assigned a number: `[CITE: url]` → `[1]`, `[2]`... | `app/agents/citation_agent.py` | `assign_citation()` |
| Bibliography built: `[{number, url, title, date}]` | `app/agents/citation_agent.py` | `annotate()` |
| `annotated_narrative` + `bibliography` written to state | `app/graph/nodes.py` | `cite()` |

---

#### STEP 10 — Word document generated

```
Graph node: generate_document
```

| What happens | File | Function called |
|---|---|---|
| Output folder created: `output/20260328_143022_impact_of_ai/` | `app/graph/nodes.py` | `generate_document()` |
| 5-section `.docx` written to disk | `app/agents/document_generator.py` | `DocumentGenerator.generate()` |
| Citation numbers `[N]` rendered in blue bold | `app/agents/document_generator.py` | `_add_inline_runs()` |
| Bibliography appended as numbered list | `app/agents/document_generator.py` | `generate()` |
| `document_path` written to state | `app/graph/nodes.py` | `generate_document()` returns `{"document_path": "..."}` |

---

#### STEP 11 — Job marked complete

```
Graph nodes: save_to_project → respond → END
```

| What happens | File | Function called |
|---|---|---|
| Job store updated with file path and source count | `app/utils/job_store.py` | `job_store.update_job(document_path=...)` |
| Final response assembled (duration, token usage, snippet) | `app/graph/nodes.py` | `respond()` |
| Job status set to `"complete"` | `app/utils/job_store.py` | `job_store.update_job(status="complete")` |
| Graph execution ends | `app/graph/graph.py` | reaches `END` node |

---

#### STEP 12 — User downloads the report

```
User → GET /download/abc-123
```

| What happens | File | Function called |
|---|---|---|
| Route handler looks up job in store | `app/api/routes.py` | `download_report()` |
| Verifies status is `"complete"` and file exists on disk | `app/api/routes.py` | `download_report()` |
| Streams `.docx` file as binary download | `app/api/routes.py` | returns `FileResponse(doc_path)` |

**User gets:** `report.docx` — a 5-section Word document with inline citations and bibliography.

---

### Scenario B: Refine Then Approve, With Synthesis Rework

```
User: POST /research {"query": "EV market Europe", "depth": "deep"}
```

| Step | Node | Key State Change |
|------|------|-----------------|
| 1–3 | Same as above | Graph pauses at HITL |
| 4 | Human: `POST /approve {"decision":"refine","feedback":"Focus on Germany and France only"}` | `hitl_status = "refining"`, `hitl_feedback = "Focus on Germany..."`, `hitl_round = 1` |
| 5 | `plan_research` | Opus replans with feedback injected → new `research_plan` |
| 6 | `await_human_approval` | Graph pauses again |
| 7 | Human: `POST /approve {"decision":"approved"}` | Resumes → `spawn_subagents` |
| 8–10 | Research + collect + evaluate | `sufficiency_signal = "sufficient"` |
| 11 | `synthesize` | Opus synthesis with [CITE:] markers |
| 12 | `review_synthesis` | Opus: `"needs_rework"` — Sub-topic 2 coverage thin → `synthesis_review_count = 1` |
| 13 | `targeted_rework` | Sub-agent 2 re-runs with gap instruction, 3 new searches |
| 14 | `synthesize` | Re-synthesises with rework results incorporated |
| 15 | `review_synthesis` | Opus: `"approved"` → routes to `cite` |
| 16–end | cite → document → respond | `job_store.status = "complete"` |

---

## 7. Tools & External Integrations

### Search Provider Abstraction Layer

The system uses a **provider-neutral search interface** so the underlying search service can be swapped without touching agent code.

| File | Purpose |
|------|---------|
| [app/tools/base.py](../app/tools/base.py) | `SearchProvider` abstract base class + `SearchResult` dataclass |
| [app/tools/tavily_provider.py](../app/tools/tavily_provider.py) | Tavily implementation (current default) |
| [app/tools/factory.py](../app/tools/factory.py) | Reads `SEARCH_PROVIDER` env var, lazy-imports and returns the right provider |

### SearchResult — Normalised Data Model

All providers return this shape regardless of their underlying API response format:

```python
@dataclass
class SearchResult:
    url: str
    title: str
    content: str               # snippet or extracted body
    published_date: Optional[str] = None
    score: Optional[float] = None
```

### SearchProvider Interface

```python
class SearchProvider(ABC):
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...
    def fetch(self, url: str) -> str: ...
    def format_search_results(self, results: list[SearchResult]) -> str: ...  # shared utility
```

### Tavily Provider

| | |
|-|-|
| **File** | [app/tools/tavily_provider.py](../app/tools/tavily_provider.py) |
| **Auth** | `TAVILY_API_KEY` env var. Free tier: 1000 searches/month at https://tavily.com |
| **search()** | Calls `TavilyClient.search(query, search_depth, max_results)` |
| **fetch()** | Calls `TavilyClient.extract(urls=[url])` → falls back to `urllib` if extract fails |
| **Search depth** | Controlled by `TAVILY_SEARCH_DEPTH` env var: `"basic"` (default) or `"advanced"` |
| **Rate limits** | Subject to Tavily plan limits. No retry logic in MVP. ⚠️ |
| **Error handling** | `search()` returns `[]` on any exception — pipeline continues with partial results |

### Manual Tool-Use Loop in SubAgent

The sub-agent does **not** use Anthropic server-side tools. It runs a standard tool-use loop where our code executes the tools and passes results back to Claude:

```python
# app/agents/sub_agent.py
for _ in range(max_tool_rounds):          # hard cap: 10 rounds
    response = _client.messages.create(
        tools=_TOOL_DEFINITIONS,          # standard JSON Schema — not provider-specific
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        break                             # Claude is satisfied — done

    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # OUR code calls the provider — not Anthropic's servers
                result = _execute_tool(block.name, block.input, all_sources)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

**Why this pattern is provider-neutral:**
- `_TOOL_DEFINITIONS` uses standard JSON Schema — works with Claude, GPT-4, Mistral, Gemini
- Search execution is isolated in `_execute_tool()` which calls `SearchProvider`
- Swapping LLM: change `_SONNET` model string and client
- Swapping search: change `SEARCH_PROVIDER` env var

### Adding a New Search Provider

1. Create `app/tools/{name}_provider.py`:
```python
from app.tools.base import SearchProvider, SearchResult

class BingSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        # call Bing API, map response to list[SearchResult]
        ...
    def fetch(self, url: str) -> str:
        # fetch URL content
        ...
```

2. Register in [app/tools/factory.py](../app/tools/factory.py):
```python
_PROVIDERS = {
    "tavily": "app.tools.tavily_provider.TavilySearchProvider",
    "bing":   "app.tools.bing_provider.BingSearchProvider",   # add this line
}
```

3. Set in `.env`: `SEARCH_PROVIDER=bing`

---

## 8. Configuration & Environment

**File:** [.env.example](../.env.example)

### Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `GROQ_API_KEY` | — | **Yes** | Groq API key. Required for all LLM calls. Free tier: 6,000 TPM, 100k TPD. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | No | Model for LeadResearcher and CitationAgent. |
| `GROQ_SUB_AGENT_MODEL` | `llama-3.3-70b-versatile` | No | Model for ResearchSubAgent. Same as lead — required for reliable tool use. |
| `GROQ_CALL_INTERVAL` | `3` | No | Minimum seconds between consecutive Groq calls. Set to `0` on a paid tier. Prevents 429s. |
| `TAVILY_API_KEY` | — | **Yes** | Tavily search API key. Used by `TavilySearchProvider`. Free tier: 1000 searches/month. |
| `OUTPUT_DIR` | `./output` | No | Directory where `.docx` reports are saved. Created on startup if missing. |
| `MAX_CONCURRENT_JOBS` | `3` | No | Max simultaneous graph executions in the ThreadPoolExecutor. |
| `MAX_SUBAGENTS` | `2` | No | Maximum sub-agents per job. Actual count is `min(tasks, MAX_SUBAGENTS)`. |
| `MAX_ITERATIONS` | `3` | No | Hard cap on research loop iterations before forcing synthesis. |
| `MAX_SYNTHESIS_REVIEW_ROUNDS` | `1` | No | Max self-review rounds before force-proceeding to citation. Default 1 = one review pass, then proceed. |
| `MAX_HITL_REFINE_ROUNDS` | `3` | No | Max times a human can request plan refinement before force-proceeding to research. |
| `MAX_COST_PER_QUERY` | `0.10` | No | Per-query budget cap in USD. Triggers 🔴 in Observability UI when exceeded. |
| `COST_WARNING_THRESHOLD` | `0.05` | No | Warning threshold in USD. Triggers ⚠️ in Observability UI. |
| `HITL_TIMEOUT_SECONDS` | `180` | No | Seconds before an awaiting-approval job is auto-approved. |

### Where Variables Are Read

| Variable | Read in |
|----------|---------|
| `GROQ_API_KEY` | `app/agents/lead_researcher.py`, `sub_agent.py`, `citation_agent.py` — all at module import |
| `GROQ_MODEL` | `app/config.py` → `app/agents/lead_researcher.py`, `citation_agent.py`, `runner.py` |
| `GROQ_SUB_AGENT_MODEL` | `app/config.py` → `app/agents/sub_agent.py`, `runner.py` |
| `GROQ_CALL_INTERVAL` | `app/utils/groq_retry.py` — controls throttle sleep between calls |
| `OUTPUT_DIR` | `app/main.py` (startup), `app/graph/nodes.py` (generate_document) |
| `MAX_SUBAGENTS` | `app/graph/nodes.py` (spawn_subagents) |
| `MAX_ITERATIONS` | `app/graph/nodes.py` (evaluate_sufficiency), `app/graph/runner.py` |
| `MAX_SYNTHESIS_REVIEW_ROUNDS` | `app/graph/nodes.py` (review_synthesis, _route_after_review) |
| `MAX_HITL_REFINE_ROUNDS` | `app/graph/nodes.py` (_route_after_hitl), `app/api/routes.py` |
| `MAX_CONCURRENT_JOBS` | `app/api/routes.py` (ThreadPoolExecutor) |
| `MAX_COST_PER_QUERY`, `COST_WARNING_THRESHOLD` | `app/graph/runner.py`, `app/ui/gradio_app.py` |

### Prompt Templates

Prompts are **inline strings** inside each agent method — there are no external prompt template files. Locations:

| Prompt | Location |
|--------|----------|
| Planning prompt | `LeadResearcher.plan()` — [lead_researcher.py:28](../app/agents/lead_researcher.py) |
| Sufficiency evaluation | `LeadResearcher.evaluate_sufficiency()` — [lead_researcher.py:110](../app/agents/lead_researcher.py) |
| Synthesis prompt | `LeadResearcher.synthesize()` — [lead_researcher.py:163](../app/agents/lead_researcher.py) |
| Self-review prompt | `LeadResearcher.review_synthesis()` — [lead_researcher.py:219](../app/agents/lead_researcher.py) |
| Sub-agent system prompt | `ResearchSubAgent.execute()` — [sub_agent.py:34](../app/agents/sub_agent.py) |
| Citation LLM fallback | `CitationAgent._llm_annotate()` — [citation_agent.py:73](../app/agents/citation_agent.py) |

### Model Parameters

| Agent | Model (env var) | max_tokens | Notes |
|-------|-----------------|-----------|-------|
| `LeadResearcher.plan()` | `GROQ_MODEL` (70b) | 4000 | JSON plan output |
| `LeadResearcher.evaluate_sufficiency()` | `GROQ_MODEL` (70b) | 200 | Outputs one signal word |
| `LeadResearcher.synthesize()` | `GROQ_MODEL` (70b) | 8000 | Narrative synthesis |
| `LeadResearcher.review_synthesis()` | `GROQ_MODEL` (70b) | 1000 | JSON review result |
| `ResearchSubAgent.execute()` | `GROQ_SUB_AGENT_MODEL` (8b) | 4000 | Agentic tool loop |
| `CitationAgent._llm_annotate()` | `GROQ_MODEL` (70b) | 8000 | Citation annotation |

All calls go through `groq_chat()` in `app/utils/groq_retry.py` which throttles to one call per `GROQ_CALL_INTERVAL` seconds and retries up to 5 times on rate-limit errors with exponential backoff (10s, 20s, 40s, 80s, 160s).

---

## 9. Entry Points & How to Run

### Start the Server

```bash
# Development (with auto-reload)
python -m app.main

# Production-style
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

⚠️ Use `--workers 1` only. Multiple workers do not share the in-memory `JobStore` or `MemorySaver`.

### Entry Function Chain

```
python -m app.main
  → app/main.py: if __name__ == "__main__": uvicorn.run("app.main:app", ...)
  → create_app() → FastAPI instance with router mounted
  → startup() event → creates ./output directory

POST /research
  → routes.start_research()
  → job_store.create_job()
  → background_tasks.add_task(_run_in_background, ...)
  → runner.run_research_job()
  → research_graph.invoke(initial_state, config)
  → [graph executes nodes until HITL interrupt]

POST /approve/{job_id}
  → routes.approve_research()
  → background_tasks.add_task(_resume_in_background, ...)
  → runner.resume_research_job()
  → research_graph.invoke(Command(resume=decision_data), config)
  → [graph continues from interrupt]
```

### Interactive API Docs

FastAPI auto-generates Swagger UI at: `http://localhost:8000/docs`
All endpoints, schemas, and example payloads are browsable and executable there.

---

## 10. Error Handling & Observability

### Error Handling Patterns

| Location | Pattern |
|----------|---------|
| `runner.run_research_job()` | `try/except` wraps entire `graph.invoke()` — on exception, sets `job_store.status = "failed"` and `error = str(exc)`, then re-raises |
| `runner.resume_research_job()` | Same pattern |
| `spawn_subagents` node | Individual sub-agent futures caught; failed agents produce a stub result with `confidence=0.0` and `summary="[Sub-agent failed: ...]"`. Pipeline continues with partial results. |
| `targeted_rework` node | Same stub pattern — rework failure is silently skipped (the original results remain) |
| `_parse_json()` in lead_researcher.py | Tries direct JSON parse → markdown fence extraction → brace-boundary extraction → returns `{}` as final fallback |
| `GET /approve/{id}` | 404 if job unknown, 409 if not in `awaiting_approval` state, 400 if `refine` without feedback |

### Logging
⚠️ **No structured logging is implemented in the MVP.** All output goes to stdout via Python's default print/exception behaviour. For production, add:
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
```

### Observability (Phase 3)

A full observability system is implemented and visible in the **🔭 Observability** tab of the Gradio UI (three sub-tabs):

**Sub-tab A: System Dashboard** (auto-refreshes every 5s)
- Last-24h aggregate stats from SQLite (`job_traces` + `jobs` tables)
- Two side-by-side tables: **Requests & Performance** and **Cost & Token Health**
- ✅/⚠️/🔴 status icons with legend

**Sub-tab B: Job Inspector** (per completed job)
- Dropdown of 10 most recent traced jobs (job ID + query preview + date)
- Manual UUID entry for older jobs
- **Budget summary bar**: allocated / used / remaining / status
- **Pipeline Health table** (11 columns): Step · Order · Duration · Time% · Tokens · Cost · Cost% · Total LLM Calls · Retries · Failed · Health
- Health icon per step: ✅ = clean · ⚠️ = retries (recovered) · 🔴 = failures

**Sub-tab C: Activity Log**
- Per-job event feed with timestamps, compact header showing Job ID + query preview

**Implementation files:**

| File | Responsibility |
|------|---------------|
| `app/utils/groq_retry.py` | Thread-local call context (`set_trace_context`/`clear_trace_context`); per-job, per-step call stats registry (`_call_stats`); `init_job_stats`, `get_job_stats`, `clear_job_stats` |
| `app/utils/tracer.py` | `RequestTracer` — spans with start/end/duration; `tracer_registry` singleton keyed by job_id |
| `app/utils/cost_tracker.py` | `CostTracker` — token → USD conversion by model using prefix-matched pricing table |
| `app/utils/metrics.py` | Process-level counters (uptime, total requests, failures) served via `GET /metrics` |
| `app/db/database.py` | `job_traces` table: `job_id`, `query`, `trace_summary`, `timeline`, `cost_summary`, `call_stats`, `saved_at` |
| `app/utils/job_store.py` | `save_job_trace()`, `get_job_trace()`, `list_recent_job_traces()`, `get_dashboard_stats()` |
| `app/graph/runner.py` | Calls `init_job_stats()` on start; populates cost_tracker from `token_usage`; persists trace+cost+call_stats to SQLite on completion |
| `app/graph/nodes.py` | `set_trace_context(job_id, step)` + `clear_trace_context()` wrapping each LLM call; `tracer.start_span()`/`end_span()` for pipeline timing |
| `app/agents/sub_agent.py` | Sets thread-local context to `"spawn_subagents"` inside `execute()` — each sub-agent thread attributes its calls correctly |

### Retry Logic

**Implemented in `app/utils/groq_retry.py`** — all Groq API calls route through `groq_chat()`:

```
Throttle: sleep(max(0, GROQ_CALL_INTERVAL - elapsed)) before each call
Retry: up to 5 attempts on 429/RateLimitError
Backoff: 10s → 20s → 40s → 80s → 160s (base 10, doubles per attempt)
Other errors: re-raised immediately (no retry)
Stats: every call increments _call_stats[job_id][step]["calls"]
       every retry increments ["retries"]
       every permanent failure increments ["failed"]
```

---

## 11. Key Design Decisions

| Decision | Why | Tradeoff |
|----------|-----|----------|
| **MemorySaver for HITL state** | Allows `graph.invoke()` to be called twice (start + resume) for the same job while preserving full graph state between calls | State lost on server restart. Production fix: `PostgresSaver`. |
| **ThreadPoolExecutor for sub-agents** | Anthropic SDK is synchronous. Threads give true parallelism without rewriting SDK calls | Not ideal at high concurrency. Alternative: `asyncio` with an async Anthropic client. |
| **Inline prompts (no template files)** | Faster development; prompts are always co-located with the code that uses them | Harder to A/B test prompts without code changes. Future: extract to `config/prompts.py`. |
| **Annotated `operator.add` for sub_agent_results** | Enables parallel sub-agent fan-in — multiple threads return results that all survive the merge | Results accumulate across rework rounds too, so `sub_agent_results` grows with each review cycle. |
| **Citation before document generation** | Avoids running CitationAgent on poor synthesis that gets reworked. Only cited once, on final approved synthesis. | Synthesis review must complete before any document work begins. |
| **Sonnet for CitationAgent** | Citation is a mechanical cross-reference task — no deep reasoning required. Sonnet is sufficient and 3-5x cheaper. | If Opus synthesis uses unusual citation formats, Sonnet may miss them. |
| **No frontend in MVP** | Scope decision — API-first allows any frontend to integrate | Requires curl or Postman to operate. |
| **`operator.add` reducer not used for `source_map`** | Source map is a dict (not list), so standard replace is correct. Sub-agents merge their sources explicitly in the node. | Manual merge logic needed in `spawn_subagents` and `targeted_rework`. |

---

## 12. Quick Reference Cheat Sheet

### "I want to change X" → "Edit this"

| Goal | File | Function / Section |
|------|------|--------------------|
| Change the planning prompt | `app/agents/lead_researcher.py` | `LeadResearcher.plan()` — the `prompt` string |
| Change the synthesis prompt | `app/agents/lead_researcher.py` | `LeadResearcher.synthesize()` |
| Change the self-review criteria | `app/agents/lead_researcher.py` | `LeadResearcher.review_synthesis()` |
| Change sub-agent search behaviour | `app/agents/sub_agent.py` | `ResearchSubAgent.execute()` — system prompt + user_prompt |
| Swap search provider (e.g. to Bing) | `.env` + `app/tools/factory.py` | Set `SEARCH_PROVIDER=bing`; add provider class to factory registry |
| Add a new search provider | `app/tools/{name}_provider.py` | Subclass `SearchProvider`, implement `search()` and `fetch()` |
| Change Tavily search depth | `.env` | `TAVILY_SEARCH_DEPTH=advanced` |
| Add a new research depth level | `app/agents/lead_researcher.py` | `depth_map` dict; also add to `ResearchDepth` enum in `schemas.py` |
| Change max sub-agents | `.env` | `MAX_SUBAGENTS=N` |
| Change report section structure | `app/agents/document_generator.py` | `DocumentGenerator.generate()` |
| Add a new pipeline node | `app/graph/nodes.py` + `app/graph/graph.py` | Write node fn; add `add_node` + `add_edge` |
| Add a new state field | `app/graph/state.py` | Add field to `ResearchState`; initialise in `runner.py` |
| Change routing logic | `app/graph/nodes.py` | `_route_after_hitl`, `_route_after_sufficiency`, `_route_after_review` |
| Add a new API endpoint | `app/api/routes.py` + `app/api/schemas.py` | Add route fn + Pydantic models |
| Swap to persistent job storage | `app/utils/job_store.py` | Replace `dict` with Redis/DB client |
| Swap to persistent graph state | `app/graph/graph.py` | Replace `MemorySaver()` with `PostgresSaver` |
| Add structured logging | All agent files | Add `logger = logging.getLogger(__name__)` at module level |
| Change LLM model (orchestrator) | `.env` | Set `GROQ_MODEL=<model-id>` |
| Change LLM model (sub-agents) | `.env` | Set `GROQ_SUB_AGENT_MODEL=<model-id>` |
| Disable call throttling (paid tier) | `.env` | Set `GROQ_CALL_INTERVAL=0` |
| View pipeline health + LLM call stats | `http://localhost:8000/ui` | Observability tab → Job Inspector |
| View 24h system dashboard | `http://localhost:8000/ui` | Observability tab → System Dashboard |

### Job Lifecycle at a Glance

```
queued → planning → awaiting_approval → running → complete
                                               ↘ failed
                         ↘ cancelled (rejected)
```

### Key Singleton Objects

| Object | Defined in | Imported by |
|--------|-----------|-------------|
| `research_graph` | `app/graph/graph.py` | `app/graph/runner.py` |
| `checkpointer` | `app/graph/graph.py` | `app/graph/runner.py` |
| `job_store` | `app/utils/job_store.py` | `app/api/routes.py`, `app/graph/nodes.py`, `app/graph/runner.py` |
| `_executor` (ThreadPoolExecutor) | `app/api/routes.py` | Local to routes module |
| `_lead`, `_sub_agent`, `_citation`, `_doc_gen` | `app/graph/nodes.py` | Local to nodes module |

---

## 13. Web UI Design

**File:** `app/ui/gradio_app.py`

**Mounted by:** `gr.mount_gradio_app(app, demo, path="/ui")` in `app/main.py`, where `app` is the FastAPI application instance and `demo` is the `gr.Blocks` object defined in `gradio_app.py`. Gradio serves its own JavaScript, CSS, and component assets — no `StaticFiles` mount or `app/ui/static/` folder is needed.

### Why Gradio

Gradio is chosen over a hand-written vanilla HTML/CSS/JS single-page application because:

- **Professional components out of the box** — `gr.Textbox`, `gr.Dropdown`, `gr.Dataframe`, `gr.File`, `gr.Markdown`, `gr.Button` provide polished, accessible UI elements with no CSS authoring.
- **Native FastAPI integration** — `gr.mount_gradio_app()` attaches the Gradio app directly to the existing FastAPI instance; no separate server or process.
- **Built-in polling support** — `gr.Timer` fires a Python callback on a configurable interval, driving status refresh without a custom SSE endpoint or browser-side `EventSource` code.
- **Native file download** — `gr.File` handles `.docx` downloads natively; the component streams the file to the browser without a custom `StreamingResponse`.
- **Hours not days** — the full four-tab UI is buildable in a single Python file with no build toolchain, no `npm install`, and no CSS.

### Tab Structure

The UI is a `gr.Blocks` layout with four tabs inside a `gr.TabbedInterface` (or equivalent `gr.Tab` blocks):

#### Tab 1: New Research

| Component | Type | Purpose |
|-----------|------|---------|
| Research query | `gr.Textbox` (multiline) | Free-text research question |
| Depth | `gr.Dropdown` | Options: `simple`, `moderate`, `deep` |
| Submit | `gr.Button` | Calls `POST /research`; displays returned `job_id` |
| Job ID output | `gr.Textbox` (read-only) | Shows the `job_id` for the submitted job |

On Submit click, the handler calls `POST /research` with the query and depth, receives `job_id`, and writes it to the Job ID output component. The user copies this `job_id` to use in subsequent tabs.

#### Tab 2: Job Status

| Component | Type | Purpose |
|-----------|------|---------|
| Job ID input | `gr.Textbox` | User enters the `job_id` to monitor |
| Check Status | `gr.Button` | Manual trigger for a status fetch |
| Status display | `gr.Markdown` | Shows current `status`, `phase`, timestamps, token usage |
| Research plan | `gr.Markdown` | Rendered when `hitl_status == awaiting_approval`; shows plan details |
| Decision buttons | `gr.Button` × 3 | **Approve**, **Refine**, **Reject** — only visible when `awaiting_approval` |
| Feedback | `gr.Textbox` | Free-text feedback field, shown alongside the Refine button |
| Auto-refresh timer | `gr.Timer` | Fires every 5 seconds; calls the same status-fetch handler to update the display automatically |

**Approval flow:** When status is `awaiting_approval`, the research plan is rendered in the plan display component and the three decision buttons become active. Clicking Approve calls `POST /approve/{job_id}` with `{"decision": "approved"}`. Clicking Refine sends `{"decision": "refine", "feedback": <feedback text>}`. Clicking Reject sends `{"decision": "rejected"}`. After any decision, the status display updates on the next timer tick.

#### Tab 3: Job History

| Component | Type | Purpose |
|-----------|------|---------|
| History table | `gr.Dataframe` | Columns: `job_id`, `query`, `status`, `created_at`, `duration_seconds`. Loaded from SQLite via `job_store.list_jobs()`. Shows ALL past jobs ever submitted — data persists across server restarts. |
| Refresh | `gr.Button` | Re-queries SQLite and refreshes the table |

The history table is populated by calling `job_store.list_jobs()` directly from the Gradio handler — no HTTP round-trip needed because the Gradio app runs in the same Python process as the FastAPI backend. All jobs ever submitted are shown, including jobs from previous server runs, because records are durable in `data/jobs.db`.

#### Tab 4: Download Report

| Component | Type | Purpose |
|-----------|------|---------|
| Job ID input | `gr.Textbox` | User enters `job_id` of a completed job |
| Download | `gr.Button` | Fetches the file path from `job_store.get_job()`, returns the `.docx` file |
| File output | `gr.File` | Streams the `.docx` to the browser for download |

The handler looks up the `document_path` field for the given `job_id` in SQLite. If the job is complete and the `.docx` exists in `output/`, it is returned as a `gr.File` value which Gradio serves as a browser download. Users can re-download reports from any previous run as long as the `.docx` file still exists in `output/`.

### How Gradio Mounts onto FastAPI

```python
# app/main.py  (pseudocode — design reference)
from fastapi import FastAPI
import gradio as gr
from app.ui.gradio_app import demo   # gr.Blocks instance

app = FastAPI()

# ... register API routes ...

gr.mount_gradio_app(app, demo, path="/ui")
```

After mounting, the Gradio UI is accessible at `http://localhost:8000/ui`. All existing FastAPI API endpoints (`/research`, `/status/{job_id}`, `/approve/{job_id}`, `/download/{job_id}`) remain unchanged and continue to work for programmatic access.

### API Calls Made by the Gradio Handlers

Gradio handlers are plain Python functions running server-side. They can call the FastAPI endpoints via `httpx` (loopback HTTP) or call the underlying service functions directly (avoiding HTTP overhead). The recommended pattern for MVP is **direct function calls** to `job_store` and `runner` modules:

```
New Research tab   → job_store.create_job() + runner.run_research_job() (via BackgroundTasks)
Job Status tab     → job_store.get_job(job_id) + POST /approve/{job_id} for decisions
Job History tab    → job_store.list_jobs()
Download tab       → job_store.get_job(job_id) → document_path → gr.File(document_path)
```

This avoids a loopback HTTP call and keeps the Gradio handlers simple and testable.

### Styling

No custom CSS is needed. Gradio's default theme provides a professional, clean appearance. If visual customization is desired, `gr.themes.Soft()` or `gr.themes.Monochrome()` can be passed to `gr.Blocks(theme=...)` without writing any CSS.

---

## 14. Job Persistence (SQLite)

**Files:** `app/db/database.py`, `app/db/models.py`, `app/utils/job_store.py`

### Schema

Database file: `data/jobs.db` (created on server startup if it does not exist).

```sql
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT    PRIMARY KEY,
    query            TEXT    NOT NULL,
    depth            TEXT    NOT NULL DEFAULT 'moderate',
    status           TEXT    NOT NULL DEFAULT 'queued',
    phase            TEXT,
    research_plan    TEXT,        -- JSON string (serialized dict or null)
    document_path    TEXT,
    error            TEXT,
    created_at       TEXT    NOT NULL,   -- ISO 8601 UTC string
    updated_at       TEXT    NOT NULL,   -- ISO 8601 UTC string, updated on every write
    token_usage      TEXT,        -- JSON string: {"lead": N, "sub_agents": N, "total": N}
    hitl_status      TEXT,
    hitl_round       INTEGER DEFAULT 0,
    iteration_count  INTEGER DEFAULT 0,
    source_count     INTEGER DEFAULT 0,
    citation_count   INTEGER DEFAULT 0,
    duration_seconds REAL,
    summary_snippet  TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
```

The index on `status` supports `list_jobs(status_filter=...)` queries. The index on `created_at` supports descending-order listing of recent jobs.

### Connection Handling

**File:** `app/db/database.py`

SQLite connections are not thread-safe when shared across threads. The correct pattern for a multi-threaded FastAPI application is to use `check_same_thread=False` (safe because all writes go through `threading.RLock` in `job_store.py`) or to open a new connection per operation.

The recommended pattern for MVP is **connection-per-operation** with a context manager:

```python
# app/db/database.py  (pseudocode — design reference, not implementation)

DB_PATH = os.path.join("data", "jobs.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # enables dict-style column access
    return conn

def init_db():
    """Called once at server startup. Creates tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
```

`sqlite3.Row` row factory allows columns to be accessed by name (`row["job_id"]`) rather than index, which makes `job_store.py` methods readable without a separate mapping step.

### Migration from In-Memory Dict

`job_store.py` is the only file that changes. Its public interface remains identical:

| Method | Signature (unchanged) | Implementation change |
|--------|-----------------------|----------------------|
| `create_job(job_id, query, depth, ...)` | unchanged | `INSERT INTO jobs ...` |
| `get_job(job_id)` | returns `dict \| None` | `SELECT * FROM jobs WHERE job_id = ?` → `dict(row)` |
| `update_job(job_id, **kwargs)` | unchanged | `UPDATE jobs SET col=? ... WHERE job_id = ?` |
| `list_jobs(limit, offset, status_filter)` | unchanged | `SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?` |

**JSON field handling in `job_store.py`:**
- On write: `json.dumps(value)` for `research_plan` and `token_usage` fields before passing to SQLite.
- On read: `json.loads(row["research_plan"])` if the column is not null; otherwise `None`.

This serialization/deserialization is encapsulated in `job_store.py` — callers receive plain Python dicts, never JSON strings.

### Startup Initialization

In `app/main.py`, the `startup` event handler calls `init_db()` before the server begins accepting requests:

```python
# app/main.py  (pseudocode — design reference)
@app.on_event("startup")
async def startup():
    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    init_db()   # creates data/jobs.db and runs CREATE TABLE IF NOT EXISTS
```

### Adding New Columns

Because no migration framework is used, new columns are added via `ALTER TABLE ... ADD COLUMN` statements in `init_db()`:

```sql
-- Safe to run on every startup (SQLite silently ignores it if the column already exists
-- in SQLite 3.37.0+; wrap in try/except for older versions)
ALTER TABLE jobs ADD COLUMN new_field TEXT;
```

This is sufficient for the evolution pace of an internal tool. For a production deployment with schema migration history, `alembic` is the appropriate tool.

### Upgrade Path to PostgreSQL

The upgrade from SQLite to PostgreSQL requires changes in only two files:

1. **`app/db/database.py`**: Replace `sqlite3.connect(DB_PATH)` with `psycopg2.connect(DSN)`. Change `?` placeholders to `%s`. Replace `executescript()` with individual `execute()` calls.
2. **`.env`**: Add `DATABASE_URL=postgresql://user:pass@host:5432/dbname`.

`app/utils/job_store.py`, `app/api/routes.py`, `app/graph/nodes.py`, and all other files are unchanged. The interface contract (`create_job`, `get_job`, `update_job`, `list_jobs`) is the migration path.

---

## 15. Real-time Progress (Gradio Polling)

**File:** `app/ui/gradio_app.py` (Job Status tab, `gr.Timer` component)

### Approach: Polling via gr.Timer

Real-time progress updates are delivered by periodic polling from the Gradio frontend rather than by a server-sent event (SSE) stream. Gradio's `gr.Timer` component fires a Python callback on a configurable interval; the callback calls `job_store.get_job(job_id)` and returns the current job state to update the UI components.

This replaces the original SSE design (a `GET /stream/{job_id}` endpoint backed by per-job `queue.Queue` objects and a `StreamingResponse` generator). The Gradio polling approach is simpler because:

- No dedicated SSE endpoint (`app/api/sse.py`) is needed — no custom server infrastructure.
- No `queue.Queue` per job, no `_event_queues` module-level dict, no `emit_event()` calls in graph nodes.
- No browser-side `EventSource` code — Gradio handles all polling client logic internally.
- The polling callback is a plain Python function that reads directly from `job_store` — fully testable with no HTTP layer.

### How gr.Timer Works in the Job Status Tab

```python
# app/ui/gradio_app.py  (pseudocode — design reference)
with gr.Tab("Job Status"):
    job_id_input = gr.Textbox(label="Job ID")
    status_display = gr.Markdown()
    plan_display = gr.Markdown(visible=False)
    approve_btn = gr.Button("Approve", visible=False)
    refine_btn = gr.Button("Refine", visible=False)
    reject_btn = gr.Button("Reject", visible=False)
    feedback_input = gr.Textbox(label="Refinement feedback", visible=False)

    timer = gr.Timer(value=5)   # fires every 5 seconds

    def refresh_status(job_id):
        if not job_id:
            return gr.update(), gr.update(visible=False), ...
        job = job_store.get_job(job_id)
        if not job:
            return "Job not found.", gr.update(visible=False), ...
        awaiting = job["hitl_status"] == "awaiting_approval"
        status_md = f"**Status:** {job['status']}  |  **Phase:** {job.get('phase', '—')}"
        plan_md = format_plan(job["research_plan"]) if awaiting else ""
        return (
            status_md,
            gr.update(value=plan_md, visible=awaiting),
            gr.update(visible=awaiting),   # approve_btn
            gr.update(visible=awaiting),   # refine_btn
            gr.update(visible=awaiting),   # reject_btn
            gr.update(visible=awaiting),   # feedback_input
        )

    timer.tick(
        fn=refresh_status,
        inputs=[job_id_input],
        outputs=[status_display, plan_display, approve_btn, refine_btn, reject_btn, feedback_input],
    )
```

The `timer.tick` event fires every 5 seconds and calls `refresh_status`, which reads the current job record from SQLite and updates all output components atomically. The timer runs only while the browser tab is active — Gradio pauses timer callbacks automatically when the component is not visible.

### Polling Interval

The default polling interval is **5 seconds**. This is appropriate for a research job that runs for 3–8 minutes. The interval is configurable by changing the `gr.Timer(value=N)` argument in `gradio_app.py`.

A 5-second polling interval means the UI is at most 5 seconds behind the actual job state. This is acceptable for the progress display use case — the user does not need sub-second precision on phase transitions.

### What the UI Shows at Each Phase

| Job status / hitl_status | Status display | Plan display | Decision buttons |
|--------------------------|---------------|--------------|-----------------|
| `queued` | "Queued" | hidden | hidden |
| `running` + any phase | Phase name + timestamp | hidden | hidden |
| `running` + `awaiting_approval` | "Awaiting your approval" | visible (full plan) | visible (Approve / Refine / Reject) |
| `complete` | "Complete" + duration + source count | hidden | hidden |
| `failed` | "Failed" + error message | hidden | hidden |
| `cancelled` | "Cancelled" | hidden | hidden |

### Comparison to SSE Design

| Aspect | SSE (original design) | Gradio polling (current design) |
|--------|-----------------------|--------------------------------|
| Server infrastructure | `app/api/sse.py`, per-job `queue.Queue`, `emit_event()` calls in nodes | None — `job_store.get_job()` only |
| New files | `app/api/sse.py` | None |
| Browser client | `EventSource` + event handler wiring in `app.js` | Built into Gradio (`gr.Timer`) |
| Update latency | Near-real-time (sub-second) | 5 seconds (configurable) |
| Reliability on reconnect | Events missed during disconnect window | No missed events — always reads current DB state |
| Complexity | High (thread/async bridge, queue lifecycle) | Low (read from SQLite on timer tick) |

For a research job measured in minutes, a 5-second polling interval is indistinguishable from real-time in practice. The SSE complexity is not justified at this scale.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_POLL_INTERVAL` | `5` | Seconds between `gr.Timer` ticks in the Job Status tab. Set in `gradio_app.py`. |
