# Multi-Agent AI Research System

> A production-grade multi-agent AI research pipeline that turns a research question into a professionally cited `.docx` report — automatically.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-FF6B35?style=flat)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-built--in-003B57?style=flat&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What It Does

You submit a research question. The system autonomously:

1. **Plans** — A LeadResearcher agent decomposes your query into parallel sub-topics and writes a detailed research plan
2. **Reviews with you** — You approve, refine, or reject the plan before any expensive work begins *(Human-in-the-Loop)*
3. **Researches in parallel** — Multiple ResearchSubAgents independently search the web, each covering a different sub-topic simultaneously
4. **Synthesizes** — The LeadResearcher weaves all findings into a coherent narrative and self-reviews for coverage gaps
5. **Cites** — A CitationAgent converts raw source URLs into numbered inline citations `[1]`, `[2]`...
6. **Delivers** — A DocumentGenerator produces a clean `.docx` report with 5 structured sections and a full bibliography

The output resembles what a skilled analyst would produce after a full day of research — not a chatbot response.

---

## Demo

```
POST /research  →  {"query": "Compare LangGraph and CrewAI for multi-agent systems", "depth": "moderate"}

  [Planning]     LeadResearcher decomposes into 2 sub-topics
  [HITL]         You review the plan → ✅ Approve
  [Researching]  2 sub-agents run in parallel (12 web searches, 8 sources)
  [Synthesizing] LeadResearcher synthesizes findings (1 self-review pass)
  [Citing]       CitationAgent adds [1]–[8] inline citations
  [Generating]   python-docx writes the .docx report

GET /download/{job_id}  →  report.docx  (5 sections, 8 citations, ~4 min total)
```

---

## Architecture

### Pipeline Flow

```
POST /research
      │
      ▼
┌─────────────────┐
│  plan_research  │  LeadResearcher (70b) decomposes query → sub-topics + task briefs
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ await_human_approval │◄── HITL: review plan → approved / refine (feedback) / rejected
└────────┬─────────────┘    (refine loops back; hard cap: MAX_HITL_REFINE_ROUNDS)
         │ approved
         ▼
┌─────────────────┐
│ spawn_subagents │  1–3 sub-agents (8b) run in parallel threads
└────────┬────────┘  each: web_search → web_fetch → structured findings
         ▼
┌──────────────────┐
│ collect_results  │  Fan-in: aggregates all sub-agent findings
└────────┬─────────┘
         ▼
┌──────────────────────┐
│ evaluate_sufficiency │  Enough data? → sufficient / needs_more / force_stop
└────────┬─────────────┘  (needs_more loops back to spawn_subagents)
         ▼
┌────────────┐
│ synthesize │  LeadResearcher (70b) writes unified narrative with [CITE: url] markers
└─────┬──────┘
      ▼
┌──────────────────┐
│ review_synthesis │  Self-review: coverage gaps? → approved / needs_rework / force_proceed
└────────┬─────────┘  (needs_rework → targeted_rework → re-synthesize; cap: MAX_SYNTHESIS_REVIEW_ROUNDS)
         ▼
┌──────┐
│ cite │  CitationAgent: [CITE: url] → [N] inline citations + deduplicated bibliography
└──┬───┘
   ▼
┌───────────────────┐
│ generate_document │  python-docx → 5-section .docx report
└────────┬──────────┘
         ▼
┌─────────┐
│ respond │  Job marked complete → GET /download/{job_id}
└─────────┘
```

### Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Workflow orchestration | **LangGraph** `StateGraph` | HITL interrupt, state checkpointing, conditional routing |
| LeadResearcher + CitationAgent | **Groq** `llama-3.3-70b-versatile` | Planning, synthesis, self-review, citation |
| ResearchSubAgents | **Groq** `llama-3.1-8b-instant` | Web search agentic loop (5× higher daily quota) |
| Web search | **Tavily** (swappable) | AI-optimised search, clean LLM-ready content |
| API layer | **FastAPI** + BackgroundTasks | Non-blocking job dispatch, Pydantic validation |
| HITL state | **LangGraph MemorySaver** | Graph state persisted across interrupt/resume |
| Job persistence | **SQLite** `data/jobs.db` | Jobs survive server restarts |
| Web UI | **Gradio** (mounted on FastAPI at `/ui`) | 6-tab browser interface |
| Report output | **python-docx** | Clean `.docx` with inline citations |
| Eval judge | **Anthropic SDK** (configurable) | LLM-as-judge quality scoring |

---

## Web UI

Access the browser UI at `http://localhost:8000/ui` after starting the server.

| Tab | What it does |
|-----|-------------|
| 🔬 **New Research** | Submit a query, choose depth, get a job ID |
| 📊 **Job Status** | Live status + activity feed + HITL approval buttons |
| 📋 **Job History** | Browse all past jobs from SQLite |
| 🧪 **Eval History** | Browse all evaluation run results |
| 📥 **Download Report** | Download `.docx` for any completed job |
| 🔭 **Observability** | Pipeline health, cost, LLM call tracking, activity logs |

### Observability Tab

Three sub-tabs give full pipeline visibility:

**System Dashboard** (auto-refreshes every 5s)
- Last-24h aggregate stats: requests, success rate, avg duration, total cost, budget health
- Two side-by-side tables with ✅ ⚠️ 🔴 status indicators

**Job Inspector** (per completed job)
- Select from 10 most recent traced jobs or paste any job ID
- Budget summary bar: allocated / used / remaining / status
- **Pipeline Health table** — per step: Duration · Time% · Tokens · Cost $ · LLM Calls · Retries · Failed · Health
- Health icons: ✅ clean &nbsp; ⚠️ retries (recovered) &nbsp; 🔴 permanent failures

**Activity Log**
- Full timestamped event feed per job

---

## Project Structure

```
Multi-Agent-AI-Research-System/
│
├── README.md                        ← You are here
├── IDEAS.md                         ← Architecture source of truth (HLD)
├── docs/technical_design.md         ← Developer implementation reference (LLD)
├── requirements.txt
├── .env.example
│
├── data/                            ← SQLite database (git-ignored)
│   └── jobs.db
│
├── output/                          ← Generated .docx reports (git-ignored)
│
├── app/
│   ├── main.py                      ← FastAPI app factory + startup
│   ├── config.py                    ← All env var loading
│   │
│   ├── api/
│   │   ├── schemas.py               ← Pydantic request/response models
│   │   └── routes.py                ← 5 REST endpoints
│   │
│   ├── graph/
│   │   ├── state.py                 ← ResearchState TypedDict
│   │   ├── graph.py                 ← LangGraph StateGraph assembly
│   │   ├── nodes.py                 ← 12 node functions + routing logic
│   │   └── runner.py                ← run_research_job() + resume_research_job()
│   │
│   ├── agents/
│   │   ├── lead_researcher.py       ← plan / evaluate / synthesize / review
│   │   ├── sub_agent.py             ← web search agentic loop
│   │   ├── citation_agent.py        ← [CITE: url] → [N] citations
│   │   └── document_generator.py   ← python-docx .docx report writer
│   │
│   ├── db/
│   │   └── database.py              ← SQLite schema + connection factory
│   │
│   ├── ui/
│   │   └── gradio_app.py            ← 6-tab Gradio UI
│   │
│   └── utils/
│       ├── job_store.py             ← Thread-safe SQLite-backed job tracker
│       ├── groq_retry.py            ← Groq call wrapper: retry + throttle + LLM call stats
│       ├── tracer.py                ← Per-job span tracer (pipeline timing)
│       ├── cost_tracker.py          ← Token → USD cost tracker by model
│       └── metrics.py               ← Process-level counters (/metrics endpoint)
│
├── evals/
│   ├── rubric.py                    ← 6 weighted scoring criteria
│   ├── test_cases.py                ← 15-case pool across 4 tiers
│   ├── judge.py                     ← LLM-as-judge scoring
│   ├── runner.py                    ← Sequential eval runner + EvalSummary
│   ├── report.py                    ← Human-readable report generator
│   └── results/                     ← Eval run outputs (git-ignored)
│
└── scripts/
    └── run_eval.py                  ← CLI entry point for evals
```

---

## Prerequisites & Setup

### Requirements
- Python 3.11+
- **Groq API key** — free tier at [console.groq.com](https://console.groq.com) (all LLM calls)
- **Tavily API key** — free tier at [tavily.com](https://tavily.com) (web search)
- **Anthropic API key** — only needed for running evals (judge model)

### Installation

```bash
# Clone the repo
git clone https://github.com/mgpt166/Multi-Agent-AI-Research-System.git
cd Multi-Agent-AI-Research-System

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# → edit .env with your API keys
```

### Configure `.env`

```env
# ── Required ────────────────────────────────────────────────────────
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=sk-ant-...    # Only needed for evals

# ── LLM Models (configurable, no code changes needed) ───────────────
GROQ_MODEL=llama-3.3-70b-versatile       # LeadResearcher + CitationAgent
GROQ_SUB_AGENT_MODEL=llama-3.1-8b-instant # ResearchSubAgents (5× higher daily limit)
GROQ_CALL_INTERVAL=3                      # Seconds between Groq calls (set 0 on paid tier)

# ── Pipeline Limits ─────────────────────────────────────────────────
MAX_SUBAGENTS=2
MAX_ITERATIONS=3
MAX_SYNTHESIS_REVIEW_ROUNDS=1
HITL_TIMEOUT_SECONDS=180

# ── Cost Controls ────────────────────────────────────────────────────
MAX_COST_PER_QUERY=0.10
COST_WARNING_THRESHOLD=0.05

# ── Output ────────────────────────────────────────────────────────────
OUTPUT_DIR=./output
```

---

## Quick Start

### Option 1: Web UI (recommended)

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/ui` → submit a query from the **New Research** tab.

### Option 2: REST API (curl)

```bash
# 1. Start server
uvicorn app.main:app --reload --port 8000

# 2. Submit a research job
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest breakthroughs in quantum computing in 2025?", "depth": "moderate"}'
# → {"job_id": "abc-123", "status": "queued"}

# 3. Poll until awaiting_approval
curl http://localhost:8000/status/abc-123

# 4. Approve the plan
curl -X POST http://localhost:8000/approve/abc-123 \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved"}'

# 5. Poll until complete (~3–8 min), then download
curl -o report.docx http://localhost:8000/download/abc-123
```

---

## API Reference

### `POST /research` — Start a research job

```json
{
  "query": "What are the latest breakthroughs in quantum computing?",
  "depth": "moderate"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | Research question |
| `depth` | `simple` \| `moderate` \| `deep` | `moderate` | Controls number of sub-agents |
| `output_folder` | string | `./output` | Custom output directory |
| `max_iterations` | int 1–5 | env var | Override max research rounds |

### `POST /approve/{job_id}` — HITL decision

| Decision | Payload | Effect |
|----------|---------|--------|
| `approved` | `{"decision": "approved"}` | Research begins immediately |
| `refine` | `{"decision": "refine", "feedback": "..."}` | Plan revised with your feedback, re-presented |
| `rejected` | `{"decision": "rejected"}` | Job cancelled, no research cost incurred |

### `GET /status/{job_id}` — Poll job progress

| Status | Meaning |
|--------|---------|
| `queued` | Created, not started |
| `planning` | LeadResearcher decomposing query |
| `awaiting_approval` | Plan ready — your action needed |
| `running` | Sub-agents searching / synthesizing |
| `complete` | Report ready |
| `failed` | Error (check `error` field) |
| `cancelled` | Rejected by human |

### `GET /download/{job_id}` — Download the `.docx` report

### `GET /jobs` — List all jobs (from SQLite, persists across restarts)

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model for LeadResearcher + CitationAgent |
| `GROQ_SUB_AGENT_MODEL` | `llama-3.1-8b-instant` | Model for ResearchSubAgents |
| `GROQ_CALL_INTERVAL` | `3` | Min seconds between Groq calls. Set `0` on paid tier |
| `TAVILY_API_KEY` | — | **Required.** Tavily search key |
| `ANTHROPIC_API_KEY` | — | Required for evals only |
| `MAX_SUBAGENTS` | `2` | Max sub-agents per job |
| `MAX_ITERATIONS` | `3` | Max research rounds before force-stop |
| `MAX_SYNTHESIS_REVIEW_ROUNDS` | `1` | Max self-review passes before proceeding |
| `MAX_HITL_REFINE_ROUNDS` | `3` | Max plan refinement rounds |
| `HITL_TIMEOUT_SECONDS` | `180` | Auto-approve HITL after N seconds |
| `MAX_COST_PER_QUERY` | `0.10` | Per-query budget cap (USD) |
| `COST_WARNING_THRESHOLD` | `0.05` | Cost warning level (USD) |
| `OUTPUT_DIR` | `./output` | Report output directory |
| `EVAL_JUDGE_MODEL` | `claude-haiku-4-5` | LLM-as-judge model for evals (Anthropic or Groq) |

### Depth vs. Cost vs. Time

| Depth | Sub-Agents | Est. Tokens | Est. Cost (8b) | Est. Time |
|-------|-----------|-------------|----------------|-----------|
| `simple` | 1 | ~15,000 | ~$0.001 | 1–2 min |
| `moderate` | 2 | ~40,000 | ~$0.003 | 3–5 min |
| `deep` | 3 | ~80,000 | ~$0.006 | 6–10 min |

> Costs based on `llama-3.1-8b-instant` Groq pricing. Switching to `llama-3.3-70b-versatile` for sub-agents is ~10× more expensive but still significantly cheaper than most hosted APIs.

---

## Evaluation Framework

The eval framework measures research quality using an **LLM-as-judge** approach across 15 test cases in 4 tiers.

```bash
# Quick 4-case smoke test (1 per tier)
python scripts/run_eval.py --count 4

# Full default run (15 cases)
python scripts/run_eval.py

# Single tier
python scripts/run_eval.py --tier simple

# Single case by ID
python scripts/run_eval.py --case medium_03

# Preview without running
python scripts/run_eval.py --count 8 --list
```

### Scoring Rubric

| Criteria | Weight | What it measures |
|----------|--------|-----------------|
| `factual_accuracy` | 25% | Claims are correct and source-supported |
| `citation_quality` | 20% | Every claim cited; sources look real |
| `completeness` | 20% | All requested topics covered |
| `source_quality` | 15% | Authoritative sources (not SEO farms) |
| `structure_clarity` | 10% | Well-organised, readable report |
| `efficiency` | 10% | Reasonable searches for query complexity |

**Pass thresholds:** ≥ 0.85 = strong pass · ≥ 0.70 = pass · < 0.70 = fail

Results saved to `evals/results/{run_id}/` — also visible in the **Eval History** tab of the UI and queryable via `GET /evals` in the API.

---

## Rate Limits (Groq Free Tier)

| Model | TPM | Daily Limit | Used for |
|-------|-----|-------------|---------|
| `llama-3.3-70b-versatile` | 6,000 | 100k tokens/day | LeadResearcher, CitationAgent |
| `llama-3.1-8b-instant` | 6,000 | 500k tokens/day | ResearchSubAgents |

The `groq_retry.py` wrapper handles 429 errors automatically with exponential backoff (10s → 20s → 40s → 80s → 160s, max 5 retries). Set `GROQ_CALL_INTERVAL=3` (default) to throttle calls and stay well under the 6k TPM limit.

---

## Extending the System

### Swap the search provider
Set `SEARCH_PROVIDER=bing` in `.env` and add a `BingSearchProvider` class in `app/tools/` that implements `search()` and `fetch()`.

### Add a new pipeline node
1. Write the node function in `app/graph/nodes.py` (receives `ResearchState`, returns partial dict)
2. Register it in `app/graph/graph.py` with `builder.add_node()` + `builder.add_edge()`
3. Add any new state fields to `app/graph/state.py`

### Use persistent graph state (production)
In `app/graph/graph.py`, replace `MemorySaver()` with `PostgresSaver.from_conn_string(...)` to survive server restarts.

### Change the report format
Replace `app/agents/document_generator.py` with a PDF/Markdown/HTML writer. The pipeline passes `annotated_narrative` + `bibliography` — the rest is format-agnostic.

---

## Known Limitations

| Limitation | Impact | Future Fix |
|-----------|--------|-----------|
| MemorySaver checkpointer | HITL state lost on server restart | Replace with `PostgresSaver` |
| No authentication | Any caller can access any job | Add API key / OAuth middleware |
| Threads for parallelism | Not ideal for high concurrency | LangGraph native parallel branches |
| Web search only | Cannot search internal documents | Add RAG / document upload |
| Groq free tier daily limits | Eval runs may hit 100k TPD on 70b | Use 8b for evals, or upgrade to paid tier |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Setup, quick start, API reference (this file) |
| [IDEAS.md](IDEAS.md) | Architecture source of truth — design decisions, agent roles, system philosophy |
| [docs/technical_design.md](docs/technical_design.md) | Developer reference — state schema, node definitions, config, error handling |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

> Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [Groq](https://groq.com) · [Tavily](https://tavily.com) · [FastAPI](https://fastapi.tiangolo.com) · [Gradio](https://gradio.app)
>
> Last updated: 2026-03-29
