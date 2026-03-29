# Multi-Agent AI Research System — Architecture & Vision

> **Document Purpose**: This is the single source of truth for the system's architecture, patterns, and design decisions. It contains no code and no implementation details. All implementation choices must trace back to a decision documented here.
>
> **Audience**: AI engineers familiar with LLMs, LangGraph, agentic systems, and FastAPI.
>
> **Last Updated**: 2026-03-29

---

## Table of Contents

1. [Vision & Problem Statement](#section-1-vision--problem-statement)
2. [Agent Roles & Information Flow](#section-2-agent-roles--information-flow)
3. [System Architecture](#section-3-system-architecture)
4. [Prompt Engineering Guide](#section-4-prompt-engineering-guide)
5. [Evaluation Strategy](#section-5-evaluation-strategy)
6. [Production Readiness](#section-6-production-readiness)
7. [Phase 2: Enhanced User Experience](#section-7-phase-2-enhanced-user-experience)
   - 7.1 [Web UI](#71-web-ui)
   - 7.2 [Job Persistence](#72-job-persistence)
   - 7.3 [Real-time Progress (Gradio Polling)](#73-real-time-progress-gradio-polling)

---

## Section 1: Vision & Problem Statement

### Why Multi-Agent?

Single-agent and RAG-based systems plateau quickly when a research task demands breadth, parallelism, and iterative refinement. A single LLM call — even with a very large context window — cannot simultaneously hold 50 source documents, evaluate their quality, reconcile contradictions, and produce a well-structured synthesis without degrading in coherence and factual reliability.

Per Anthropic's Research engineering team, their multi-agent system outperformed a single-agent baseline by **90.2%** on research benchmarks. Critically, **token usage explains approximately 80% of performance variance** — the system that used more tokens (because it searched more broadly, evaluated more sources, and reasoned more carefully) consistently produced better outputs. This is not a coincidence: research quality is proportional to cognitive effort, and multi-agent systems are the mechanism that makes sustained, high-effort reasoning economically tractable.

The tradeoff is real: agents consume roughly **15x more tokens than a standard chat interaction**. This is a feature for tasks where quality matters and a bug for tasks where it doesn't. The architecture must be selective about when to engage full multi-agent capacity.

**When multi-agent architecture shines:**

- Research tasks requiring synthesis across 5+ heterogeneous sources
- Questions where sub-topics are naturally parallelizable (e.g., "compare X, Y, and Z" — each can be independently researched)
- Tasks where iterative refinement improves quality (first pass identifies gaps, second pass fills them)
- Extended research sessions where context management for a single agent would degrade quality
- Fact-gathering requiring cross-validation across independent sources

**When to avoid this architecture:**

- Simple factual Q&A answerable in a single search
- Single-document analysis or summarization
- Tightly-coupled sequential pipelines where Step N requires Step N-1's exact output
- Real-time or latency-sensitive tasks (multi-agent adds coordination overhead)
- Tasks requiring shared mutable state across agents in real time
- Use cases where the 15x token cost is not justified by output quality requirements

> **Decision Log**: The choice to use multi-agent rather than a single large-context agent is grounded in Anthropic's empirical finding that performance scales with token usage, and parallelism is the mechanism that unlocks that token budget without making latency unacceptable. A single agent doing 15x work serially would be too slow. Parallelism is the key.

---

### What This System Can Solve

The system is designed for **open-ended, multi-source research tasks** where a human researcher might spend hours gathering, evaluating, and synthesizing information. Concrete use cases:

- **Competitive intelligence**: Gather product positioning, pricing, feature comparisons, and market sentiment across multiple competitors simultaneously.
- **Market research**: Synthesize industry size estimates, growth trends, key players, and analyst perspectives from diverse sources.
- **Literature reviews**: Survey recent publications, extract key findings, and identify consensus vs. contested areas in a domain.
- **Due diligence**: Compile background on companies, founders, regulatory environment, and competitive dynamics for investment or partnership decisions.
- **Multi-source fact-gathering**: Answer complex questions that require corroboration across independent, authoritative sources.
- **Healthcare option navigation**: Research treatment approaches, specialist types, and patient-reported outcomes across medical literature and community sources.
- **Technical deep-dives**: Survey the state of a technology, its tradeoffs, leading implementations, and community sentiment.

The unifying thread: the user has a **research goal** that requires broad information gathering, source evaluation, and coherent synthesis. The output should resemble what a skilled junior analyst would produce after a full day of research — not a chatbot response.

---

### What This System Should NOT Be Used For

This architecture is inappropriate for the following use cases, and attempts to force-fit them will produce worse results than simpler approaches:

- **Simple Q&A**: "What year was X founded?" does not require multi-agent orchestration.
- **Single-document analysis**: Summarizing one PDF should use a single-agent call with the document in context.
- **Tightly sequential workflows**: If Task B's inputs are precisely Task A's outputs and there's no opportunity for parallelism, a linear chain is simpler and more reliable.
- **Real-time data needs**: This system is optimized for research quality, not latency. Do not use it for anything requiring sub-second responses.
- **Shared mutable state across agents**: Sub-agents in this system communicate through structured outputs passed via the orchestrator, not via shared memory. Tasks that require agents to update a shared database mid-run are a poor fit.
- **High-volume automation at scale**: The 15x token multiplier makes this expensive at scale. Use it for high-value, low-frequency research tasks.

---

### Core Design Philosophy

The system should mimic how skilled human researchers operate, not how early LLM wrappers were built. Key philosophical commitments:

**Instill heuristics, not rigid rules.** The lead agent should be trained to make judgment calls — when a source is credible, when enough evidence has been gathered, when a sub-topic warrants deeper exploration. Rigid rule-following breaks on edge cases. Judgment generalizes.

**Decompose before acting.** The best researchers do not immediately search for the first thing that comes to mind. They spend time understanding the question, identifying what they need to know, and planning how to find it. The lead agent must do the same.

**Breadth before depth, then targeted depth.** Start with broad orienting searches to build a map of the information landscape. Then narrow to fill specific gaps. This mirrors how expert human researchers work and prevents the trap of going deep on a narrow path early and missing the broader picture.

**Know when to stop.** Endless searching is a failure mode. The system must have explicit stopping criteria: diminishing returns on new searches, citation saturation, coverage of all planned sub-topics.

**Every factual claim must be traceable.** A research output without citations is not a research output — it is a hallucination delivery mechanism. Citation is not a nice-to-have; it is a core functional requirement enforced architecturally.

---

## Section 2: Agent Roles & Information Flow

### Agent Roles & Responsibilities

The system defines four agent roles, all active in every research run. **One Human-in-the-Loop (HITL) checkpoint** gates the pipeline before any expensive work begins:

- **HITL-1 (Plan Review)**: After `plan_research`, before `spawn_subagents` — human approves, refines, or rejects the research plan before any sub-agent cost is incurred.

After that, the **LeadResearcher autonomously reviews its own synthesis** against the sub-agent task assignment map. If the synthesis falls short, it sends targeted rework instructions back to specific sub-agents. This loop is hard-capped at `MAX_SYNTHESIS_REVIEW_ROUNDS` rounds (default **1**) — after the cap is reached, output proceeds to the CitationAgent regardless. This prevents runaway loops while still allowing the system to self-correct.

---

#### LeadResearcher (Orchestrator)

The LeadResearcher is the strategic center of the system. It is the only agent with a global view of the research goal, the accumulated findings, and the quality of work done so far.

**Responsibilities:**

- Parse and analyze the user's research query to understand intent, scope, and implicit requirements
- Identify major sub-topics or research dimensions that can be explored in parallel
- Decompose the research goal into 2-3 concrete, bounded sub-tasks for sub-agents
- Write detailed, unambiguous task descriptions for each sub-agent (see Section 4: Prompt Engineering Guide)
- Evaluate the sufficiency and quality of sub-agent results
- Decide whether additional research rounds are needed (loop) or whether synthesis can begin
- Synthesize sub-agent findings into a coherent, unified research narrative
- **Self-review the synthesis** against the original sub-agent task assignment map: did every assigned scope get adequately represented? Are there contradictions, gaps, or low-confidence areas that a targeted rework could fix?
- If synthesis falls short and `synthesis_review_count < MAX_SYNTHESIS_REVIEW_ROUNDS`: issue targeted rework instructions to the specific sub-agents whose coverage was inadequate, then re-synthesize
- If `synthesis_review_count >= MAX_SYNTHESIS_REVIEW_ROUNDS` (default 1): force-proceed to citation with a completeness caveat — no further loops
- Hand off the approved synthesized content and source map to the CitationAgent
- Respond to the API with job status and output metadata

**What it does NOT do:**

- Execute web searches directly (delegates to sub-agents)
- Manage citations (delegates to CitationAgent)
- Produce the final document (delegates to DocumentGeneratorAgent)

**Model**: `llama-3.3-70b-versatile` via Groq (configurable via `GROQ_MODEL` env var) — highest quality model available on the Groq free tier, appropriate for strategic reasoning, synthesis, and judgment calls.

> **Decision Log**: The LeadResearcher uses the larger 70b model because orchestration quality is the primary bottleneck. Poor task decomposition and weak synthesis are the most common failure modes. Sub-agents are more mechanical (search → extract → summarise) and can run effectively on the faster, higher-throughput 8b model — which also has a 5× higher daily token limit on the Groq free tier (500k vs 100k TPD), making it essential for parallel sub-agent execution.

---

#### Human-in-the-Loop Plan Review (HITL Checkpoint)

This is not an agent — it is a mandatory **interrupt point** in the LangGraph graph that pauses execution after `plan_research` and before `spawn_subagents`. The human reviews the LeadResearcher's proposed research plan and decides how to proceed before any sub-agent work (and therefore any significant token cost) begins.

**What the human sees at this checkpoint:**

- The interpreted research goal (how the LeadResearcher understood the query)
- The proposed sub-topic decomposition (e.g., "Sub-topic 1: Market sizing, Sub-topic 2: Competitor landscape, Sub-topic 3: Regulatory environment")
- The proposed sub-agent count and their assigned scope
- Estimated depth and token budget for the run
- Any ambiguities or assumptions the LeadResearcher flagged

**Human actions available:**

| Action | Effect |
|---|---|
| **Approve** | Graph resumes; sub-agents are dispatched immediately |
| **Reject** | Job is terminated; no sub-agent work is performed; job status set to `cancelled` |
| **Refine** | Human provides free-text feedback; graph loops back to `plan_research` with the feedback injected into the LeadResearcher's context; a new plan is generated for review |

**Refine feedback examples:**

- "Focus only on North American markets, ignore Europe"
- "Add a sub-topic on regulatory risk; remove the technology deep-dive"
- "The query is about the *company* Stripe, not payment processing in general"
- "Prioritize academic sources over news coverage"

The LeadResearcher treats refinement feedback as a first-class input — not a hint — and rewrites the plan accordingly before presenting it for re-review. The loop repeats until the human approves or rejects.

**Why this checkpoint is placed here:**

The `plan_research` step costs a small number of tokens (Opus planning call). Sub-agent execution costs 10-20x more. Catching a misunderstood query, wrong scope, or poor decomposition at the plan stage is the highest-leverage, lowest-cost intervention point in the entire pipeline. Once sub-agents are running, correction requires re-running research from scratch.

> **Decision Log**: Human approval is placed after planning and before execution — not before planning (too early, no plan to review) and not after execution (too late, cost already spent). The refine loop is bounded: if the human has not approved after 3 refinement rounds, the system surfaces a warning. This prevents infinite loops on genuinely ambiguous queries.

---

#### ResearchSubAgent (Worker)

Each ResearchSubAgent is a focused, independent research unit. It receives a single bounded task and executes it using web tools. It has no knowledge of other sub-agents' tasks.

**Responsibilities:**

- Execute web searches using the provided search tool
- Evaluate source quality and credibility before extracting information
- Fetch and parse the full content of high-value sources
- Compress findings into a structured summary appropriate for the LeadResearcher to consume
- Record all source URLs, titles, and publication dates used
- Report confidence levels and coverage gaps back to the orchestrator
- Know when to stop: if 3 search rounds yield no new information, stop and report what was found

**Scope constraints (per-task):**

- Focus only on the specific sub-topic assigned; do not drift into adjacent areas
- Do not attempt to synthesize across sub-topics — that is the LeadResearcher's job
- Report facts with their source attribution inline; do not paraphrase away citation information

**Model**: `llama-3.1-8b-instant` via Groq (configurable via `GROQ_SUB_AGENT_MODEL` env var) — sufficient for focused search-and-extract tasks, and has a 5× higher daily token limit than the 70b model on the Groq free tier.

---

#### CitationAgent (Post-Processor)

The CitationAgent is a mandatory post-processing step between synthesis and document generation. Its sole function is to ensure every factual claim in the synthesized research is traceable to a source.

**Responsibilities:**

- Parse the synthesized research narrative from the LeadResearcher
- Map every factual claim to a source URL and title from the sub-agent source records
- Assign sequential inline citation numbers (e.g., [1], [2], [3]) to each claim
- Produce a deduplicated, numbered bibliography
- Flag claims that cannot be traced to a provided source (these are candidates for hallucination — the human reviewer must evaluate them)
- Return the citation-annotated narrative and bibliography to the DocumentGeneratorAgent

**What it does NOT do:**

- Re-search for sources — it works only with sources already gathered by sub-agents
- Evaluate source quality — that is the sub-agent's job during research
- Modify the substance of claims — only annotate them

> **Decision Log**: The CitationAgent is a separate role rather than a responsibility of the LeadResearcher because citation enforcement requires a different mode of reasoning (systematic cross-referencing) than synthesis (narrative construction). Mixing them in one agent leads to citation gaps when the synthesizing model prioritizes narrative coherence over citation completeness. Architectural separation enforces the requirement.

---

#### DocumentGeneratorAgent (Output Producer)

The DocumentGeneratorAgent transforms the structured, citation-annotated research into a clean, professional `.docx` report saved to the project output folder.

**Responsibilities:**

- Accept the citation-annotated narrative, bibliography, and metadata (query, date, token usage) from the CitationAgent
- Structure the content into a well-organized Word document following the standard report template (see Section 3: Document Generation Pipeline)
- Apply clean, readable formatting: proper headings hierarchy, readable tables where data warrants it, consistent citation style
- Save the file to the configured output folder using the naming convention `{timestamp}_{query_slug}/report.docx`
- Return the file path and document metadata to the LeadResearcher for API response construction

**What it does NOT do:**

- Add branding, logos, or decorative elements
- Modify the substance of the content — the narrative and citations arrive fully formed
- Produce any format other than `.docx` (extensions for PDF export are a future consideration)

---

---

### Information Flow

The following describes the data that flows between agents at each handoff point.

**User → LeadResearcher:**
- Raw research query (natural language)
- Optional constraints: depth level, source type preferences, output folder path

**LeadResearcher → Human (HITL checkpoint — via API):**
- Interpreted research goal (how the system understood the query)
- Proposed sub-topic list with one-line description per sub-topic
- Proposed sub-agent assignments and scope boundaries
- Estimated token budget and depth level
- Assumptions or ambiguities flagged by the LeadResearcher

**Human → LeadResearcher (HITL response — via API):**
- Decision: `approved` / `rejected` / `refine`
- If `refine`: free-text feedback string injected into the next `plan_research` call
- If `rejected`: optional rejection reason (recorded in job metadata, no further work done)

**LeadResearcher → ResearchSubAgent (per sub-agent):**
- Sub-task description (detailed, specific, bounded)
- Research objective for this sub-task
- Explicit output format specification (structured JSON with: summary, key facts, sources list, confidence score, coverage gaps)
- Tool guidance: which search strategies to prioritize
- Scope boundaries: what NOT to research (prevents overlap with other sub-agents)
- Token budget hint (e.g., "this is a narrow fact-finding task, 5-8 searches should be sufficient")

**ResearchSubAgent → LeadResearcher:**
- Structured findings summary (compressed, not raw search results)
- Ordered list of sources: URL, title, publication date, relevance score
- Confidence signal: how thoroughly was this sub-topic covered?
- Coverage gaps: what was searched for but not found?
- Tool call count: how many searches/fetches were executed?

**LeadResearcher → ResearchSubAgent (targeted rework — if review fails):**
- Specific sub-agent ID being asked to rework
- What was inadequate in the original output (gap description, not a general redo)
- What to fix: reframe existing findings, fill a specific gap, clarify a contradictory point
- Constraint: use already-gathered source material unless explicitly told to search again
- `synthesis_review_count` context so the sub-agent knows this is a rework pass

**LeadResearcher → CitationAgent:**
- Approved synthesized research narrative (prose, organized by sub-topic)
- Aggregated source map: all sources from all sub-agents, deduplicated
- Research metadata: query, sub-agent count, iteration count, synthesis review count, total token usage
- Any completeness caveats to be included in the Limitations section (if `force_proceed` was triggered)

**CitationAgent → DocumentGeneratorAgent:**
- Citation-annotated narrative: prose with inline [N] references
- Numbered bibliography: title, URL, publication date for each source
- Flagged claims: any claims without source attribution

**DocumentGeneratorAgent → LeadResearcher:**
- File path to saved `.docx` report
- Document metadata: page count, citation count, word count
- Save status (success or error details)

**LeadResearcher → FastAPI → User:**
- Job status (complete / failed)
- File path to `.docx` report
- Research metadata: query, duration, total token usage, estimated cost, sub-agent count
- Executive summary snippet (first 200 words of the report)

---

### Key Design Principles

These principles are drawn from Anthropic's Research engineering team's learnings and should inform every design and implementation decision.

**1. Delegate with surgical precision.**
The lead agent's task descriptions for sub-agents must be detailed and unambiguous. Vague delegation is the most common cause of sub-agent failure. A task description should specify: the objective, the required output format, the scope boundaries, the search strategy, and the stopping criteria.

**2. Scale effort to query complexity.**
A simple factual question does not need three sub-agents and five research rounds. The lead agent must assess complexity and size the response accordingly. Simple queries get one sub-agent and 3-10 tool calls. Comparison queries get two sub-agents. Deep research gets three. Spawning unnecessary agents is waste, not thoroughness.

**3. Breadth before depth.**
The first search pass should orient the research landscape. Narrow, deep searches come after the broad structure is established. This mirrors expert research practice and prevents premature depth on tangential sub-topics.

**4. Extended thinking as planning scratchpad.**
The LeadResearcher should use extended thinking (Claude's internal reasoning capability) for query decomposition, complexity assessment, and sub-agent role definition. Thinking before acting produces better plans and better task descriptions.

**5. Let agents self-improve.**
Use Claude to diagnose prompt failures. When a sub-agent consistently produces poor outputs, have Claude analyze the task descriptions and suggest rewrites. Per Anthropic's Research engineering team, having an agent use a tool dozens of times and then rewrite the tool description based on failures produces meaningfully better descriptions than human-authored descriptions alone.

**6. Parallel tool calling for speed.**
Where a sub-agent needs to fetch multiple sources, it should issue parallel fetch requests rather than sequential ones. Per Anthropic's Research engineering team, parallel tool calling can reduce research time by up to 90% for fetch-heavy tasks.

**7. Human oversight before commitment.**
The most expensive part of the pipeline — sub-agent execution — must not begin without explicit human approval of the research plan. This is not a UX nicety; it is a cost and quality control mechanism. A misunderstood query caught at the plan stage costs one Opus planning call. The same mistake caught after sub-agent execution costs the full research budget and requires a complete re-run.

---

## Section 3: System Architecture

### Search Provider Abstraction

The system uses a **provider-neutral search abstraction layer** (`app/tools/`) so the underlying web search service can be swapped without changing any agent code. The active provider is selected at runtime via the `SEARCH_PROVIDER` environment variable.

**Architecture:**

```
ResearchSubAgent
      │
      ▼
SearchProvider (abstract interface — app/tools/base.py)
      │
      ├── TavilySearchProvider   (current default — app/tools/tavily_provider.py)
      ├── BingSearchProvider     (future)
      ├── SerpAPISearchProvider  (future)
      └── BraveSearchProvider    (future)
```

**Why not Anthropic's built-in web tools (`web_search_20260209`)?**

Anthropic's server-side tools are convenient for quick prototypes but create hard vendor lock-in:
- They only work with Anthropic's API — switching to another LLM provider (OpenAI, Mistral, etc.) breaks web search entirely
- No control over search depth, domain filtering, or result quality
- Cannot be tested in isolation without a live Anthropic API call

The abstraction layer means:
- Search provider is swappable via a single env var: `SEARCH_PROVIDER=tavily`
- LLM provider is independently swappable — the agent loop is a standard tool-use pattern
- New providers require only one new file (`app/tools/{name}_provider.py`) and one line in the factory registry

**Current provider: Tavily**

Tavily is an AI-optimised search API built specifically for LLM applications. It returns clean, relevant content without HTML parsing. Free tier: 1000 searches/month. Get a key at https://tavily.com.

**To add a new search provider:**
1. Create `app/tools/{name}_provider.py` implementing `SearchProvider` (base class in `app/tools/base.py`)
2. Register it in `app/tools/factory.py` `_PROVIDERS` dict
3. Set `SEARCH_PROVIDER={name}` in `.env`

**Design principle:** The `SearchProvider` abstraction extends naturally to LLM provider neutrality. The manual tool-use loop in `ResearchSubAgent` (where our code calls the search provider and returns results as `tool_result` blocks) works identically with any model that supports tool/function calling — Claude, GPT-4, Mistral, Gemini, etc.

---

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER / CLIENT                             │
│   POST /research              POST /approve/{job_id}               │
│   {query, options}            {decision, feedback?}                │
└──────────────┬────────────────────────┬────────────────────────────┘
               │                        │ (approve / reject / refine)
               ▼                        │
┌─────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                             │
│  • Job queue management                                             │
│  • Auth / rate limiting                                             │
│  • Async task dispatch                                              │
│  • GET /status/{job_id}    GET /download/{job_id}                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH RESEARCH GRAPH                         │
│                                                                     │
│  ┌──────────────────────────┐                                       │
│  │  LeadResearcher (70b)    │                                       │
│  │      plan_research       │ ◄── refine feedback loops back here  │
│  └────────────┬─────────────┘                                       │
│               │  research plan + sub-task descriptions              │
│               ▼                                                     │
│  ╔══════════════════════════╗                                       │
│  ║  ⏸  await_human_approval ║ ◄── GRAPH PAUSES HERE                │
│  ║   (HITL interrupt node)  ║     job status → "awaiting_approval"  │
│  ╚══════╤═══════╤═══════════╝                                       │
│         │       │       │                                           │
│      approve  refine  reject                                        │
│         │       │       │                                           │
│         │       └───────┼──► plan_research (re-plan with feedback) │
│         │               └──► respond (job cancelled)               │
│         ▼                                                           │
│  ┌──────────────────────────┐                                       │
│  │     spawn_subagents      │ (parallel fan-out, 1–3 agents)        │
│  └──────┬──────┬────────────┘                                       │
│         │      │        │                                           │
│         ▼      ▼        ▼                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│  │SubAgent 1│ │SubAgent 2│ │SubAgent 3│                           │
│  │  (8b)    │ │  (8b)    │ │  (8b)    │                           │
│  │ search   │ │ search   │ │ search   │                           │
│  │ fetch    │ │ fetch    │ │ fetch    │                           │
│  │ compress │ │ compress │ │ compress │                           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                           │
│       └────────────┴────────────┘                                  │
│                    │ collect_results                                 │
│                    ▼                                                 │
│         evaluate_sufficiency                                         │
│                    │                                                 │
│         ┌──────────┴──────────┐                                     │
│      needs_more           sufficient / force_stop                   │
│         │                     │                                     │
│         └──► spawn_subagents  ▼                                     │
│              (refined tasks) synthesize                             │
│                               │                                     │
│                               ▼                                     │
│                    review_synthesis  ◄──────────────────┐           │
│                    (LeadResearcher)                      │           │
│                               │                         │           │
│               ┌───────────────┼───────────────┐         │           │
│           approved       needs_rework     force_proceed  │           │
│               │          (round < 3)     (round == 3)   │           │
│               │               │               │         │           │
│               │               └──► targeted   │         │           │
│               │                    sub-agent  │         │           │
│               │                    rework ────┘         │           │
│               │                    + re-synthesize ─────┘           │
│               ▼                                                      │
│              ┌─────────────────────────┐                           │
│              │     CitationAgent       │                           │
│              │  annotate + bibliography│                           │
│              └────────────┬────────────┘                           │
│                            ▼                                        │
│              ┌─────────────────────────┐                           │
│              │  DocumentGeneratorAgent │                           │
│              │  produces report.docx   │                           │
│              └────────────┬────────────┘                           │
│                            ▼                                        │
│              ┌─────────────────────────┐                           │
│              │    save_to_project      │                           │
│              │ ./output/{ts}_{slug}/   │                           │
│              └────────────┬────────────┘                           │
└───────────────────────────┼─────────────────────────────────────────┘
                             ▼
               API Response: { file_path, metadata, summary }
```

---

### LangGraph State Design

The LangGraph state object is the single shared data structure that flows through all graph nodes. It must capture the complete research session — enough to resume from any checkpoint without data loss.

**State fields:**

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Unique identifier for the research run |
| `query` | string | Original user research query |
| `research_plan` | object | LeadResearcher's decomposition: sub-topics, sub-agent assignments, complexity assessment |
| `sub_agent_tasks` | list[object] | Task descriptions dispatched to each sub-agent |
| `sub_agent_results` | list[object] | Structured findings returned by each sub-agent |
| `accumulated_findings` | string | Running synthesis across all completed rounds |
| `source_map` | dict[url → source_metadata] | Deduplicated map of all sources encountered |
| `citation_map` | dict[claim_id → citation_number] | CitationAgent's output: claim-to-reference mapping |
| `bibliography` | list[object] | Ordered numbered bibliography |
| `hitl_status` | enum | `awaiting_approval` / `approved` / `rejected` / `refining` |
| `hitl_feedback` | string | Human's free-text refinement instruction (null if approved/rejected) |
| `hitl_round` | int | How many refine loops have occurred (bounded at 3) |
| `iteration_count` | int | How many sub-agent research rounds have been executed |
| `sufficiency_signal` | enum | `needs_more` / `sufficient` / `force_stop` |
| `synthesis_review_count` | int | How many synthesis self-review rounds have occurred (hard cap: `MAX_SYNTHESIS_REVIEW_ROUNDS`, default 1) |
| `synthesis_review_signal` | enum | `approved` / `needs_rework` / `force_proceed` |
| `synthesis_rework_instructions` | list[object] | Per-sub-agent targeted rework instructions from the LeadResearcher |
| `synthesized_narrative` | string | LeadResearcher's synthesized output |
| `annotated_narrative` | string | CitationAgent's output with inline [N] references |
| `document_path` | string | Absolute path to the saved `.docx` file |
| `token_usage` | object | Per-agent token consumption |
| `error_log` | list[object] | Tool failures, retries, and recovery actions |
| `start_time` | timestamp | For duration tracking and timeout enforcement |

> **Decision Log**: The state is intentionally large. LangGraph's checkpointing relies on the state being serializable and complete. A minimal state would reduce memory usage but would make recovery from mid-run failures impossible. Given the 15x token cost of these runs, losing a partially-complete research session is more expensive than the overhead of a comprehensive state object.

---

### Graph Topology

The LangGraph graph has the following nodes and edges:

```
plan_research
     │
     ▼
await_human_approval  ◄──────────────────────────────────┐
     │                                                    │
     ├── [approved]  ──────────────────────────────────── │ ──────────┐
     │                                                    │           │
     ├── [refine]  ──► plan_research (with feedback) ────┘           │
     │                                                                │
     └── [rejected] ──► respond (cancelled)                          │
                                                                      │
                                                             spawn_subagents
                                                          (parallel fan-out, 1-3)
                                                                      │
                                                             collect_results
                                                          (fan-in: wait for all)
                                                                      │
                                                          evaluate_sufficiency
                                                                      │
                                                   ┌──────────────────┴───────────┐
                                                needs_more                  sufficient /
                                                   │                        force_stop
                                                   └──► spawn_subagents          │
                                                        (refined tasks)          ▼
                                                                            synthesize
                                                                                 │
                                                                                 ▼
                                                                       review_synthesis ◄──────────┐
                                                                       (LeadResearcher)             │
                                                                                 │                  │
                                                              ┌──────────────────┼──────────┐       │
                                                           approved         needs_rework  force_     │
                                                              │             (round < 3)  proceed     │
                                                              │                  │      (round=3)    │
                                                              │                  └──► targeted_rework│
                                                              │                       + synthesize ──┘
                                                              ▼
                                                           cite
                                                                                 │
                                                                                 ▼
                                                                       generate_document
                                                                                 │
                                                                                 ▼
                                                                        save_to_project
                                                                                 │
                                                                                 ▼
                                                                             respond
```

**Node descriptions:**

- `plan_research`: LeadResearcher analyzes the query (and any refinement feedback from the human), assesses complexity, decomposes into sub-topics, writes sub-agent task descriptions, and sets the iteration budget. On re-entry after `refine`, the human's feedback is injected at the top of the prompt.
- `await_human_approval`: **LangGraph interrupt node.** Graph execution pauses here. The research plan is surfaced to the human via the API (job status transitions to `awaiting_approval`). Execution resumes only when the human POSTs a decision to `/approve/{job_id}`. This node has no LLM call — it is a pure state-wait.
- `spawn_subagents`: Dispatches 1-3 sub-agents in parallel. Each receives its task description, tools, and output format spec.
- `collect_results`: Waits for all sub-agents to return. Handles timeouts — if a sub-agent exceeds the timeout, its slot is marked as incomplete and the graph continues with available results.
- `evaluate_sufficiency`: LeadResearcher assesses whether the accumulated findings adequately address the original query. Checks: coverage of all sub-topics, source quality, confidence signals from sub-agents, iteration count vs. budget.
- `synthesize`: LeadResearcher produces a coherent narrative from all sub-agent findings. Resolves contradictions, fills small gaps, structures by sub-topic.
- `review_synthesis`: LeadResearcher checks its own synthesis against the original sub-agent task assignment map. Asks: did every sub-agent's assigned scope get adequately represented? Are there contradictions or confidence gaps a targeted rework could fix? Produces one of three signals: `approved` (proceed to citation), `needs_rework` (issue targeted instructions to specific sub-agents and re-synthesize — only if `synthesis_review_count < 3`), or `force_proceed` (cap reached — proceed to citation with a completeness note in the Limitations section).
- `targeted_rework`: The LeadResearcher dispatches rework instructions only to the sub-agents whose coverage was found inadequate. Sub-agents refine their outputs using already-gathered source material — no new web searches unless the rework instruction explicitly requires it. Results feed back into `synthesize`.
- `cite`: CitationAgent maps all factual claims to source references. Produces annotated narrative and bibliography.
- `generate_document`: DocumentGeneratorAgent structures and formats the `.docx` report.
- `save_to_project`: Writes the file to the configured output folder. Updates state with file path.
- `respond`: Assembles the final API response payload.

**Conditional edges:**

The `await_human_approval` node has three outgoing edges — the only human-controlled branch in the graph:

- `approved`: Human accepts the plan as-is → proceed to `spawn_subagents`
- `refine`: Human provides feedback → loop back to `plan_research` with feedback in context; `hitl_round` increments; if `hitl_round` reaches 3, surface a warning alongside the plan
- `rejected`: Human cancels the job → jump to `respond` with status `cancelled`; no sub-agent work is performed; only the planning token cost is incurred

The `evaluate_sufficiency` node has three outgoing edges — controls whether more *research* is needed:

- `sufficient`: Coverage adequate, confidence positive → proceed to `synthesize`
- `needs_more`: Specific gaps remain, iteration count below budget → spawn another round with gap-filling tasks
- `force_stop`: Iteration budget exhausted or diminishing returns detected → proceed to `synthesize` with a completeness caveat

The `review_synthesis` node has three outgoing edges — controls whether the *synthesis quality* is acceptable (no human involved):

- `approved`: LeadResearcher is satisfied the synthesis covers all assigned sub-topics → proceed to `cite`
- `needs_rework`: Synthesis has gaps or quality issues AND `synthesis_review_count < MAX_SYNTHESIS_REVIEW_ROUNDS` → issue targeted rework instructions to specific sub-agents, re-run `synthesize`, increment `synthesis_review_count`
- `force_proceed`: `synthesis_review_count >= MAX_SYNTHESIS_REVIEW_ROUNDS` — cap reached regardless of quality → proceed to `cite`; a note is added to the report's Limitations section flagging which sub-topics had unresolved coverage gaps

> **Decision Log**: The hard cap on synthesis review is non-negotiable. Without it, a poorly-scoped query or a domain with sparse web coverage could cause the system to loop indefinitely. The cap is configurable via `MAX_SYNTHESIS_REVIEW_ROUNDS` (default 1 — one review pass then proceed). After the cap the marginal improvement from another rework cycle is almost always negligible, and the token cost is not. The Limitations section caveat ensures the human reader knows the output is incomplete rather than receiving a silently partial document.

---

### Document Generation Pipeline

The DocumentGeneratorAgent produces a single `.docx` file following a standard structure.

**Report Structure:**

```
[Document Title]
Research Report: {query}
Generated: {date}  |  Sources: {N}  |  Research Depth: {simple/moderate/deep}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Executive Summary
   [3-5 paragraph synthesis of key findings]

2. Research Findings
   2.1 {Sub-topic A}
       [Findings with inline citations e.g., "Market size is $4.2B [1]"]
       [Tables where comparative data warrants it]
   2.2 {Sub-topic B}
       ...
   2.3 {Sub-topic C}
       ...

3. Recommendations
   [Actionable conclusions drawn from the findings]

4. Limitations & Gaps
   [What was not found; coverage gaps; low-confidence areas]

5. References / Bibliography
   [1] Title of Source. URL. Published: {date if available}.
   [2] ...
   ...
```

**Formatting principles:**

- Headings use a clear hierarchy: H1 for document title, H2 for section numbers, H3 for sub-topics
- Tables are used only where structured data (comparisons, statistics, feature matrices) genuinely benefits from tabular format — not for prose
- Inline citations use bracketed sequential numbers: [1], [2], [3]
- The bibliography is the final section, with each entry on its own line
- No branding, no color themes, no decorative elements — clean and readable in default Word styling
- Font: consistent throughout (system default or a clean sans-serif)

**Citation handling:**

- Every factual claim in the findings section carries an inline citation
- The bibliography maps each number to: source title, URL, and publication date where available
- If a claim cannot be cited from available sources, it is flagged with [UNVERIFIED] rather than silently uncited
- Duplicate sources use the same citation number throughout the document

**File output:**

```
./output/
  └── {YYYYMMDD_HHMMSS}_{query_slug}/
        ├── report.docx          ← the deliverable
        └── metadata.json        ← job metadata for API response
```

The `query_slug` is the first 40 characters of the query with spaces replaced by underscores and special characters stripped. The timestamp uses UTC.

---

### Agent-Tool Interface Design

Each agent type has a specific tool set. Tools are not shared freely — each agent gets only what it needs for its role.

**ResearchSubAgent tools:**

| Tool | Purpose | Key design note |
|---|---|---|
| `web_search` | Execute a search query, return top N result titles/URLs/snippets | Must specify: max results (5-10), recency preference |
| `web_fetch` | Fetch and parse the full text content of a URL | Must specify: max token budget for the fetched content |
| `result_parser` | Extract structured data (tables, lists, key facts) from raw page content | Reduces token usage vs. passing raw HTML to the model |

**CitationAgent tools:**

| Tool | Purpose |
|---|---|
| `source_lookup` | Given a factual claim string, fuzzy-match it against the source map to find the best attribution |

**DocumentGeneratorAgent tools:**

| Tool | Purpose |
|---|---|
| `docx_writer` | Write a structured Word document from a document model (sections, headings, paragraphs, tables, bibliography) |
| `file_save` | Save the document to the output folder with the standard naming convention |

**On tool description quality:**

Per Anthropic's Research engineering team, imprecise tool descriptions are a leading cause of sub-agent failures — accounting for up to 40% of avoidable errors. A poor `web_search` description that doesn't clarify what makes a good search query causes agents to use overly verbose, SEO-optimized phrasing that returns poor results. Every tool description must specify: what the tool does, what good inputs look like, what bad inputs look like, and how to interpret the output.

> **Decision Log**: The sub-agents intentionally do not have access to `docx_writer` or `file_save`. Giving workers access to output tools would allow them to bypass the citation and document generation pipeline, producing uncited outputs. Tool access restriction enforces the architectural constraint that citation is mandatory.

---

### Context Window Management

Long research sessions — particularly deep research with multiple rounds — can exhaust the lead agent's context window. The following strategies mitigate this:

**LeadResearcher context management:**

- After each research round, the LeadResearcher compresses completed sub-agent outputs into a structured `accumulated_findings` summary stored in state
- Raw sub-agent outputs are not retained in the working context after summarization
- The research plan (decomposition and sub-topic map) is saved to state at `plan_research` and referenced by field rather than reconstructed

**Sub-agent context isolation:**

- Each sub-agent is initialized with a fresh context containing only its task description, tools, and output format specification
- Sub-agents have no visibility into other sub-agents' work or the LeadResearcher's synthesis
- This isolation is not just a context optimization — it prevents cross-contamination between sub-agent research tracks

**Source map management:**

- The source map in state is deduplicated at every collection step
- Sources are indexed by URL; duplicate fetches of the same URL are suppressed
- The source map stores metadata (title, date, relevance score) rather than full content

---

### Parallel Execution Strategy

The system runs 2-3 sub-agents in parallel within LangGraph using its native fan-out and fan-in patterns.

**Fan-out (spawn_subagents):**
The `spawn_subagents` node creates N sub-agent task invocations simultaneously. LangGraph's parallel node execution handles the dispatch. Each sub-agent is an independent LangGraph subgraph or a structured tool call, depending on implementation choice.

**Fan-in (collect_results):**
The `collect_results` node waits for all dispatched sub-agents. Results are collected into `sub_agent_results` in state as they arrive.

**Handling slow or stuck sub-agents:**
- Each sub-agent invocation has a maximum wall-clock timeout (configurable, default: 90 seconds)
- If a sub-agent exceeds its timeout, the collect step marks it as `timed_out` and proceeds with available results
- The LeadResearcher is informed of the timeout and factors the missing coverage into its sufficiency evaluation
- A timed-out sub-topic may be re-assigned in the next iteration with a simpler, faster task description

**Result merging:**
Sub-agent results arrive as structured objects (not prose). The LeadResearcher synthesizes them in the `synthesize` node. There is no automatic merging — synthesis is a reasoning step, not a concatenation step.

---

### Error Handling & Recovery

**Checkpoint/resume strategy:**
LangGraph's built-in checkpointing persists the full state object after each node completes. If the system fails mid-run, it can resume from the last successful checkpoint rather than restarting from scratch. Given the token cost of research runs, this is essential.

**Tool failure handling:**
When a tool call fails (network error, rate limit, malformed response), the agent receives a structured error object rather than a raw exception. The agent is instructed to:
1. Log the failure in the error log
2. Attempt an alternative approach (different search query, different URL)
3. After N retries, mark the sub-topic as partially covered and continue

**Retry logic:**
- Tool retries use exponential backoff: 1s, 2s, 4s, 8s (max 3 retries per tool call)
- After exhausting retries, the failure is recorded in state and the agent adapts its strategy
- Rate limit errors (HTTP 429) use the Retry-After header if provided

**Agent-level failure:**
If a sub-agent fails to return a valid structured result after timeout, the slot is marked failed and the LeadResearcher decides whether to: re-assign the sub-topic, narrow the task, or proceed without it.

---

### API Design (FastAPI)

**Endpoints:**

```
POST   /research                 Submit a new research job
POST   /approve/{job_id}         Submit human decision on the research plan (HITL)
GET    /status/{job_id}          Poll job status and partial results
GET    /download/{job_id}        Download the completed .docx report
DELETE /cancel/{job_id}          Cancel an in-progress job
GET    /jobs                     List recent jobs for authenticated user
```

**POST /research — Request:**
```
{
  "query": "string (required) — the research question",
  "depth": "simple | moderate | deep (default: moderate)",
  "output_folder": "string (optional) — override default output path",
  "max_iterations": "int (optional) — override default iteration budget"
}
```

**POST /research — Response (202 Accepted):**
```
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_duration_seconds": 120,
  "poll_url": "/status/{job_id}"
}
```

**POST /approve/{job_id} — Request:**
```
{
  "decision": "approved | rejected | refine",
  "feedback": "string (required when decision == refine) — free-text instruction for the LeadResearcher"
}
```

**POST /approve/{job_id} — Response (200):**
```
{
  "job_id": "uuid",
  "decision": "approved",
  "status": "running",                // immediately transitions if approved
  "hitl_round": 1                     // increments on each refine; capped at 3
}
```

**GET /status/{job_id} — Response (200):**
```
{
  "job_id": "uuid",
  "status": "queued | planning | awaiting_approval | running | complete | failed | cancelled",
  "hitl": {                                          // only present when status == awaiting_approval
    "research_plan": {
      "interpreted_goal": "...",
      "sub_topics": [
        { "id": 1, "title": "...", "scope": "...", "assigned_to": "SubAgent 1" },
        { "id": 2, "title": "...", "scope": "...", "assigned_to": "SubAgent 2" }
      ],
      "sub_agent_count": 2,
      "depth": "moderate",
      "estimated_tokens": 35000,
      "assumptions": ["query refers to US market only", "..."]
    },
    "approve_url": "/approve/{job_id}",
    "hitl_round": 1,
    "max_refine_rounds": 3
  },
  "progress": {
    "phase": "planning | awaiting_approval | researching | synthesizing | citing | generating",
    "iterations_completed": 1,
    "sub_agents_active": 2
  },
  "result": {                         // only present when status == "complete"
    "file_path": "./output/..../report.docx",
    "download_url": "/download/{job_id}",
    "metadata": {
      "query": "...",
      "duration_seconds": 118,
      "token_usage": { "total": 48000, "lead": 12000, "sub_agents": 36000 },
      "estimated_cost_usd": 0.47,
      "source_count": 23,
      "citation_count": 41,
      "sub_agent_count": 3,
      "iterations": 2
    },
    "summary_snippet": "first 200 words of executive summary..."
  },
  "error": null                       // populated on failure
}
```

**GET /download/{job_id}:**
Returns the `.docx` file as a binary attachment with `Content-Disposition: attachment; filename="report.docx"`.

**Synchronous vs. async:**
All research jobs are inherently async given their duration (60-300 seconds). The `/research` endpoint returns 202 immediately. Clients poll `/status/{job_id}`. The HITL checkpoint introduces a second natural pause in the polling loop: clients should watch for `status == awaiting_approval` and surface the `hitl.research_plan` payload to the user before calling `/approve/{job_id}`. A streaming endpoint (`GET /stream/{job_id}` via Server-Sent Events) is a future consideration for real-time progress updates and HITL push notifications.

**Typical client flow with HITL:**
```
POST /research  →  202 { job_id }
  poll GET /status  →  status: "planning"
  poll GET /status  →  status: "awaiting_approval"  ← show plan to user
POST /approve/{job_id} { decision: "approved" }
  poll GET /status  →  status: "running"
  poll GET /status  →  status: "complete"
GET /download/{job_id}  →  report.docx
```

> **Open Question**: Should the API support webhooks (callback URL on job completion AND on plan-ready) in addition to polling? Particularly useful for the HITL case — the client could receive a push notification when the plan is ready for review rather than polling. Adds complexity; defer to v2 unless a UI integration drives the requirement.

---

## Section 4: Prompt Engineering Guide

### Lead Agent Prompt Principles

The LeadResearcher prompt is the most consequential prompt in the system. It determines the quality of task decomposition, the precision of sub-agent instructions, and the judgment applied at every decision point.

**Query decomposition:**
Instruct the lead agent to first articulate what it needs to know before deciding how to find it. The decomposition step should produce: (a) a list of 2-3 distinct sub-topics, (b) a rationale for why each is necessary, (c) an assessment of which can be researched in parallel vs. which depend on others.

**Writing sub-agent task descriptions:**
The lead agent must be instructed to write task descriptions that include all of the following:
- The specific objective of this sub-task (one sentence)
- The output format expected (structured, with field names)
- The scope boundary (what NOT to research)
- Suggested search strategies (starting search queries or keywords)
- Stopping criteria (when is enough enough for this sub-task)
- The question this sub-task is answering in service of the larger goal

**Complexity judgment:**
The lead agent should assess complexity before decomposing. A 3-point rubric works well:
- **Simple**: Answerable from a single authoritative source, limited factual surface area → 1 sub-agent
- **Moderate**: Requires synthesis across 3-5 sources, 1-2 distinct sub-topics → 2 sub-agents
- **Deep**: Multi-dimensional, contested or rapidly-evolving domain, 3+ sub-topics → 3 sub-agents

**Stopping criteria for the lead agent:**
Explicitly instruct the lead agent on when to stop requesting more research:
- All planned sub-topics have been covered with acceptable confidence
- New research rounds are returning facts already present in accumulated findings
- The iteration budget is exhausted
- The source map has reached saturation (same sources appearing repeatedly)

---

### Sub-Agent Prompt Principles

**Objective clarity:**
Every sub-agent prompt opens with a single-sentence objective. Not "research the competitive landscape" but "identify the top 5 competitors to [Company X] by market share, including their key product differentiators and publicly available pricing information."

**Output format enforcement:**
Sub-agents must return structured outputs, not prose narratives. Specify the exact fields required. Unstructured output creates parsing failures downstream and makes the LeadResearcher's synthesis job significantly harder.

**Search strategy:**
Instruct sub-agents to begin with broad orienting searches, then narrow. The first 2 searches should establish the landscape. Subsequent searches should fill specific gaps identified from the initial results.

**Source evaluation heuristics to embed:**
- Prefer sources with known authorship and institutional affiliation
- Prefer recent publication dates for rapidly-evolving topics
- Treat Wikipedia as a starting point for context, not as a primary citation
- Flag sources from obvious SEO content farms (thin content, excessive ads, no clear authorship)
- Prefer primary sources (company announcements, research papers, regulatory filings) over secondary coverage where available

**When to give up on a sub-topic:**
If 3 consecutive search rounds yield no new information and no primary sources can be found, instruct the agent to report this as a coverage gap rather than continuing to search. Endless searching for nonexistent sources is a documented failure mode (per Anthropic's Research engineering team).

---

### Scaling Rules to Embed

These rules should be embedded in the LeadResearcher's prompt as explicit heuristics:

| Query Type | Sub-Agents | Tool Calls per Agent | Notes |
|---|---|---|---|
| Simple fact-finding | 1 | 3–10 | Single authoritative source usually sufficient |
| Direct comparison (A vs. B) | 2 | 10–15 each | One agent per subject |
| Market/industry overview | 2–3 | 10–15 each | Divide by dimension (players, trends, data) |
| Deep research / literature review | 3 | 15–20 each | Clearly divided sub-topics |

The lead agent should be instructed: "Spawning more sub-agents than the task warrants is not thoroughness — it is waste. A simple factual query answered by 3 agents costs 3x more and produces no better output."

---

### Anti-Patterns to Prevent

Per Anthropic's Research engineering team, these failure modes appear consistently without explicit countermeasures in the prompt:

**Over-spawning for simple queries:**
Prevent by requiring the lead agent to state its complexity assessment before spawning sub-agents. If complexity is "simple," the prompt should constrain to 1 sub-agent.

**Endless searching for nonexistent sources:**
Prevent by building stopping criteria into the sub-agent prompt: "If you have not found a relevant source after 3 search attempts on this specific question, record it as a coverage gap and move on."

**Duplicate work across sub-agents:**
Prevent by including explicit scope boundaries in each sub-agent's task description. The lead agent's prompt should instruct it to define non-overlapping research territories before dispatching.

**Overly verbose or specific search queries:**
Prevent by including examples in the sub-agent prompt. Bad: "What are the most recent quarterly earnings results for technology companies in the enterprise software sector for fiscal year 2025?" Good: "enterprise software earnings Q4 2025".

**Choosing SEO content farms over authoritative sources:**
Prevent by including source quality heuristics in the sub-agent prompt (see above). Instruct agents to check: who wrote this, when, and for what purpose?

**Hallucinating citations:**
Prevent architecturally (via the CitationAgent's flag for unverified claims) and in the sub-agent prompt (instruct agents to record source URLs alongside every fact extracted, not as an afterthought).

---

### Self-Improvement Loop

The system should support a diagnostic loop for prompt improvement:

1. Collect failed or low-quality research runs (identified via evaluation scoring — see Section 5)
2. For each failure, pass the full run trace (task descriptions, sub-agent outputs, quality scores) to Claude with the prompt: "Identify the most likely cause of this research failure. Was it the task description, the search strategy, the stopping criteria, or the synthesis step? Suggest a specific prompt modification."
3. Cluster similar diagnoses across multiple failures to identify systematic prompt weaknesses
4. Revise prompts based on clustered recommendations and re-run evaluation suite

Per Anthropic's Research engineering team, having an agent use a tool dozens of times and then rewrite the tool description based on observed failures produces significantly better descriptions than human-written descriptions alone. Apply this same principle to task description templates.

---

### Thinking/Scratchpad Strategy

Extended thinking (Claude's internal reasoning before producing output) should be activated for the following LeadResearcher steps:

- **Query decomposition**: Reasoning about what the question is really asking, what ambiguities exist, and what sub-topics are truly independent
- **Complexity assessment**: Reasoning about how much is known about this domain and how many sources will likely be needed
- **Sub-agent role definition**: Reasoning about how to divide research territory with minimal overlap
- **Sufficiency evaluation**: Reasoning about whether gaps are material or acceptable

Extended thinking should NOT be activated for sub-agents — they are executing well-defined tasks, not making strategic judgments. Thinking overhead on sub-agents adds cost without improving the quality of search-and-extract work.

---

## Section 5: Evaluation Strategy

### Start Small

Per Anthropic's Research engineering team, begin evaluation with approximately 20 representative queries before building a large evaluation suite. Effect sizes in early-stage multi-agent systems are large — a single prompt change can move success rates from 30% to 80%. Running hundreds of evaluation cases before the system is stable wastes tokens and obscures signal.

The initial 20 queries should span the complexity tiers defined in Section 4 (simple, moderate, deep) and cover several domains (technology, business, healthcare, general knowledge). Ensure at least 2-3 queries known to be edge cases or likely failure modes.

---

### LLM-as-Judge Framework

Each completed research run is evaluated by a single LLM judge call. Per Anthropic's Research engineering team, a single LLM judge with a structured rubric was more consistent than a multi-judge panel — multi-judge approaches introduced disagreement noise that made iteration signals unclear.

**Evaluation rubric (0.0 – 1.0 per dimension):**

| Dimension | What's being measured |
|---|---|
| Factual accuracy | Are the claims made in the report accurate and consistent with the cited sources? |
| Citation completeness | Does every factual claim have an inline citation? No exceptions. |
| Citation validity | Do the cited URLs resolve and actually contain the cited information? |
| Completeness | Does the report adequately address all aspects of the original query? |
| Source quality | Are the sources authoritative, recent, and relevant (not SEO farms or low-authority blogs)? |
| Tool efficiency | Was the number of tool calls appropriate for the query complexity? No wasteful over-searching. |
| Document structure | Clear headings? Logical flow? Executive summary present? Bibliography complete? |
| Synthesis quality | Does the report coherently connect findings across sub-topics? No mere concatenation? |

**Aggregate score**: Weighted average. Citation completeness and factual accuracy carry higher weights than document structure.

**Judge prompt structure:**
The judge receives: the original query, the completed report, and the source map. It produces a score for each dimension with a one-sentence rationale. The rationale is essential — score trends are easy to track, but actionable diagnosis requires knowing why a score is low.

---

### Human Evaluation Layer

Automated evaluation misses what LLM judges cannot reliably detect:

- **Hallucinations on unusual queries**: LLM judges can miss subtle hallucinations in niche domains where the judge's training data is thin
- **Source selection bias**: Judges may not recognize that a source is an SEO content farm if the content surface appears authoritative
- **Edge case failure modes**: Novel query types that fall outside the evaluation suite
- **Coherence at the document level**: LLM judges evaluate paragraphs well but may miss macro-level incoherence (the sections don't tell a consistent story)

Human evaluation should be conducted on a sample of completed runs — particularly those near the automated score decision boundary (e.g., scores between 0.5 and 0.7) and any run flagged with UNVERIFIED claims by the CitationAgent.

---

### Key Metrics to Track

Per Anthropic's Research engineering team, token usage explains 80% of performance variance. Token tracking is therefore not just a cost metric — it is a quality signal.

| Metric | Why it matters |
|---|---|
| Total tokens per query | Primary quality proxy; track by complexity tier |
| Token split: lead vs. sub-agents | Imbalance may indicate over-burdened orchestrator or idle sub-agents |
| Tool calls per sub-agent | Efficiency proxy; too few = shallow research, too many = looping |
| Research duration (wall clock) | User experience and timeout calibration |
| Source quality distribution | Ratio of authoritative to low-quality sources |
| Citation completeness rate | Hard requirement; any score below 1.0 is a failure mode |
| Cost per query | Track by complexity tier to set user-facing cost expectations |
| Iterations to sufficiency | How many rounds does the system need before stopping? |

---

### Evaluation Categories by Complexity

Different query types warrant different evaluation expectations:

**Simple fact-finding:**
- Expected: 1 sub-agent, 5-10 tool calls, 1 iteration, high citation completeness
- Failure indicators: multiple sub-agents spawned, >15 tool calls, low factual accuracy

**Direct comparisons:**
- Expected: 2 sub-agents with clear scope separation, parallel execution, comparable coverage depth on both subjects
- Failure indicators: one subject covered more thoroughly than the other, sub-agents' research overlapping

**Multi-faceted research:**
- Expected: 3 sub-agents, 2+ iterations, broad source diversity, strong synthesis in the executive summary
- Failure indicators: coverage gaps in major sub-topics, weak cross-topic synthesis, low completeness score

**Cross-domain synthesis:**
- Expected: Sources from multiple domains (technical, regulatory, business), explicit handling of conflicting information from different domains
- Failure indicators: single-domain perspective presented as complete, contradictions between sub-topics not acknowledged

---

### Regression Testing

Multi-agent systems exhibit emergent behavior: a prompt change to the LeadResearcher can unpredictably affect sub-agent behavior through the task descriptions it generates. This makes regression testing non-trivial.

**Regression prevention strategy:**

1. Maintain a curated set of 20-30 reference queries with expected quality scores
2. After any prompt change, run the full reference set and compare score distributions
3. Flag any query where the score drops by more than 0.1 on any dimension
4. Treat score regressions on previously-passing queries as bugs, not acceptable tradeoffs
5. Track prompt versions in the evaluation database so regressions can be traced to specific changes

Per Anthropic's Research engineering team, small lead agent prompt changes can have large, non-obvious downstream effects on sub-agent behavior. Never evaluate prompt changes on a subset of the reference suite.

---

### Observability

The system includes a built-in observability layer (Phase 3), surfaced in the **🔭 Observability** tab of the Gradio UI. It is designed to monitor agent behaviour without exposing research content in monitoring systems.

**What is monitored:**
- Per-job pipeline timing: each major node (plan_research, spawn_subagents, synthesize, cite) is wrapped in a span with start/end/duration
- Per-job cost: token consumption per agent, converted to USD using model-specific pricing, accumulated in `CostTracker`
- Per-job, per-step LLM call health: total calls, retries, and failures tracked via thread-local context in `groq_retry.py`
- System-level 24h aggregates: request counts, success rate, average duration, total cost, budget-exceeded count
- Job activity feed: timestamped events emitted by each node and sub-agent during execution, persisted to SQLite

**What is NOT monitored:**
- Research content or user query text in dashboards — queries appear only in the job-specific Activity Log and Job Inspector views

**Implementation:**
All observability data is persisted to the `job_traces` table in `data/jobs.db` on job completion. This means the Observability UI remains fully functional after server restarts and across multiple runs. The three-sub-tab UI design provides a clear separation of concerns: system health, job-level inspection, and raw event logs.

---

## Section 6: Production Readiness

### Statefulness Challenges

Multi-agent research systems are stateful across many tool calls and many minutes of execution. This creates compounding failure risks that do not exist in stateless API call patterns:

- **Error propagation**: A poor sub-agent result in iteration 1 shapes the LeadResearcher's iteration 2 task descriptions, potentially compounding the error
- **Context drift**: As context windows fill, model behavior can drift subtly — later decisions may not be coherent with early decisions
- **Partial failure**: One sub-agent timing out affects synthesis quality for the entire run
- **Non-determinism**: The same query run twice will produce different results due to model temperature and web content variability

Mitigations for each of these are described throughout this document (checkpointing in Section 3, context management in Section 3, timeout handling in Section 3, evaluation rubric in Section 5).

---

### Deployment Strategy

**Rainbow deployments:**
Per Anthropic's Research engineering team, standard blue/green deployments are inappropriate for long-running agentic jobs. Cutting over mid-run to a new version can leave in-flight jobs in an inconsistent state if the new version has different state schema or prompt behavior.

Rainbow deployments allow multiple versions to run simultaneously. In-flight jobs continue on the version they started with. New jobs start on the current version. Versions are retired only after all in-flight jobs on that version complete.

**Version management:**
Three independent versioning axes must be tracked:
1. **Prompt versions**: Each agent's prompt is versioned independently. A prompt version change requires evaluation against the regression suite before promotion to production.
2. **Tool versions**: Tool descriptions and tool implementations are versioned. Tool description changes follow the same evaluation gate as prompt changes.
3. **Graph execution logic**: Changes to node behavior, edge conditions, or state schema require careful migration planning — state objects from previous versions must remain readable.

**Environment parity:**
The development environment must be able to reproduce production behavior. This means the evaluation suite must run against the same tool endpoints (real web search, real web fetch) rather than mocked versions. Mock-based tests are useful for unit testing graph logic, but they cannot validate research quality.

---

### Cost Management

The 15x token multiplier is a structural reality of this architecture. At moderate Claude API pricing:
- A simple query: ~5,000-10,000 tokens → $0.05-0.10
- A moderate query: ~20,000-40,000 tokens → $0.20-0.40
- A deep research query: ~60,000-100,000 tokens → $0.60-1.00

**Cost control mechanisms:**

- **Token budget per query**: Set at job creation based on depth parameter. The LeadResearcher is informed of its budget and must stay within it.
- **Iteration hard cap**: Maximum iterations enforced at the graph level — not just as a prompt instruction. A prompt instruction can be ignored; a graph-level constraint cannot.
- **Sub-agent timeout**: Prevents runaway sub-agents from consuming unlimited tokens
- **Complexity routing**: Automatically route simple queries (as assessed by a cheap classification call) to the 1-agent path, bypassing orchestration overhead

> **Open Question**: Should cost per query be exposed in the API response for user-visible billing transparency? Yes in principle, but requires accurate token-to-cost mapping that may need updating as API pricing changes. Design the metadata schema to include it now; populate it with an estimate that can be made precise later.

---

### Rate Limiting & Quotas

**Anthropic API rate limits:**
The Anthropic API imposes rate limits at the organization level (requests per minute, tokens per minute). With 3 parallel sub-agents each making multiple tool calls, these limits can be hit faster than expected. Mitigations:
- Implement exponential backoff on 429 responses (see Section 3: Error Handling)
- Track tokens-per-minute consumption in real time; slow sub-agent dispatch if approaching limits
- Consider a token-rate-aware dispatcher that staggers sub-agent launches by a small delay

**Concurrent research job limits:**
Running multiple deep research jobs simultaneously multiplies the token rate. Implement:
- A global concurrent job cap (configurable; start conservatively at 2-3 simultaneous deep jobs)
- A per-user concurrent job cap to prevent single users from monopolizing capacity
- A queue with position feedback in the `/status` response

**User-level quotas:**
For any multi-tenant deployment:
- Daily token budget per user tier
- Per-query cost cap (reject or warn before starting jobs that would exceed the cap)
- Historical usage visibility via API

---

### Debugging & Observability

Multi-agent systems are significantly harder to debug than single-agent systems because failures are often emergent and non-deterministic.

**Full production tracing:**
Every job must produce a complete trace: every node execution, every tool call, every state mutation, every agent decision. This trace must be retained and queryable. LangSmith or equivalent tracing infrastructure is required from day one — retrofitting tracing into a production system is painful.

**Agent decision logging:**
Beyond tool calls, log the key judgment calls: complexity assessment at `plan_research`, sufficiency signal at `evaluate_sufficiency`, coverage gap list at `collect_results`. These are the inflection points where the system succeeds or fails, and they are the first place to look when debugging.

**Failure mode dashboards:**
Track failure modes by category:
- `tool_error`: Web search or fetch failures
- `timeout`: Sub-agent exceeded wall-clock budget
- `format_error`: Sub-agent returned malformed output
- `loop_excess`: System hit iteration cap without reaching sufficiency
- `citation_gap`: CitationAgent flagged uncited claims
- `save_error`: Document file write failed

**Non-determinism management:**
The same query will not produce identical results across runs. This is expected and acceptable — it is not a bug. However, it makes regression testing noisier. Mitigate by: running evaluation queries multiple times and using median scores, maintaining a "ground truth" set of unambiguously correct answers for simple fact-finding queries, and treating non-determinism as a reason to run more evaluation rather than less.

---

### Known Limitations & Future Work

**Current limitations:**

- **Synchronous sub-agent execution within a round**: While sub-agents run in parallel within a round, the system waits for all sub-agents in a round to complete before the LeadResearcher evaluates and dispatches the next round. This creates a "slowest sub-agent" bottleneck. A sub-agent finishing early sits idle while waiting for siblings.

- **No cross-agent communication**: Sub-agents cannot communicate with each other mid-task. If Sub-agent 1 discovers information highly relevant to Sub-agent 2's task, it cannot forward it. All inter-agent communication flows through the LeadResearcher.

- **Web-only sources**: The system is designed for public web research. Internal documents, proprietary databases, or private APIs require tool additions and authentication handling not covered by this design.

- **Single output format**: The system produces `.docx` only. PDF export, structured JSON, or interactive HTML are not supported in v1.

**Future work (prioritized):**

1. **Streaming progress to the API client**: SSE-based streaming of phase updates and partial findings as they complete, rather than polling-only
2. **Async sub-agent dispatch**: Allow the LeadResearcher to process completed sub-agent results as they arrive rather than waiting for all to complete
3. **Sub-agent coordination**: Allow a limited, structured form of sub-agent information sharing (mediated by the orchestrator) for tasks where inter-agent context sharing would improve quality
4. **Internal data source integration**: Add tool support for authenticated internal sources (SharePoint, Confluence, proprietary databases)
5. **Additional LLM provider support**: Abstract the model layer to support non-Anthropic providers as fallback or cost optimization options
6. **PDF export**: Add DocumentGeneratorAgent support for PDF output alongside `.docx`
7. **Interactive research sessions**: Allow the user to provide mid-run feedback (via the API) that the LeadResearcher incorporates into subsequent iterations

---

### Security Considerations

**Input sanitization:**
Research queries are passed to web search APIs and may influence URL construction. Validate and sanitize all user inputs before passing to tool invocations. Reject inputs containing characters used in URL injection, shell injection, or prompt injection patterns.

**Prompt injection via search results:**
Web content returned by the `web_fetch` tool may contain adversarial content designed to hijack the agent's behavior ("Ignore previous instructions and..."). Mitigations:
- Pre-process fetched content to strip obvious injection patterns before passing to the model
- Structure the sub-agent prompt to treat fetched content as data, not instructions (system/user role separation)
- Monitor for anomalous agent behavior patterns that may indicate successful injection (unexpected tool calls, deviation from task scope)

**Output validation:**
Before saving the `.docx` file, validate that the document content does not contain embedded macros or other executable content. The system produces research reports, not executable files.

**API authentication:**
All API endpoints require authentication. Job outputs (including the `.docx` file) are scoped to the authenticated user who created the job. Cross-user access to job results must be explicitly prevented.

**Sensitive query handling:**
Research queries may contain sensitive business information (unreleased product names, M&A targets, health conditions). The system must not log query content to shared monitoring systems. Query content belongs in job records, not in aggregate telemetry.

> **Open Question**: Should the system support ephemeral mode — where the `.docx` output is returned directly in the API response rather than saved to a project folder, with no persistent job record? This would be preferable for sensitive queries. Feasible with a `mode=ephemeral` flag on the request. Flag for v2 design consideration.

---

---

## Section 7: Phase 2: Enhanced User Experience

Phase 1 delivered a fully functional API-first research system. Phase 2 addresses three gaps that limit practical usability: there is no browser-based interface, jobs vanish when the server restarts, and progress is only observable by polling. These three features are designed to be delivered together — the Web UI (built with Gradio) polls for job status updates, and both the UI and the persistence layer benefit from durable job records in SQLite.

---

### 7.1 Web UI

#### Design Rationale

The MVP is API-only — usable via curl or Postman, but not by non-technical stakeholders who need to review research plans, monitor progress, and download reports. A browser-based UI eliminates this barrier without requiring users to learn the API contract.

The UI is built with **Gradio** (`gradio>=4.0.0`), mounted directly onto the FastAPI application via `gr.mount_gradio_app(app, demo, path="/ui")`. Gradio serves its own assets — no `app/ui/static/` folder, no `StaticFiles` mount, and no HTML, CSS, or JavaScript files to maintain.

#### Tech Choice: Gradio

**Why Gradio and not vanilla HTML/CSS/JS:**

- **Professional components with zero CSS authoring** — `gr.Textbox`, `gr.Dropdown`, `gr.Dataframe`, `gr.File`, `gr.Button`, and `gr.Markdown` provide polished, accessible UI elements that would take days to replicate by hand.
- **Native FastAPI integration** — `gr.mount_gradio_app()` attaches the Gradio app directly to the existing FastAPI instance with one line of code in `app/main.py`. No separate server, no port conflict, no CORS configuration.
- **Built-in file download** — `gr.File` streams a `.docx` to the browser natively. The vanilla approach required a custom `StreamingResponse` endpoint and an anchor tag with a `download` attribute.
- **Built-in polling support** — `gr.Timer` fires a Python callback on a configurable interval, driving status refresh from the server side. This replaces the SSE `EventSource` client logic that vanilla JS required.
- **Hours, not days** — the full four-tab UI is buildable in a single Python file (`app/ui/gradio_app.py`) with no build toolchain, no `npm install`, and no CSS.
- **Looks professional out of the box** — Gradio's default theme requires no visual polish work. An optional theme parameter (`gr.themes.Soft()`) can be passed if visual customization is desired.

#### Layout Description

The UI is a `gr.Blocks` layout with four tabs:

**Tab 1 — New Research:**
- `gr.Textbox` (multiline) for the research query
- `gr.Dropdown` for depth selection: `simple`, `moderate`, `deep`
- `gr.Button` — Submit
- `gr.Textbox` (read-only) — displays the returned `job_id`

**Tab 2 — Job Status:**
- `gr.Textbox` for job ID input
- `gr.Button` — Check Status (manual trigger)
- `gr.Markdown` — current status, phase, timestamps, token usage
- `gr.Markdown` — research plan display (visible only when `awaiting_approval`)
- `gr.Button` × 3 — Approve, Refine, Reject (visible only when `awaiting_approval`)
- `gr.Textbox` — feedback text for Refine (visible only when `awaiting_approval`)
- `gr.Timer` — fires every 5 seconds to auto-refresh status

**Tab 3 — Job History:**
- `gr.Dataframe` — table of all past jobs (`job_id`, `query`, `status`, `created_at`, `duration_seconds`) loaded from SQLite; shows all runs ever submitted, persisted across server restarts
- `gr.Button` — Refresh

**Tab 4 — Download Report:**
- `gr.Textbox` for job ID input
- `gr.Button` — Download
- `gr.File` — streams the `.docx` report to the browser

#### Interaction Flow

1. User opens the browser at `http://localhost:8000/ui`.
2. User enters a query and depth in the **New Research** tab and clicks Submit. The handler calls `POST /research` and displays the returned `job_id`.
3. User pastes the `job_id` into the **Job Status** tab. The `gr.Timer` begins polling `GET /status/{job_id}` every 5 seconds and updates the status display.
4. When `hitl_status` transitions to `awaiting_approval`, the research plan is rendered and the Approve / Refine / Reject buttons appear.
5. User approves (or refines or rejects) the plan. The handler calls `POST /approve/{job_id}`. The timer continues polling and the status display updates as the job progresses.
6. When the job completes, the user switches to the **Download Report** tab and enters the `job_id`. Clicking Download calls `GET /download/{job_id}` and Gradio delivers the file to the browser.
7. The **Job History** tab shows all past jobs at any time — the user can click Refresh to reload from SQLite.

#### Report Re-download

Any completed job can be re-downloaded from the **Download Report** tab as long as the `.docx` file still exists in `output/`. Job records persist in `data/jobs.db` across server restarts, so the history tab shows all previous runs. Users are not limited to jobs from the current server session.

> **Decision Log**: Gradio was chosen over vanilla HTML/CSS/JS because it provides professional-grade UI components without requiring frontend expertise, integrates natively with FastAPI via a single mount call, handles file downloads through a built-in component, and ships with a polling mechanism (`gr.Timer`) that replaces the SSE infrastructure the vanilla approach required. The tradeoff is a dependency on the Gradio library; this is acceptable because Gradio is a well-maintained, widely-used Python library with a stable API since 4.0. For a tool-internal application, the productivity gain far outweighs the dependency cost.

---

### 7.2 Job Persistence

#### Why SQLite

The MVP uses an in-memory Python dict (`job_store.py`) for job state. This is fast and zero-configuration, but all job records are lost when the server restarts. For any multi-session research workflow — including the HITL checkpoint, which may sit in `awaiting_approval` overnight — this is a critical reliability gap.

SQLite is the correct choice for the Phase 2 persistence layer:

- **Zero external dependencies**: SQLite is part of Python's standard library (`sqlite3`). No server, no Docker container, no configuration.
- **Durable by default**: The database file (`data/jobs.db`) survives server restarts, crashes, and deployments.
- **Single-writer, multi-reader**: The system runs with `--workers 1`, so SQLite's write serialization is not a bottleneck. SQLite's concurrency model is a perfect match for this deployment model.
- **Queryable**: Job records are inspectable with any SQLite client without writing custom tooling.
- **Right-sized**: The system is not a high-throughput database workload. SQLite handles thousands of job records and hundreds of concurrent readers without performance concerns.

No ORM is used. Python's `sqlite3` module is sufficient — it provides parameterized queries, transaction control, and row factories. An ORM would add a dependency and a learning surface for a schema that has one table and a handful of columns.

#### Schema Design

The `jobs` table stores all fields currently held in the in-memory `JobStore` dict:

```
jobs (
    job_id          TEXT    PRIMARY KEY,
    query           TEXT    NOT NULL,
    depth           TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    phase           TEXT,
    research_plan   TEXT,           -- JSON blob: full research_plan dict
    document_path   TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL,   -- ISO 8601 UTC
    updated_at      TEXT    NOT NULL,   -- ISO 8601 UTC
    token_usage     TEXT,           -- JSON blob: {"lead": N, "sub_agents": N, "total": N}
    hitl_status     TEXT,
    hitl_round      INTEGER,
    iteration_count INTEGER,
    source_count    INTEGER,
    citation_count  INTEGER,
    duration_seconds REAL,
    summary_snippet TEXT
)
```

JSON blobs (`research_plan`, `token_usage`) are stored as serialized strings and deserialized on read. This avoids schema complexity for fields that are only ever read/written as a whole unit. The `updated_at` field is set on every `UPDATE` and serves as a coarse audit log.

No migration framework is used in MVP. Schema is created idempotently at server startup with `CREATE TABLE IF NOT EXISTS`. Columns can be added with `ALTER TABLE ... ADD COLUMN` in startup logic when new fields are needed — this is sufficient for the evolution pace of an internal tool.

#### Practical Persistence Guarantees

- **Database file location**: `data/jobs.db` in the project root. Created automatically on first server startup — no manual setup required.
- **Durability across restarts**: Data persists across all server restarts. Jobs are never lost due to a process restart, crash, or redeployment. The Job History tab in the Gradio UI shows every job ever submitted, not just the current server session.
- **Report re-download**: Users can re-download old `.docx` reports from any previous run as long as the file still exists in `output/`. The `document_path` column in `jobs.db` stores the absolute path; the Download Report tab in the UI uses this to serve the file.
- **`.gitignore` placement**: The `data/` folder must be in `.gitignore`. It contains user-generated research data, not source code. The folder should be tracked with a `.gitkeep` placeholder so the directory exists in a fresh clone but the database file is never committed.
- **No extra dependencies**: `sqlite3` is part of Python's standard library. No database server, no Docker container, and no additional Python package is needed. `pip install` does not change for Phase 2 SQLite support.

#### Backward Compatibility via Same Interface

`job_store.py` exposes four methods: `create_job()`, `get_job()`, `update_job()`, and `list_jobs()`. The SQLite replacement retains this exact interface. No other module is aware of how jobs are stored. This means the swap from dict to SQLite requires changes only in `job_store.py` — zero changes to `routes.py`, `nodes.py`, `runner.py`, or any other file.

This interface contract also defines the upgrade path to PostgreSQL: replace the `sqlite3` connection in `job_store.py` with `psycopg2` (or `asyncpg` for async). The table schema maps directly to PostgreSQL with no changes beyond connection string and driver. All calling code remains unchanged.

#### Upgrade Path to PostgreSQL

When the system moves to multi-worker deployment or a managed cloud environment, PostgreSQL replaces SQLite by:

1. Creating the equivalent table in PostgreSQL (identical schema)
2. Replacing `sqlite3.connect()` with `psycopg2.connect()` in `job_store.py`
3. Adjusting placeholder syntax (`?` → `%s`) and JSON handling
4. Migrating existing records with a one-time `INSERT INTO ... SELECT FROM ...` script

No application code outside `job_store.py` changes. The interface contract is the migration path.

> **Decision Log**: SQLite was chosen over Redis because Redis introduces an external service dependency that is not justified for a single-worker deployment. SQLite files are simpler to back up, inspect, and migrate than a Redis keyspace. The `job_store.py` interface abstraction means the storage backend is swappable at any time — the architectural commitment is to the interface, not the storage engine.

---

### 7.3 Real-time Progress (Gradio Polling)

#### Approach

Progress updates are delivered by periodic polling rather than a persistent server-sent event (SSE) stream. The Gradio `gr.Timer` component fires a Python callback every 5 seconds; the callback calls `job_store.get_job(job_id)` and returns the current job state to update the UI components in the Job Status tab.

This is simpler than SSE because:

- No dedicated streaming endpoint (`GET /stream/{job_id}`) is needed on the FastAPI side.
- No per-job `queue.Queue` infrastructure, no module-level `_event_queues` dict, and no `emit_event()` calls in graph nodes.
- No browser-side `EventSource` object or event handler wiring.
- The polling callback is a plain Python function reading directly from SQLite — straightforward to understand, test, and debug.

#### Why Polling is Sufficient

Research jobs run for 3–8 minutes. At a 5-second polling interval, the UI is at most 5 seconds behind the actual job state. For a task measured in minutes, a 5-second lag is imperceptible to users. The primary observable state transitions (queued → planning → awaiting_approval → running → complete) each last tens of seconds to minutes — none are so brief that a 5-second poll misses them.

An SSE stream provides sub-second latency. For this application, that precision provides no practical benefit and adds significant implementation complexity: a custom streaming endpoint, a thread-to-async-event-loop bridge via `queue.Queue`, heartbeat events to keep proxies alive, and reconnection logic. The Gradio polling approach delivers equivalent user experience at a fraction of the implementation surface.

#### Polling vs. SSE: Architecture Comparison

| Aspect | SSE | Gradio polling |
|--------|-----|---------------|
| Server infrastructure | Custom endpoint + per-job `queue.Queue` + `emit_event()` in every node | None — `job_store.get_job()` only |
| New files required | `app/api/sse.py` | None |
| Browser client code | `EventSource` + event handlers | Built into Gradio (`gr.Timer`) |
| Update latency | Sub-second | 5 seconds (configurable) |
| Missed events on reconnect | Possible | None — always reads current DB state |
| Nodes must emit events | Yes — every phase-changing node must call `emit_event()` | No — nodes write to `job_store` only (already required) |

#### How Graph Nodes Report Progress

Graph nodes do not need to emit events for the UI to reflect current state. Each node already calls `job_store.update_job(job_id, status=..., phase=...)` as part of its normal operation. The Gradio polling timer reads these updates directly from SQLite. Progress visibility is a consequence of correct `job_store` writes — no additional instrumentation is required.

> **Decision Log**: Gradio's `gr.Timer` polling was chosen over a custom SSE implementation because it delivers equivalent user experience (status updates within 5 seconds) with zero server-side streaming infrastructure. The original SSE design required a new file (`app/api/sse.py`), per-job in-memory queues, a thread-to-asyncio bridge, and `emit_event()` calls in every graph node — significant complexity for a benefit (sub-second latency) that is irrelevant when jobs run for minutes. Polling reads directly from the durable SQLite record, which also means it correctly reflects state after a server restart, something an in-memory SSE queue cannot do.

---

*End of IDEAS.md — This document is the single source of truth for the Multi-Agent AI Research System architecture. All implementation decisions should trace back to a section here. When a decision is made that is not covered by this document, update this document first.*
