"""
app/agents/sub_agent.py
========================
ResearchSubAgent — a focused web researcher powered by Groq llama-3.3-70b-versatile.

Each sub-agent is assigned one sub-topic task brief by the LeadResearcher and
independently searches the web until it has gathered sufficient evidence or
hits the stopping criteria.

Search provider abstraction:
    This agent does NOT use any server-side web tools.
    Instead, it uses the SearchProvider abstraction (app/tools/) so the
    underlying search service (Tavily, Bing, SerpAPI, etc.) can be swapped
    by changing the SEARCH_PROVIDER environment variable — no code changes needed.

Agentic loop pattern (manual tool execution with Groq function calling):
    1. Send task to Groq with tool/function schemas (web_search, web_fetch)
    2. Groq returns tool_calls blocks indicating what to search/fetch
    3. We execute those calls via the SearchProvider
    4. Results are returned to Groq as tool role messages
    5. Loop continues until finish_reason == "stop"

    This pattern works with Groq's OpenAI-compatible API and is provider-agnostic
    because we control the loop, not the API server.

Methods:
    execute(task)   Run a fresh research task. Returns structured findings dict.
    rework(task)    Re-run a task with targeted gap-filling instructions from
                    LeadResearcher's synthesis review. Reuses existing sources.

Output format (per task):
    {
        "task_id":       int,
        "title":         str,
        "summary":       str,         — full narrative of findings
        "key_facts":     list[str],   — bullet-point facts for easy scanning
        "sources":       list[dict],  — [{url, title, date, relevance_score}]
        "confidence":    float,       — 0.0–1.0 self-assessed coverage confidence
        "coverage_gaps": list[str],   — topics the agent could not find good sources for
        "token_usage":   dict,        — {"sub_agents": total_tokens}
    }
"""

from __future__ import annotations
import json
import re
from groq import Groq
from app.utils.groq_retry import groq_chat, set_trace_context, clear_trace_context
from app.config import GROQ_API_KEY, GROQ_SUB_AGENT_MODEL, LLM_MAX_TOKENS_SUBAGENT, SEARCH_MAX_RESULTS, MAX_TOOL_ROUNDS

from app.tools.factory import get_search_provider
from app.tools.base import SearchProvider
from app.utils.job_store import job_store

_client = Groq(api_key=GROQ_API_KEY)
_MODEL = GROQ_SUB_AGENT_MODEL

# Lazy-initialised singleton — created on first sub-agent call, reused across all
_provider: SearchProvider | None = None


def _get_provider() -> SearchProvider:
    """Return the module-level search provider singleton, initialising if needed."""
    global _provider
    if _provider is None:
        _provider = get_search_provider()
    return _provider


# ── Tool schema definitions (OpenAI/Groq function calling format) ─────────────
# These are sent to Groq so it knows what tools are available.
# The actual execution is handled by the SearchProvider, not Groq's servers.

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information on a topic. "
                "Returns a ranked list of results with titles, URLs, and content snippets. "
                "Use this to find relevant sources and understand what information is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific for better results.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch the full text content of a specific web page by URL. "
                "Use this after web_search to read the complete content of a high-value source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


class ResearchSubAgent:
    """
    Focused research agent that searches the web for a single assigned sub-topic.

    Uses a manual agentic loop: Groq requests tool calls, we execute them via
    the configured SearchProvider, and return results until Groq is satisfied.
    """

    def execute(self, task: dict) -> dict:
        """
        Execute a research task end-to-end using web search.

        Args:
            task: Task brief dict from LeadResearcher.plan(), containing:
                    id, title, objective, scope, search_strategy,
                    output_format, stopping_criteria

        Returns:
            dict: Structured findings with summary, key_facts, sources,
                  confidence, coverage_gaps, token_usage.
        """
        task_id = task.get("id", 0)
        title = task.get("title", "Research")
        job_id = task.get("job_id", "")
        objective = task.get("objective", "")
        scope = task.get("scope", "")
        search_strategy = task.get("search_strategy", [])
        output_format = task.get("output_format", "")
        stopping_criteria = task.get("stopping_criteria", "")

        if job_id:
            job_store.emit_event(job_id, f"🤖 SubAgent-{task_id} [{title}] starting...")

        # Set thread-local context so groq_retry attributes calls to this job/step
        if job_id:
            set_trace_context(job_id, "spawn_subagents")

        search_hints = "\n".join(f"  - {q}" for q in search_strategy[:3])

        system = """You are a focused research sub-agent. Your job is to search the web \
thoroughly for specific information, evaluate sources, and return structured findings.

Rules:
- Search broadly first, then narrow to fill specific gaps
- Prefer authoritative sources (official docs, research papers, reputable news, announcements)
- Avoid SEO content farms (thin content, no clear authorship)
- Record the URL of every source you use
- Stop after finding 5+ distinct authoritative sources or after 10 searches (whichever first)
- Be specific and factual — no vague generalities"""

        user_prompt = f"""## Research Task
Title: {title}
Objective: {objective}
Scope: {scope}

## Suggested Starting Searches
{search_hints}

## Required Output Format
{output_format}

## Stopping Criteria
{stopping_criteria}

Research this topic now. After gathering sufficient information, provide your structured findings."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        all_sources: list[dict] = []
        token_total = 0
        max_tool_rounds = MAX_TOOL_ROUNDS  # hard cap on tool call rounds

        # ── Agentic loop ──────────────────────────────────────────────────────
        for _ in range(max_tool_rounds):
            response = groq_chat(_client,
                model=_MODEL,
                max_tokens=LLM_MAX_TOKENS_SUBAGENT,
                tools=_TOOL_DEFINITIONS,
                tool_choice="auto",
                messages=messages,
            )
            token_total += _count_tokens(response)

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # Groq is done — no more tool calls
            if finish_reason == "stop":
                break

            # Groq wants to call tools — execute them via the provider
            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls or []

                # Append assistant message with tool_calls to conversation history
                messages.append({
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # Execute each tool call and append results as tool messages
                for tc in tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}

                    result_content = _execute_tool(tc.function.name, tool_input, all_sources, job_id=job_id, task_id=task_id)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })
                continue

            # Unexpected finish reason — record as coverage gap and exit loop
            all_sources.append({
                "url": "",
                "title": f"[Search truncated: finish_reason={finish_reason}]",
                "date": None,
                "relevance_score": 0.0,
            })
            break

        final_text = _extract_text(response)
        structured = _parse_findings(final_text, task_id, title, all_sources)
        structured["token_usage"] = {"sub_agents": token_total}
        if job_id:
            job_store.emit_event(job_id, f"✅ SubAgent-{task_id} [{title}] done — {len(all_sources)} source(s) found")
            clear_trace_context()
        return structured

    def rework(self, task: dict) -> dict:
        """
        Targeted rework of a specific sub-topic using existing sources plus minimal new searches.

        Called by the targeted_rework node when LeadResearcher's self-review
        identifies coverage gaps in the synthesis.

        Args:
            task: Modified task dict with additional fields:
                    rework_instruction: str — specific gap to address
                    existing_sources:   list[dict] — sources already gathered

        Returns:
            dict: Same format as execute() — structured findings dict.
        """
        rework_instruction = task.get("rework_instruction", "")
        existing_sources = task.get("existing_sources", [])

        # Inject rework context into the task objective
        task["objective"] = f"REWORK: {rework_instruction}\n\nOriginal objective: {task.get('objective', '')}"
        task["stopping_criteria"] = "Stop after 3–5 additional targeted searches."

        # Hint about already-gathered sources to avoid redundant re-research
        if existing_sources:
            src_hint = "\n".join(
                f"  - {s.get('url', '')}: {s.get('title', '')}"
                for s in existing_sources[:5]
            )
            task["scope"] = (
                task.get("scope", "")
                + f"\n\nAlready have these sources — focus on filling gaps not covered by them:\n{src_hint}"
            )

        return self.execute(task)


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(
    tool_name: str,
    tool_input: dict,
    all_sources: list[dict],
    job_id: str = "",
    task_id: int = 0,
) -> str:
    """
    Execute a tool call requested by Groq using the configured SearchProvider.

    Mutates all_sources in-place to accumulate sources across all tool calls.

    Args:
        tool_name:   "web_search" or "web_fetch"
        tool_input:  Dict with "query" (search) or "url" (fetch)
        all_sources: Mutable list to append discovered sources to
        job_id:      Optional job UUID for activity logging.
        task_id:     Optional task ID for activity logging.

    Returns:
        str: Formatted result string to return as tool message content.
    """
    provider = _get_provider()

    if tool_name == "web_search":
        query = tool_input.get("query", "")
        if job_id:
            job_store.emit_event(job_id, f"🔍 SubAgent-{task_id} searching: '{query[:60]}'")
        results = provider.search(query, max_results=SEARCH_MAX_RESULTS)

        # Accumulate sources discovered during this search
        existing_urls = {s["url"] for s in all_sources}
        for r in results:
            if r.url and r.url not in existing_urls:
                all_sources.append({
                    "url": r.url,
                    "title": r.title,
                    "date": r.published_date,
                    "relevance_score": r.score or 0.8,
                })
                existing_urls.add(r.url)

        return provider.format_search_results(results)

    elif tool_name == "web_fetch":
        url = tool_input.get("url", "")
        if job_id:
            job_store.emit_event(job_id, f"🌐 SubAgent-{task_id} fetching: {url[:80]}")
        content = provider.fetch(url)
        if not content:
            return f"Could not fetch content from {url}"
        return f"Content from {url}:\n\n{content[:8000]}"

    return f"Unknown tool: {tool_name}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """Extract text content from a Groq chat completion response."""
    return response.choices[0].message.content or ""


def _count_tokens(response) -> int:
    """Sum prompt and completion tokens from a Groq response usage block."""
    if hasattr(response, "usage") and response.usage:
        return (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
    return 0


def _parse_findings(text: str, task_id: int, title: str, all_sources: list[dict]) -> dict:
    """
    Parse structured JSON findings from the agent's final response.

    Groq is instructed to return a JSON block. If it does, parse it and
    merge in any sources collected during the tool loop. Falls back to an
    unstructured dict if JSON is missing or malformed.

    Args:
        text:        Groq's final text response.
        task_id:     Sub-agent task ID for traceability.
        title:       Sub-topic title.
        all_sources: Sources accumulated during the tool loop.

    Returns:
        dict: Structured findings with task_id, title, summary, key_facts,
              sources, confidence, coverage_gaps.
    """
    # Try to parse a JSON block from Groq's response
    json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            # Merge tool-loop sources not already in Groq's JSON output
            existing_urls = {s.get("url") for s in data.get("sources", [])}
            for src in all_sources:
                if src.get("url") and src["url"] not in existing_urls:
                    data.setdefault("sources", []).append(src)
                    existing_urls.add(src["url"])
            data["task_id"] = task_id
            data["title"] = title
            return data
        except json.JSONDecodeError:
            pass

    # Fallback: treat entire response as unstructured summary
    return {
        "task_id": task_id,
        "title": title,
        "summary": text,
        "key_facts": [],
        "sources": all_sources,
        "confidence": 0.6,
        "coverage_gaps": [],
        "tool_call_count": len(all_sources),
    }
