"""
app/agents/lead_researcher.py
==============================
LeadResearcher agent — the orchestrator and chief analyst of the pipeline.

Powered by Groq's llama-3.3-70b-versatile model via the Groq Python SDK,
which provides an OpenAI-compatible API interface.

Responsibilities:
    plan()                  Decompose the user's query into N sub-topics and write
                            task briefs for each ResearchSubAgent. Returns a structured
                            research plan and a list of task dicts.

    evaluate_sufficiency()  After sub-agents return findings, decide whether coverage
                            is adequate or whether another research round is needed.
                            Returns one of: "sufficient" | "needs_more" | "force_stop".

    synthesize()            Weave all sub-agent findings into a coherent narrative.
                            Marks every factual claim with [CITE: url] for the
                            CitationAgent to process downstream.

    review_synthesis()      Self-review the synthesis against the original task map.
                            Returns a signal: "approved" | "needs_rework" | "force_proceed"
                            plus per-task rework instructions if signal is "needs_rework".

Input/Output summary:
    All methods take structured Python dicts/strings and return dicts.
    No LangGraph state is accessed directly — this class is pure agent logic,
    keeping it testable independently of the graph.

Model:
    llama-3.3-70b-versatile via Groq API (fast inference, strong reasoning).
"""
from __future__ import annotations
import json
from groq import Groq
from app.utils.groq_retry import groq_chat
from app.config import (
    GROQ_API_KEY, GROQ_MODEL,
    LLM_MAX_TOKENS_PLAN, LLM_MAX_TOKENS_EVALUATE,
    LLM_MAX_TOKENS_SYNTHESIZE, LLM_MAX_TOKENS_REVIEW,
    FINDINGS_TRUNCATE_CHARS, EVALUATE_TRUNCATE_CHARS, REVIEW_TRUNCATE_CHARS,
)

_client = Groq(api_key=GROQ_API_KEY)
_MODEL = GROQ_MODEL


class LeadResearcher:

    # ─────────────────────────────────────────────────────────────────
    # plan()
    # ─────────────────────────────────────────────────────────────────
    def plan(self, query: str, depth: str, hitl_feedback: str = "", hitl_round: int = 0) -> dict:
        """Decompose the query into sub-topics and write sub-agent task descriptions."""
        depth_map = {"simple": 1, "moderate": 2, "deep": 3}
        n_agents = depth_map.get(depth, 2)

        feedback_section = ""
        if hitl_feedback:
            feedback_section = f"\n\n## REFINEMENT FEEDBACK FROM HUMAN (round {hitl_round})\n{hitl_feedback}\nIncorporate this feedback completely into your revised plan."

        prompt = f"""You are the LeadResearcher for a multi-agent research system.

## Research Query
{query}

## Requested Depth
{depth} → plan for {n_agents} sub-agent(s){feedback_section}

## Your Task
Produce a research plan with:
1. An interpreted_goal (1-2 sentences: what you understand the user is asking)
2. {n_agents} distinct sub_topics, each with: title, scope (what to research), search_strategy (2-3 starting search queries)
3. Complexity assessment
4. List of assumptions you are making
5. Estimated token usage (rough: simple=10000, moderate=30000, deep=60000)

Then produce one task_description per sub-topic for a ResearchSubAgent, each containing:
- id (int, 1-based)
- title
- objective (what to find, one sentence)
- scope (what to cover, what NOT to cover)
- search_strategy (3 example search queries as a list)
- output_format: "Return a JSON with: summary (str), key_facts (list[str]), sources (list[{{url,title,date}}]), confidence (0-1), coverage_gaps (list[str])"
- stopping_criteria: "Stop after 8-12 web searches or when you have 5+ distinct authoritative sources."

Return ONLY valid JSON in this exact structure:
{{
  "plan": {{
    "interpreted_goal": "...",
    "sub_topics": [
      {{"id": 1, "title": "...", "scope": "...", "assigned_to": "SubAgent 1"}}
    ],
    "sub_agent_count": {n_agents},
    "depth": "{depth}",
    "estimated_tokens": 30000,
    "assumptions": ["..."]
  }},
  "tasks": [
    {{
      "id": 1,
      "title": "...",
      "objective": "...",
      "scope": "...",
      "search_strategy": ["query1", "query2", "query3"],
      "output_format": "...",
      "stopping_criteria": "..."
    }}
  ]
}}"""

        response = groq_chat(_client,
            model=_MODEL,
            max_tokens=LLM_MAX_TOKENS_PLAN,
            messages=[{"role": "user", "content": prompt}],
        )

        text = _extract_text(response)
        parsed = _parse_json(text)

        tasks = parsed.get("tasks", [])

        # Fallback: if LLM returned sub_topics but no tasks, derive tasks from sub_topics
        if not tasks:
            sub_topics = parsed.get("plan", {}).get("sub_topics", [])
            tasks = [
                {
                    "id": t.get("id", i + 1),
                    "title": t.get("title", f"Sub-topic {i + 1}"),
                    "objective": t.get("scope", f"Research: {t.get('title', query)}"),
                    "scope": t.get("scope", ""),
                    "search_strategy": t.get("search_strategy", [query, f"{query} overview"]),
                    "output_format": (
                        'Return a JSON with: summary (str), key_facts (list[str]), '
                        'sources (list[{url,title,date}]), confidence (0-1), coverage_gaps (list[str])'
                    ),
                    "stopping_criteria": "Stop after 8-12 web searches or 5+ distinct authoritative sources.",
                }
                for i, t in enumerate(sub_topics[:n_agents])
            ]

        # Last resort: one generic task for the whole query
        if not tasks:
            tasks = [{
                "id": 1,
                "title": query[:60],
                "objective": f"Research: {query}",
                "scope": "Comprehensive research on the topic",
                "search_strategy": [query, f"{query} overview", f"{query} 2025"],
                "output_format": (
                    'Return a JSON with: summary (str), key_facts (list[str]), '
                    'sources (list[{url,title,date}]), confidence (0-1), coverage_gaps (list[str])'
                ),
                "stopping_criteria": "Stop after 8-12 web searches or 5+ distinct authoritative sources.",
            }]

        return {
            "plan": parsed.get("plan", {}),
            "tasks": tasks,
            "token_usage": {"lead": _count_tokens(response)},
        }

    # ─────────────────────────────────────────────────────────────────
    # evaluate_sufficiency()
    # ─────────────────────────────────────────────────────────────────
    def evaluate_sufficiency(
        self,
        query: str,
        findings: str,
        sub_agent_results: list[dict],
        iteration_count: int,
        max_iterations: int,
    ) -> str:
        """Returns 'needs_more', 'sufficient', or 'force_stop'."""
        coverages = [r.get("confidence", 0.5) for r in sub_agent_results]
        avg_confidence = sum(coverages) / len(coverages) if coverages else 0
        gaps = [g for r in sub_agent_results for g in r.get("coverage_gaps", [])]

        prompt = f"""You are evaluating research sufficiency.

Query: {query}

Iteration: {iteration_count}/{max_iterations}
Average sub-agent confidence: {avg_confidence:.2f}
Coverage gaps identified: {json.dumps(gaps[:10])}

Findings so far (first 2000 chars):
{findings[:EVALUATE_TRUNCATE_CHARS]}

Decide: is the research sufficient to write a quality report, or does it need another round?
- "sufficient": coverage is adequate, proceed to synthesis
- "needs_more": specific gaps that a targeted round would fix (only if iteration < max)
- "force_stop": return this if avg_confidence > 0.7 OR iteration >= {max_iterations - 1}

Return ONLY one of: sufficient / needs_more / force_stop"""

        response = groq_chat(_client,
            model=_MODEL,
            max_tokens=LLM_MAX_TOKENS_EVALUATE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(response).strip().lower()
        if "needs_more" in text and iteration_count < max_iterations - 1:
            return "needs_more"
        if "force_stop" in text:
            return "force_stop"
        return "sufficient"

    # ─────────────────────────────────────────────────────────────────
    # synthesize()
    # ─────────────────────────────────────────────────────────────────
    def synthesize(
        self,
        query: str,
        findings: str,
        sub_agent_results: list[dict],
        sub_agent_tasks: list[dict],
        source_map: dict,
        rework_instructions: list[dict],
    ) -> dict:
        """Synthesize sub-agent findings into a coherent narrative."""
        rework_note = ""
        if rework_instructions:
            rework_note = f"\n\n## REWORK INSTRUCTIONS\nAddress these specific gaps:\n" + "\n".join(
                f"- {i.get('instruction', '')}" for i in rework_instructions
            )

        sources_list = "\n".join(
            f"- [{s.get('title', url)}]({url})" for url, s in list(source_map.items())[:30]
        )

        prompt = f"""You are the LeadResearcher synthesizing research findings.

## Original Research Query
{query}

## Sub-Agent Findings
{findings[:FINDINGS_TRUNCATE_CHARS]}

## Available Sources
{sources_list}{rework_note}

## Instructions
Write a comprehensive research synthesis with these sections:

1. **Executive Summary** (3-5 paragraphs covering the most important findings)
2. **Research Findings** (organized by sub-topic, with specific facts and data points; mark each factual claim with [CITE: url] where you use a source)
3. **Recommendations** (actionable conclusions)
4. **Limitations & Gaps** (what was not found or is uncertain)

Rules:
- Every specific fact, statistic, or claim must be tagged [CITE: <url>] with the source URL from the available sources
- Be specific and substantive — no vague generalities
- Organize clearly by sub-topic under Research Findings
- Use the actual data from the findings, not generic statements

Write the full synthesis now:"""

        response = groq_chat(_client,
            model=_MODEL,
            max_tokens=LLM_MAX_TOKENS_SYNTHESIZE,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "narrative": _extract_text(response),
            "token_usage": {"lead": _count_tokens(response)},
        }

    # ─────────────────────────────────────────────────────────────────
    # review_synthesis()
    # ─────────────────────────────────────────────────────────────────
    def review_synthesis(
        self,
        query: str,
        narrative: str,
        sub_agent_tasks: list[dict],
        sub_agent_results: list[dict],
    ) -> dict:
        """Self-review synthesis against task assignment map. Returns signal + rework instructions."""
        tasks_summary = "\n".join(
            f"- Task {t['id']}: {t['title']} — objective: {t.get('objective', '')}"
            for t in sub_agent_tasks
        )
        confidences = {r.get("task_id"): r.get("confidence", 0.5) for r in sub_agent_results}

        prompt = f"""You are the LeadResearcher reviewing your own synthesis.

## Original Query
{query}

## Sub-agent tasks that were assigned:
{tasks_summary}

## Sub-agent confidence scores: {json.dumps(confidences)}

## Synthesis (first 3000 chars):
{narrative[:REVIEW_TRUNCATE_CHARS]}

## Review Criteria
Check:
1. Did every sub-topic get covered in the synthesis? (not just mentioned but substantively covered)
2. Are there contradictions or unsupported claims?
3. Is the overall quality sufficient for a professional research report?

Decision:
- "approved": synthesis is good enough, proceed to citation
- "needs_rework": specific addressable gaps exist — provide rework instructions per task_id
- "force_proceed": synthesis has issues but you've already reviewed twice, accept it

Return ONLY valid JSON:
{{
  "signal": "approved" | "needs_rework" | "force_proceed",
  "rework_instructions": [
    {{"task_id": 1, "instruction": "The section on X is missing Y. Re-research Z."}}
  ]
}}"""

        response = groq_chat(_client,
            model=_MODEL,
            max_tokens=LLM_MAX_TOKENS_REVIEW,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _parse_json(_extract_text(response))
        signal = parsed.get("signal", "approved")
        if signal not in ("approved", "needs_rework", "force_proceed"):
            signal = "approved"

        return {
            "signal": signal,
            "rework_instructions": parsed.get("rework_instructions", []),
            "token_usage": {"lead": _count_tokens(response)},
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """Extract the text content from a Groq chat completion response."""
    return response.choices[0].message.content or ""


def _count_tokens(response) -> int:
    """Sum prompt and completion tokens from a Groq response usage block."""
    if hasattr(response, "usage") and response.usage:
        return (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
    return 0


def _parse_json(text: str) -> dict:
    """Extract JSON from text, handling markdown code fences."""
    import re
    # Try to find JSON block in code fence
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}
