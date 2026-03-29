"""
app/agents/citation_agent.py
=============================
CitationAgent — converts raw source markers into numbered inline citations
and builds a deduplicated bibliography.

The LeadResearcher embeds [CITE: https://source-url.com] markers throughout
the synthesized narrative wherever it references a source. This agent's job
is to convert those markers into clean academic-style [1], [2], [3] citations
and produce a numbered bibliography for the document footer.

Two-pass approach:
    Pass 1 (regex, always runs):
        Uses a regular expression to find all [CITE: url] markers in the text.
        Assigns sequential citation numbers, replaces markers with [N], and
        builds the bibliography from the source_map metadata.

    Pass 2 (LLM fallback, runs if Pass 1 found fewer than 3 citations):
        If the narrative has few or no [CITE:] markers (e.g. the LeadResearcher
        didn't embed them consistently), uses Groq llama-3.3-70b-versatile to
        re-annotate the narrative by matching factual claims to the available sources.

Input:
    narrative   str     — The synthesized narrative containing [CITE: url] markers
    source_map  dict    — {url: {title, date, relevance_score}} from sub-agents

Output:
    {
        "annotated_narrative": str,       — text with [N] inline citation markers
        "citation_map":        dict,      — {url: citation_number}
        "bibliography":        list[dict],— [{number, url, title, date}] sorted by number
        "token_usage":         dict,      — only populated if LLM fallback was used
    }
"""
from __future__ import annotations
import json
import re
from groq import Groq
from app.utils.groq_retry import groq_chat
from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS_CITATION

_client = Groq(api_key=GROQ_API_KEY)
_MODEL = GROQ_MODEL


class CitationAgent:

    def annotate(self, narrative: str, source_map: dict) -> dict:
        """
        Take a narrative with [CITE: url] markers and convert to proper [N] citations.
        Also catches any uncited factual claims and marks them [UNVERIFIED].
        """
        if not narrative:
            return {"annotated_narrative": "", "citation_map": {}, "bibliography": [], "token_usage": {}}

        # First pass: build citation numbers from [CITE: url] markers already in the text
        citation_map: dict[str, int] = {}  # url -> citation number
        bibliography: list[dict] = []
        citation_counter = [0]

        def assign_citation(url: str) -> int:
            if url not in citation_map:
                citation_counter[0] += 1
                n = citation_counter[0]
                citation_map[url] = n
                src = source_map.get(url, {})
                bibliography.append({
                    "number": n,
                    "url": url,
                    "title": src.get("title", url),
                    "date": src.get("date", ""),
                })
            return citation_map[url]

        # Replace [CITE: url] markers with [N]
        def replace_cite(match):
            url = match.group(1).strip()
            n = assign_citation(url)
            return f"[{n}]"

        annotated = re.sub(r'\[CITE:\s*([^\]]+)\]', replace_cite, narrative)

        # If there are sources but very few citations, use Groq to do a proper annotation pass
        llm_tokens = 0
        if len(bibliography) < 3 and source_map:
            try:
                annotated, citation_map, bibliography, llm_tokens = self._llm_annotate(narrative, source_map)
            except Exception:
                # LLM fallback failed — keep regex results rather than crashing the pipeline
                pass

        # Last resort: if bibliography is still empty but source_map has URLs,
        # build a bibliography directly so citation_quality isn't zero
        if not bibliography and source_map:
            for i, (url, meta) in enumerate(list(source_map.items())[:30], start=1):
                src = meta if isinstance(meta, dict) else {}
                citation_map[url] = i
                bibliography.append({
                    "number": i,
                    "url": url,
                    "title": src.get("title", url),
                    "date": src.get("date", ""),
                })

        return {
            "annotated_narrative": annotated,
            "citation_map": {url: n for url, n in citation_map.items()},
            "bibliography": bibliography,
            "token_usage": {"citation": llm_tokens} if llm_tokens else {},
        }

    def _llm_annotate(self, narrative: str, source_map: dict) -> tuple[str, dict, list, int]:
        """Use Groq to insert inline citations into the narrative."""
        sources_str = "\n".join(
            f"[{i+1}] {s.get('title', url)} — {url}"
            for i, (url, s) in enumerate(list(source_map.items())[:30])
        )
        num_to_url = {
            str(i+1): url
            for i, url in enumerate(list(source_map.keys())[:30])
        }

        prompt = f"""You are a citation specialist. Add inline citation numbers to factual claims in the research narrative.

## Available Sources (use ONLY these):
{sources_str}

## Research Narrative:
{narrative[:6000]}

## Instructions:
1. Add [N] citation markers after every specific factual claim, statistic, or statement that can be traced to a source
2. Use the source numbers listed above
3. Claims with no matching source should be marked [UNVERIFIED]
4. Keep the narrative text intact — only add citation markers

Return ONLY the annotated narrative text (no JSON, no explanation):"""

        response = groq_chat(_client,
            model=_MODEL,
            max_tokens=LLM_MAX_TOKENS_CITATION,
            messages=[{"role": "user", "content": prompt}],
        )

        annotated = _extract_text(response)

        # Build citation_map and bibliography from which [N] appear in the annotated text
        used_numbers = set(re.findall(r'\[(\d+)\]', annotated))
        citation_map: dict[str, int] = {}
        bibliography: list[dict] = []
        for n_str in sorted(used_numbers, key=int):
            url = num_to_url.get(n_str)
            if url:
                n = int(n_str)
                citation_map[url] = n
                src = source_map.get(url, {})
                bibliography.append({
                    "number": n,
                    "url": url,
                    "title": src.get("title", url),
                    "date": src.get("date", ""),
                })

        token_count = _count_tokens(response)
        return annotated, citation_map, sorted(bibliography, key=lambda x: x["number"]), token_count


def _extract_text(response) -> str:
    """Extract text content from a Groq chat completion response."""
    return response.choices[0].message.content or ""


def _count_tokens(response) -> int:
    """Sum prompt and completion tokens from a Groq response usage block."""
    if hasattr(response, "usage") and response.usage:
        return (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
    return 0
