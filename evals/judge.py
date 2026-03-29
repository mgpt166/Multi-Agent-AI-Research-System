"""
evals/judge.py
==============
LLM-as-judge scoring logic for research output evaluation.

Supports two providers, auto-detected from the model name:
  - Groq  (any model that does NOT start with "claude")  — uses the Groq SDK
  - Anthropic (models starting with "claude")            — uses the Anthropic SDK

A single LLM call is made per evaluation — per Anthropic's own finding, a single
detailed judge call is more consistent than multiple passes.

The judge evaluates:
    - 6 rubric criteria (factual accuracy, citation quality, completeness, etc.)
    - must_cover items (binary: was this topic mentioned?)
    - must_not_contain items (binary: was this forbidden string absent?)

Output is returned as a structured JudgeResult dataclass.

Judge model: configurable via EVAL_JUDGE_MODEL in config (default: llama-3.3-70b-versatile).
The judge is NOT told the pass/fail threshold to prevent score anchoring.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass

from app.config import ANTHROPIC_API_KEY, GROQ_API_KEY, EVAL_JUDGE_MODEL
from evals.rubric import RubricCriteria, compute_weighted_score, PASS_THRESHOLD

_logger = logging.getLogger(__name__)


def _is_anthropic_model(model: str) -> bool:
    """Return True if model name indicates an Anthropic/Claude model."""
    return model.strip().lower().startswith("claude")


def _call_groq(model: str, prompt: str) -> tuple[str, int]:
    """Call Groq API and return (response_text, tokens_used)."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    tokens = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
    return text, tokens


def _call_anthropic(model: str, prompt: str) -> tuple[str, int]:
    """Call Anthropic API and return (response_text, tokens_used)."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else ""
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return text, tokens


@dataclass
class JudgeResult:
    """Structured output from a single LLM judge evaluation."""
    test_case_id: str
    scores: dict[str, float]               # criteria_name -> 0.0–1.0
    weighted_score: float                  # overall weighted score
    passed: bool                           # True if weighted_score >= PASS_THRESHOLD
    verdict: str                           # "strong_pass" | "pass" | "fail"
    must_cover_results: dict[str, bool]    # required topic -> was it found?
    must_not_contain_results: dict[str, bool]  # forbidden string -> was it absent?
    reasoning: str                         # judge's overall explanation
    judge_tokens_used: int

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "scores": self.scores,
            "weighted_score": self.weighted_score,
            "passed": self.passed,
            "verdict": self.verdict,
            "must_cover_results": self.must_cover_results,
            "must_not_contain_results": self.must_not_contain_results,
            "reasoning": self.reasoning,
            "judge_tokens_used": self.judge_tokens_used,
        }


def judge_research_output(
    test_case,                  # TestCase
    research_output: dict,
    rubric: list[RubricCriteria],
    model: str | None = None,
) -> JudgeResult:
    """
    Evaluate a research output against the rubric using an LLM as judge.

    Auto-detects the provider from the model name:
      - Models starting with "claude" → Anthropic SDK
      - All other models              → Groq SDK

    Makes a single LLM call with a detailed prompt covering all criteria.
    Parses the XML response into structured scores.

    Args:
        test_case:        TestCase with query, must_cover, must_not_contain.
        research_output:  Dict with keys: narrative, bibliography, source_map, summary.
        rubric:           List of RubricCriteria to score against.
        model:            Model ID. Defaults to EVAL_JUDGE_MODEL from config.

    Returns:
        JudgeResult with per-criteria scores, weighted score, and pass/fail verdict.
    """
    model = model or EVAL_JUDGE_MODEL
    prompt = _build_judge_prompt(test_case, research_output, rubric)

    provider = "Anthropic" if _is_anthropic_model(model) else "Groq"
    _logger.info("Judging %s with %s (%s)", test_case.id, model, provider)

    if _is_anthropic_model(model):
        response_text, tokens_used = _call_anthropic(model, prompt)
    else:
        response_text, tokens_used = _call_groq(model, prompt)

    return _parse_judge_response(
        response_text=response_text,
        test_case_id=test_case.id,
        rubric=rubric,
        must_cover=test_case.must_cover,
        must_not_contain=test_case.must_not_contain,
        research_output=research_output,
        judge_tokens_used=tokens_used,
    )


def _build_judge_prompt(test_case, research_output: dict, rubric: list[RubricCriteria]) -> str:
    """Construct the detailed judge prompt."""
    # Format rubric criteria section
    rubric_section = "\n".join(
        f"""<criterion name="{c.name}" weight="{c.weight}">
  <description>{c.description}</description>
  <scoring_guide>{c.scoring_guide}</scoring_guide>
</criterion>"""
        for c in rubric
    )

    # Format must_cover items
    must_cover_section = (
        "\n".join(f"- {item}" for item in test_case.must_cover)
        if test_case.must_cover
        else "(none)"
    )

    # Format must_not_contain items
    must_not_section = (
        "\n".join(f"- {item}" for item in test_case.must_not_contain)
        if test_case.must_not_contain
        else "(none)"
    )

    # Prepare the research output text
    narrative = research_output.get("narrative", "")
    summary = research_output.get("summary", "")
    bibliography = research_output.get("bibliography", [])
    source_map = research_output.get("source_map", {})

    # Truncate narrative to avoid exceeding context
    narrative_excerpt = narrative[:6000] + ("...[truncated]" if len(narrative) > 6000 else "")

    # Format source list
    source_lines = []
    for i, (url, meta) in enumerate(list(source_map.items())[:30]):
        title = meta.get("title", url) if isinstance(meta, dict) else url
        source_lines.append(f"  [{i+1}] {title} — {url}")
    sources_section = "\n".join(source_lines) if source_lines else "  (no sources recorded)"

    # Format bibliography
    bib_lines = []
    for entry in bibliography[:30]:
        n = entry.get("number", "?")
        title = entry.get("title", entry.get("url", "Unknown"))
        url = entry.get("url", "")
        bib_lines.append(f"  [{n}] {title} — {url}")
    bib_section = "\n".join(bib_lines) if bib_lines else "  (no bibliography)"

    tool_call_count = research_output.get("tool_call_count", "unknown")
    token_usage = research_output.get("token_usage", {})
    total_tokens = sum(v for v in token_usage.values() if isinstance(v, (int, float))) if token_usage else 0

    return f"""You are an expert research quality evaluator. Your job is to rigorously assess a research report produced by an AI research system.

## Original Research Query
{test_case.query}

## Research Output

### Summary
{summary or "(no summary available)"}

### Full Narrative (may be truncated)
{narrative_excerpt}

### Sources Used
{sources_section}

### Bibliography
{bib_section}

### System Metrics
- Approximate searches performed: {tool_call_count}
- Total tokens used by research pipeline: {total_tokens:,}

---

## Evaluation Instructions

Score each criterion independently on a scale of 0.0 to 1.0. Use the scoring guides as anchors. Do NOT round to nice numbers — use precise scores like 0.72 or 0.88 based on your actual assessment.

### Rubric Criteria
{rubric_section}

### Required Coverage Check (must_cover)
These topics/entities MUST appear meaningfully in the output. Check each:
{must_cover_section}

### Forbidden Content Check (must_not_contain)
These strings/patterns should NOT appear in the output. Check each:
{must_not_section}

---

## Output Format

Respond ONLY with valid XML in this exact structure. No text before or after the XML.

<evaluation>
  <criterion name="factual_accuracy">
    <score>0.00</score>
    <reasoning>Brief explanation of this score (1-3 sentences).</reasoning>
  </criterion>
  <criterion name="citation_quality">
    <score>0.00</score>
    <reasoning>Brief explanation.</reasoning>
  </criterion>
  <criterion name="completeness">
    <score>0.00</score>
    <reasoning>Brief explanation.</reasoning>
  </criterion>
  <criterion name="source_quality">
    <score>0.00</score>
    <reasoning>Brief explanation.</reasoning>
  </criterion>
  <criterion name="structure_clarity">
    <score>0.00</score>
    <reasoning>Brief explanation.</reasoning>
  </criterion>
  <criterion name="efficiency">
    <score>0.00</score>
    <reasoning>Brief explanation.</reasoning>
  </criterion>
  <must_cover>
    <!-- For each item in the must_cover list, add a line like this: -->
    <!-- <item name="LangGraph" found="true"/> -->
  </must_cover>
  <must_not_contain>
    <!-- For each item in the must_not_contain list, add a line like this: -->
    <!-- <item name="definitely" found="false"/> -->
    <!-- found="true" means the forbidden string WAS found (bad). found="false" means it was absent (good). -->
  </must_not_contain>
  <overall_reasoning>
    2-4 sentence summary of the research quality. What did it do well? What were the main weaknesses?
  </overall_reasoning>
</evaluation>"""


def _parse_judge_response(
    response_text: str,
    test_case_id: str,
    rubric: list[RubricCriteria],
    must_cover: list[str],
    must_not_contain: list[str],
    research_output: dict,
    judge_tokens_used: int,
) -> JudgeResult:
    """Parse the XML judge response into a JudgeResult."""
    # Strip any preamble before the <evaluation> block (Groq models sometimes add one)
    eval_start = response_text.find("<evaluation>")
    if eval_start > 0:
        response_text = response_text[eval_start:]

    scores: dict[str, float] = {}
    per_criterion_reasoning: dict[str, str] = {}

    # Extract per-criterion scores and reasoning
    for criterion in rubric:
        pattern = (
            rf'<criterion\s+name="{re.escape(criterion.name)}">'
            rf'.*?<score>([\d.]+)</score>'
            rf'.*?<reasoning>(.*?)</reasoning>'
            rf'.*?</criterion>'
        )
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                scores[criterion.name] = min(1.0, max(0.0, float(match.group(1))))
                per_criterion_reasoning[criterion.name] = match.group(2).strip()
            except ValueError:
                scores[criterion.name] = 0.0
                per_criterion_reasoning[criterion.name] = "Parse error"
        else:
            scores[criterion.name] = 0.0
            per_criterion_reasoning[criterion.name] = "Not evaluated"
            _logger.warning("Could not parse score for criterion '%s'", criterion.name)

    # Extract must_cover results
    must_cover_results: dict[str, bool] = {}
    narrative_lower = (research_output.get("narrative", "") + research_output.get("summary", "")).lower()
    for item in must_cover:
        # First try to read from XML response
        xml_pattern = rf'<item\s+name="{re.escape(item)}"\s+found="(true|false)"'
        xml_match = re.search(xml_pattern, response_text, re.IGNORECASE)
        if xml_match:
            must_cover_results[item] = xml_match.group(1).lower() == "true"
        else:
            # Fallback: check text directly
            must_cover_results[item] = item.lower() in narrative_lower

    # Extract must_not_contain results (found=True means BAD — it was present)
    must_not_contain_results: dict[str, bool] = {}
    for item in must_not_contain:
        xml_pattern = rf'<item\s+name="{re.escape(item)}"\s+found="(true|false)"'
        xml_match = re.search(xml_pattern, response_text, re.IGNORECASE)
        if xml_match:
            # found="false" in XML means string is ABSENT (good) → result is True (absent = pass)
            must_not_contain_results[item] = xml_match.group(1).lower() == "false"
        else:
            # Fallback: check text directly — True means it's absent (good)
            must_not_contain_results[item] = item.lower() not in narrative_lower

    # Extract overall reasoning
    reasoning_match = re.search(
        r'<overall_reasoning>(.*?)</overall_reasoning>', response_text, re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."

    # Apply must_cover penalty: each missing required topic reduces score
    if must_cover:
        missing = sum(1 for v in must_cover_results.values() if not v)
        if missing > 0:
            penalty = (missing / len(must_cover)) * 0.15
            scores["completeness"] = max(0.0, scores.get("completeness", 0.0) - penalty)
            _logger.info(
                "%s: %d/%d must_cover items missing — completeness penalised by %.2f",
                test_case_id, missing, len(must_cover), penalty
            )

    # Apply must_not_contain penalty: each present forbidden item reduces accuracy
    if must_not_contain:
        violations = sum(1 for v in must_not_contain_results.values() if not v)
        if violations > 0:
            penalty = (violations / len(must_not_contain)) * 0.15
            scores["factual_accuracy"] = max(0.0, scores.get("factual_accuracy", 0.0) - penalty)

    weighted_score = compute_weighted_score(scores)
    passed = weighted_score >= PASS_THRESHOLD

    from evals.rubric import classify_result
    verdict = classify_result(weighted_score)

    # Enrich reasoning with per-criterion notes
    detail_lines = [
        f"{name}: {score:.2f} — {per_criterion_reasoning.get(name, '')}"
        for name, score in scores.items()
    ]
    full_reasoning = reasoning + "\n\nPer-criterion:\n" + "\n".join(detail_lines)

    return JudgeResult(
        test_case_id=test_case_id,
        scores=scores,
        weighted_score=weighted_score,
        passed=passed,
        verdict=verdict,
        must_cover_results=must_cover_results,
        must_not_contain_results=must_not_contain_results,
        reasoning=full_reasoning,
        judge_tokens_used=judge_tokens_used,
    )
