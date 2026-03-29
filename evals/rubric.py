"""
evals/rubric.py
===============
Scoring rubric used by the LLM judge to evaluate research outputs.

The rubric consists of 6 weighted criteria. Each criterion is scored
0.0–1.0 independently by the judge. The final weighted score determines
whether a test case passes.

Pass thresholds:
    STRONG_PASS >= 0.85
    PASS        >= 0.70
    FAIL        <  0.70
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RubricCriteria:
    """A single evaluation criterion used by the LLM judge."""
    name: str
    weight: float          # 0.0–1.0; all weights in RUBRIC must sum to 1.0
    description: str       # What the judge evaluates
    scoring_guide: str     # Anchored descriptions for 0.0, 0.4, 0.7, 1.0


# ── Rubric definition ─────────────────────────────────────────────────────────

RUBRIC: list[RubricCriteria] = [
    RubricCriteria(
        name="factual_accuracy",
        weight=0.25,
        description="Are the claims factually correct based on the sources cited?",
        scoring_guide=(
            "1.0 = all claims accurate and well-supported by cited sources. "
            "0.7 = minor inaccuracies or one unsupported claim. "
            "0.4 = significant factual errors or multiple unsupported claims. "
            "0.0 = mostly wrong, fabricated, or contradicted by its own sources."
        ),
    ),
    RubricCriteria(
        name="citation_quality",
        weight=0.20,
        description="Does every factual claim have a citation? Are sources real and relevant?",
        scoring_guide=(
            "1.0 = every claim cited, all source URLs look real and relevant. "
            "0.7 = most claims cited, sources appear legitimate. "
            "0.4 = many uncited claims or several dubious/off-topic sources. "
            "0.0 = no citations present, or sources appear fabricated."
        ),
    ),
    RubricCriteria(
        name="completeness",
        weight=0.20,
        description="Does the research cover all aspects asked in the query? Are there gaps?",
        scoring_guide=(
            "1.0 = all requested topics covered thoroughly with depth. "
            "0.7 = most topics covered, minor gaps that don't undermine the value. "
            "0.4 = significant topics missing or only surface-level coverage. "
            "0.0 = barely addresses the query or ignores major requested topics."
        ),
    ),
    RubricCriteria(
        name="source_quality",
        weight=0.15,
        description="Are sources authoritative? Primary sources preferred over aggregator/SEO content.",
        scoring_guide=(
            "1.0 = authoritative primary sources (official docs, research papers, reputable news). "
            "0.7 = mix of good and mediocre sources; a few thin-content sites. "
            "0.4 = mostly low-quality aggregator or SEO-farm sites. "
            "0.0 = unreliable sources, no URLs, or sources that don't exist."
        ),
    ),
    RubricCriteria(
        name="structure_clarity",
        weight=0.10,
        description="Is the report well-organised with clear headings, logical flow, and readable prose?",
        scoring_guide=(
            "1.0 = excellent structure, flows logically, easy to navigate. "
            "0.7 = decent structure with minor readability issues. "
            "0.4 = disorganised or hard to follow in places. "
            "0.0 = incoherent, wall-of-text, or no discernible structure."
        ),
    ),
    RubricCriteria(
        name="efficiency",
        weight=0.10,
        description="Did the system use a reasonable number of searches/tokens for the query complexity?",
        scoring_guide=(
            "1.0 = efficient; appropriate depth without wasted searches. "
            "0.7 = slightly over-searched but not problematically so. "
            "0.4 = significant over-searching relative to query complexity. "
            "0.0 = wildly inefficient (e.g. 50 searches for a simple factual query)."
        ),
    ),
]

# Validate weights sum to 1.0
_total_weight = sum(c.weight for c in RUBRIC)
assert abs(_total_weight - 1.0) < 1e-6, f"RUBRIC weights must sum to 1.0, got {_total_weight}"

# ── Pass/fail thresholds ───────────────────────────────────────────────────────

STRONG_PASS_THRESHOLD = 0.85
PASS_THRESHOLD = 0.70


def compute_weighted_score(scores: dict[str, float]) -> float:
    """
    Compute the overall weighted score from per-criteria scores.

    Args:
        scores: dict mapping criteria name -> 0.0–1.0 score.

    Returns:
        float: Weighted sum. Missing criteria score 0.0.
    """
    total = 0.0
    for criterion in RUBRIC:
        total += criterion.weight * scores.get(criterion.name, 0.0)
    return round(total, 4)


def classify_result(weighted_score: float) -> str:
    """Return 'strong_pass', 'pass', or 'fail' based on weighted score."""
    if weighted_score >= STRONG_PASS_THRESHOLD:
        return "strong_pass"
    if weighted_score >= PASS_THRESHOLD:
        return "pass"
    return "fail"
