"""
evals/test_cases.py
===================
Test case pool and auto-distribution logic for the evaluation framework.

Test cases are organised into 4 complexity tiers:
    simple       (~27% of run) — single-fact lookups, 1 sub-agent
    medium       (~27% of run) — comparison/multi-source, 2 sub-agents
    complex      (~27% of run) — deep multi-entity research, 3+ sub-agents
    adversarial  (~19% of run) — edge cases, unanswerable or ambiguous queries

Usage:
    cases = select_test_cases(15)   # returns 4 simple + 4 medium + 4 complex + 3 adversarial

Adding more cases:
    Add to the relevant tier pool below. The selector always picks the first N
    cases from each pool, so existing selections are stable as the pool grows.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single test case for the evaluation framework."""
    id: str                          # e.g. "simple_01"
    query: str                       # The research query
    tier: str                        # "simple" | "medium" | "complex" | "adversarial"
    description: str                 # What this case validates
    expected_sub_agents: int         # Approximate expected sub-agent count
    must_cover: list[str]            # Topics/entities that MUST appear in output
    must_not_contain: list[str]      # Strings whose presence indicates failure
    max_acceptable_cost: float       # Cost ceiling in USD
    max_acceptable_duration: int     # Duration ceiling in seconds


# ── Tier 1: Simple ────────────────────────────────────────────────────────────
# Basic fact-finding, 1 sub-agent, <$0.10 each

_SIMPLE_POOL: list[TestCase] = [
    TestCase(
        id="simple_01",
        query="What is LangGraph and who created it?",
        tier="simple",
        description="Validates basic fact-finding; single source should be sufficient.",
        expected_sub_agents=1,
        must_cover=["LangGraph", "LangChain"],
        must_not_contain=[],
        max_acceptable_cost=0.10,
        max_acceptable_duration=120,
    ),
    TestCase(
        id="simple_02",
        query="What is the current population of Tokyo?",
        tier="simple",
        description="Validates simple factual lookup and number accuracy.",
        expected_sub_agents=1,
        must_cover=["Tokyo", "population", "million"],
        must_not_contain=[],
        max_acceptable_cost=0.10,
        max_acceptable_duration=120,
    ),
    TestCase(
        id="simple_03",
        query="Who is the CEO of Anthropic and when was the company founded?",
        tier="simple",
        description="Validates person + date fact retrieval.",
        expected_sub_agents=1,
        must_cover=["Dario Amodei", "Anthropic", "2021"],
        must_not_contain=[],
        max_acceptable_cost=0.10,
        max_acceptable_duration=120,
    ),
    TestCase(
        id="simple_04",
        query="What programming language is Rust and what is it used for?",
        tier="simple",
        description="Validates technical concept explanation from authoritative sources.",
        expected_sub_agents=1,
        must_cover=["Rust", "systems programming", "memory safety"],
        must_not_contain=[],
        max_acceptable_cost=0.10,
        max_acceptable_duration=120,
    ),
]

# ── Tier 2: Medium ────────────────────────────────────────────────────────────
# Comparison and multi-source synthesis, 2 sub-agents, <$0.20 each

_MEDIUM_POOL: list[TestCase] = [
    TestCase(
        id="medium_01",
        query=(
            "Compare LangGraph and CrewAI for building multi-agent systems. "
            "Cover architecture, ease of use, and community support."
        ),
        tier="medium",
        description="Validates fair comparison and multiple perspectives.",
        expected_sub_agents=2,
        must_cover=["LangGraph", "CrewAI", "architecture", "community"],
        must_not_contain=[],
        max_acceptable_cost=0.20,
        max_acceptable_duration=240,
    ),
    TestCase(
        id="medium_02",
        query=(
            "What are the main approaches to reducing LLM inference costs in production? "
            "Cover at least 3 techniques."
        ),
        tier="medium",
        description="Validates technical breadth and coverage of multiple techniques.",
        expected_sub_agents=2,
        must_cover=["quantization", "caching"],
        must_not_contain=[],
        max_acceptable_cost=0.20,
        max_acceptable_duration=240,
    ),
    TestCase(
        id="medium_03",
        query="What are the latest developments in AI regulation in the US and EU as of 2025?",
        tier="medium",
        description="Validates current events, geographic comparison, and timeliness.",
        expected_sub_agents=2,
        must_cover=["EU", "AI Act", "United States"],
        must_not_contain=[],
        max_acceptable_cost=0.20,
        max_acceptable_duration=240,
    ),
    TestCase(
        id="medium_04",
        query=(
            "How do vector databases work and which ones are most popular in 2025? "
            "Compare at least 3."
        ),
        tier="medium",
        description="Validates concept explanation combined with product comparison.",
        expected_sub_agents=2,
        must_cover=["vector", "embedding", "similarity"],
        must_not_contain=[],
        max_acceptable_cost=0.20,
        max_acceptable_duration=240,
    ),
]

# ── Tier 3: Complex ───────────────────────────────────────────────────────────
# Deep multi-entity research, 3+ sub-agents, <$0.40 each

_COMPLEX_POOL: list[TestCase] = [
    TestCase(
        id="complex_01",
        query=(
            "Research the competitive landscape of AI coding assistants in 2025. "
            "Cover GitHub Copilot, Cursor, Claude Code, and Windsurf. "
            "Include pricing, key features, and market positioning."
        ),
        tier="complex",
        description="Validates multi-entity research and structured comparison.",
        expected_sub_agents=3,
        must_cover=["Copilot", "Cursor", "Claude Code", "Windsurf", "pricing"],
        must_not_contain=[],
        max_acceptable_cost=0.40,
        max_acceptable_duration=480,
    ),
    TestCase(
        id="complex_02",
        query=(
            "Analyze the current state of open-source large language models in 2025. "
            "Cover Llama, Mistral, and Qwen families. "
            "Include model sizes, benchmarks, licensing, and commercial adoption."
        ),
        tier="complex",
        description="Validates technical depth and multi-family model analysis.",
        expected_sub_agents=3,
        must_cover=["Llama", "Mistral", "Qwen", "open-source", "license"],
        must_not_contain=[],
        max_acceptable_cost=0.40,
        max_acceptable_duration=480,
    ),
    TestCase(
        id="complex_03",
        query=(
            "What are the most promising AI applications in healthcare as of 2025? "
            "Cover diagnostics, drug discovery, and clinical operations. "
            "Include real companies and products."
        ),
        tier="complex",
        description="Validates domain research and sourcing of real-world examples.",
        expected_sub_agents=3,
        must_cover=["diagnostics", "drug discovery", "clinical"],
        must_not_contain=[],
        max_acceptable_cost=0.40,
        max_acceptable_duration=480,
    ),
    TestCase(
        id="complex_04",
        query=(
            "Research how companies are implementing AI agents in production. "
            "Cover architecture patterns, common challenges, and at least 5 real-world case studies."
        ),
        tier="complex",
        description="Validates pattern identification and case study sourcing.",
        expected_sub_agents=3,
        must_cover=["agent", "production", "challenges"],
        must_not_contain=[],
        max_acceptable_cost=0.40,
        max_acceptable_duration=480,
    ),
]

# ── Tier 4: Adversarial ───────────────────────────────────────────────────────
# Edge cases, unanswerable or ambiguous queries — tests graceful failure handling

_ADVERSARIAL_POOL: list[TestCase] = [
    TestCase(
        id="adversarial_01",
        query="What is the airspeed velocity of an unladen swallow?",
        tier="adversarial",
        description=(
            "Validates graceful handling of a humorous/unanswerable query. "
            "The system should not hallucinate a precise answer."
        ),
        expected_sub_agents=1,
        must_cover=[],
        must_not_contain=["definitely", "exactly"],
        max_acceptable_cost=0.10,
        max_acceptable_duration=180,
    ),
    TestCase(
        id="adversarial_02",
        query="Compare the AI strategies of companies ABC Corp, XYZ Ltd, and DEF Inc.",
        tier="adversarial",
        description=(
            "Validates handling of queries about non-existent companies. "
            "The system should report it cannot find information rather than fabricating."
        ),
        expected_sub_agents=1,
        must_cover=[],
        must_not_contain=[],
        max_acceptable_cost=0.10,
        max_acceptable_duration=180,
    ),
    TestCase(
        id="adversarial_03",
        query="What will the stock price of NVIDIA be in 2027?",
        tier="adversarial",
        description=(
            "Validates refusal to make specific future predictions. "
            "Should provide analysis rather than fabricating a precise future price."
        ),
        expected_sub_agents=1,
        must_cover=[],
        must_not_contain=["will be $", "predicted to reach exactly"],
        max_acceptable_cost=0.10,
        max_acceptable_duration=180,
    ),
]

# ── Distribution logic ────────────────────────────────────────────────────────

_TIER_POOLS: dict[str, list[TestCase]] = {
    "simple": _SIMPLE_POOL,
    "medium": _MEDIUM_POOL,
    "complex": _COMPLEX_POOL,
    "adversarial": _ADVERSARIAL_POOL,
}


def _distribute(count: int) -> dict[str, int]:
    """
    Calculate per-tier allocation for a given total test count.

    Distribution ratios: simple 27%, medium 27%, complex 27%, adversarial 19%.
    Minimum 1 per tier when count >= 4.

    Args:
        count: Total number of test cases to run.

    Returns:
        dict mapping tier name -> number of cases to select.
    """
    if count < 4:
        # Distribute 1 per tier up to count — so --count 1 = 1 case, --count 2 = 2 cases, etc.
        tiers = ["simple", "medium", "complex", "adversarial"]
        result = {t: 0 for t in tiers}
        for i in range(min(count, 4)):
            result[tiers[i]] = 1
        return result

    per3 = max(1, int(count * 0.27))
    adversarial = max(1, count - 3 * per3)
    return {
        "simple": per3,
        "medium": per3,
        "complex": per3,
        "adversarial": adversarial,
    }


def select_test_cases(count: int) -> list[TestCase]:
    """
    Select test cases from the pool according to the auto-distribution logic.

    Always picks from the START of each tier pool so selections are stable
    as the pool grows. Users can add more cases to any pool without changing
    which cases are selected for the current count.

    Args:
        count: Total number of test cases to select.

    Returns:
        list[TestCase]: Ordered list of selected cases (simple → medium → complex → adversarial).

    Examples:
        select_test_cases(15) → 4 simple + 4 medium + 4 complex + 3 adversarial
        select_test_cases(8)  → 2 simple + 2 medium + 2 complex + 2 adversarial
        select_test_cases(4)  → 1 simple + 1 medium + 1 complex + 1 adversarial
    """
    allocation = _distribute(count)
    selected: list[TestCase] = []
    for tier in ("simple", "medium", "complex", "adversarial"):
        pool = _TIER_POOLS[tier]
        n = min(allocation[tier], len(pool))
        selected.extend(pool[:n])
    return selected


def get_all_cases() -> list[TestCase]:
    """Return the full test case pool across all tiers."""
    cases: list[TestCase] = []
    for pool in _TIER_POOLS.values():
        cases.extend(pool)
    return cases


def get_tier_pool(tier: str) -> list[TestCase]:
    """Return all test cases in a specific tier pool."""
    if tier not in _TIER_POOLS:
        raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {list(_TIER_POOLS)}")
    return list(_TIER_POOLS[tier])


def get_case_by_id(case_id: str) -> TestCase:
    """Return a single test case by its ID."""
    for pool in _TIER_POOLS.values():
        for case in pool:
            if case.id == case_id:
                return case
    raise ValueError(f"No test case with ID '{case_id}'")
