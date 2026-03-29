"""
evals/report.py
===============
Generates a human-readable summary report from evaluation results.

The report is printed to console AND saved as report.txt in the run directory.

Sections:
    1. Header        — run ID, timestamp, total duration, total cost
    2. Overall       — pass rate, average weighted score
    3. Tier table    — tier | cases | passed | avg score | avg cost | avg duration
    4. Criteria table— criteria | weight | avg score | weakest case
    5. Case table    — ID | query | score | verdict | cost | duration
    6. Failures      — per-failure score breakdown + judge reasoning
    7. Recommendations — based on lowest-scoring criteria
"""

from __future__ import annotations
from pathlib import Path

from evals.runner import EvalSummary, EvalResult
from evals.rubric import RUBRIC


def generate_eval_report(summary: EvalSummary, run_dir: Path | None = None) -> str:
    """
    Generate a formatted text report from evaluation results.

    Prints the report to stdout and optionally saves it as report.txt
    in the provided run directory.

    Args:
        summary:  EvalSummary returned by EvalRunner.run_all().
        run_dir:  Directory to save report.txt. If None, only prints.

    Returns:
        str: The full report text.
    """
    lines: list[str] = []

    def _line(text: str = "") -> None:
        lines.append(text)

    def _header(text: str) -> None:
        _line()
        _line("=" * 60)
        _line(f"  {text}")
        _line("=" * 60)

    def _subheader(text: str) -> None:
        _line()
        _line(f"── {text} {'─' * (55 - len(text))}")

    total_duration = sum(r.duration_seconds for r in summary.results)

    # ── 1. Header ─────────────────────────────────────────────────────────────
    _line("=" * 60)
    _line("  MULTI-AGENT RESEARCH SYSTEM — EVALUATION REPORT")
    _line("=" * 60)
    _line(f"  Run ID   : {summary.run_id}")
    _line(f"  Date     : {summary.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    _line(f"  Duration : {total_duration:.0f}s ({total_duration/60:.1f} min)")
    _line(f"  Judge    : {summary.judge_model}")
    _line(f"  Total $  : ${summary.total_cost:.4f}")

    # ── 2. Overall scores ─────────────────────────────────────────────────────
    _subheader("OVERALL")
    _line(f"  Cases run     : {summary.total_cases}")
    _line(f"  Passed        : {summary.passed}  ({summary.pass_rate*100:.0f}%)")
    _line(f"  Failed        : {summary.failed}")
    _line(f"  Avg score     : {summary.avg_weighted_score:.3f}")
    _line(f"  Avg cost/case : ${summary.avg_cost_per_query:.4f}")
    _line(f"  Avg duration  : {summary.avg_duration_per_query:.0f}s")
    _line(f"  Total tokens  : {summary.total_tokens_used:,}")

    # ── 3. Per-tier table ─────────────────────────────────────────────────────
    _subheader("BY TIER")
    _line(f"  {'Tier':<14} {'Cases':>5} {'Passed':>6} {'Avg Score':>9} {'Pass Rate':>9}")
    _line(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*9} {'-'*9}")
    tier_order = ["simple", "medium", "complex", "adversarial"]
    for tier in tier_order:
        if tier not in summary.tier_scores:
            continue
        tier_results = [r for r in summary.results if r.test_case.tier == tier]
        passed = sum(1 for r in tier_results if r.judge_result and r.judge_result.passed)
        _line(
            f"  {tier:<14} {len(tier_results):>5} {passed:>6} "
            f"{summary.tier_scores[tier]:>9.3f} "
            f"{summary.tier_pass_rates.get(tier, 0)*100:>8.0f}%"
        )

    # ── 4. Per-criteria table ─────────────────────────────────────────────────
    _subheader("BY CRITERIA")
    scored_results = [r for r in summary.results if r.judge_result]

    _line(f"  {'Criteria':<22} {'Weight':>6} {'Avg Score':>9} {'Weakest Case':<20}")
    _line(f"  {'-'*22} {'-'*6} {'-'*9} {'-'*20}")
    for criterion in RUBRIC:
        cname = criterion.name
        avg = summary.criteria_scores.get(cname, 0.0)
        # Find case with lowest score for this criterion
        worst_case = "—"
        worst_score = 1.1
        for r in scored_results:
            s = r.judge_result.scores.get(cname, 0.0)
            if s < worst_score:
                worst_score = s
                worst_case = r.test_case.id
        _line(
            f"  {cname:<22} {criterion.weight:>6.2f} {avg:>9.3f} "
            f"{worst_case:<20}"
        )

    # ── 5. Individual results table ───────────────────────────────────────────
    _subheader("ALL RESULTS")
    _line(f"  {'ID':<18} {'Score':>5} {'Verdict':<12} {'Cost':>7} {'Dur':>5}  Query")
    _line(f"  {'-'*18} {'-'*5} {'-'*12} {'-'*7} {'-'*5}  {'-'*35}")

    for r in summary.results:
        if r.pipeline_error:
            score_str = "ERR"
            verdict_str = "pipeline_err"
        elif r.judge_result:
            score_str = f"{r.judge_result.weighted_score:.3f}"
            verdict_str = r.judge_result.verdict
        else:
            score_str = "?"
            verdict_str = "no_judge"

        query_short = r.test_case.query[:35] + ("…" if len(r.test_case.query) > 35 else "")
        _line(
            f"  {r.test_case.id:<18} {score_str:>5} {verdict_str:<12} "
            f"${r.total_cost:.4f} {r.duration_seconds:>4.0f}s  {query_short}"
        )

    # ── 6. Failed cases detail ────────────────────────────────────────────────
    failed = [r for r in summary.results if not r.judge_result or not r.judge_result.passed]
    if failed:
        _subheader(f"FAILED CASES ({len(failed)})")
        for r in failed:
            _line()
            _line(f"  ▸ {r.test_case.id} — {r.test_case.query[:70]}")
            if r.pipeline_error:
                _line(f"    Pipeline error: {r.pipeline_error}")
            elif r.judge_result:
                jr = r.judge_result
                _line(f"    Weighted score: {jr.weighted_score:.3f}")
                for cname, score in sorted(jr.scores.items(), key=lambda x: x[1]):
                    _line(f"      {cname:<22}: {score:.2f}")
                _line(f"    Must-cover misses: " +
                      ", ".join(k for k, v in jr.must_cover_results.items() if not v) or "none")
                _line(f"    Judge reasoning: {jr.reasoning[:200]}...")

    # ── 7. Recommendations ────────────────────────────────────────────────────
    _subheader("RECOMMENDATIONS")
    _recommend(summary, lines)

    _line()
    _line("=" * 60)
    _line("  END OF REPORT")
    _line("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    if run_dir is not None:
        report_path = run_dir / "report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def _recommend(summary: EvalSummary, lines: list[str]) -> None:
    """Append actionable recommendations based on eval results."""
    cscores = summary.criteria_scores

    recs = {
        "factual_accuracy": (
            "Factual accuracy is weakest — consider tightening the LeadResearcher "
            "synthesis prompt to require explicit source verification before each claim."
        ),
        "citation_quality": (
            "Citation quality is weakest — the CitationAgent may be missing claims. "
            "Try increasing LLM_MAX_TOKENS_CITATION in .env and reviewing the citation prompt."
        ),
        "completeness": (
            "Completeness is weakest — sub-agents may not be covering all requested topics. "
            "Increase MAX_SUBAGENTS or MAX_ITERATIONS in .env for deeper coverage."
        ),
        "source_quality": (
            "Source quality is weakest — consider adding domain filters to the search provider "
            "or instructing sub-agents to prefer .gov, .edu, and official docs."
        ),
        "structure_clarity": (
            "Structure clarity is weakest — review the synthesize() prompt in lead_researcher.py "
            "to enforce clearer section headings and logical flow."
        ),
        "efficiency": (
            "Efficiency is weakest — sub-agents may be over-searching. "
            "Reduce MAX_TOOL_ROUNDS in .env or tighten the stopping criteria in sub-agent prompts."
        ),
    }

    if not cscores:
        lines.append("  No scores available to generate recommendations.")
        return

    # Top recommendation: weakest criterion
    worst_criterion = summary.lowest_scoring_criteria
    worst_score = cscores.get(worst_criterion, 0.0)
    lines.append(f"  Primary issue (lowest criterion: {worst_criterion} = {worst_score:.2f}):")
    lines.append(f"  → {recs.get(worst_criterion, 'Review the pipeline for this area.')}")

    # Secondary: worst tier
    worst_tier = summary.lowest_scoring_tier
    worst_tier_score = summary.tier_scores.get(worst_tier, 0.0)
    lines.append(f"")
    lines.append(f"  Worst-performing tier: {worst_tier} (avg {worst_tier_score:.2f})")
    if worst_tier == "adversarial":
        lines.append("  → Add graceful fallback handling for ambiguous/unanswerable queries.")
    elif worst_tier == "complex":
        lines.append("  → Increase MAX_SUBAGENTS and MAX_ITERATIONS for deep research tasks.")
    else:
        lines.append("  → Review agent prompts for this tier's query type.")

    # Overall pass rate comment
    if summary.pass_rate < 0.5:
        lines.append("")
        lines.append("  ⚠️  Pass rate below 50% — major quality issues need addressing.")
    elif summary.pass_rate < 0.7:
        lines.append("")
        lines.append("  ℹ️  Pass rate moderate — targeted improvements recommended above.")
    else:
        lines.append("")
        lines.append("  ✅ Pass rate looks healthy. Focus on the weakest criterion above.")
