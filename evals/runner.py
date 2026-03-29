"""
evals/runner.py
===============
Evaluation runner — orchestrates test cases through the full research pipeline,
then scores each output with the LLM judge.

Design:
    - Runs test cases SEQUENTIALLY to maintain consistent token budgets and
      avoid rate-limit collisions.
    - Auto-approves the HITL checkpoint so evals are fully unattended.
    - Extracts the full research state from the LangGraph checkpointer after
      completion (includes narrative, bibliography, source_map not in job_store).
    - Saves each EvalResult to JSON as it completes; summary saved at the end.
    - A failing test case never stops the run — errors are captured and marked.

Usage:
    runner = EvalRunner()
    summary = runner.run_all(test_cases)
"""

from __future__ import annotations
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import EVAL_JUDGE_MODEL, EVAL_TEST_COUNT
from app.graph.graph import research_graph
from app.graph.runner import run_research_job, resume_research_job
from app.utils.job_store import job_store
from evals.judge import judge_research_output, JudgeResult
from evals.rubric import RUBRIC, PASS_THRESHOLD, STRONG_PASS_THRESHOLD
from evals.test_cases import TestCase

_logger = logging.getLogger(__name__)

# Results base directory (relative to project root)
_RESULTS_DIR = Path(__file__).parent / "results"

# Map tier -> research depth
_TIER_DEPTH = {
    "simple": "simple",
    "medium": "moderate",
    "complex": "deep",
    "adversarial": "simple",
}

# Rough token cost rates (USD per token) for cost estimation
_GROQ_COST_PER_TOKEN = 0.065 / 1_000_000      # llama-3.1-8b-instant blended
_ANTHROPIC_COST_PER_TOKEN = 9.0 / 1_000_000   # claude-sonnet-4-6 blended (input+output avg)

def _judge_cost_per_token(model: str) -> float:
    """Return the per-token cost rate for the judge model."""
    return _ANTHROPIC_COST_PER_TOKEN if model.strip().lower().startswith("claude") else _GROQ_COST_PER_TOKEN


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of running a single test case through the full pipeline + judge."""
    test_case: TestCase
    research_output: Optional[dict]        # Full research output; None if pipeline failed
    judge_result: Optional[JudgeResult]    # None if judge call failed
    pipeline_error: Optional[str]          # Error message if pipeline crashed
    duration_seconds: float
    total_tokens: int
    total_cost: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "test_case": {
                "id": self.test_case.id,
                "query": self.test_case.query,
                "tier": self.test_case.tier,
                "description": self.test_case.description,
                "expected_sub_agents": self.test_case.expected_sub_agents,
                "must_cover": self.test_case.must_cover,
                "must_not_contain": self.test_case.must_not_contain,
                "max_acceptable_cost": self.test_case.max_acceptable_cost,
                "max_acceptable_duration": self.test_case.max_acceptable_duration,
            },
            "research_output": self.research_output,
            "judge_result": self.judge_result.to_dict() if self.judge_result else None,
            "pipeline_error": self.pipeline_error,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvalSummary:
    """Aggregate summary of a full evaluation run."""
    run_id: str
    timestamp: datetime
    results: list[EvalResult]
    judge_model: str
    eval_test_count: int

    # Aggregate metrics
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_weighted_score: float
    avg_cost_per_query: float
    avg_duration_per_query: float
    total_tokens_used: int
    total_cost: float

    # Per-tier breakdown
    tier_scores: dict[str, float]
    tier_pass_rates: dict[str, float]

    # Per-criteria breakdown
    criteria_scores: dict[str, float]

    # Weakest areas
    lowest_scoring_criteria: str
    lowest_scoring_tier: str
    failed_cases: list[str]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "judge_model": self.judge_model,
            "eval_test_count": self.eval_test_count,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_weighted_score": round(self.avg_weighted_score, 4),
            "avg_cost_per_query": round(self.avg_cost_per_query, 6),
            "avg_duration_per_query": round(self.avg_duration_per_query, 2),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 6),
            "tier_scores": {k: round(v, 4) for k, v in self.tier_scores.items()},
            "tier_pass_rates": {k: round(v, 4) for k, v in self.tier_pass_rates.items()},
            "criteria_scores": {k: round(v, 4) for k, v in self.criteria_scores.items()},
            "lowest_scoring_criteria": self.lowest_scoring_criteria,
            "lowest_scoring_tier": self.lowest_scoring_tier,
            "failed_cases": self.failed_cases,
        }


# ── EvalRunner ────────────────────────────────────────────────────────────────

class EvalRunner:
    """
    Orchestrates test case execution, judging, and result persistence.

    Each test case is run synchronously (sequential, not parallel) through
    the full research pipeline. The HITL checkpoint is auto-approved.
    """

    def __init__(self, results_dir: Path = _RESULTS_DIR):
        self.results_dir = results_dir
        self.judge_model = EVAL_JUDGE_MODEL

    def run_single(self, test_case: TestCase, run_dir: Path) -> EvalResult:
        """
        Run one test case through the full pipeline and judge the output.

        Args:
            test_case: The test case to execute.
            run_dir:   Directory where this result JSON should be saved.

        Returns:
            EvalResult with scores and metadata.
        """
        _logger.info("Running test case %s: %s", test_case.id, test_case.query[:60])
        start = time.time()

        research_output = None
        judge_result = None
        pipeline_error = None
        total_tokens = 0
        total_cost = 0.0

        try:
            # ── Phase 1: Create job and run until HITL ────────────────
            depth = _TIER_DEPTH.get(test_case.tier, "moderate")
            job_id = job_store.create_job(
                query=test_case.query,
                depth=depth,
            )
            run_research_job(job_id, test_case.query, depth)

            # ── Phase 2: Auto-approve HITL ────────────────────────────
            job = job_store.get_job(job_id)
            if job and job.get("status") == "awaiting_approval":
                _logger.info("%s: auto-approving HITL for job %s", test_case.id, job_id[:8])
                resume_research_job(job_id, {
                    "decision": "approved",
                    "feedback": "Eval framework auto-approved",
                })

            # ── Phase 3: Extract full state from checkpointer ─────────
            config = {"configurable": {"thread_id": job_id}}
            state_snapshot = research_graph.get_state(config)
            state = state_snapshot.values if state_snapshot else {}

            # Refresh job from store for metadata
            job = job_store.get_job(job_id) or {}

            narrative = state.get("annotated_narrative") or state.get("synthesized_narrative", "")
            bibliography = state.get("bibliography", [])
            source_map = state.get("source_map", {})
            token_usage = job.get("token_usage", {})

            research_output = {
                "query": test_case.query,
                "narrative": narrative,
                "summary": job.get("summary_snippet", ""),
                "bibliography": bibliography,
                "source_map": source_map,
                "token_usage": token_usage,
                "tool_call_count": len(source_map),
                "status": job.get("status", "unknown"),
                "job_id": job_id,
            }

            pipeline_tokens = sum(v for v in token_usage.values() if isinstance(v, (int, float))) if token_usage else 0

            # ── Phase 4: Judge the output ─────────────────────────────
            judge_result = judge_research_output(
                test_case=test_case,
                research_output=research_output,
                rubric=RUBRIC,
                model=self.judge_model,
            )

            judge_tokens = judge_result.judge_tokens_used
            total_tokens = pipeline_tokens + judge_tokens
            total_cost = _estimate_cost(pipeline_tokens, judge_tokens, self.judge_model)

        except Exception as exc:
            _logger.error("Test case %s failed: %s", test_case.id, exc, exc_info=True)
            pipeline_error = str(exc)
            total_cost = 0.0

        duration = time.time() - start

        result = EvalResult(
            test_case=test_case,
            research_output=research_output,
            judge_result=judge_result,
            pipeline_error=pipeline_error,
            duration_seconds=duration,
            total_tokens=total_tokens,
            total_cost=total_cost,
            timestamp=datetime.utcnow(),
        )

        # Save result to JSON and SQLite
        _save_result(result, run_dir)
        try:
            job_store.save_eval_result(run_dir.name, result)
        except Exception as exc:
            _logger.warning("Could not save eval result to SQLite: %s", exc)
        return result

    def run_all(self, test_cases: list[TestCase], run_id: str | None = None) -> EvalSummary:
        """
        Run all provided test cases sequentially and return a summary.

        Prints progress to console as each case completes.
        Saves individual results + summary to evals/results/{run_id}/.

        Args:
            test_cases: List of TestCase to run.
            run_id:     Optional run identifier. Auto-generated UUID if not provided.

        Returns:
            EvalSummary with aggregate statistics.
        """
        run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        total = len(test_cases)
        results: list[EvalResult] = []

        print(f"\n{'='*60}")
        print(f"  Eval Run: {run_id}")
        print(f"  Cases:    {total}")
        print(f"  Judge:    {self.judge_model}")
        print(f"{'='*60}\n")

        for i, test_case in enumerate(test_cases, start=1):
            print(f"[{i}/{total}] {test_case.id} ({test_case.tier}) — {test_case.query[:55]}...")
            result = self.run_single(test_case, run_dir)
            results.append(result)

            # Print inline progress
            if result.pipeline_error:
                print(f"         [FAIL] PIPELINE ERROR: {result.pipeline_error[:80]}")
            elif result.judge_result:
                jr = result.judge_result
                status = "PASS" if jr.passed else "FAIL"
                verdict = jr.verdict.replace("_", " ").upper()
                print(
                    f"         [{status}] ({verdict}) "
                    f"score={jr.weighted_score:.2f}  "
                    f"dur={result.duration_seconds:.0f}s  "
                    f"cost=${result.total_cost:.4f}"
                )
            else:
                print(f"         [WARN] No judge result")

        summary = _compute_summary(run_id, results, self.judge_model)

        # Save summary JSON
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        _logger.info("Summary saved to %s", summary_path)

        # Save summary to SQLite
        try:
            job_store.save_eval_run(summary)
            _logger.info("Eval run %s saved to SQLite", run_id)
        except Exception as exc:
            _logger.warning("Could not save eval run to SQLite: %s", exc)

        return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_cost(pipeline_tokens: int, judge_tokens: int, judge_model: str = "") -> float:
    """Rough USD cost estimate combining pipeline (Groq) and judge tokens."""
    judge_rate = _judge_cost_per_token(judge_model) if judge_model else _GROQ_COST_PER_TOKEN
    return (pipeline_tokens * _GROQ_COST_PER_TOKEN) + (judge_tokens * judge_rate)


def _save_result(result: EvalResult, run_dir: Path) -> None:
    """Save an EvalResult to JSON in the run directory."""
    path = run_dir / f"{result.test_case.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def _compute_summary(run_id: str, results: list[EvalResult], judge_model: str) -> EvalSummary:
    """Compute aggregate statistics from a list of EvalResult."""
    total = len(results)
    scored = [r for r in results if r.judge_result is not None]
    errors = [r for r in results if r.pipeline_error is not None]

    passed_results = [r for r in scored if r.judge_result.passed]
    failed_results = [r for r in results if not r.judge_result or not r.judge_result.passed]

    pass_count = len(passed_results)
    fail_count = total - pass_count
    pass_rate = pass_count / total if total else 0.0

    avg_score = (
        sum(r.judge_result.weighted_score for r in scored) / len(scored)
        if scored else 0.0
    )
    avg_cost = sum(r.total_cost for r in results) / total if total else 0.0
    avg_duration = sum(r.duration_seconds for r in results) / total if total else 0.0
    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.total_cost for r in results)

    # Per-tier stats
    tiers = list({r.test_case.tier for r in results})
    tier_scores: dict[str, float] = {}
    tier_pass_rates: dict[str, float] = {}
    for tier in tiers:
        tier_results = [r for r in scored if r.test_case.tier == tier]
        tier_all = [r for r in results if r.test_case.tier == tier]
        if tier_results:
            tier_scores[tier] = sum(r.judge_result.weighted_score for r in tier_results) / len(tier_results)
        else:
            tier_scores[tier] = 0.0
        tier_pass = sum(1 for r in tier_all if r.judge_result and r.judge_result.passed)
        tier_pass_rates[tier] = tier_pass / len(tier_all) if tier_all else 0.0

    # Per-criteria stats
    criteria_scores: dict[str, float] = {}
    for criterion in RUBRIC:
        cname = criterion.name
        vals = [
            r.judge_result.scores.get(cname, 0.0)
            for r in scored
            if r.judge_result
        ]
        criteria_scores[cname] = sum(vals) / len(vals) if vals else 0.0

    lowest_criteria = (
        min(criteria_scores, key=criteria_scores.get) if criteria_scores else "unknown"
    )
    lowest_tier = min(tier_scores, key=tier_scores.get) if tier_scores else "unknown"
    failed_ids = [r.test_case.id for r in failed_results]

    return EvalSummary(
        run_id=run_id,
        timestamp=datetime.utcnow(),
        results=results,
        judge_model=judge_model,
        eval_test_count=EVAL_TEST_COUNT,
        total_cases=total,
        passed=pass_count,
        failed=fail_count,
        pass_rate=pass_rate,
        avg_weighted_score=avg_score,
        avg_cost_per_query=avg_cost,
        avg_duration_per_query=avg_duration,
        total_tokens_used=total_tokens,
        total_cost=total_cost,
        tier_scores=tier_scores,
        tier_pass_rates=tier_pass_rates,
        criteria_scores=criteria_scores,
        lowest_scoring_criteria=lowest_criteria,
        lowest_scoring_tier=lowest_tier,
        failed_cases=failed_ids,
    )
