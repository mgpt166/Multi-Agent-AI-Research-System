"""
app/utils/job_store.py
======================
Thread-safe SQLite-backed store for tracking research job state.

Replaces the previous in-memory dict implementation. The public interface
(create_job, get_job, update_job, list_jobs) is identical — no other module
needs to change.

Data is persisted to data/jobs.db and survives server restarts. All past
research runs remain queryable and their reports remain downloadable as long
as the .docx files exist in output/.

Design notes:
    - Each method opens and closes its own connection (simple, safe for low concurrency).
    - threading.RLock serialises concurrent writes from graph background threads.
    - JSON fields (research_plan, token_usage) are serialised as TEXT in SQLite.
    - sqlite3.Row factory allows dict-style field access on read.

Exports:
    job_store   — singleton instance used throughout the application
"""

from __future__ import annotations
import json
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from app.db.database import get_connection, init_db

# JSON fields that must be serialised on write and deserialised on read
_JSON_FIELDS = {"research_plan", "token_usage"}


class JobStore:
    """
    SQLite-backed job registry with the same interface as the former in-memory store.

    All public methods are thread-safe via a reentrant lock. Each method manages
    its own database connection lifecycle (open → use → close).
    """

    def __init__(self):
        self._lock = threading.RLock()
        # Ensure the schema exists before any method is called
        init_db()

    def create_job(
        self,
        query: str,
        depth: str = "moderate",
        output_folder: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> str:
        """
        Register a new research job in the database and return its unique ID.

        Args:
            query:          The research question.
            depth:          Research depth level (simple | moderate | deep).
            output_folder:  Optional custom output directory for the report.
            max_iterations: Optional override for max research rounds.

        Returns:
            str: UUID4 job identifier.
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = get_connection()
            try:
                conn.execute("""
                    INSERT INTO jobs (
                        job_id, query, depth, output_folder, max_iterations,
                        status, phase, hitl_round, iteration_count,
                        synthesis_review_count, sub_agents_active, sub_agent_count,
                        source_count, citation_count, duration_seconds,
                        research_plan, document_path, summary_snippet,
                        token_usage, error, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        'queued', 'queued', 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        NULL, NULL, '',
                        '{}', NULL, ?, ?
                    )
                """, (job_id, query, depth, output_folder, max_iterations, now, now))
                conn.commit()
            finally:
                conn.close()

        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        """
        Retrieve a job's current state as a plain dict.

        JSON fields (research_plan, token_usage) are automatically deserialised.

        Args:
            job_id: The job's UUID.

        Returns:
            dict if found, None if job_id is not in the database.
        """
        with self._lock:
            conn = get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return _row_to_dict(row)

    def update_job(self, job_id: str, **kwargs) -> None:
        """
        Merge keyword arguments into the job's stored state.

        Automatically serialises JSON fields and sets updated_at to now.
        Silently ignores updates for unknown job_ids.

        Args:
            job_id: The job's UUID.
            **kwargs: Fields to update (e.g. status="running", phase="synthesizing").
        """
        if not kwargs:
            return

        # Serialise any JSON fields present in the update
        serialised = {}
        for k, v in kwargs.items():
            if k in _JSON_FIELDS:
                serialised[k] = json.dumps(v) if v is not None else None
            else:
                serialised[k] = v

        serialised["updated_at"] = datetime.utcnow().isoformat()

        # Build dynamic SET clause from provided fields
        set_clause = ", ".join(f"{col} = ?" for col in serialised)
        values = list(serialised.values()) + [job_id]

        with self._lock:
            conn = get_connection()
            try:
                conn.execute(
                    f"UPDATE jobs SET {set_clause} WHERE job_id = ?",
                    values,
                )
                conn.commit()
            finally:
                conn.close()

    def list_jobs(self) -> list[dict]:
        """
        Return all jobs ordered by creation date (newest first).

        Returns:
            list[dict]: All job records as plain dicts, newest first.
        """
        with self._lock:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            finally:
                conn.close()

        return [_row_to_dict(row) for row in rows]

    def emit_event(self, job_id: str, message: str) -> None:
        """
        Record a timestamped activity event for a job.

        Called by graph nodes and agents to log progress that can be
        displayed in the UI activity feed.

        Args:
            job_id:  The job's UUID.
            message: Human-readable event description (e.g. "🤖 SubAgent-1 starting...").
        """
        import logging
        now = datetime.utcnow().isoformat()
        logging.getLogger(__name__).info("[%s] %s", job_id[:8], message)
        with self._lock:
            conn = get_connection()
            try:
                conn.execute(
                    "INSERT INTO events (job_id, timestamp, message) VALUES (?, ?, ?)",
                    (job_id, now, message),
                )
                conn.commit()
            finally:
                conn.close()

    def get_events(self, job_id: str) -> list[dict]:
        """
        Return all activity events for a job ordered by time ascending.

        Args:
            job_id: The job's UUID.

        Returns:
            list[dict]: Each dict has keys: id, job_id, timestamp, message.
        """
        with self._lock:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT id, job_id, timestamp, message FROM events WHERE job_id = ? ORDER BY id ASC",
                    (job_id,),
                ).fetchall()
            finally:
                conn.close()
        return [dict(row) for row in rows]

    # ── Eval persistence ──────────────────────────────────────────────────────

    def save_eval_run(self, summary) -> None:
        """
        Persist an EvalSummary to the eval_runs table.

        Args:
            summary: EvalSummary dataclass from evals/runner.py.
        """
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO eval_runs (
                        run_id, timestamp, judge_model,
                        total_cases, passed, failed, pass_rate,
                        avg_weighted_score, avg_cost_per_query, avg_duration_per_query,
                        total_tokens_used, total_cost,
                        lowest_scoring_criteria, lowest_scoring_tier,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.run_id,
                    summary.timestamp.isoformat(),
                    summary.judge_model,
                    summary.total_cases,
                    summary.passed,
                    summary.failed,
                    summary.pass_rate,
                    summary.avg_weighted_score,
                    summary.avg_cost_per_query,
                    summary.avg_duration_per_query,
                    summary.total_tokens_used,
                    summary.total_cost,
                    summary.lowest_scoring_criteria,
                    summary.lowest_scoring_tier,
                    now,
                ))
                conn.commit()
            finally:
                conn.close()

    def save_eval_result(self, run_id: str, result) -> None:
        """
        Persist a single EvalResult to the eval_results table.

        Args:
            run_id: The eval run this result belongs to.
            result: EvalResult dataclass from evals/runner.py.
        """
        jr = result.judge_result
        scores = jr.scores if jr else {}
        must_cover_passed = sum(1 for v in jr.must_cover_results.values() if v) if jr else 0
        must_cover_total = len(jr.must_cover_results) if jr else 0
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = get_connection()
            try:
                conn.execute("""
                    INSERT INTO eval_results (
                        run_id, case_id, tier, query,
                        weighted_score, verdict, passed,
                        cost, duration_seconds, pipeline_error,
                        factual_accuracy, citation_quality, completeness,
                        source_quality, structure_clarity, efficiency,
                        must_cover_passed, must_cover_total, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    result.test_case.id,
                    result.test_case.tier,
                    result.test_case.query,
                    jr.weighted_score if jr else None,
                    jr.verdict if jr else None,
                    1 if (jr and jr.passed) else 0,
                    result.total_cost,
                    result.duration_seconds,
                    result.pipeline_error,
                    scores.get("factual_accuracy"),
                    scores.get("citation_quality"),
                    scores.get("completeness"),
                    scores.get("source_quality"),
                    scores.get("structure_clarity"),
                    scores.get("efficiency"),
                    must_cover_passed,
                    must_cover_total,
                    now,
                ))
                conn.commit()
            finally:
                conn.close()

    def list_eval_runs(self) -> list[dict]:
        """Return all eval runs ordered by timestamp descending."""
        with self._lock:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM eval_runs ORDER BY timestamp DESC"
                ).fetchall()
            finally:
                conn.close()
        return [dict(row) for row in rows]

    def get_eval_results(self, run_id: str) -> list[dict]:
        """Return all individual test case results for a given run."""
        with self._lock:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id ASC",
                    (run_id,),
                ).fetchall()
            finally:
                conn.close()
        return [dict(row) for row in rows]

    # ── Dashboard stats ────────────────────────────────────────────────────────

    def get_dashboard_stats(self, hours: int = 24) -> dict:
        """
        Return aggregated job and cost stats for the last N hours from SQLite.

        Queries the jobs table (for request/duration stats) and job_traces table
        (for cost/token stats) so data survives server restarts.

        Args:
            hours: Lookback window in hours (default 24).

        Returns:
            dict with keys: total, completed, failed, success_rate,
            avg_duration, fastest, slowest, total_cost, avg_cost,
            total_tokens, avg_tokens, budget_exceeded_count,
            sub_agent_failures, tool_call_failures.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._lock:
            conn = get_connection()
            try:
                job_rows = conn.execute(
                    "SELECT status, duration_seconds FROM jobs WHERE created_at >= ?",
                    (cutoff,),
                ).fetchall()
                trace_rows = conn.execute(
                    "SELECT cost_summary FROM job_traces WHERE saved_at >= ?",
                    (cutoff,),
                ).fetchall()
            finally:
                conn.close()

        jobs = [dict(r) for r in job_rows]
        total = len(jobs)
        completed = [j for j in jobs if j["status"] == "complete"]
        failed = [j for j in jobs if j["status"] == "failed"]
        durations = [j["duration_seconds"] for j in completed if j.get("duration_seconds")]

        total_cost = 0.0
        total_tokens = 0
        budget_exceeded = 0
        for row in trace_rows:
            try:
                cs = json.loads(row["cost_summary"] or "{}")
                total_cost += cs.get("total", 0.0)
                total_tokens += (
                    cs.get("total_input_tokens", 0) + cs.get("total_output_tokens", 0)
                )
                if cs.get("budget_exceeded"):
                    budget_exceeded += 1
            except (json.JSONDecodeError, TypeError):
                pass

        n_completed = len(completed)
        n_traces = len(trace_rows)

        return {
            "total": total,
            "completed": n_completed,
            "failed": len(failed),
            "success_rate": n_completed / total * 100 if total else 0.0,
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "fastest": min(durations) if durations else 0.0,
            "slowest": max(durations) if durations else 0.0,
            "total_cost": total_cost,
            "avg_cost": total_cost / n_traces if n_traces else 0.0,
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / n_traces if n_traces else 0.0,
            "budget_exceeded_count": budget_exceeded,
        }

    # ── Observability traces ───────────────────────────────────────────────────

    def save_job_trace(
        self,
        job_id: str,
        query: str,
        trace_summary: dict,
        timeline: list,
        cost_summary: dict,
        call_stats: dict | None = None,
    ) -> None:
        """
        Persist observability data for a completed job.

        Called by runner._record_completion() after the graph finishes.
        Uses INSERT OR REPLACE so re-runs overwrite the previous trace.

        Args:
            job_id:        The job's UUID.
            query:         Original research question.
            trace_summary: Output of RequestTracer.get_summary().
            timeline:      Spans-only list from RequestTracer.get_timeline().
            cost_summary:  Output of CostTracker.get_summary().
            call_stats:    Per-step LLM call stats from groq_retry.get_job_stats().
        """
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO job_traces
                        (job_id, query, trace_summary, timeline, cost_summary, call_stats, saved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    query,
                    json.dumps(trace_summary),
                    json.dumps(timeline),
                    json.dumps(cost_summary),
                    json.dumps(call_stats or {}),
                    now,
                ))
                conn.commit()
            finally:
                conn.close()

    def get_job_trace(self, job_id: str) -> Optional[dict]:
        """
        Return the observability trace for a job, or None if not found.

        JSON fields (trace_summary, timeline, cost_summary) are automatically
        deserialised back to Python objects.

        Args:
            job_id: The job's UUID.
        """
        with self._lock:
            conn = get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM job_traces WHERE job_id = ?", (job_id,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        d = dict(row)
        for field in ("trace_summary", "timeline", "cost_summary", "call_stats"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def list_recent_job_traces(self, limit: int = 10) -> list[dict]:
        """
        Return the most recent completed job traces (lightweight — no JSON blobs).

        Used to populate the Observability tab job selector dropdown.

        Args:
            limit: Maximum number of traces to return (default 10).

        Returns:
            list[dict]: Each dict has job_id, query, saved_at — newest first.
        """
        with self._lock:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT job_id, query, saved_at FROM job_traces ORDER BY saved_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        return [dict(row) for row in rows]


def _row_to_dict(row: Any) -> dict:
    """
    Convert a sqlite3.Row to a plain dict, deserialising JSON fields.

    Args:
        row: sqlite3.Row from a jobs table query.

    Returns:
        dict with all fields; JSON fields parsed back to Python objects.
    """
    d = dict(row)
    for field in _JSON_FIELDS:
        if field in d and d[field] is not None:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass  # Leave as-is if parsing fails
    return d


# Module-level singleton — imported directly by routes and graph nodes
job_store = JobStore()
