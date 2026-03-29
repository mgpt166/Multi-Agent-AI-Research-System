"""
app/db/database.py
==================
SQLite connection management and schema initialisation.

The database file is stored at data/jobs.db (relative to project root).
The directory is created automatically if it does not exist.

All connections use sqlite3.Row as row_factory so rows can be accessed
as dicts (row["field"]) rather than by index.

Functions:
    get_connection()   Open and return a new sqlite3 connection.
    init_db()          Create all tables if they do not exist (idempotent).

Tables:
    jobs          Research job records
    events        Per-job activity log (live feed)
    eval_runs     One row per evaluation run (aggregate metrics)
    eval_results  One row per test case per run (per-criteria scores)
    job_traces    Observability trace + cost data per completed job
"""

from __future__ import annotations
import os
import sqlite3

# Path to the SQLite database file — relative to project root
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jobs.db")


def get_connection() -> sqlite3.Connection:
    """
    Open and return a new SQLite connection to the jobs database.
    Uses sqlite3.Row factory so rows support dict-style field access.
    The caller is responsible for closing the connection.
    """
    db_path = os.path.abspath(_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the jobs table if it does not already exist (idempotent — safe to call on every startup).

    Schema notes:
        - research_plan and token_usage are stored as JSON strings (TEXT)
        - All nullable fields default to NULL
        - created_at is set once; updated_at is updated on every write
    """
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id              TEXT PRIMARY KEY,
                query               TEXT NOT NULL,
                depth               TEXT NOT NULL DEFAULT 'moderate',
                output_folder       TEXT,
                max_iterations      INTEGER,
                status              TEXT NOT NULL DEFAULT 'queued',
                phase               TEXT NOT NULL DEFAULT 'queued',
                hitl_round          INTEGER NOT NULL DEFAULT 0,
                iteration_count     INTEGER NOT NULL DEFAULT 0,
                synthesis_review_count INTEGER NOT NULL DEFAULT 0,
                sub_agents_active   INTEGER NOT NULL DEFAULT 0,
                sub_agent_count     INTEGER NOT NULL DEFAULT 0,
                source_count        INTEGER NOT NULL DEFAULT 0,
                citation_count      INTEGER NOT NULL DEFAULT 0,
                duration_seconds    REAL NOT NULL DEFAULT 0,
                research_plan       TEXT,
                document_path       TEXT,
                summary_snippet     TEXT NOT NULL DEFAULT '',
                token_usage         TEXT NOT NULL DEFAULT '{}',
                error               TEXT,
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id    TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                message   TEXT    NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_job_id ON events(job_id)")

        # ── Eval runs — one row per evaluation run ────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id                  TEXT PRIMARY KEY,
                timestamp               TEXT NOT NULL,
                judge_model             TEXT NOT NULL,
                total_cases             INTEGER NOT NULL DEFAULT 0,
                passed                  INTEGER NOT NULL DEFAULT 0,
                failed                  INTEGER NOT NULL DEFAULT 0,
                pass_rate               REAL NOT NULL DEFAULT 0,
                avg_weighted_score      REAL NOT NULL DEFAULT 0,
                avg_cost_per_query      REAL NOT NULL DEFAULT 0,
                avg_duration_per_query  REAL NOT NULL DEFAULT 0,
                total_tokens_used       INTEGER NOT NULL DEFAULT 0,
                total_cost              REAL NOT NULL DEFAULT 0,
                lowest_scoring_criteria TEXT NOT NULL DEFAULT '',
                lowest_scoring_tier     TEXT NOT NULL DEFAULT '',
                created_at              TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs(timestamp)")

        # ── Eval results — one row per test case per run ──────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              TEXT NOT NULL,
                case_id             TEXT NOT NULL,
                tier                TEXT NOT NULL,
                query               TEXT NOT NULL,
                weighted_score      REAL,
                verdict             TEXT,
                passed              INTEGER NOT NULL DEFAULT 0,
                cost                REAL NOT NULL DEFAULT 0,
                duration_seconds    REAL NOT NULL DEFAULT 0,
                pipeline_error      TEXT,
                factual_accuracy    REAL,
                citation_quality    REAL,
                completeness        REAL,
                source_quality      REAL,
                structure_clarity   REAL,
                efficiency          REAL,
                must_cover_passed   INTEGER NOT NULL DEFAULT 0,
                must_cover_total    INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON eval_results(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_case_id ON eval_results(case_id)")

        # ── Job traces — observability data per completed job ──────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_traces (
                job_id          TEXT PRIMARY KEY,
                query           TEXT NOT NULL DEFAULT '',
                trace_summary   TEXT NOT NULL DEFAULT '{}',
                timeline        TEXT NOT NULL DEFAULT '[]',
                cost_summary    TEXT NOT NULL DEFAULT '{}',
                call_stats      TEXT NOT NULL DEFAULT '{}',
                saved_at        TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_traces_saved_at ON job_traces(saved_at)")
        # Migration: add call_stats column to existing databases that lack it
        try:
            conn.execute("ALTER TABLE job_traces ADD COLUMN call_stats TEXT NOT NULL DEFAULT '{}'")
            conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore

        conn.commit()
    finally:
        conn.close()
