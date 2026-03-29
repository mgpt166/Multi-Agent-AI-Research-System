"""
app/utils/groq_retry.py
=======================
Thin wrapper around the Groq chat completions API with automatic
retry-with-exponential-backoff on rate limit (429) errors.

Also tracks per-job, per-step call statistics (total calls, retries, failures)
via a thread-local context so the Observability UI can show LLM call health.

Usage:
    from app.utils.groq_retry import groq_chat, set_trace_context, clear_trace_context

    set_trace_context(job_id, "plan_research")
    response = groq_chat(client, model=..., max_tokens=..., messages=[...])
    clear_trace_context()

Retry behaviour:
    - Detects 429 / RateLimitError from the Groq SDK
    - Waits base_delay * 2^attempt seconds between retries (10s, 20s, 40s, 80s)
    - Gives up after max_retries attempts and re-raises the original error
    - All other errors are re-raised immediately (no retry)
"""

from __future__ import annotations
import logging
import threading
import time

_logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 10.0   # seconds — doubles each attempt: 10, 20, 40, 80, 160

# Minimum seconds to wait between consecutive Groq calls to stay well under
# the free-tier 6,000 TPM limit. Set to 0 to disable throttling (paid tier).
_MIN_CALL_INTERVAL = float(__import__("os").environ.get("GROQ_CALL_INTERVAL", "3"))

_last_call_time: float = 0.0

# ── Per-job call stats registry ───────────────────────────────────────────────
# Structure: {job_id: {step_name: {"calls": N, "retries": N, "failed": N}}}
_call_stats: dict[str, dict[str, dict]] = {}
_stats_lock = threading.Lock()

# Thread-local context so each thread knows its job_id and step name
_ctx = threading.local()


def set_trace_context(job_id: str, step: str) -> None:
    """Set the current thread's job_id and step for call stat attribution."""
    _ctx.job_id = job_id
    _ctx.step = step


def clear_trace_context() -> None:
    """Clear the current thread's trace context after a node finishes."""
    _ctx.job_id = None
    _ctx.step = None


def init_job_stats(job_id: str) -> None:
    """Initialise an empty stats dict for a new job. Called by runner.py."""
    with _stats_lock:
        _call_stats[job_id] = {}


def get_job_stats(job_id: str) -> dict[str, dict]:
    """Return the full call stats dict for a job. Called by runner.py on completion."""
    with _stats_lock:
        return dict(_call_stats.get(job_id, {}))


def clear_job_stats(job_id: str) -> None:
    """Remove stats for a completed/failed job to free memory."""
    with _stats_lock:
        _call_stats.pop(job_id, None)


def _record(job_id: str | None, step: str | None, key: str) -> None:
    """Increment a counter in the call stats registry."""
    if not job_id or not step:
        return
    with _stats_lock:
        job = _call_stats.setdefault(job_id, {})
        entry = job.setdefault(step, {"calls": 0, "retries": 0, "failed": 0})
        entry[key] += 1


def groq_chat(client, **kwargs):
    """
    Call client.chat.completions.create(**kwargs) with retry on rate limits.

    Throttles calls to at most one every GROQ_CALL_INTERVAL seconds (default 3s)
    to stay below the free-tier 6,000 TPM limit before retries are needed.

    Increments call stats for the current thread's job_id/step context.

    Args:
        client:   Groq client instance.
        **kwargs: Passed directly to chat.completions.create()
                  (model, max_tokens, messages, etc.)

    Returns:
        Groq chat completion response object.

    Raises:
        The last exception if all retries are exhausted.
    """
    global _last_call_time
    import time as _time

    job_id = getattr(_ctx, "job_id", None)
    step = getattr(_ctx, "step", None)

    # Count this as one LLM call attempt
    _record(job_id, step, "calls")

    # Throttle: ensure minimum gap between calls
    if _MIN_CALL_INTERVAL > 0:
        elapsed = _time.monotonic() - _last_call_time
        wait = _MIN_CALL_INTERVAL - elapsed
        if wait > 0:
            _time.sleep(wait)

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        _last_call_time = _time.monotonic()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "rate limit" in err_str.lower()
                or type(exc).__name__ == "RateLimitError"
            )
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2 ** attempt)
                _logger.warning(
                    "Groq rate limit hit (attempt %d/%d) — waiting %.0fs before retry. Error: %s",
                    attempt + 1, _MAX_RETRIES, delay, err_str[:120],
                )
                _record(job_id, step, "retries")
                time.sleep(delay)
                last_exc = exc
            else:
                # Permanent failure — record and re-raise
                _record(job_id, step, "failed")
                raise
    # All retries exhausted
    _record(job_id, step, "failed")
    raise last_exc  # type: ignore[misc]
