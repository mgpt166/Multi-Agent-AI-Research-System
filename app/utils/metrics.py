"""
app/utils/metrics.py
====================
In-memory metrics store for the research API.

Simple thread-safe counters and rolling averages. Resets on server restart.
Exposed at GET /metrics.

Metrics tracked:
    requests_total          — all research requests received
    requests_completed      — successfully completed jobs
    requests_failed         — jobs that errored
    avg_duration_seconds    — rolling average over last N requests
    avg_cost_per_query      — rolling average
    avg_tokens_per_query    — rolling average
    total_tokens_used       — lifetime token counter
    total_cost              — lifetime cost in USD
    sub_agent_failures      — individual sub-agent errors
    tool_call_failures      — individual search/fetch errors
    budget_exceeded_count   — jobs that hit the cost ceiling
    uptime_seconds          — seconds since the MetricsStore was created
"""
from __future__ import annotations
import threading
import time
from collections import deque


class MetricsStore:
    """
    Thread-safe in-memory metrics store.

    Args:
        rolling_window: Number of recent requests to include in rolling averages.
    """

    def __init__(self, rolling_window: int = 100) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._rolling_window = rolling_window

        # Counters
        self.requests_total: int = 0
        self.requests_completed: int = 0
        self.requests_failed: int = 0
        self.sub_agent_failures: int = 0
        self.tool_call_failures: int = 0
        self.budget_exceeded_count: int = 0
        self.total_tokens_used: int = 0
        self.total_cost: float = 0.0

        # Rolling window deques for averages
        self._recent_durations: deque[float] = deque(maxlen=rolling_window)
        self._recent_costs: deque[float] = deque(maxlen=rolling_window)
        self._recent_tokens: deque[int] = deque(maxlen=rolling_window)

    def record_request_start(self) -> None:
        """Increment requests_total when a new research job is submitted."""
        with self._lock:
            self.requests_total += 1

    def record_request_complete(
        self, duration_seconds: float, cost: float, tokens: int
    ) -> None:
        """
        Record a successfully completed job.

        Args:
            duration_seconds: Wall-clock time from submit to completion.
            cost:             Estimated USD cost for this job.
            tokens:           Total tokens used by this job.
        """
        with self._lock:
            self.requests_completed += 1
            self.total_cost += cost
            self.total_tokens_used += tokens
            self._recent_durations.append(duration_seconds)
            self._recent_costs.append(cost)
            self._recent_tokens.append(tokens)

    def record_request_failed(self) -> None:
        """Increment requests_failed when a job errors out."""
        with self._lock:
            self.requests_failed += 1

    def record_sub_agent_failure(self) -> None:
        """Increment when a sub-agent task raises an exception."""
        with self._lock:
            self.sub_agent_failures += 1

    def record_tool_call_failure(self) -> None:
        """Increment when a web search or fetch tool fails."""
        with self._lock:
            self.tool_call_failures += 1

    def record_budget_exceeded(self) -> None:
        """Increment when a job hits the MAX_COST_PER_QUERY ceiling."""
        with self._lock:
            self.budget_exceeded_count += 1

    def get_all(self) -> dict:
        """
        Return a complete snapshot of all metrics.

        Returns:
            dict suitable for the GET /metrics JSON response.
        """
        with self._lock:
            avg_dur = (
                sum(self._recent_durations) / len(self._recent_durations)
                if self._recent_durations else 0.0
            )
            avg_cost = (
                sum(self._recent_costs) / len(self._recent_costs)
                if self._recent_costs else 0.0
            )
            avg_tok = (
                sum(self._recent_tokens) / len(self._recent_tokens)
                if self._recent_tokens else 0
            )
            return {
                "requests_total": self.requests_total,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "avg_duration_seconds": round(avg_dur, 2),
                "avg_cost_per_query": round(avg_cost, 6),
                "avg_tokens_per_query": round(avg_tok),
                "total_tokens_used": self.total_tokens_used,
                "total_cost": round(self.total_cost, 6),
                "sub_agent_failures": self.sub_agent_failures,
                "tool_call_failures": self.tool_call_failures,
                "budget_exceeded_count": self.budget_exceeded_count,
                "uptime_seconds": round(time.time() - self._start_time),
            }


# Module-level singleton — imported directly by routes.py and runner.py
metrics = MetricsStore()
