"""Unit tests for MetricsStore — no API calls required."""
from app.utils.metrics import MetricsStore


def test_initial_state():
    m = MetricsStore()
    data = m.get_all()
    assert data["requests_total"] == 0
    assert data["requests_completed"] == 0
    assert data["requests_failed"] == 0
    assert data["total_cost"] == 0.0
    assert data["total_tokens_used"] == 0


def test_request_start_increments_total():
    m = MetricsStore()
    m.record_request_start()
    m.record_request_start()
    assert m.get_all()["requests_total"] == 2


def test_request_complete_increments_completed():
    m = MetricsStore()
    m.record_request_complete(duration_seconds=60.0, cost=0.10, tokens=10_000)
    data = m.get_all()
    assert data["requests_completed"] == 1
    assert abs(data["total_cost"] - 0.10) < 1e-9
    assert data["total_tokens_used"] == 10_000


def test_request_failed_increments_failed():
    m = MetricsStore()
    m.record_request_failed()
    assert m.get_all()["requests_failed"] == 1


def test_rolling_average_duration():
    m = MetricsStore(rolling_window=3)
    m.record_request_complete(10.0, 0.0, 0)
    m.record_request_complete(20.0, 0.0, 0)
    m.record_request_complete(30.0, 0.0, 0)
    assert abs(m.get_all()["avg_duration_seconds"] - 20.0) < 0.1


def test_rolling_average_cost():
    m = MetricsStore(rolling_window=2)
    m.record_request_complete(0, 0.10, 0)
    m.record_request_complete(0, 0.20, 0)
    assert abs(m.get_all()["avg_cost_per_query"] - 0.15) < 1e-6


def test_rolling_window_evicts_oldest():
    m = MetricsStore(rolling_window=2)
    m.record_request_complete(100.0, 0.0, 0)  # evicted
    m.record_request_complete(10.0, 0.0, 0)
    m.record_request_complete(20.0, 0.0, 0)
    # Average of last 2: (10+20)/2 = 15
    assert abs(m.get_all()["avg_duration_seconds"] - 15.0) < 0.1


def test_failure_counters():
    m = MetricsStore()
    m.record_sub_agent_failure()
    m.record_sub_agent_failure()
    m.record_tool_call_failure()
    m.record_budget_exceeded()
    data = m.get_all()
    assert data["sub_agent_failures"] == 2
    assert data["tool_call_failures"] == 1
    assert data["budget_exceeded_count"] == 1


def test_uptime_is_positive():
    m = MetricsStore()
    assert m.get_all()["uptime_seconds"] >= 0


def test_get_all_has_all_required_keys():
    m = MetricsStore()
    data = m.get_all()
    required = [
        "requests_total", "requests_completed", "requests_failed",
        "avg_duration_seconds", "avg_cost_per_query", "avg_tokens_per_query",
        "total_tokens_used", "total_cost", "sub_agent_failures",
        "tool_call_failures", "budget_exceeded_count", "uptime_seconds",
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"
