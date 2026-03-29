"""Unit tests for CostTracker — no API calls required."""
import pytest
from app.utils.cost_tracker import CostTracker, get_model_pricing


def _tracker(max_budget=1.0, warning=0.7):
    return CostTracker(max_budget=max_budget, warning_threshold=warning)


def test_initial_state():
    t = _tracker()
    assert t.total_cost == 0.0
    assert t.remaining_budget == 1.0
    assert not t.is_budget_exceeded()
    assert not t.is_warning_threshold()


def test_add_usage_calculates_cost_correctly():
    # claude-sonnet-4-6: $3/1M input — 1M tokens should cost $3
    t = _tracker()
    t.add_usage(1_000_000, 0, "test", model="claude-sonnet-4-6")
    assert abs(t.total_cost - 3.0) < 1e-6


def test_output_tokens_priced_separately():
    # claude-sonnet-4-6: $15/1M output
    t = _tracker()
    t.add_usage(0, 1_000_000, "test", model="claude-sonnet-4-6")
    assert abs(t.total_cost - 15.0) < 1e-6


def test_multiple_calls_accumulate():
    t = _tracker()
    t.add_usage(500_000, 0, "a", model="claude-sonnet-4-6")   # $1.50
    t.add_usage(500_000, 0, "b", model="claude-sonnet-4-6")   # $1.50
    assert abs(t.total_cost - 3.0) < 1e-6


def test_dynamic_pricing_groq_8b():
    inp, out = get_model_pricing("llama-3.1-8b-instant")
    assert inp == 0.05
    assert out == 0.08


def test_dynamic_pricing_groq_70b():
    inp, out = get_model_pricing("llama-3.3-70b-versatile")
    assert inp == 0.59
    assert out == 0.79


def test_dynamic_pricing_claude_sonnet():
    inp, out = get_model_pricing("claude-sonnet-4-6")
    assert inp == 3.00
    assert out == 15.00


def test_dynamic_pricing_gpt4o_mini():
    inp, out = get_model_pricing("gpt-4o-mini")
    assert inp == 0.15
    assert out == 0.60


def test_mixed_models_costed_correctly():
    # Groq 8b for research + Claude sonnet for judge — should use different prices
    t = _tracker()
    t.add_usage(1_000_000, 0, "sub_agent", model="llama-3.1-8b-instant")   # $0.05
    t.add_usage(1_000_000, 0, "judge",     model="claude-sonnet-4-6")       # $3.00
    assert abs(t.total_cost - 3.05) < 1e-6
    assert t.get_summary()["breakdown"]["sub_agent"]["model"] == "llama-3.1-8b-instant"
    assert t.get_summary()["breakdown"]["judge"]["model"] == "claude-sonnet-4-6"


def test_per_agent_breakdown_recorded():
    t = _tracker()
    t.add_usage(100, 50, "agent_a")
    t.add_usage(200, 100, "agent_b")
    summary = t.get_summary()
    assert "agent_a" in summary["breakdown"]
    assert "agent_b" in summary["breakdown"]
    assert summary["breakdown"]["agent_a"]["input_tokens"] == 100


def test_same_agent_calls_aggregated():
    t = _tracker()
    t.add_usage(100_000, 0, "lead", model="llama-3.1-8b-instant")
    t.add_usage(100_000, 0, "lead", model="llama-3.1-8b-instant")
    summary = t.get_summary()
    assert summary["breakdown"]["lead"]["input_tokens"] == 200_000


def test_budget_exceeded():
    t = _tracker(max_budget=0.001)
    t.add_usage(1_000_000, 0, "test", model="claude-sonnet-4-6")  # costs $3 >> $0.001
    assert t.is_budget_exceeded()
    assert t.remaining_budget == 0.0
    assert t.get_summary()["budget_exceeded"] is True


def test_warning_threshold():
    t = _tracker(max_budget=10.0, warning=0.5)
    t.add_usage(1_000_000, 0, "test", model="claude-sonnet-4-6")  # costs $3 > $0.5
    assert t.is_warning_threshold()
    assert t.get_summary()["warning_triggered"] is True


def test_remaining_budget_never_negative():
    t = _tracker(max_budget=0.01)
    t.add_usage(1_000_000, 0, "test", model="claude-sonnet-4-6")  # way over budget
    assert t.remaining_budget == 0.0


def test_summary_structure():
    t = _tracker()
    t.add_usage(500, 250, "lead_plan")
    s = t.get_summary()
    assert "total" in s
    assert "total_input_tokens" in s
    assert "total_output_tokens" in s
    assert "breakdown" in s
    assert "budget_remaining" in s
    assert "budget_exceeded" in s
    assert "warning_triggered" in s
