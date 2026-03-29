"""Unit tests for RequestTracer — no API calls required."""
import time
import pytest
from app.utils.tracer import RequestTracer, tracer_registry


def test_span_records_duration():
    t = RequestTracer(trace_id="test-123", query="test query")
    sid = t.start_span("my_span")
    time.sleep(0.05)
    t.end_span(sid)
    spans = [x for x in t.get_timeline() if x["type"] == "span"]
    assert len(spans) == 1
    assert spans[0]["duration_ms"] is not None
    assert spans[0]["duration_ms"] >= 40


def test_unended_span_duration_is_none():
    t = RequestTracer(trace_id="t", query="q")
    t.start_span("open_span")
    spans = [x for x in t.get_timeline() if x["type"] == "span"]
    assert spans[0]["duration_ms"] is None


def test_events_recorded_in_order():
    t = RequestTracer(trace_id="test-456", query="test")
    t.log_event("tool_call", "search started")
    t.log_event("tool_call", "search ended")
    t.log_event("budget_warning", "70% budget used")
    events = [x for x in t.get_timeline() if x["type"] == "event"]
    assert len(events) == 3
    assert events[0]["message"] == "search started"
    assert events[2]["event_type"] == "budget_warning"


def test_span_metadata_merged_on_end():
    t = RequestTracer(trace_id="t", query="q")
    sid = t.start_span("s", metadata={"start_key": "start_val"})
    t.end_span(sid, metadata={"end_key": "end_val"})
    spans = [x for x in t.get_timeline() if x["type"] == "span"]
    assert spans[0]["metadata"]["start_key"] == "start_val"
    assert spans[0]["metadata"]["end_key"] == "end_val"


def test_summary_has_required_fields():
    t = RequestTracer(trace_id="test-789", query="test")
    sid = t.start_span("node_a")
    t.end_span(sid)
    t.log_event("info", "done")
    summary = t.get_summary()
    assert "trace_id" in summary
    assert "total_duration_ms" in summary
    assert "node_durations_ms" in summary
    assert "event_counts" in summary
    assert "span_count" in summary


def test_summary_node_durations():
    t = RequestTracer(trace_id="t", query="q")
    sid = t.start_span("my_node")
    time.sleep(0.03)
    t.end_span(sid)
    summary = t.get_summary()
    assert "my_node" in summary["node_durations_ms"]
    assert summary["node_durations_ms"]["my_node"] >= 20


def test_summary_event_counts():
    t = RequestTracer(trace_id="t", query="q")
    t.log_event("tool_call", "a")
    t.log_event("tool_call", "b")
    t.log_event("error", "oops")
    summary = t.get_summary()
    assert summary["event_counts"]["tool_call"] == 2
    assert summary["event_counts"]["error"] == 1


def test_registry_create_and_get():
    tracer_registry.create("job-001", "test query")
    t = tracer_registry.get("job-001")
    assert t is not None
    assert t.trace_id == "job-001"
    tracer_registry.remove("job-001")
    assert tracer_registry.get("job-001") is None


def test_registry_get_missing_returns_none():
    assert tracer_registry.get("nonexistent-job") is None
