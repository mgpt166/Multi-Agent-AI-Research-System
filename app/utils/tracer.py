"""
app/utils/tracer.py
===================
Lightweight per-request tracing for the research pipeline.

Creates a unique trace_id per research job and records a timeline of:
  - Spans  — timed operations (e.g. "plan_research", "sub_agent_01")
  - Events — point-in-time occurrences (e.g. "tool_call", "budget_warning")

No external dependencies — pure Python. The tracer is stored in a
module-level registry keyed by job_id so any part of the pipeline can
access it without threading it through the LangGraph state.

Usage:
    from app.utils.tracer import tracer_registry

    # At job start (runner.py):
    tracer_registry.create(job_id, query)

    # Inside a node or agent:
    t = tracer_registry.get(job_id)
    if t:
        sid = t.start_span("synthesize")
        ...
        t.end_span(sid, {"tokens": 1200})

    # At job end:
    summary = tracer_registry.get_summary(job_id)
    tracer_registry.remove(job_id)
"""
from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A timed operation within the pipeline."""
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "type": "span",
            "span_id": self.span_id,
            "name": self.name,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3) if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class TraceEvent:
    """A point-in-time event."""
    event_type: str
    message: str
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": "event",
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": round(self.timestamp, 3),
            "metadata": self.metadata,
        }


class RequestTracer:
    """
    Timeline recorder for a single research request.

    Args:
        trace_id: Unique identifier for this request (usually the job_id).
        query:    The original research query.
    """

    def __init__(self, trace_id: str, query: str) -> None:
        self.trace_id = trace_id
        self.query = query
        self._start_time = time.time()
        self._spans: dict[str, Span] = {}
        self._timeline: list[Span | TraceEvent] = []

    def start_span(self, name: str, metadata: dict | None = None) -> str:
        """
        Start a timed span.

        Args:
            name:     Human-readable span name (e.g. "plan_research").
            metadata: Optional key-value pairs attached to this span.

        Returns:
            span_id: Pass this to end_span() to close the span.
        """
        span_id = str(uuid.uuid4())[:8]
        span = Span(
            span_id=span_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._spans[span_id] = span
        self._timeline.append(span)
        _logger.debug("trace=%s span_start name=%s id=%s", self.trace_id, name, span_id)
        return span_id

    def end_span(self, span_id: str, metadata: dict | None = None) -> None:
        """
        Close a previously opened span and record its duration.

        Args:
            span_id:  ID returned by start_span().
            metadata: Optional additional metadata to merge into the span.
        """
        span = self._spans.get(span_id)
        if span is None:
            return
        span.end_time = time.time()
        if metadata:
            span.metadata.update(metadata)
        _logger.debug(
            "trace=%s span_end name=%s duration_ms=%.0f",
            self.trace_id, span.name, span.duration_ms or 0,
        )

    def log_event(self, event_type: str, message: str, metadata: dict | None = None) -> None:
        """
        Record a point-in-time event.

        Args:
            event_type: Category (e.g. "tool_call", "budget_warning", "error").
            message:    Human-readable description.
            metadata:   Optional key-value context.
        """
        event = TraceEvent(
            event_type=event_type,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._timeline.append(event)
        _logger.info("trace=%s event=%s %s", self.trace_id, event_type, message)

    def get_timeline(self) -> list[dict]:
        """Return all spans and events in chronological order."""
        return [item.to_dict() for item in self._timeline]

    def get_summary(self) -> dict:
        """
        Return a high-level summary of the request timeline.

        Returns:
            dict with trace_id, total duration, per-node durations, event counts.
        """
        total_ms = round((time.time() - self._start_time) * 1000, 2)
        node_durations = {
            s.name: s.duration_ms
            for s in self._spans.values()
            if s.end_time is not None
        }
        event_counts: dict[str, int] = {}
        for item in self._timeline:
            if isinstance(item, TraceEvent):
                event_counts[item.event_type] = event_counts.get(item.event_type, 0) + 1
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": total_ms,
            "node_durations_ms": node_durations,
            "event_counts": event_counts,
            "span_count": len(self._spans),
        }


class _TracerRegistry:
    """Thread-safe registry mapping job_id → RequestTracer."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._tracers: dict[str, RequestTracer] = {}

    def create(self, job_id: str, query: str) -> RequestTracer:
        t = RequestTracer(trace_id=job_id, query=query)
        with self._lock:
            self._tracers[job_id] = t
        return t

    def get(self, job_id: str) -> Optional[RequestTracer]:
        with self._lock:
            return self._tracers.get(job_id)

    def get_summary(self, job_id: str) -> Optional[dict]:
        t = self.get(job_id)
        return t.get_summary() if t else None

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._tracers.pop(job_id, None)


# Module-level singleton — import and use directly
tracer_registry = _TracerRegistry()
