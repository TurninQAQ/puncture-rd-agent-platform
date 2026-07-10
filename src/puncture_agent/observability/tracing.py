"""Minimal tracing API that mirrors concepts later exported via OpenTelemetry."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Mapping, Protocol
from uuid import uuid4


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    started_at_unix_ns: int
    ended_at_unix_ns: int | None = None
    duration_ms: float | None = None
    status: str = "UNSET"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceExporter(Protocol):
    def export(self, span: SpanRecord) -> None:
        ...


class InMemoryTraceExporter:
    """Thread-safe exporter used by unit tests and mock demos."""

    def __init__(self) -> None:
        self._spans: list[SpanRecord] = []
        self._lock = Lock()

    def export(self, span: SpanRecord) -> None:
        with self._lock:
            self._spans.append(span)

    def spans(self) -> list[SpanRecord]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class JsonLinesTraceExporter:
    """Append one JSON object per completed span for local replay/debugging."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def export(self, span: SpanRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(span.to_dict(), ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


_span_stack: ContextVar[tuple[SpanRecord, ...]] = ContextVar(
    "puncture_agent_span_stack", default=()
)


class TraceRecorder:
    """Create nested spans without requiring the OpenTelemetry package."""

    def __init__(self, exporter: TraceExporter | None = None) -> None:
        self.exporter = exporter or InMemoryTraceExporter()

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanRecord]:
        stack = _span_stack.get()
        parent = stack[-1] if stack else None
        span = SpanRecord(
            trace_id=parent.trace_id if parent else uuid4().hex,
            span_id=uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
            name=name,
            started_at_unix_ns=time.time_ns(),
            attributes=dict(attributes or {}),
        )
        token = _span_stack.set(stack + (span,))
        started_perf_ns = time.perf_counter_ns()
        try:
            yield span
        except Exception as exc:
            span.status = "ERROR"
            span.error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        else:
            span.status = "OK"
        finally:
            ended_perf_ns = time.perf_counter_ns()
            span.ended_at_unix_ns = time.time_ns()
            span.duration_ms = (ended_perf_ns - started_perf_ns) / 1_000_000
            _span_stack.reset(token)
            self.exporter.export(span)

    def add_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        stack = _span_stack.get()
        if not stack:
            raise RuntimeError("add_event requires an active span")
        stack[-1].events.append(
            {
                "name": name,
                "timestamp_unix_ns": time.time_ns(),
                "attributes": dict(attributes or {}),
            }
        )

    def current_span(self) -> SpanRecord | None:
        stack = _span_stack.get()
        return stack[-1] if stack else None
