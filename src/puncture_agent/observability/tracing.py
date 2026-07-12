"""Minimal tracing API that mirrors concepts later exported via OpenTelemetry.

The dependency-free :class:`TraceRecorder` remains the production-facing facade
used by graph runtimes. Optional OpenTelemetry export is layered on top via
:class:`OpenTelemetryTraceExporter` without coupling business code to a vendor
SDK.
"""

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

from puncture_agent.observability.attributes import sanitize_attributes
from puncture_agent.observability.propagation import (
    TraceContext,
    context_from_span_ids,
    extract_trace_context,
    inject_trace_context,
    new_trace_context,
)


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

    def as_trace_context(self) -> TraceContext:
        return context_from_span_ids(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
        )


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

    def by_name(self, name: str) -> list[SpanRecord]:
        with self._lock:
            return [span for span in self._spans if span.name == name]


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


class CompositeTraceExporter:
    """Fan-out completed spans to multiple exporters; failures are isolated."""

    def __init__(self, exporters: list[TraceExporter]) -> None:
        self.exporters = list(exporters)
        self._failure_count = 0
        self._lock = Lock()

    def export(self, span: SpanRecord) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(span)
            except Exception:
                with self._lock:
                    self._failure_count += 1

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count


class InMemoryOtlpTraceExporter:
    """In-memory OTLP-shaped collector used by unit tests.

    Records a simplified OTLP resource/scope/span payload without requiring the
    OpenTelemetry SDK. Suitable for asserting hierarchy, attributes, and status
    before wiring a live collector.
    """

    def __init__(
        self,
        *,
        service_name: str = "puncture-rd-agent",
        enqueue_timeout_ms: float = 50.0,
    ) -> None:
        self.service_name = service_name
        self.enqueue_timeout_ms = enqueue_timeout_ms
        self._records: list[dict[str, Any]] = []
        self._lock = Lock()
        self._blocked = False
        self._enqueue_failures = 0

    def block(self) -> None:
        """Simulate an unavailable remote collector for timeout tests."""

        self._blocked = True

    def unblock(self) -> None:
        self._blocked = False

    def export(self, span: SpanRecord) -> None:
        if self._blocked:
            # Application paths must not block on remote collector availability
            # beyond the configured enqueue timeout. We simulate immediate
            # failure rather than sleeping.
            with self._lock:
                self._enqueue_failures += 1
            raise TimeoutError(
                f"OTLP enqueue exceeded {self.enqueue_timeout_ms} ms"
            )
        payload = {
            "resource": {"service.name": self.service_name},
            "scope": {"name": "puncture_agent.observability", "version": "1"},
            "span": {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "parentSpanId": span.parent_span_id,
                "name": span.name,
                "startTimeUnixNano": span.started_at_unix_ns,
                "endTimeUnixNano": span.ended_at_unix_ns,
                "status": {"code": span.status},
                "attributes": dict(span.attributes),
                "events": list(span.events),
                "error": span.error,
                "duration_ms": span.duration_ms,
            },
        }
        with self._lock:
            self._records.append(payload)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def spans(self) -> list[dict[str, Any]]:
        return [item["span"] for item in self.records()]

    @property
    def enqueue_failure_count(self) -> int:
        with self._lock:
            return self._enqueue_failures

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._enqueue_failures = 0


class OpenTelemetryTraceExporter:
    """Optional bridge to the real OpenTelemetry SDK when installed.

    Business code never imports OpenTelemetry directly. If the SDK is missing,
    construction raises :class:`ImportError` so callers can fall back to the
    in-memory/JSONL exporters used by CI.
    """

    def __init__(
        self,
        *,
        service_name: str = "puncture-rd-agent",
        endpoint: str | None = None,
        headers: Mapping[str, str] | None = None,
        use_batch: bool = True,
    ) -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
                SpanExporter,
                SpanExportResult,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "opentelemetry-sdk is required for OpenTelemetryTraceExporter; "
                "install puncture-rd-agent-platform[implementation] or use "
                "InMemoryOtlpTraceExporter in tests"
            ) from exc

        self._otel_trace = otel_trace
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        self._provider = provider
        self._tracer = provider.get_tracer("puncture_agent.observability", "1")
        self._endpoint = endpoint
        self._headers = dict(headers or {})

        class _RecordingExporter(SpanExporter):
            def __init__(self, outer: "OpenTelemetryTraceExporter") -> None:
                self._outer = outer

            def export(self, spans):  # type: ignore[no-untyped-def]
                # When no OTLP transport is configured we only keep spans in the
                # SDK provider; callers can still use force_flush.
                del spans
                return SpanExportResult.SUCCESS

            def shutdown(self) -> None:
                return None

        processor_cls = BatchSpanProcessor if use_batch else SimpleSpanProcessor
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp = OTLPSpanExporter(
                    endpoint=endpoint,
                    headers=self._headers or None,
                )
                provider.add_span_processor(processor_cls(otlp))
            except ImportError:
                provider.add_span_processor(processor_cls(_RecordingExporter(self)))
        else:
            provider.add_span_processor(processor_cls(_RecordingExporter(self)))
        otel_trace.set_tracer_provider(provider)

    def export(self, span: SpanRecord) -> None:
        # Bridge completed TraceRecorder spans into the SDK as standalone spans.
        # Parent linkage uses OTEL context when the parent is still active; for
        # already-finished parents we still emit the W3C IDs as attributes so
        # collectors can reconstruct hierarchy from the facade records.
        tracer = self._tracer
        attributes = {
            **dict(span.attributes),
            "puncture.trace_id": span.trace_id,
            "puncture.span_id": span.span_id,
        }
        if span.parent_span_id:
            attributes["puncture.parent_span_id"] = span.parent_span_id
        with tracer.start_as_current_span(span.name, attributes=attributes) as otel_span:
            if span.status == "ERROR":
                from opentelemetry.trace import Status, StatusCode

                message = None
                if span.error:
                    message = str(span.error.get("message") or span.error.get("type"))
                otel_span.set_status(Status(StatusCode.ERROR, description=message))
                if span.error:
                    otel_span.record_exception(
                        RuntimeError(str(span.error.get("message") or "error"))
                    )
            for event in span.events:
                otel_span.add_event(
                    str(event.get("name") or "event"),
                    attributes=dict(event.get("attributes") or {}),
                )

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        return bool(self._provider.force_flush(timeout_millis))

    def shutdown(self) -> None:
        self._provider.shutdown()


# Stable low-cardinality span names used across graph, RAG, model, MCP and
# checkpoint instrumentation. Adapters should prefer these constants.
SPAN_AGENT_GRAPH = "agent.graph"
SPAN_AGENT_NODE = "agent.node"
SPAN_RAG_REWRITE = "rag.rewrite"
SPAN_RAG_RETRIEVE = "rag.retrieve"
SPAN_RAG_RERANK = "rag.rerank"
SPAN_MODEL_GENERATE = "model.generate"
SPAN_MCP_TOOL = "mcp.tool"
SPAN_VERIFIER = "agent.verifier"
SPAN_CHECKPOINT = "agent.checkpoint"


_span_stack: ContextVar[tuple[SpanRecord, ...]] = ContextVar(
    "puncture_agent_span_stack", default=()
)
_active_trace_context: ContextVar[TraceContext | None] = ContextVar(
    "puncture_agent_active_trace_context", default=None
)


class TraceRecorder:
    """Create nested spans without requiring the OpenTelemetry package."""

    def __init__(
        self,
        exporter: TraceExporter | None = None,
        *,
        sanitize: bool = True,
    ) -> None:
        self.exporter = exporter or InMemoryTraceExporter()
        self.sanitize = sanitize
        self._export_failure_count = 0
        self._export_failure_lock = Lock()

    def _prepare_attributes(
        self, attributes: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        raw = dict(attributes or {})
        if not self.sanitize:
            return raw
        return sanitize_attributes(raw)

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        remote_parent: TraceContext | None = None,
    ) -> Iterator[SpanRecord]:
        stack = _span_stack.get()
        parent = stack[-1] if stack else None
        if trace_id is not None and (
            not isinstance(trace_id, str) or not trace_id.strip()
        ):
            raise ValueError("trace_id must be a non-empty string when provided")

        if parent is not None:
            resolved_trace_id = parent.trace_id
            parent_span_id = parent.span_id
        elif remote_parent is not None:
            resolved_trace_id = remote_parent.trace_id
            parent_span_id = remote_parent.span_id
        else:
            resolved_trace_id = (trace_id or uuid4().hex).lower()
            # Accept non-hex legacy IDs from state.metadata for continuity, but
            # normalize empty values out above.
            parent_span_id = None
            if len(resolved_trace_id) != 32 or any(
                ch not in "0123456789abcdef" for ch in resolved_trace_id
            ):
                # Preserve caller-provided opaque IDs (e.g. demo strings) for
                # single-process correlation; propagation helpers still require
                # W3C hex when crossing process boundaries.
                pass

        span = SpanRecord(
            trace_id=resolved_trace_id,
            span_id=uuid4().hex[:16],
            parent_span_id=parent_span_id,
            name=name,
            started_at_unix_ns=time.time_ns(),
            attributes=self._prepare_attributes(attributes),
        )
        token = _span_stack.set(stack + (span,))
        context_token = _active_trace_context.set(span.as_trace_context())
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
            _active_trace_context.reset(context_token)
            try:
                self.exporter.export(span)
            except Exception:
                # Observability is deliberately non-fatal: a trace sink outage
                # must not turn a completed tool/graph transition into a replay.
                with self._export_failure_lock:
                    self._export_failure_count += 1

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
                "attributes": self._prepare_attributes(attributes),
            }
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a privacy-filtered attribute on the active span."""

        stack = _span_stack.get()
        if not stack:
            raise RuntimeError("set_attribute requires an active span")
        prepared = self._prepare_attributes({key: value})
        stack[-1].attributes.update(prepared)

    def current_span(self) -> SpanRecord | None:
        stack = _span_stack.get()
        return stack[-1] if stack else None

    def current_trace_context(self) -> TraceContext | None:
        span = self.current_span()
        if span is not None:
            return span.as_trace_context()
        return _active_trace_context.get()

    def inject_headers(
        self,
        headers: dict[str, str] | None = None,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Inject the active (or new) trace context into outbound carriers."""

        context = self.current_trace_context() or new_trace_context()
        return inject_trace_context(context, headers if headers is not None else {}, metadata=metadata)

    def attach_remote_parent(
        self,
        headers: Mapping[str, str] | None = None,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> TraceContext | None:
        """Extract an inbound remote parent without creating a span."""

        return extract_trace_context(headers, metadata=metadata)

    @property
    def export_failure_count(self) -> int:
        with self._export_failure_lock:
            return self._export_failure_count


def start_instrumented_span(
    tracer: TraceRecorder,
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    remote_parent: TraceContext | None = None,
    trace_id: str | None = None,
):
    """Convenience wrapper used by adapters for stable span names."""

    return tracer.start_span(
        name,
        attributes=attributes,
        remote_parent=remote_parent,
        trace_id=trace_id,
    )
