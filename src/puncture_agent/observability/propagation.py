"""W3C Trace Context style propagation helpers for HTTP / gRPC / MCP boundaries.

These helpers are dependency-free. They encode and decode a single request's
``traceparent`` (and optional baggage) so graph, RAG, model, and MCP clients
share one ``trace_id`` across process boundaries without requiring the
OpenTelemetry SDK at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, MutableMapping
from uuid import uuid4

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
TRACE_ID_HEADER = "x-puncture-trace-id"
SPAN_ID_HEADER = "x-puncture-span-id"
BAGGAGE_HEADER = "baggage"

# W3C: version-traceid-spanid-flags  (version 00)
_TRACEPARENT_RE = re.compile(
    r"^"
    r"(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})"
    r"$",
    re.IGNORECASE,
)

_HEX32 = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
_HEX16 = re.compile(r"^[0-9a-f]{16}$", re.IGNORECASE)


@dataclass(frozen=True)
class TraceContext:
    """In-process representation of a distributed trace context."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: str = "01"
    trace_state: str | None = None
    baggage: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        tid = self.trace_id.lower()
        sid = self.span_id.lower()
        if not _HEX32.match(tid) or tid == "0" * 32:
            raise ValueError("trace_id must be 32 lowercase hex chars, non-zero")
        if not _HEX16.match(sid) or sid == "0" * 16:
            raise ValueError("span_id must be 16 lowercase hex chars, non-zero")
        object.__setattr__(self, "trace_id", tid)
        object.__setattr__(self, "span_id", sid)
        object.__setattr__(self, "trace_flags", (self.trace_flags or "01").lower())

    def child(self, *, span_id: str | None = None) -> "TraceContext":
        child_span = (span_id or uuid4().hex[:16]).lower()
        if not _HEX16.match(child_span) or child_span == "0" * 16:
            raise ValueError("child span_id must be 16 non-zero hex chars")
        return TraceContext(
            trace_id=self.trace_id,
            span_id=child_span,
            parent_span_id=self.span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state,
            baggage=dict(self.baggage) if self.baggage else None,
        )


def new_trace_context(*, sampled: bool = True) -> TraceContext:
    return TraceContext(
        trace_id=uuid4().hex,
        span_id=uuid4().hex[:16],
        parent_span_id=None,
        trace_flags="01" if sampled else "00",
    )


def format_traceparent(context: TraceContext) -> str:
    return (
        f"00-{context.trace_id}-{context.span_id}-{context.trace_flags}"
    )


def parse_traceparent(value: str) -> TraceContext:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("traceparent must be a non-empty string")
    match = _TRACEPARENT_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid traceparent: {value!r}")
    version = match.group("version").lower()
    if version != "00":
        raise ValueError(f"unsupported traceparent version: {version}")
    return TraceContext(
        trace_id=match.group("trace_id").lower(),
        span_id=match.group("span_id").lower(),
        parent_span_id=None,
        trace_flags=match.group("flags").lower(),
    )


def _normalize_header_map(
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    if not headers:
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def extract_trace_context(
    headers: Mapping[str, str] | None,
    *,
    metadata: Mapping[str, str] | None = None,
) -> TraceContext | None:
    """Extract a TraceContext from HTTP headers and/or MCP/gRPC metadata.

    Preference order:
    1. W3C ``traceparent``
    2. Explicit ``x-puncture-trace-id`` + optional span id
    3. Metadata keys with the same names
    """

    combined: dict[str, str] = {}
    combined.update(_normalize_header_map(headers))
    if metadata:
        combined.update(_normalize_header_map(metadata))

    raw_parent = combined.get(TRACEPARENT_HEADER)
    if raw_parent:
        try:
            context = parse_traceparent(raw_parent)
        except ValueError:
            context = None
        else:
            state = combined.get(TRACESTATE_HEADER)
            if state:
                return TraceContext(
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    parent_span_id=context.parent_span_id,
                    trace_flags=context.trace_flags,
                    trace_state=state,
                    baggage=_parse_baggage(combined.get(BAGGAGE_HEADER)),
                )
            return TraceContext(
                trace_id=context.trace_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                trace_flags=context.trace_flags,
                baggage=_parse_baggage(combined.get(BAGGAGE_HEADER)),
            )

    tid = combined.get(TRACE_ID_HEADER)
    if tid and _HEX32.match(tid) and tid.lower() != "0" * 32:
        sid = combined.get(SPAN_ID_HEADER) or uuid4().hex[:16]
        if not _HEX16.match(sid):
            sid = uuid4().hex[:16]
        return TraceContext(
            trace_id=tid.lower(),
            span_id=sid.lower(),
            baggage=_parse_baggage(combined.get(BAGGAGE_HEADER)),
        )
    return None


def inject_trace_context(
    context: TraceContext,
    headers: MutableMapping[str, str] | None = None,
    *,
    metadata: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject trace context into HTTP headers and optional MCP/gRPC metadata."""

    carrier = headers if headers is not None else {}
    carrier[TRACEPARENT_HEADER] = format_traceparent(context)
    carrier[TRACE_ID_HEADER] = context.trace_id
    carrier[SPAN_ID_HEADER] = context.span_id
    if context.trace_state:
        carrier[TRACESTATE_HEADER] = context.trace_state
    if context.baggage:
        carrier[BAGGAGE_HEADER] = _format_baggage(context.baggage)
    if metadata is not None:
        metadata[TRACEPARENT_HEADER] = carrier[TRACEPARENT_HEADER]
        metadata[TRACE_ID_HEADER] = context.trace_id
        metadata[SPAN_ID_HEADER] = context.span_id
        if context.trace_state:
            metadata[TRACESTATE_HEADER] = context.trace_state
        if context.baggage:
            metadata[BAGGAGE_HEADER] = carrier[BAGGAGE_HEADER]
    return dict(carrier)


def _parse_baggage(value: str | None) -> dict[str, str] | None:
    if not value or not value.strip():
        return None
    result: dict[str, str] = {}
    for part in value.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        # Never put credentials or PHI into baggage propagation.
        from puncture_agent.observability.attributes import is_denied_key

        if not key or is_denied_key(key):
            continue
        result[key] = raw.strip()[:128]
        if len(result) >= 16:
            break
    return result or None


def _format_baggage(baggage: Mapping[str, str]) -> str:
    from puncture_agent.observability.attributes import is_denied_key

    parts: list[str] = []
    for key, value in list(baggage.items())[:16]:
        if not key or is_denied_key(str(key)):
            continue
        parts.append(f"{key}={str(value)[:128]}")
    return ",".join(parts)


def context_from_span_ids(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None = None,
) -> TraceContext:
    """Build a TraceContext from already-recorded span identifiers."""

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
    )
