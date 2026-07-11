"""Bounded Server-Sent Events replay helpers for committed Run events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import re
from threading import Lock, RLock
import time
from typing import Any, AsyncIterator, Callable, Literal, Mapping
from urllib.parse import unquote_plus

from starlette.responses import StreamingResponse

from puncture_agent.runtime import RunEvent, RunEventPage, RunServiceError, RunSnapshot
from puncture_agent.runtime.models import RunStatus

from .http_contracts import RunEventResponse


_CURSOR = re.compile(r"0|[1-9][0-9]*")
_QVALUE = re.compile(r"(?:0(?:\.[0-9]{0,3})?|1(?:\.0{0,3})?)")
_MEDIA_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_MAX_CURSOR = 2**63 - 1
_TERMINAL_STATUSES = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


class SseNegotiationError(ValueError):
    """The Accept header cannot select JSON or SSE."""


class SsePageContractError(RuntimeError):
    """A repository page cannot be represented as the public SSE contract."""


async def _run_before_deadline(
    deadline: float,
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(
        asyncio.to_thread(function, *args),
        timeout=remaining,
    )


@dataclass(frozen=True, slots=True)
class SseConfig:
    page_size: int = 128
    poll_interval_seconds: float = 1.0
    heartbeat_seconds: float = 15.0
    max_connection_seconds: float = 600.0
    max_connections: int = 200
    max_connections_per_tenant: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or self.page_size < 1
            or self.page_size > 512
        ):
            raise ValueError("SSE page_size must be between 1 and 512")
        for name, value, minimum, maximum in (
            (
                "poll_interval_seconds",
                self.poll_interval_seconds,
                0.01,
                10.0,
            ),
            ("heartbeat_seconds", self.heartbeat_seconds, 0.01, 60.0),
            (
                "max_connection_seconds",
                self.max_connection_seconds,
                0.05,
                3600.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < minimum
                or float(value) > maximum
            ):
                raise ValueError(
                    f"SSE {name} must be between {minimum} and {maximum}"
                )
        if self.heartbeat_seconds < self.poll_interval_seconds:
            raise ValueError("SSE heartbeat must not be shorter than poll interval")
        for name, value in (
            ("max_connections", self.max_connections),
            ("max_connections_per_tenant", self.max_connections_per_tenant),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"SSE {name} must be a positive integer")
        if self.max_connections_per_tenant > self.max_connections:
            raise ValueError(
                "SSE per-tenant connection limit cannot exceed the global limit"
            )


@dataclass(frozen=True, slots=True)
class CursorResolution:
    cursor: int
    source: Literal["none", "query", "header"]


def _raw_header_values(scope: Mapping[str, Any], name: bytes) -> list[bytes]:
    return [
        value
        for key, value in scope.get("headers", ())
        if isinstance(key, bytes)
        and isinstance(value, bytes)
        and key.lower() == name
    ]


def _parse_qvalue(raw: str) -> float:
    if not _QVALUE.fullmatch(raw):
        raise SseNegotiationError("Accept q-value is invalid")
    return float(raw)


def negotiate_event_representation(
    scope: Mapping[str, Any],
) -> Literal["json", "sse"]:
    values = _raw_header_values(scope, b"accept")
    if not values:
        return "json"
    try:
        combined = ",".join(value.decode("ascii") for value in values)
    except UnicodeDecodeError as exc:
        raise SseNegotiationError("Accept header is invalid") from exc
    ranges: list[tuple[str, str, float]] = []
    for raw_item in combined.split(","):
        item = raw_item.strip()
        if not item:
            raise SseNegotiationError("Accept header is invalid")
        parts = [part.strip() for part in item.split(";")]
        media_type = parts[0].lower()
        media_main, separator, media_subtype = media_type.partition("/")
        if (
            not separator
            or "/" in media_subtype
            or not media_main
            or not media_subtype
            or (media_main == "*" and media_subtype != "*")
            or (
                media_main != "*"
                and not _MEDIA_TOKEN.fullmatch(media_main)
            )
            or (
                media_subtype != "*"
                and not _MEDIA_TOKEN.fullmatch(media_subtype)
            )
        ):
            raise SseNegotiationError("Accept header is invalid")
        quality = 1.0
        quality_seen = False
        for parameter in parts[1:]:
            if not parameter:
                raise SseNegotiationError("Accept header is invalid")
            name, separator, value = parameter.partition("=")
            if (
                name.strip().lower() != "q"
                or not separator
                or quality_seen
            ):
                raise SseNegotiationError("Accept header is invalid")
            quality_seen = True
            quality = _parse_qvalue(value.strip())
        ranges.append((media_main, media_subtype, quality))

    def effective_quality(target_type: str, target_subtype: str) -> tuple[float, int]:
        matches: list[tuple[int, float]] = []
        for media_main, media_subtype, quality in ranges:
            if media_main == target_type and media_subtype == target_subtype:
                specificity = 2
            elif media_main == target_type and media_subtype == "*":
                specificity = 1
            elif media_main == "*" and media_subtype == "*":
                specificity = 0
            else:
                continue
            matches.append((specificity, quality))
        if not matches:
            return 0.0, -1
        specificity = max(item[0] for item in matches)
        return (
            max(quality for item_specificity, quality in matches if item_specificity == specificity),
            specificity,
        )

    json_quality, json_specificity = effective_quality("application", "json")
    sse_quality, sse_specificity = effective_quality("text", "event-stream")
    if sse_quality > json_quality and sse_quality > 0:
        return "sse"
    if (
        sse_quality > 0
        and sse_quality == json_quality
        and sse_specificity > json_specificity
    ):
        return "sse"
    if json_quality > 0:
        return "json"
    if sse_quality > 0:
        return "sse"
    raise SseNegotiationError("no supported event representation is acceptable")


def _canonical_cursor(value: str) -> int:
    if len(value) > 19 or not _CURSOR.fullmatch(value):
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid")
    cursor = int(value)
    if cursor > _MAX_CURSOR:
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid")
    return cursor


def resolve_event_cursor(
    scope: Mapping[str, Any],
    *,
    after_sequence: int | None,
    representation: Literal["json", "sse"],
) -> CursorResolution:
    try:
        raw_query = scope.get("query_string", b"").decode("ascii")
    except UnicodeDecodeError as exc:
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid") from exc
    raw_query_values: list[str] = []
    for component in raw_query.split("&") if raw_query else ():
        raw_name, separator, raw_value = component.partition("=")
        try:
            decoded_name = unquote_plus(raw_name, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RunServiceError(
                "INVALID_ARGUMENT",
                "event cursor is invalid",
            ) from exc
        if decoded_name != "after_sequence":
            continue
        if not separator or raw_name != "after_sequence":
            raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid")
        raw_query_values.append(raw_value)
    if len(raw_query_values) > 1:
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is ambiguous")
    query_cursor: int | None = None
    if raw_query_values:
        query_cursor = _canonical_cursor(raw_query_values[0])
        if after_sequence is None or query_cursor != after_sequence:
            raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid")
    elif after_sequence is not None:
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is ambiguous")

    last_event_ids = _raw_header_values(scope, b"last-event-id")
    if representation == "json" and last_event_ids:
        raise RunServiceError(
            "INVALID_ARGUMENT",
            "Last-Event-ID is only valid for event streams",
        )
    if len(last_event_ids) > 1:
        raise RunServiceError("INVALID_ARGUMENT", "event cursor is ambiguous")
    header_cursor: int | None = None
    if last_event_ids:
        try:
            header_cursor = _canonical_cursor(last_event_ids[0].decode("ascii"))
        except UnicodeDecodeError as exc:
            raise RunServiceError("INVALID_ARGUMENT", "event cursor is invalid") from exc
    if (
        query_cursor is not None
        and header_cursor is not None
        and query_cursor != header_cursor
    ):
        raise RunServiceError("INVALID_ARGUMENT", "event cursors conflict")
    if header_cursor is not None:
        return CursorResolution(header_cursor, "header")
    if query_cursor is not None:
        return CursorResolution(query_cursor, "query")
    return CursorResolution(0, "none")


def encode_event_page(
    page: RunEventPage,
    *,
    run_id: str,
    after_sequence: int,
) -> tuple[tuple[RunEvent, bytes], ...]:
    if not page.events and after_sequence < page.high_water_sequence:
        raise SsePageContractError("event page omitted committed events")
    expected = after_sequence + 1
    encoded: list[tuple[RunEvent, bytes]] = []
    for event in page.events:
        if event.run_id != run_id or event.sequence != expected:
            raise SsePageContractError("event page is not contiguous")
        public = RunEventResponse.from_runtime(event)
        data = public.model_dump_json().encode("utf-8")
        frame = (
            f"id: {event.sequence}\n"
            f"event: {event.event_type.value}\n"
            "data: "
        ).encode("utf-8") + data + b"\n\n"
        encoded.append((event, frame))
        expected += 1
    if page.events and page.events[-1].sequence > page.high_water_sequence:
        raise SsePageContractError("event page exceeds high-water sequence")
    return tuple(encoded)


class SseConnectionLease:
    def __init__(self, limiter: "SseConnectionLimiter", tenant_id: str) -> None:
        self._limiter = limiter
        self._tenant_id = tenant_id
        self._released = False
        self._lock = Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._limiter._release(self._tenant_id)


class SseConnectionLimiter:
    def __init__(self, *, max_connections: int, max_per_tenant: int) -> None:
        if max_connections < 1 or max_per_tenant < 1:
            raise ValueError("SSE connection limits must be positive")
        if max_per_tenant > max_connections:
            raise ValueError("tenant SSE limit cannot exceed global limit")
        self._max_connections = max_connections
        self._max_per_tenant = max_per_tenant
        self._active = 0
        self._tenant_active: dict[str, int] = {}
        self._lock = Lock()

    def try_acquire(self, tenant_id: str) -> SseConnectionLease:
        with self._lock:
            tenant_active = self._tenant_active.get(tenant_id, 0)
            if (
                self._active >= self._max_connections
                or tenant_active >= self._max_per_tenant
            ):
                raise RunServiceError(
                    "SSE_CAPACITY_UNAVAILABLE",
                    "event stream capacity is unavailable",
                    retryable=True,
                )
            self._active += 1
            self._tenant_active[tenant_id] = tenant_active + 1
        return SseConnectionLease(self, tenant_id)

    def _release(self, tenant_id: str) -> None:
        with self._lock:
            tenant_active = self._tenant_active.get(tenant_id, 0)
            if self._active < 1 or tenant_active < 1:
                return
            self._active -= 1
            if tenant_active == 1:
                self._tenant_active.pop(tenant_id, None)
            else:
                self._tenant_active[tenant_id] = tenant_active - 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


class SseStreamingResponse(StreamingResponse):
    """Release stream resources even when ASGI sending fails before iteration."""

    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        lease: SseConnectionLease,
        headers: Mapping[str, str],
    ) -> None:
        self._lease = lease
        super().__init__(
            content,
            media_type="text/event-stream",
            headers=dict(headers),
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if callable(close):
                try:
                    await close()
                except (Exception, asyncio.CancelledError):
                    pass
            self._lease.release()


class SseMetrics:
    _OUTCOMES = ("terminal", "disconnect", "timeout", "revoked", "error")
    _SOURCES = ("none", "query", "header")

    def __init__(self) -> None:
        self._lock = RLock()
        self._active = 0
        self._opened = {source: 0 for source in self._SOURCES}
        self._closed = {outcome: 0 for outcome in self._OUTCOMES}
        self._events: dict[str, int] = {}
        self._heartbeats = 0

    def opened(self, source: str) -> None:
        with self._lock:
            self._active += 1
            self._opened[source if source in self._opened else "none"] += 1

    def event(self, event_type: str) -> None:
        with self._lock:
            self._events[event_type] = self._events.get(event_type, 0) + 1

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeats += 1

    def closed(self, outcome: str) -> None:
        normalized = outcome if outcome in self._closed else "error"
        with self._lock:
            self._active = max(0, self._active - 1)
            self._closed[normalized] += 1

    def render(self) -> str:
        with self._lock:
            active = self._active
            opened = dict(self._opened)
            closed = dict(self._closed)
            events = dict(self._events)
            heartbeats = self._heartbeats
        lines = [
            "# HELP puncture_api_sse_connections_current Active SSE connections.",
            "# TYPE puncture_api_sse_connections_current gauge",
            f"puncture_api_sse_connections_current {active}",
            "# HELP puncture_api_sse_connections_total SSE connections by cursor source.",
            "# TYPE puncture_api_sse_connections_total counter",
        ]
        for source in self._SOURCES:
            lines.append(
                "puncture_api_sse_connections_total"
                f'{{source="{source}"}} {opened[source]}'
            )
        lines.extend(
            (
                "# HELP puncture_api_sse_connection_outcomes_total Closed SSE connections by outcome.",
                "# TYPE puncture_api_sse_connection_outcomes_total counter",
            )
        )
        for outcome in self._OUTCOMES:
            lines.append(
                "puncture_api_sse_connection_outcomes_total"
                f'{{outcome="{outcome}"}} {closed[outcome]}'
            )
        lines.extend(
            (
                "# HELP puncture_api_sse_events_total SSE events by fixed EventType.",
                "# TYPE puncture_api_sse_events_total counter",
            )
        )
        for event_type, count in sorted(events.items()):
            lines.append(
                "puncture_api_sse_events_total"
                f'{{event_type="{event_type}"}} {count}'
            )
        lines.extend(
            (
                "# HELP puncture_api_sse_heartbeats_total SSE heartbeat comments.",
                "# TYPE puncture_api_sse_heartbeats_total counter",
                f"puncture_api_sse_heartbeats_total {heartbeats}",
            )
        )
        return "\n".join(lines) + "\n"


async def stream_event_pages(
    *,
    run_id: str,
    initial_page: RunEventPage,
    initial_frames: tuple[tuple[RunEvent, bytes], ...],
    after_sequence: int,
    authorize: Callable[[], RunSnapshot],
    read_page: Callable[[int, int], RunEventPage],
    config: SseConfig,
    lease: SseConnectionLease,
    metrics: SseMetrics,
    cursor_source: str,
) -> AsyncIterator[bytes]:
    cursor = after_sequence
    page = initial_page
    frames = initial_frames
    started = time.monotonic()
    deadline = started + config.max_connection_seconds
    last_emitted = started
    outcome = "disconnect"
    metrics.opened(cursor_source)
    try:
        while True:
            for event, frame in frames:
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    return
                yield frame
                cursor = event.sequence
                last_emitted = time.monotonic()
                metrics.event(event.event_type.value)
            if page.status in _TERMINAL_STATUSES and cursor >= page.high_water_sequence:
                outcome = "terminal"
                return
            if time.monotonic() >= deadline:
                outcome = "timeout"
                return
            waited = not page.has_more
            if waited:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    outcome = "timeout"
                    return
                await asyncio.sleep(min(config.poll_interval_seconds, remaining))
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    return
            try:
                await _run_before_deadline(deadline, authorize)
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    return
                page = await _run_before_deadline(
                    deadline,
                    read_page,
                    cursor,
                    config.page_size,
                )
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    return
                frames = encode_event_page(
                    page,
                    run_id=run_id,
                    after_sequence=cursor,
                )
                if (
                    waited
                    and not frames
                    and page.status not in _TERMINAL_STATUSES
                ):
                    now = time.monotonic()
                    if (
                        now < deadline
                        and now - last_emitted >= config.heartbeat_seconds
                    ):
                        yield b": heartbeat\n\n"
                        last_emitted = now
                        metrics.heartbeat()
            except asyncio.TimeoutError:
                outcome = "timeout"
                return
            except RunServiceError as exc:
                outcome = (
                    "revoked"
                    if exc.code in {"FORBIDDEN", "NOT_FOUND"}
                    else "error"
                )
                return
            except Exception:
                outcome = "error"
                return
    except asyncio.CancelledError:
        outcome = "disconnect"
        raise
    finally:
        lease.release()
        metrics.closed(outcome)


__all__ = [
    "CursorResolution",
    "SseConfig",
    "SseConnectionLease",
    "SseConnectionLimiter",
    "SseMetrics",
    "SseNegotiationError",
    "SsePageContractError",
    "SseStreamingResponse",
    "encode_event_page",
    "negotiate_event_representation",
    "resolve_event_cursor",
    "stream_event_pages",
]
