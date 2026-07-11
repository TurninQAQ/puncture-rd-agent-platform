"""Small low-cardinality Prometheus exposition for the HTTP boundary."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
import time
from typing import Any, Awaitable, Callable, MutableMapping


AsgiMessage = MutableMapping[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiScope = MutableMapping[str, Any]
AsgiApp = Callable[[AsgiScope, AsgiReceive, AsgiSend], Awaitable[None]]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class HttpMetrics:
    """Thread-safe counters keyed only by bounded transport dimensions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._request_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duration_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._duration_sums: dict[tuple[str, str], float] = defaultdict(float)

    def observe(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        normalized_method = method if method in {"GET", "POST"} else "OTHER"
        normalized_route = route if route.startswith("/") else "unmatched"
        status_class = (
            f"{status_code // 100}xx"
            if 100 <= status_code <= 599
            else "unknown"
        )
        safe_duration = max(0.0, float(duration_seconds))
        with self._lock:
            self._request_counts[
                (normalized_method, normalized_route, status_class)
            ] += 1
            self._duration_counts[(normalized_method, normalized_route)] += 1
            self._duration_sums[(normalized_method, normalized_route)] += safe_duration

    def render(self) -> str:
        with self._lock:
            request_counts = sorted(self._request_counts.items())
            duration_counts = sorted(self._duration_counts.items())
            duration_sums = dict(self._duration_sums)

        lines = [
            "# HELP puncture_api_http_requests_total HTTP requests by fixed route and status class.",
            "# TYPE puncture_api_http_requests_total counter",
        ]
        for (method, route, status_class), count in request_counts:
            lines.append(
                "puncture_api_http_requests_total"
                f'{{method="{_escape_label(method)}",'
                f'route="{_escape_label(route)}",'
                f'status_class="{_escape_label(status_class)}"}} {count}'
            )
        lines.extend(
            (
                "# HELP puncture_api_http_request_duration_seconds HTTP request duration by fixed route.",
                "# TYPE puncture_api_http_request_duration_seconds summary",
            )
        )
        for (method, route), count in duration_counts:
            labels = (
                f'method="{_escape_label(method)}",'
                f'route="{_escape_label(route)}"'
            )
            lines.append(
                "puncture_api_http_request_duration_seconds_count"
                f"{{{labels}}} {count}"
            )
            lines.append(
                "puncture_api_http_request_duration_seconds_sum"
                f"{{{labels}}} {duration_sums[(method, route)]:.9f}"
            )
        return "\n".join(lines) + "\n"


class HttpMetricsMiddleware:
    """Observe HTTP responses without using raw paths as metric labels."""

    def __init__(self, app: AsgiApp, *, metrics: HttpMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(
        self,
        scope: AsgiScope,
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status_code = 500

        async def observe_send(message: AsgiMessage) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                candidate = message.get("status", 500)
                if isinstance(candidate, int):
                    status_code = candidate
                path = str(scope.get("path", ""))
                if path.startswith("/api/") or path in {"/health", "/metrics"}:
                    headers = list(message.get("headers", ()))
                    lowered = {
                        key.lower()
                        for key, _ in headers
                        if isinstance(key, bytes)
                    }
                    if b"cache-control" not in lowered:
                        headers.append((b"cache-control", b"no-store"))
                    if b"x-content-type-options" not in lowered:
                        headers.append((b"x-content-type-options", b"nosniff"))
                    message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, observe_send)
        finally:
            path = str(scope.get("path", ""))
            if path != "/metrics":
                route_object = scope.get("route")
                route = getattr(route_object, "path", None)
                if not isinstance(route, str):
                    route = "unmatched"
                self.metrics.observe(
                    method=str(scope.get("method", "OTHER")).upper(),
                    route=route,
                    status_code=status_code,
                    duration_seconds=time.perf_counter() - started,
                )


__all__ = ["HttpMetrics", "HttpMetricsMiddleware"]
