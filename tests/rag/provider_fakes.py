"""Scripted provider transport used by production-adapter unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from puncture_agent.rag.provider_http import (
    ProviderEndpoint,
    ProviderHttpResponse,
)


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    json_body: Any | None
    raw_body: bytes | None
    timeout_seconds: float | None


ScriptItem = ProviderHttpResponse | BaseException | Callable[[RecordedRequest], ProviderHttpResponse]


class ScriptedTransport:
    def __init__(
        self,
        scripts: list[ScriptItem] | tuple[ScriptItem, ...],
        *,
        endpoint: ProviderEndpoint | None = None,
    ) -> None:
        self._endpoint = endpoint or ProviderEndpoint("https://provider.internal")
        self._scripts = list(scripts)
        self.requests: list[RecordedRequest] = []
        self.closed = False

    @property
    def endpoint(self) -> ProviderEndpoint:
        return self._endpoint

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any | None = None,
        raw_body: bytes | None = None,
        timeout_seconds: float | None = None,
    ) -> ProviderHttpResponse:
        request = RecordedRequest(
            method=method,
            path=path,
            headers=dict(headers or {}),
            json_body=json_body,
            raw_body=raw_body,
            timeout_seconds=timeout_seconds,
        )
        self.requests.append(request)
        if not self._scripts:
            raise AssertionError("scripted provider transport received an unexpected request")
        item = self._scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        return item

    def close(self) -> None:
        self.closed = True

    def assert_consumed(self) -> None:
        if self._scripts:
            raise AssertionError(f"{len(self._scripts)} scripted provider responses were not consumed")


def json_response(
    payload: Any,
    *,
    status: int = 200,
    content_type: str = "application/json",
) -> ProviderHttpResponse:
    return ProviderHttpResponse(
        status=status,
        headers={"Content-Type": content_type},
        body=json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8"),
    )


__all__ = ["RecordedRequest", "ScriptedTransport", "json_response"]
