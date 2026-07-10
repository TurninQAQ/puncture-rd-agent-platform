#!/usr/bin/env python3
"""Dependency-free HTTP helpers shared by deployment verification scripts."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any


class ServiceError(RuntimeError):
    """A safe HTTP/protocol error that never includes an authorization value."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_SSE_EVENT_BYTES = 1024 * 1024
MAX_SSE_RESPONSE_BYTES = 32 * 1024 * 1024


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Authentication must never follow a redirect to another origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        fp.close()
        raise ServiceError("model service redirect rejected", status=code)


# Contact the configured private endpoint directly. Ambient HTTP(S)_PROXY values
# must not receive prompts, tool schemas, or bearer credentials.
_DIRECT_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _RejectRedirects(),
)


def validate_service_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("service URL must not be empty")
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise ValueError("service URL must not contain whitespace or control characters")
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("service URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("service URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("service URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("service URL must not contain a query or fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("service URL contains an invalid port") from exc
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def normalize_base_url(value: str) -> str:
    return validate_service_url(value).rstrip("/")


def server_root(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def load_api_key() -> str:
    direct = os.getenv("VLLM_API_KEY", "")
    filename = os.getenv("VLLM_API_KEY_FILE", "")
    if direct and filename:
        raise ValueError("set only one of VLLM_API_KEY and VLLM_API_KEY_FILE")
    if filename:
        with open(filename, "r", encoding="utf-8") as handle:
            return handle.readline().strip()
    return direct


def request_headers(api_key: str = "", *, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def request_bytes(
    method: str,
    url: str,
    *,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> bytes:
    url = validate_service_url(url)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(api_key, json_body=payload is not None),
        method=method,
    )
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
            if len(raw) > MAX_JSON_RESPONSE_BYTES:
                raise ServiceError("model service response exceeds the size limit")
            return raw
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ServiceError(f"HTTP {status} from model service", status=status) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ServiceError("model service connection failed") from exc


def request_json(
    method: str,
    url: str,
    *,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    raw = request_bytes(method, url, api_key=api_key, payload=payload, timeout=timeout)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError("model service returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ServiceError("model service returned a non-object JSON response")
    return value


def iter_sse_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
    timeout: float = 120.0,
) -> Iterator[dict[str, Any]]:
    url = validate_service_url(url)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(api_key, json_body=True),
        method="POST",
    )
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type != "text/event-stream":
                raise ServiceError("model service stream has an invalid content type")
            total_bytes = 0
            while True:
                raw_line = response.readline(MAX_SSE_EVENT_BYTES + 1)
                if not raw_line:
                    raise ServiceError("model service SSE ended before [DONE]")
                if len(raw_line) > MAX_SSE_EVENT_BYTES:
                    raise ServiceError("model service SSE event exceeds the size limit")
                total_bytes += len(raw_line)
                if total_bytes > MAX_SSE_RESPONSE_BYTES:
                    raise ServiceError("model service SSE response exceeds the size limit")
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ServiceError("model service returned malformed SSE JSON") from exc
                if not isinstance(event, dict):
                    raise ServiceError("model service returned a non-object SSE event")
                yield event
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ServiceError(f"HTTP {status} from model service", status=status) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ServiceError("model service stream failed") from exc
