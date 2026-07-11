#!/usr/bin/env python3
"""Dependency-free, fail-closed HTTP helpers for OpenSearch deployment scripts."""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import pathlib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


MAX_JSON_RESPONSE_BYTES = 8 * 1024 * 1024


class ServiceError(RuntimeError):
    """Safe transport/protocol error that never includes credentials or response bodies."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        fp.close()
        raise ServiceError("OpenSearch redirect rejected", status=code)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number rejected: {value}")


def strict_json_loads(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ServiceError("OpenSearch returned invalid JSON") from exc


def validate_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OpenSearch endpoint must not be empty")
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise ValueError("OpenSearch endpoint must not contain whitespace or control characters")
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("OpenSearch endpoint must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("OpenSearch endpoint must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenSearch endpoint must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("OpenSearch endpoint must not contain a path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OpenSearch endpoint contains an invalid port") from exc
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError("plain HTTP is allowed only for a loopback OpenSearch endpoint")
    netloc = parsed.hostname
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", ""))


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def api_url(endpoint: str, path: str, query: dict[str, str] | None = None) -> str:
    base = validate_endpoint(endpoint)
    if not path.startswith("/") or "#" in path or "?" in path:
        raise ValueError("OpenSearch API path must be absolute and must not contain query syntax")
    encoded_query = urllib.parse.urlencode(query or {}, doseq=False)
    return f"{base}{path}" + (f"?{encoded_query}" if encoded_query else "")


def validate_identifier(value: str, *, label: str, versioned_index: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if value[0] in {"_", "-", "+", "."} or len(value) > 255:
        raise ValueError(f"{label} is not a safe OpenSearch identifier")
    if value.lower() != value or any(character not in allowed for character in value):
        raise ValueError(f"{label} is not a safe OpenSearch identifier")
    if any(token in value for token in ("*", "?", ",", "/", "\\", "..")):
        raise ValueError(f"{label} must not contain wildcards or path syntax")
    if versioned_index:
        stem, separator, generation = value.rpartition("-v")
        if not separator or not stem or len(generation) < 6 or not generation.isdigit():
            raise ValueError(f"{label} must end with -v followed by at least six digits")
    return value


def load_secret_file(filename: str | None = None) -> str:
    path_value = filename or os.getenv("OPENSEARCH_PASSWORD_FILE", "")
    if not path_value:
        raise ValueError("OPENSEARCH_PASSWORD_FILE must point to a secret file")
    path = pathlib.Path(path_value)
    if not path.is_file():
        raise ValueError("OpenSearch password path must be a regular file")
    with path.open("r", encoding="utf-8") as handle:
        value = handle.readline().rstrip("\r\n")
        if handle.read(1):
            raise ValueError("OpenSearch password file must contain exactly one line")
    if not value:
        raise ValueError("OpenSearch password file is empty")
    return value


def parse_bool(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError(f"{label} must be true or false")


def _ssl_context(endpoint: str, *, ca_file: str | None, insecure: bool) -> ssl.SSLContext | None:
    if urllib.parse.urlsplit(endpoint).scheme != "https":
        return None
    if insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if ca_file:
        path = pathlib.Path(ca_file)
        if not path.is_file():
            raise ValueError("OpenSearch CA path must be a regular file")
        return ssl.create_default_context(cafile=str(path))
    return ssl.create_default_context()


def _opener(endpoint: str, *, ca_file: str | None, insecure: bool) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [urllib.request.ProxyHandler({}), _RejectRedirects()]
    context = _ssl_context(endpoint, ca_file=ca_file, insecure=insecure)
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers)


def request(
    method: str,
    endpoint: str,
    path: str,
    *,
    username: str,
    password: str,
    payload: dict[str, Any] | list[Any] | None = None,
    query: dict[str, str] | None = None,
    timeout: float = 15.0,
    ca_file: str | None = None,
    insecure: bool = False,
    allowed_statuses: Iterable[int] = (200,),
) -> HttpResult:
    endpoint = validate_endpoint(endpoint)
    if not username or any(character in username for character in "\r\n"):
        raise ValueError("OpenSearch username is invalid")
    if any(character in password for character in "\r\n"):
        raise ValueError("OpenSearch password must be one line")
    if timeout <= 0:
        raise ValueError("request timeout must be positive")
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Basic {token}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    http_request = urllib.request.Request(
        api_url(endpoint, path, query),
        data=body,
        headers=headers,
        method=method.upper(),
    )
    accepted = frozenset(int(status) for status in allowed_statuses)
    opener = _opener(endpoint, ca_file=ca_file, insecure=insecure)
    try:
        with opener.open(http_request, timeout=timeout) as response:
            raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
            if len(raw) > MAX_JSON_RESPONSE_BYTES:
                raise ServiceError("OpenSearch response exceeds the size limit")
            status = int(response.status)
            if status not in accepted:
                raise ServiceError(f"unexpected HTTP {status} from OpenSearch", status=status)
            return HttpResult(status, dict(response.headers.items()), raw)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        exc.close()
        if status in accepted:
            return HttpResult(status, dict(exc.headers.items()), b"")
        raise ServiceError(f"HTTP {status} from OpenSearch", status=status) from exc
    except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
        raise ServiceError("OpenSearch connection failed") from exc


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    username: str,
    password: str,
    payload: dict[str, Any] | list[Any] | None = None,
    query: dict[str, str] | None = None,
    timeout: float = 15.0,
    ca_file: str | None = None,
    insecure: bool = False,
    allowed_statuses: Iterable[int] = (200,),
) -> tuple[int, dict[str, Any]]:
    result = request(
        method,
        endpoint,
        path,
        username=username,
        password=password,
        payload=payload,
        query=query,
        timeout=timeout,
        ca_file=ca_file,
        insecure=insecure,
        allowed_statuses=allowed_statuses,
    )
    if not result.body:
        return result.status, {}
    value = strict_json_loads(result.body)
    if not isinstance(value, dict):
        raise ServiceError("OpenSearch returned a non-object JSON response")
    return result.status, value


@dataclass(frozen=True)
class ConnectionConfig:
    endpoint: str
    username: str
    password: str
    timeout: float
    ca_file: str | None
    insecure: bool


def connection_from_environment() -> ConnectionConfig:
    endpoint = validate_endpoint(os.getenv("OPENSEARCH_ENDPOINT", "https://127.0.0.1:9200"))
    username = os.getenv("OPENSEARCH_USERNAME", "admin")
    password = load_secret_file()
    timeout = float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "15"))
    ca_file = os.getenv("OPENSEARCH_CA_FILE", "") or None
    insecure = parse_bool(os.getenv("OPENSEARCH_INSECURE", "false"), label="OPENSEARCH_INSECURE")
    return ConnectionConfig(endpoint, username, password, timeout, ca_file, insecure)


def request_json_with(config: ConnectionConfig, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
    return request_json(
        method,
        config.endpoint,
        path,
        username=config.username,
        password=config.password,
        timeout=config.timeout,
        ca_file=config.ca_file,
        insecure=config.insecure,
        **kwargs,
    )
