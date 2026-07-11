"""Secure, injectable HTTP boundary shared by production RAG providers.

The RAG adapters use a deliberately small transport protocol so their request and
response validation can be tested without OpenSearch, vLLM, model downloads, or a
network connection.  The production implementation lazily imports ``httpx`` and
reuses one pooled client for its lifetime.
"""

from __future__ import annotations

import base64
import importlib
import ipaddress
import json
import math
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit


class ProviderHttpError(RuntimeError):
    """Base class for normalized provider transport failures."""


class ProviderSecurityError(ProviderHttpError):
    """A URL, redirect, header, credential, or TLS policy failed closed."""


class ProviderTimeoutError(ProviderHttpError):
    """The provider exceeded the caller's configured request deadline."""


class ProviderUnavailableError(ProviderHttpError):
    """The provider could not be reached over the configured transport."""


class ProviderProtocolError(ProviderHttpError):
    """The provider returned an invalid or unsafe protocol response."""


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def normalize_base_url(base_url: str, *, allow_insecure_http: bool = False) -> str:
    """Validate a provider base URL and return a stable normalized value."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("provider base_url must be a non-empty string")
    value = base_url.strip()
    if _has_control_characters(value) or any(character.isspace() for character in value):
        raise ValueError("provider base_url must not contain whitespace or control characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider base_url contains an invalid host or port") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("provider base_url must use http or https")
    if not parsed.hostname:
        raise ValueError("provider base_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("provider base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("provider base_url must not contain a query or fragment")
    if ".." in parsed.path.split("/"):
        raise ValueError("provider base_url path must not contain parent traversal")
    if scheme == "http" and not (allow_insecure_http and _is_loopback(parsed.hostname)):
        raise ValueError("plain HTTP is allowed only for an explicitly enabled loopback endpoint")

    hostname = parsed.hostname.casefold()
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def validate_request_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("provider request path must be absolute")
    if _has_control_characters(path) or any(character.isspace() for character in path):
        raise ValueError("provider request path must not contain whitespace or control characters")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("provider request path must not contain an origin, query, or fragment")
    if ".." in parsed.path.split("/"):
        raise ValueError("provider request path must not contain parent traversal")
    return parsed.path


@dataclass(frozen=True)
class ProviderEndpoint:
    """Connection, authentication, TLS, and resource limits for one provider."""

    base_url: str
    auth_kind: str = "none"
    username: str | None = None
    secret: str | None = field(default=None, repr=False)
    ca_bundle_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = field(default=None, repr=False)
    timeout_seconds: float = 10.0
    max_response_bytes: int = 8 * 1024 * 1024
    max_connections: int = 100
    max_keepalive_connections: int = 20
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_insecure_http, bool):
            raise ValueError("allow_insecure_http must be a boolean")
        object.__setattr__(
            self,
            "base_url",
            normalize_base_url(
                self.base_url,
                allow_insecure_http=self.allow_insecure_http,
            ),
        )
        if self.auth_kind not in {"none", "basic", "bearer"}:
            raise ValueError("auth_kind must be none, basic, or bearer")
        if self.auth_kind == "none":
            if self.username is not None or self.secret is not None:
                raise ValueError("none authentication must not include credentials")
        elif self.auth_kind == "basic":
            self._validate_credential(self.username, "username")
            self._validate_credential(self.secret, "secret")
            if ":" in str(self.username):
                raise ValueError("basic authentication username must not contain a colon")
        else:
            if self.username is not None:
                raise ValueError("bearer authentication must not include a username")
            self._validate_credential(self.secret, "secret")
        if (self.client_cert_path is None) != (self.client_key_path is None):
            raise ValueError("client certificate and key paths must be configured together")
        for value, label in (
            (self.ca_bundle_path, "ca_bundle_path"),
            (self.client_cert_path, "client_cert_path"),
            (self.client_key_path, "client_key_path"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{label} must be a non-empty string or None")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        for value, label in (
            (self.max_response_bytes, "max_response_bytes"),
            (self.max_connections, "max_connections"),
            (self.max_keepalive_connections, "max_keepalive_connections"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError("max_keepalive_connections must not exceed max_connections")

    @staticmethod
    def _validate_credential(value: str | None, label: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be configured for the selected authentication kind")
        if _has_control_characters(value):
            raise ValueError(f"{label} must not contain control characters")

    def authorization_header(self) -> str | None:
        if self.auth_kind == "none":
            return None
        if self.auth_kind == "bearer":
            return f"Bearer {self.secret}"
        raw = f"{self.username}:{self.secret}".encode("utf-8")
        return f"Basic {base64.b64encode(raw).decode('ascii')}"


@dataclass(frozen=True)
class ProviderHttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class ProviderHttpTransport(Protocol):
    @property
    def endpoint(self) -> ProviderEndpoint: ...

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any | None = None,
        raw_body: bytes | None = None,
        timeout_seconds: float | None = None,
    ) -> ProviderHttpResponse: ...

    def close(self) -> None: ...


class HttpxProviderTransport:
    """Pooled production transport with redirects and environment proxies disabled."""

    def __init__(self, endpoint: ProviderEndpoint) -> None:
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise RuntimeError("httpx is required for production RAG provider transports") from exc
        self._httpx = httpx
        self._endpoint = endpoint
        verify: bool | ssl.SSLContext = True
        if endpoint.ca_bundle_path is not None:
            verify = ssl.create_default_context(cafile=endpoint.ca_bundle_path)
        cert: tuple[str, str] | None = None
        if endpoint.client_cert_path is not None and endpoint.client_key_path is not None:
            cert = (endpoint.client_cert_path, endpoint.client_key_path)
        self._client = httpx.Client(
            verify=verify,
            cert=cert,
            timeout=httpx.Timeout(endpoint.timeout_seconds),
            limits=httpx.Limits(
                max_connections=endpoint.max_connections,
                max_keepalive_connections=endpoint.max_keepalive_connections,
            ),
            trust_env=False,
            follow_redirects=False,
        )

    @property
    def endpoint(self) -> ProviderEndpoint:
        return self._endpoint

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxProviderTransport":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

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
        request_path = validate_request_path(path)
        if json_body is not None and raw_body is not None:
            raise ValueError("only one provider request body representation may be supplied")
        if raw_body is not None and not isinstance(raw_body, bytes):
            raise ValueError("provider raw request body must be bytes")
        timeout = self._endpoint.timeout_seconds if timeout_seconds is None else timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("provider request timeout must be a positive finite number")
        request_headers = self._build_headers(headers)
        content = raw_body
        if json_body is not None:
            try:
                content = json.dumps(
                    json_body,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("provider JSON request body is not serializable") from exc
            request_headers.setdefault("Content-Type", "application/json")
        url = f"{self._endpoint.base_url}{request_path}"
        try:
            with self._client.stream(
                method.upper(),
                url,
                headers=request_headers,
                content=content,
                timeout=float(timeout),
            ) as response:
                status = int(response.status_code)
                if 300 <= status < 400:
                    raise ProviderSecurityError("provider redirect was rejected")
                body = self._read_bounded(response.iter_bytes())
                return ProviderHttpResponse(
                    status=status,
                    headers=dict(response.headers.items()),
                    body=body,
                )
        except ProviderHttpError:
            raise
        except Exception as exc:
            self._raise_transport_error(exc)
            raise AssertionError("unreachable")

    def _build_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        result = {"Accept": "application/json"}
        for key, value in dict(headers or {}).items():
            if not isinstance(key, str) or not key or _has_control_characters(key):
                raise ValueError("provider header name is invalid")
            if not isinstance(value, str) or _has_control_characters(value):
                raise ValueError("provider header value is invalid")
            if key.casefold() == "authorization":
                raise ValueError("provider Authorization header is owned by endpoint configuration")
            result[key] = value
        authorization = self._endpoint.authorization_header()
        if authorization is not None:
            result["Authorization"] = authorization
        return result

    def _read_bounded(self, chunks: Any) -> bytes:
        result: list[bytes] = []
        total = 0
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ProviderProtocolError("provider response chunks must be bytes")
            total += len(chunk)
            if total > self._endpoint.max_response_bytes:
                raise ProviderProtocolError("provider response exceeds the configured size limit")
            result.append(chunk)
        return b"".join(result)

    def _raise_transport_error(self, exc: BaseException) -> None:
        if self._has_tls_cause(exc):
            raise ProviderSecurityError("provider TLS negotiation or certificate validation failed") from exc
        if isinstance(exc, self._httpx.TimeoutException):
            raise ProviderTimeoutError("provider HTTP request timed out") from exc
        if isinstance(exc, self._httpx.TransportError):
            raise ProviderUnavailableError("provider HTTP endpoint is unavailable") from exc
        raise exc

    @staticmethod
    def _has_tls_cause(exc: BaseException) -> bool:
        current: BaseException | None = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            if isinstance(current, (ssl.SSLError, ssl.CertificateError)):
                return True
            visited.add(id(current))
            current = current.__cause__ or current.__context__
        return False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def strict_json_loads(raw: bytes, *, max_bytes: int = 8 * 1024 * 1024) -> Any:
    if not isinstance(raw, bytes):
        raise ProviderProtocolError("provider response body must be bytes")
    if len(raw) > max_bytes:
        raise ProviderProtocolError("provider response exceeds the configured size limit")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderProtocolError("provider returned invalid JSON") from exc


def decode_json_response(
    response: ProviderHttpResponse,
    *,
    max_bytes: int,
    require_object: bool = True,
) -> Any:
    content_type = ""
    for key, value in response.headers.items():
        if key.casefold() == "content-type":
            content_type = value.split(";", 1)[0].strip().casefold()
            break
    if content_type != "application/json" and not content_type.endswith("+json"):
        raise ProviderProtocolError("provider response Content-Type is not JSON")
    if not response.body:
        raise ProviderProtocolError("provider returned an empty JSON response")
    value = strict_json_loads(response.body, max_bytes=max_bytes)
    if require_object and not isinstance(value, dict):
        raise ProviderProtocolError("provider returned a non-object JSON response")
    return value


def ensure_existing_tls_files(endpoint: ProviderEndpoint) -> None:
    """Optional deployment-time validation kept separate from contract construction."""

    for value, label in (
        (endpoint.ca_bundle_path, "CA bundle"),
        (endpoint.client_cert_path, "client certificate"),
        (endpoint.client_key_path, "client key"),
    ):
        if value is not None and not Path(value).is_file():
            raise ProviderSecurityError(f"configured provider {label} is not a regular file")


__all__ = [
    "HttpxProviderTransport",
    "ProviderEndpoint",
    "ProviderHttpError",
    "ProviderHttpResponse",
    "ProviderHttpTransport",
    "ProviderProtocolError",
    "ProviderSecurityError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "decode_json_response",
    "ensure_existing_tls_files",
    "normalize_base_url",
    "strict_json_loads",
    "validate_request_path",
]
