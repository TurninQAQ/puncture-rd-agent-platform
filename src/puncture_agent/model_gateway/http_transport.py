"""Small injectable HTTP transport for the OpenAI-compatible vLLM API.

The gateway intentionally depends on this narrow protocol instead of exposing
``urllib`` response objects.  Unit tests can therefore script status codes,
network failures, and fragmented SSE chunks without a real server or GPU.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import importlib
import json
import socket
import ssl
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, ProxyHandler, Request, build_opener


class TransportSecurityError(RuntimeError):
    """A non-retryable TLS, URL, or redirect policy failure."""

    def __init__(self, message: str, *, code: str = "MODEL_SECURITY_ERROR") -> None:
        super().__init__(message)
        self.code = code


def validate_base_url(base_url: str) -> tuple[str, str, int]:
    """Validate an HTTP(S) provider URL and return its normalized origin."""

    if any(ord(character) < 32 or ord(character) == 127 for character in base_url):
        raise ValueError("base_url must not contain control characters")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url contains an invalid host or port") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("base_url must use http or https")
    if not parsed.hostname:
        raise ValueError("base_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    if not parsed.path.startswith("/"):
        raise ValueError("base_url path is invalid")
    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    return scheme, parsed.hostname.lower(), effective_port


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so bearer credentials never leave the configured request."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        fp.close()
        raise TransportSecurityError("provider redirect was rejected")


@dataclass
class HttpResponse:
    """Provider response independent of a concrete HTTP implementation."""

    status: int
    headers: Mapping[str, str]
    body: bytes | Iterable[bytes]

    def iter_bytes(self) -> Iterator[bytes]:
        if isinstance(self.body, bytes):
            if self.body:
                yield self.body
            return
        for chunk in self.body:
            if not isinstance(chunk, bytes):
                raise TypeError("HTTP response chunks must be bytes")
            if chunk:
                yield chunk

    def read(self, limit: int = 8 * 1024 * 1024) -> bytes:
        """Read a bounded response body.

        Provider responses are expected to be small JSON documents.  A bound
        prevents a broken or hostile endpoint from consuming unlimited memory.
        """

        chunks: list[bytes] = []
        total = 0
        for chunk in self.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise ValueError("HTTP response body exceeds configured limit")
            chunks.append(chunk)
        return b"".join(chunks)


class HttpTransport(Protocol):
    """The replaceable transport contract consumed by ``VllmModelGateway``."""

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None,
        timeout: float,
        stream: bool = False,
    ) -> HttpResponse: ...


class HttpxTransport:
    """Pooled production transport using a lazily imported ``httpx.Client``.

    Lazy import keeps contract tests importable before optional production
    dependencies are installed.  One client is reused for the adapter lifetime;
    callers should invoke ``close`` (or use the context-manager methods) during
    service shutdown.
    """

    _MAX_BODY_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        ca_bundle_path: str | None = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        validate_base_url(base_url)
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise RuntimeError("httpx is not installed") from exc
        self._httpx = httpx
        self._base_url = base_url.rstrip("/")
        verify: bool | ssl.SSLContext = True
        if ca_bundle_path is not None:
            verify = ssl.create_default_context(cafile=ca_bundle_path)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            verify=verify,
            trust_env=False,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxTransport":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None,
        timeout: float,
        stream: bool = False,
    ) -> HttpResponse:
        request_kwargs = {
            "headers": dict(headers),
            "json": json_body,
            "timeout": timeout,
        }
        context = None
        try:
            url = f"{self._base_url}/{path.lstrip('/')}"
            context = self._client.stream(method.upper(), url, **request_kwargs)
            response = context.__enter__()
            status = int(response.status_code)
            response_headers = dict(response.headers.items())
            if stream and 200 <= status < 300:
                return HttpResponse(
                    status=status,
                    headers=response_headers,
                    body=self._stream_response(response, context),
                )
            body = self._read_bounded(response.iter_bytes())
            context.__exit__(None, None, None)
            context = None
            return HttpResponse(status=status, headers=response_headers, body=body)
        except Exception as exc:
            if context is not None:
                context.__exit__(type(exc), exc, exc.__traceback__)
            self._raise_normalized_transport_error(exc)
            raise AssertionError("unreachable")

    @classmethod
    def _read_bounded(cls, chunks: Iterable[bytes]) -> bytes:
        result: list[bytes] = []
        total = 0
        for chunk in chunks:
            total += len(chunk)
            if total > cls._MAX_BODY_BYTES:
                raise TransportSecurityError("provider HTTP body exceeds the size limit")
            result.append(chunk)
        return b"".join(result)

    def _stream_response(self, response: Any, context: Any) -> Iterator[bytes]:
        try:
            yield from response.iter_bytes()
        except Exception as exc:
            self._raise_normalized_transport_error(exc)
        finally:
            context.__exit__(None, None, None)

    def _raise_normalized_transport_error(self, exc: BaseException) -> None:
        if isinstance(exc, TransportSecurityError):
            raise exc
        if self._has_tls_cause(exc):
            raise TransportSecurityError(
                "model HTTPS certificate or TLS negotiation failed",
                code="MODEL_TLS_ERROR",
            ) from exc
        if isinstance(exc, self._httpx.TimeoutException):
            raise TimeoutError("model HTTP request timed out") from exc
        if isinstance(exc, self._httpx.TransportError):
            raise ConnectionError("model HTTP endpoint is unavailable") from exc
        raise exc

    @staticmethod
    def _has_tls_cause(exc: BaseException) -> bool:
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if isinstance(current, (ssl.SSLError, ssl.CertificateError)):
                return True
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return False


def create_default_transport(
    base_url: str,
    timeout_seconds: float,
    ca_bundle_path: str | None = None,
) -> HttpTransport:
    """Create the required pooled production client.

    Missing ``httpx`` is a deployment/configuration error.  The explicit urllib
    transport remains available for constrained tests, but silently selecting a
    non-pooled fallback would hide an incorrectly built production image.
    """

    return HttpxTransport(
        base_url,
        timeout_seconds=timeout_seconds,
        ca_bundle_path=ca_bundle_path,
    )


class UrllibHttpTransport:
    """Standard-library transport with a reusable opener/client lifecycle."""

    _MAX_BODY_BYTES = 8 * 1024 * 1024

    def __init__(self, base_url: str, opener: OpenerDirector | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        validate_base_url(self._base_url)
        # Empty ProxyHandler prevents ambient HTTP(S)_PROXY variables from
        # silently routing confidential prompts through an unrelated host.
        self._opener = opener or build_opener(ProxyHandler({}), _RejectRedirectHandler())

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None,
        timeout: float,
        stream: bool = False,
    ) -> HttpResponse:
        url = f"{self._base_url}/{path.lstrip('/')}"
        data = None
        if json_body is not None:
            data = json.dumps(
                json_body,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")

        request = Request(url, data=data, headers=dict(headers), method=method.upper())
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as exc:
            # HTTP status failures are values, not transport failures.  Returning
            # them lets the gateway apply one retry/error matrix consistently.
            try:
                body = self._read_bounded(exc)
            finally:
                exc.close()
            return HttpResponse(
                status=int(exc.code),
                headers=dict(exc.headers.items()) if exc.headers is not None else {},
                body=body,
            )
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError("model HTTP request timed out") from exc
        except URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise TimeoutError("model HTTP request timed out") from exc
            if isinstance(exc.reason, (ssl.SSLError, ssl.CertificateError)):
                raise TransportSecurityError(
                    "model HTTPS certificate or TLS negotiation failed",
                    code="MODEL_TLS_ERROR",
                ) from exc
            raise ConnectionError("model HTTP endpoint is unavailable") from exc
        except (ssl.SSLError, ssl.CertificateError) as exc:
            raise TransportSecurityError(
                "model HTTPS certificate or TLS negotiation failed",
                code="MODEL_TLS_ERROR",
            ) from exc
        except TransportSecurityError:
            raise
        except OSError as exc:
            raise ConnectionError("model HTTP endpoint is unavailable") from exc

        response_headers = dict(response.headers.items())
        status = int(response.status)
        if stream:
            return HttpResponse(status=status, headers=response_headers, body=self._stream(response))
        try:
            body = self._read_bounded(response)
        finally:
            response.close()
        return HttpResponse(status=status, headers=response_headers, body=body)

    @classmethod
    def _read_bounded(cls, response: Any) -> bytes:
        result: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > cls._MAX_BODY_BYTES:
                raise TransportSecurityError("provider HTTP body exceeds the size limit")
            result.append(chunk)
        return b"".join(result)

    @staticmethod
    def _stream(response: Any, chunk_size: int = 4096) -> Iterator[bytes]:
        try:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError("model SSE stream timed out") from exc
        except (ssl.SSLError, ssl.CertificateError) as exc:
            raise TransportSecurityError(
                "model SSE TLS connection failed",
                code="MODEL_TLS_ERROR",
            ) from exc
        except OSError as exc:
            raise ConnectionError("model SSE stream disconnected") from exc
        finally:
            response.close()
