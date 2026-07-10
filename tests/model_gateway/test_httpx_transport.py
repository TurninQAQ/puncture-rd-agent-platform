"""Security and lifecycle tests for the pooled production HTTP transport."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import ssl
import sys
import threading
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.model_gateway.http_transport import (  # noqa: E402
    HttpxTransport,
    TransportSecurityError,
)


class _FakeResponse:
    def __init__(
        self,
        status: int = 200,
        chunks: Iterable[bytes] = (b'{}',),
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self._chunks = tuple(chunks)
        self.headers = dict(headers or {"Content-Type": "application/json"})

    def iter_bytes(self) -> Iterable[bytes]:
        yield from self._chunks


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.exits: list[tuple[Any, Any, Any]] = []

    def __enter__(self) -> _FakeResponse:
        return self.response

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.exits.append((exc_type, exc, traceback))


def _fake_httpx_module(response: _FakeResponse | None = None) -> types.SimpleNamespace:
    module = types.SimpleNamespace()

    class TimeoutException(Exception):
        pass

    class TransportError(Exception):
        pass

    class Timeout:
        def __init__(self, value: float) -> None:
            self.value = value

    class Limits:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            self.requests: list[dict[str, Any]] = []
            self.contexts: list[_FakeStreamContext] = []
            module.last_client = self

        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            self.requests.append({"method": method, "url": url, **kwargs})
            context = _FakeStreamContext(response or _FakeResponse())
            self.contexts.append(context)
            return context

        def close(self) -> None:
            self.closed = True

    module.TimeoutException = TimeoutException
    module.TransportError = TransportError
    module.Timeout = Timeout
    module.Limits = Limits
    module.Client = Client
    module.last_client = None
    return module


class HttpxTransportOfflineTests(unittest.TestCase):
    def test_client_disables_environment_proxy_and_redirects_and_closes(self) -> None:
        fake_httpx = _fake_httpx_module()
        with mock.patch(
            "puncture_agent.model_gateway.http_transport.importlib.import_module",
            return_value=fake_httpx,
        ):
            transport = HttpxTransport(
                "https://vllm.internal/v1",
                timeout_seconds=9.0,
                max_connections=12,
                max_keepalive_connections=4,
            )

        client = fake_httpx.last_client
        self.assertFalse(client.kwargs["trust_env"])
        self.assertFalse(client.kwargs["follow_redirects"])
        self.assertTrue(client.kwargs["verify"])
        self.assertEqual(client.kwargs["timeout"].value, 9.0)
        self.assertEqual(client.kwargs["limits"].kwargs["max_connections"], 12)
        self.assertEqual(client.kwargs["limits"].kwargs["max_keepalive_connections"], 4)

        response = transport.request(
            "GET",
            "/models",
            headers={"Authorization": "Bearer test-token"},
            json_body=None,
            timeout=2.0,
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"{}")
        self.assertEqual(client.requests[0]["url"], "https://vllm.internal/v1/models")
        self.assertTrue(client.contexts[0].exits)

        transport.close()
        self.assertTrue(client.closed)

    def test_bounded_body_and_tls_errors_are_normalized(self) -> None:
        fake_httpx = _fake_httpx_module(_FakeResponse(chunks=(b"123", b"45")))
        with mock.patch(
            "puncture_agent.model_gateway.http_transport.importlib.import_module",
            return_value=fake_httpx,
        ):
            transport = HttpxTransport("https://vllm.internal/v1", timeout_seconds=3.0)

        original_limit = HttpxTransport._MAX_BODY_BYTES
        HttpxTransport._MAX_BODY_BYTES = 4
        try:
            with self.assertRaisesRegex(TransportSecurityError, "size limit"):
                transport.request(
                    "GET",
                    "/models",
                    headers={},
                    json_body=None,
                    timeout=1.0,
                )
        finally:
            HttpxTransport._MAX_BODY_BYTES = original_limit
        self.assertTrue(fake_httpx.last_client.contexts[0].exits)

        tls_error = fake_httpx.TransportError("TLS negotiation failed")
        tls_error.__cause__ = ssl.SSLError("certificate verify failed")
        with self.assertRaises(TransportSecurityError) as context:
            transport._raise_normalized_transport_error(tls_error)
        self.assertEqual(context.exception.code, "MODEL_TLS_ERROR")

    def test_custom_ca_uses_an_explicit_ssl_context(self) -> None:
        fake_httpx = _fake_httpx_module()
        sentinel_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with (
            mock.patch(
                "puncture_agent.model_gateway.http_transport.importlib.import_module",
                return_value=fake_httpx,
            ),
            mock.patch(
                "puncture_agent.model_gateway.http_transport.ssl.create_default_context",
                return_value=sentinel_context,
            ) as create_context,
        ):
            HttpxTransport(
                "https://vllm.internal/v1",
                timeout_seconds=3.0,
                ca_bundle_path="/approved/enterprise-ca.pem",
            )
        create_context.assert_called_once_with(cafile="/approved/enterprise-ca.pem")
        self.assertIs(fake_httpx.last_client.kwargs["verify"], sentinel_context)


try:
    import httpx as _real_httpx  # noqa: F401
except ImportError:
    REAL_HTTPX_AVAILABLE = False
else:
    REAL_HTTPX_AVAILABLE = True


class _TransportHandler(BaseHTTPRequestHandler):
    capture_hits = 0

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/ok":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/capture")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/capture":
            type(self).capture_hits += 1
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/large":
            body = b"x" * 64
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


@unittest.skipUnless(REAL_HTTPX_AVAILABLE, "install httpx to run pooled transport integration tests")
class HttpxTransportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _TransportHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_ambient_proxy_is_ignored_and_redirect_is_not_followed(self) -> None:
        _TransportHandler.capture_hits = 0
        with mock.patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
            },
            clear=False,
        ):
            with HttpxTransport(self.base_url, timeout_seconds=3.0) as transport:
                ok = transport.request(
                    "GET", "/ok", headers={}, json_body=None, timeout=2.0
                )
                redirect = transport.request(
                    "GET",
                    "/redirect",
                    headers={"Authorization": "Bearer test-token"},
                    json_body=None,
                    timeout=2.0,
                )
        self.assertEqual(ok.status, 200)
        self.assertEqual(redirect.status, 302)
        self.assertEqual(_TransportHandler.capture_hits, 0)

    def test_real_client_enforces_body_limit(self) -> None:
        with HttpxTransport(self.base_url, timeout_seconds=3.0) as transport:
            original_limit = HttpxTransport._MAX_BODY_BYTES
            HttpxTransport._MAX_BODY_BYTES = 16
            try:
                with self.assertRaisesRegex(TransportSecurityError, "size limit"):
                    transport.request(
                        "GET", "/large", headers={}, json_body=None, timeout=2.0
                    )
            finally:
                HttpxTransport._MAX_BODY_BYTES = original_limit


if __name__ == "__main__":
    unittest.main()
