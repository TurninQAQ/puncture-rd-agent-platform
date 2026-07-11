from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag.provider_http import (  # noqa: E402
    HttpxProviderTransport,
    ProviderEndpoint,
    ProviderHttpResponse,
    ProviderProtocolError,
    ProviderSecurityError,
    decode_json_response,
    normalize_base_url,
    strict_json_loads,
    validate_request_path,
)
import puncture_agent.rag.provider_http as provider_http_module  # noqa: E402


class _FakeTimeoutException(Exception):
    pass


class _FakeTransportError(Exception):
    pass


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body

    def iter_bytes(self):
        yield self._body


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc, traceback):
        return None


class _FakeClient:
    def __init__(self, module: "_FakeHttpx") -> None:
        self.module = module
        self.closed = False

    def stream(self, method, url, **kwargs):
        self.module.requests.append((method, url, kwargs))
        return _FakeStreamContext(self.module.response)

    def close(self):
        self.closed = True


class _FakeHttpx:
    TimeoutException = _FakeTimeoutException
    TransportError = _FakeTransportError

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.client_kwargs = None
        self.requests = []
        self.client = None

    @staticmethod
    def Timeout(value):
        return ("timeout", value)

    @staticmethod
    def Limits(**kwargs):
        return ("limits", kwargs)

    def Client(self, **kwargs):
        self.client_kwargs = kwargs
        self.client = _FakeClient(self)
        return self.client


class ProviderEndpointTests(unittest.TestCase):
    def test_separate_auth_and_tls_configuration_is_validated_and_redacted(self) -> None:
        endpoint = ProviderEndpoint(
            "https://Search.Internal:9443/v1/",
            auth_kind="basic",
            username="rag-reader",
            secret="top-secret-password",
            ca_bundle_path="/etc/rag/ca.pem",
            client_cert_path="/etc/rag/client.pem",
            client_key_path="/etc/rag/client.key",
            timeout_seconds=3.5,
        )
        self.assertEqual(endpoint.base_url, "https://search.internal:9443/v1")
        self.assertTrue(endpoint.authorization_header().startswith("Basic "))
        rendered = repr(endpoint)
        self.assertNotIn("top-secret-password", rendered)
        self.assertNotIn("client.key", rendered)
        self.assertIn("ca.pem", rendered)

    def test_bearer_secret_is_owned_by_endpoint(self) -> None:
        endpoint = ProviderEndpoint(
            "https://embedding.internal/v1",
            auth_kind="bearer",
            secret="embedding-token",
        )
        self.assertEqual(endpoint.authorization_header(), "Bearer embedding-token")
        self.assertNotIn("embedding-token", repr(endpoint))

    def test_unsafe_urls_and_credential_shapes_are_rejected(self) -> None:
        invalid_urls = (
            "file:///tmp/provider",
            "https://user:pass@provider.internal",
            "https://provider.internal/path?token=x",
            "https://provider.internal/#fragment",
            "https://provider.internal/a/../b",
            "https://provider.internal\r\nX-Test: injected",
            "http://provider.internal",
        )
        for value in invalid_urls:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProviderEndpoint(value)

        with self.assertRaises(ValueError):
            ProviderEndpoint("https://provider.internal", auth_kind="basic", username="reader")
        with self.assertRaises(ValueError):
            ProviderEndpoint(
                "https://provider.internal",
                client_cert_path="/tmp/client.pem",
            )

    def test_plain_http_requires_both_loopback_and_explicit_enablement(self) -> None:
        self.assertEqual(
            normalize_base_url("http://127.0.0.1:9200", allow_insecure_http=True),
            "http://127.0.0.1:9200",
        )
        with self.assertRaises(ValueError):
            normalize_base_url("http://127.0.0.1:9200")
        with self.assertRaises(ValueError):
            normalize_base_url("http://10.0.0.5:9200", allow_insecure_http=True)

    def test_request_path_cannot_escape_or_replace_the_configured_origin(self) -> None:
        self.assertEqual(validate_request_path("/v1/embeddings"), "/v1/embeddings")
        for value in (
            "v1/embeddings",
            "//attacker.internal/v1",
            "/v1/../admin",
            "/v1/items?secret=x",
            "/v1/items#fragment",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request_path(value)


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_numbers_and_invalid_utf8_are_rejected(self) -> None:
        invalid = (
            b'{"a":1,"a":2}',
            b'{"value":NaN}',
            b'{"value":Infinity}',
            b"\xff",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ProviderProtocolError):
                strict_json_loads(raw)

    def test_response_size_content_type_shape_and_empty_body_are_enforced(self) -> None:
        with self.assertRaisesRegex(ProviderProtocolError, "size limit"):
            strict_json_loads(b'{"large":"value"}', max_bytes=4)

        with self.assertRaisesRegex(ProviderProtocolError, "Content-Type"):
            decode_json_response(
                ProviderHttpResponse(200, {"Content-Type": "text/html"}, b"{}"),
                max_bytes=1024,
            )
        with self.assertRaisesRegex(ProviderProtocolError, "empty"):
            decode_json_response(
                ProviderHttpResponse(200, {"Content-Type": "application/json"}, b""),
                max_bytes=1024,
            )
        with self.assertRaisesRegex(ProviderProtocolError, "non-object"):
            decode_json_response(
                ProviderHttpResponse(200, {"Content-Type": "application/problem+json"}, b"[]"),
                max_bytes=1024,
            )


class HttpxProviderTransportTests(unittest.TestCase):
    def test_pooled_client_disables_redirects_and_environment_proxies(self) -> None:
        fake_httpx = _FakeHttpx(_FakeResponse(200, b'{"ok":true}'))
        endpoint = ProviderEndpoint(
            "https://provider.internal",
            auth_kind="bearer",
            secret="provider-token",
            max_connections=12,
            max_keepalive_connections=4,
        )
        with patch.object(provider_http_module.importlib, "import_module", return_value=fake_httpx):
            transport = HttpxProviderTransport(endpoint)
            response = transport.request("GET", "/health")
            transport.close()

        self.assertEqual(response.body, b'{"ok":true}')
        self.assertIs(fake_httpx.client_kwargs["trust_env"], False)
        self.assertIs(fake_httpx.client_kwargs["follow_redirects"], False)
        method, url, kwargs = fake_httpx.requests[0]
        self.assertEqual((method, url), ("GET", "https://provider.internal/health"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer provider-token")
        self.assertTrue(fake_httpx.client.closed)

    def test_redirect_and_oversized_body_fail_closed(self) -> None:
        cases = (
            (_FakeResponse(302, b""), ProviderSecurityError),
            (_FakeResponse(200, b"12345"), ProviderProtocolError),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                fake_httpx = _FakeHttpx(response)
                endpoint = ProviderEndpoint(
                    "https://provider.internal",
                    max_response_bytes=4,
                )
                with patch.object(
                    provider_http_module.importlib,
                    "import_module",
                    return_value=fake_httpx,
                ):
                    transport = HttpxProviderTransport(endpoint)
                    with self.assertRaises(expected):
                        transport.request("GET", "/health")


if __name__ == "__main__":
    unittest.main()
