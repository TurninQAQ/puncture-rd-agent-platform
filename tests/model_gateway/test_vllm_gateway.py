"""Offline production-adapter tests for the OpenAI-compatible vLLM gateway."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from email.utils import formatdate
import json
import ssl
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.model_gateway import (  # noqa: E402
    ChatMessage,
    ModelGatewayError,
    ModelRequest,
    ToolDefinition,
    VllmGatewayConfig,
    VllmModelGateway,
)
from puncture_agent.model_gateway.http_transport import HttpResponse  # noqa: E402
from puncture_agent.model_gateway.http_transport import TransportSecurityError  # noqa: E402


SEARCH_TOOL = ToolDefinition(
    name="search_knowledge",
    description="Search versioned internal knowledge.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

STATUS_TOOL = ToolDefinition(
    name="get_case_status",
    description="Get a case status.",
    input_schema={
        "type": "object",
        "properties": {"case_id": {"type": "string"}},
        "required": ["case_id"],
        "additionalProperties": False,
    },
)


def json_response(payload: Mapping[str, Any], status: int = 200, **headers: str) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=headers,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def completion(
    *,
    content: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: Mapping[str, int] | None = None,
) -> HttpResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    payload: dict[str, Any] = {
        "id": "chatcmpl-provider-1",
        "model": "qwen-private",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = dict(usage)
    return json_response(payload)


class ScriptedTransport:
    """A fake transport that captures requests and returns a fixed script."""

    def __init__(self, *steps: HttpResponse | BaseException) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "json_body": json_body,
                "timeout": timeout,
                "stream": stream,
            }
        )
        if not self.steps:
            raise AssertionError("fake transport script exhausted")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class Clock:
    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.last = self.values[-1] if self.values else 0.0

    def __call__(self) -> float:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class VllmGatewayTests(unittest.TestCase):
    def gateway(
        self,
        transport: ScriptedTransport,
        *,
        max_retries: int = 2,
        api_key: str | None = None,
        sleeps: list[float] | None = None,
        clock: Clock | None = None,
        wall_time: Callable[[], float] | None = None,
    ) -> VllmModelGateway:
        return VllmModelGateway(
            VllmGatewayConfig(
                base_url="http://vllm.internal/v1",
                model="qwen-private",
                api_key=api_key,
                timeout_seconds=12.0,
                max_retries=max_retries,
            ),
            transport=transport,
            sleep=(sleeps.append if sleeps is not None else lambda _: None),
            jitter=lambda: 0.0,
            monotonic=clock or Clock(1.0, 1.025),
            wall_time=wall_time or (lambda: 0.0),
        )

    def test_serializes_messages_tools_and_plain_response(self) -> None:
        transport = ScriptedTransport(
            completion(
                content="analysis complete",
                usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            )
        )
        request = ModelRequest(
            request_id="req-serialize",
            messages=(
                ChatMessage(role="system", content="Follow policy", name="policy"),
                ChatMessage(role="user", content="Search spacing"),
                ChatMessage(role="assistant", content="Calling a tool"),
                ChatMessage(role="tool", content='{"result": "ok"}', tool_call_id="call-old"),
            ),
            tools=(SEARCH_TOOL,),
            temperature=0.2,
            max_tokens=321,
            metadata={"api_key": "must-not-forward", "mock_text": "ignored"},
        )

        result = self.gateway(transport).generate(request)

        sent = transport.calls[0]
        self.assertEqual(sent["method"], "POST")
        self.assertEqual(sent["path"], "/chat/completions")
        self.assertEqual(sent["headers"]["X-Request-ID"], "req-serialize")
        body = sent["json_body"]
        self.assertEqual(body["model"], "qwen-private")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 321)
        self.assertEqual(body["n"], 1)
        self.assertFalse(body["stream"])
        self.assertNotIn("metadata", body)
        self.assertNotIn("api_key", json.dumps(body))
        self.assertEqual(body["messages"][0]["name"], "policy")
        self.assertEqual(body["messages"][3]["tool_call_id"], "call-old")
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tools"][0]["function"]["name"], SEARCH_TOOL.name)
        self.assertEqual(body["tools"][0]["function"]["parameters"], SEARCH_TOOL.input_schema)
        self.assertEqual(result.text, "analysis complete")
        self.assertEqual(result.model, "qwen-private")
        self.assertEqual(result.raw_response_id, "chatcmpl-provider-1")
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertAlmostEqual(result.latency_ms, 25.0)

    def test_parses_multiple_tool_calls_and_validates_each_schema(self) -> None:
        raw_calls = [
            {
                "id": "call-search",
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "arguments": '{"query":"safe radius","top_k":3}',
                },
            },
            {
                "id": "call-status",
                "type": "function",
                "function": {"name": "get_case_status", "arguments": '{"case_id":"case-1"}'},
            },
        ]
        transport = ScriptedTransport(completion(content=None, tool_calls=raw_calls, finish_reason="tool_calls"))
        request = ModelRequest(
            request_id="req-tools",
            messages=(ChatMessage(role="user", content="run tools"),),
            tools=(SEARCH_TOOL, STATUS_TOOL),
        )

        result = self.gateway(transport).generate(request)

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual([call.name for call in result.tool_calls], ["search_knowledge", "get_case_status"])
        self.assertEqual(result.tool_calls[0].arguments["top_k"], 3)
        self.assertEqual(result.tool_calls[1].call_id, "call-status")

    def test_structured_output_is_requested_parsed_and_locally_validated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["SEARCH", "PLAN"]},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }
        transport = ScriptedTransport(completion(content='{"intent":"PLAN","confidence":0.96}'))
        request = ModelRequest(
            request_id="req-structured",
            messages=(ChatMessage(role="user", content="classify"),),
            response_schema=schema,
        )

        result = self.gateway(transport).generate(request)

        response_format = transport.calls[0]["json_body"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)
        self.assertIsNone(result.text)
        self.assertEqual(result.structured_output, {"intent": "PLAN", "confidence": 0.96})

    def test_missing_usage_is_normalized_to_explicit_zero_counts(self) -> None:
        result = self.gateway(ScriptedTransport(completion())).generate(
            ModelRequest(
                request_id="req-no-usage",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.usage.prompt_tokens, 0)
        self.assertEqual(result.usage.completion_tokens, 0)
        self.assertEqual(result.usage.total_tokens, 0)
        self.assertFalse(result.usage.usage_known)

    def test_partial_usage_and_finish_reason_mismatches_are_protocol_errors(self) -> None:
        partial_usage = completion(
            content="ok",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )
        tool_with_stop = completion(
            content=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge",
                        "arguments": '{"query":"spacing"}',
                    },
                }
            ],
            finish_reason="stop",
        )
        declared_without_call = completion(content="text", finish_reason="tool_calls")
        cases = (
            (
                partial_usage,
                ModelRequest(
                    request_id="partial-usage",
                    messages=(ChatMessage(role="user", content="hello"),),
                ),
            ),
            (
                tool_with_stop,
                ModelRequest(
                    request_id="tool-finish-mismatch",
                    messages=(ChatMessage(role="user", content="search"),),
                    tools=(SEARCH_TOOL,),
                ),
            ),
            (
                declared_without_call,
                ModelRequest(
                    request_id="missing-declared-call",
                    messages=(ChatMessage(role="user", content="hello"),),
                ),
            ),
        )
        for provider_response, request in cases:
            with self.subTest(request=request.request_id):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(provider_response)).generate(request)
                self.assertEqual(context.exception.code, "MODEL_PROTOCOL_ERROR")

    def test_stream_finish_reason_must_match_tool_call_data(self) -> None:
        mismatch_frames = (
            {
                "model": "qwen-private",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "search_knowledge",
                                        "arguments": '{"query":"spacing"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "model": "qwen-private",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        for index, frame in enumerate(mismatch_frames):
            body = [
                f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n".encode("utf-8")
            ]
            events = list(
                self.gateway(ScriptedTransport(HttpResponse(200, {}, body))).stream(
                    ModelRequest(
                        request_id=f"stream-finish-mismatch-{index}",
                        messages=(ChatMessage(role="user", content="search"),),
                        tools=(SEARCH_TOOL,) if index == 0 else (),
                        stream=True,
                    )
                )
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "error")
            self.assertEqual(events[0].error["code"], "MODEL_PROTOCOL_ERROR")

    def test_tool_and_structured_outputs_fail_closed_with_exact_codes(self) -> None:
        cases: list[tuple[str, HttpResponse, ModelRequest, str]] = [
            (
                "unknown tool",
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call-1",
                            "function": {"name": "delete_everything", "arguments": "{}"},
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ModelRequest(
                    request_id="bad-unknown",
                    messages=(ChatMessage(role="user", content="bad"),),
                    tools=(SEARCH_TOOL,),
                ),
                "UNKNOWN_TOOL",
            ),
            (
                "malformed arguments",
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call-1",
                            "function": {"name": "search_knowledge", "arguments": "{bad"},
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ModelRequest(
                    request_id="bad-json",
                    messages=(ChatMessage(role="user", content="bad"),),
                    tools=(SEARCH_TOOL,),
                ),
                "TOOL_ARGUMENT_PARSE_ERROR",
            ),
            (
                "missing required argument",
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call-1",
                            "function": {"name": "search_knowledge", "arguments": "{}"},
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ModelRequest(
                    request_id="bad-required",
                    messages=(ChatMessage(role="user", content="bad"),),
                    tools=(SEARCH_TOOL,),
                ),
                "TOOL_ARGUMENT_SCHEMA_ERROR",
            ),
            (
                "unexpected argument",
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call-1",
                            "function": {
                                "name": "search_knowledge",
                                "arguments": '{"query":"x","unsafe":true}',
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                ModelRequest(
                    request_id="bad-extra",
                    messages=(ChatMessage(role="user", content="bad"),),
                    tools=(SEARCH_TOOL,),
                ),
                "TOOL_ARGUMENT_SCHEMA_ERROR",
            ),
            (
                "structured mismatch",
                completion(content='{"unexpected":true}'),
                ModelRequest(
                    request_id="bad-structured",
                    messages=(ChatMessage(role="user", content="bad"),),
                    response_schema={
                        "type": "object",
                        "properties": {"intent": {"type": "string"}},
                        "required": ["intent"],
                        "additionalProperties": False,
                    },
                ),
                "STRUCTURED_OUTPUT_SCHEMA_ERROR",
            ),
        ]
        for label, provider_response, request, expected_code in cases:
            with self.subTest(label):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(provider_response)).generate(request)
                self.assertEqual(context.exception.code, expected_code)
                self.assertFalse(context.exception.retryable)
                self.assertEqual(context.exception.details["attempts"], 1)

    def test_malformed_and_empty_provider_responses_are_protocol_errors(self) -> None:
        malformed = HttpResponse(status=200, headers={}, body=b"not-json")
        with self.assertRaises(ModelGatewayError) as malformed_context:
            self.gateway(ScriptedTransport(malformed)).generate(
                ModelRequest(
                    request_id="malformed",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(malformed_context.exception.code, "MODEL_PROTOCOL_ERROR")

        with self.assertRaises(ModelGatewayError) as empty_context:
            self.gateway(ScriptedTransport(completion(content=None))).generate(
                ModelRequest(
                    request_id="empty",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(empty_context.exception.code, "EMPTY_MODEL_RESPONSE")

    def test_timeout_and_rate_limit_retry_then_succeed(self) -> None:
        sleeps: list[float] = []
        timeout_transport = ScriptedTransport(TimeoutError("socket timeout"), completion(content="recovered"))
        result = self.gateway(timeout_transport, sleeps=sleeps).generate(
            ModelRequest(
                request_id="retry-timeout",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(timeout_transport.calls), 2)
        self.assertEqual(sleeps, [0.25])

        sleeps = []
        limited = HttpResponse(status=429, headers={"Retry-After": "1.75"}, body=b'{"error":"busy"}')
        rate_transport = ScriptedTransport(limited, completion(content="recovered"))
        result = self.gateway(rate_transport, sleeps=sleeps).generate(
            ModelRequest(
                request_id="retry-429",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "recovered")
        self.assertEqual(sleeps, [1.75])

        sleeps = []
        current_wall_time = 1_700_000_000.0
        retry_at = formatdate(current_wall_time + 2.0, usegmt=True)
        dated_transport = ScriptedTransport(
            HttpResponse(status=429, headers={"Retry-After": retry_at}, body=b"busy"),
            completion(content="recovered-from-date"),
        )
        result = self.gateway(
            dated_transport,
            sleeps=sleeps,
            wall_time=lambda: current_wall_time,
        ).generate(
            ModelRequest(
                request_id="retry-429-http-date",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "recovered-from-date")
        self.assertEqual(sleeps, [2.0])

    def test_retry_exhaustion_and_non_retryable_http_matrix(self) -> None:
        sleeps: list[float] = []
        transport = ScriptedTransport(
            HttpResponse(503, {}, b"busy"),
            HttpResponse(503, {}, b"busy"),
            HttpResponse(503, {}, b"busy"),
        )
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(transport, max_retries=2, sleeps=sleeps).generate(
                ModelRequest(
                    request_id="retry-exhausted",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(context.exception.code, "MODEL_UNAVAILABLE")
        self.assertTrue(context.exception.retryable)
        self.assertEqual(context.exception.details["attempts"], 3)
        self.assertTrue(context.exception.details["retry_exhausted"])
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])

        non_retryable_statuses = (
            (302, "MODEL_SECURITY_ERROR"),
            (400, "MODEL_REQUEST_REJECTED"),
            (401, "MODEL_PERMISSION_DENIED"),
            (403, "MODEL_PERMISSION_DENIED"),
            (404, "MODEL_REQUEST_REJECTED"),
            (413, "MODEL_REQUEST_REJECTED"),
            (422, "MODEL_REQUEST_REJECTED"),
        )
        for status, expected_code in non_retryable_statuses:
            with self.subTest(status=status):
                one_call = ScriptedTransport(HttpResponse(status, {}, b"rejected"))
                with self.assertRaises(ModelGatewayError) as http_context:
                    self.gateway(one_call, max_retries=5).generate(
                        ModelRequest(
                            request_id=f"status-{status}",
                            messages=(ChatMessage(role="user", content="hello"),),
                        )
                    )
                self.assertEqual(http_context.exception.code, expected_code)
                self.assertFalse(http_context.exception.retryable)
                self.assertEqual(http_context.exception.details["attempts"], 1)
                self.assertEqual(len(one_call.calls), 1)

    def test_health_reports_up_degraded_and_down_without_generation(self) -> None:
        served = ScriptedTransport(
            json_response({"object": "list", "data": [{"id": "other"}, {"id": "qwen-private"}]})
        )
        health = self.gateway(served).health()
        self.assertEqual(health.status, "UP")
        self.assertEqual(served.calls[0]["path"], "/models")
        self.assertEqual(served.calls[0]["method"], "GET")
        self.assertFalse(served.calls[0]["stream"])
        self.assertLessEqual(served.calls[0]["timeout"], 5.0)

        missing = self.gateway(ScriptedTransport(json_response({"data": [{"id": "other"}]}))).health()
        self.assertEqual(missing.status, "DOWN")
        self.assertEqual(missing.details["model_check"], "configured_model_not_served")

        malformed = self.gateway(ScriptedTransport(json_response({"not_data": []}))).health()
        self.assertEqual(malformed.status, "DEGRADED")

        unavailable = self.gateway(ScriptedTransport(ConnectionError("dns failed"))).health()
        self.assertEqual(unavailable.status, "DOWN")
        self.assertEqual(unavailable.details["error_code"], "MODEL_UNAVAILABLE")

    def test_api_key_is_sent_but_never_exposed_by_errors(self) -> None:
        secret = "super-secret-private-token"
        provider = ScriptedTransport(
            HttpResponse(
                status=401,
                headers={"X-Debug-Authorization": f"Bearer {secret}"},
                body=f'{{"error":"credential {secret}"}}'.encode(),
            )
        )
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(provider, api_key=secret).generate(
                ModelRequest(
                    request_id="secret-test",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(provider.calls[0]["headers"]["Authorization"], f"Bearer {secret}")
        exposed = f"{context.exception!s} {context.exception.details!r} {context.exception!r}"
        self.assertNotIn(secret, exposed)
        self.assertEqual(context.exception.code, "MODEL_PERMISSION_DENIED")

    def test_stream_reassembles_fragmented_text_and_tool_arguments(self) -> None:
        events = [
            {
                "id": "stream-provider-1",
                "model": "qwen-private",
                "choices": [{"index": 0, "delta": {"content": "checking "}, "finish_reason": None}],
            },
            {
                "id": "stream-provider-1",
                "model": "qwen-private",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-stream",
                                    "function": {"name": "search_", "arguments": '{"query":"safe'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "stream-provider-1",
                "model": "qwen-private",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "now",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "knowledge", "arguments": ' radius","top_k":2}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "id": "stream-provider-1",
                "model": "qwen-private",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            },
        ]
        wire = "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events) + "data: [DONE]\r\n\r\n"
        raw = wire.encode("utf-8")
        # Split in the middle of JSON tokens and UTF-8/SSE framing to prove the
        # parser does not assume one HTTP chunk equals one provider event.
        cuts = [3, 19, 53, 127, 211, len(raw) - 8]
        chunks: list[bytes] = []
        start = 0
        for end in cuts:
            chunks.append(raw[start:end])
            start = end
        chunks.append(raw[start:])
        transport = ScriptedTransport(HttpResponse(200, {"Content-Type": "text/event-stream"}, chunks))
        request = ModelRequest(
            request_id="stream-tools",
            messages=(ChatMessage(role="user", content="search"),),
            tools=(SEARCH_TOOL,),
            stream=True,
        )

        result_events = list(self.gateway(transport).stream(request))

        self.assertEqual([event.sequence for event in result_events], list(range(len(result_events))))
        self.assertEqual([event.event_type for event in result_events], ["text_delta", "text_delta", "tool_call", "completed"])
        self.assertEqual(result_events[2].tool_call.name, "search_knowledge")
        self.assertEqual(result_events[2].tool_call.arguments, {"query": "safe radius", "top_k": 2})
        final = result_events[-1].response
        self.assertEqual(final.finish_reason, "tool_calls")
        self.assertEqual(final.text, "checking now")
        self.assertEqual(final.tool_calls, (result_events[2].tool_call,))
        self.assertEqual(final.raw_response_id, "stream-provider-1")
        self.assertEqual(final.model, "qwen-private")
        self.assertEqual(final.usage.total_tokens, 12)
        sent = transport.calls[0]
        self.assertTrue(sent["stream"])
        self.assertTrue(sent["json_body"]["stream"])
        self.assertEqual(sent["json_body"]["stream_options"], {"include_usage": True})

    def test_stream_emits_one_completed_event_for_plain_text(self) -> None:
        chunks = [
            b'data: {"id":"s1","model":"qwen-private","choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}\n\n',
            b'data: {"id":"s1","model":"qwen-private","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = list(
            self.gateway(ScriptedTransport(HttpResponse(200, {}, chunks))).stream(
                ModelRequest(
                    request_id="stream-text",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        )
        self.assertEqual("".join(event.delta or "" for event in events), "hello")
        self.assertEqual(sum(event.event_type == "completed" for event in events), 1)
        self.assertEqual(events[-1].response.text, "hello")

    def test_disconnect_mid_stream_is_normalized_and_never_completes(self) -> None:
        def broken_chunks() -> Iterable[bytes]:
            yield b'data: {"model":"qwen-private","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            raise ConnectionError("provider reset connection")

        stream = self.gateway(
            ScriptedTransport(HttpResponse(200, {}, broken_chunks()))
        ).stream(
            ModelRequest(
                request_id="stream-disconnect",
                messages=(ChatMessage(role="user", content="hello"),),
                stream=True,
            )
        )
        first = next(stream)
        self.assertEqual(first.event_type, "text_delta")
        remaining = list(stream)
        self.assertEqual(len(remaining), 1)
        terminal = remaining[0]
        self.assertEqual(terminal.event_type, "error")
        self.assertEqual(terminal.sequence, 1)
        self.assertEqual(terminal.error["code"], "MODEL_UNAVAILABLE")
        self.assertFalse(terminal.error["retryable"])
        self.assertEqual(terminal.error["details"]["attempts"], 1)
        self.assertTrue(terminal.error["details"]["output_visible"])
        self.assertTrue(terminal.error["details"]["upstream_retryable"])

    def test_disconnect_before_visible_stream_output_is_retried(self) -> None:
        def disconnect_before_data() -> Iterable[bytes]:
            raise ConnectionError("provider reset before first event")
            yield b""  # pragma: no cover - makes this a generator

        recovered_body = [
            b'data: {"model":"qwen-private","choices":[{"index":0,"delta":{"content":"recovered"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        sleeps: list[float] = []
        transport = ScriptedTransport(
            HttpResponse(200, {}, disconnect_before_data()),
            HttpResponse(200, {}, recovered_body),
        )
        events = list(
            self.gateway(
                transport,
                max_retries=1,
                sleeps=sleeps,
            ).stream(
                ModelRequest(
                    request_id="stream-previsible-retry",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        )
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual([event.event_type for event in events], ["text_delta", "completed"])
        self.assertEqual(events[-1].response.text, "recovered")

    def test_previsible_stream_disconnect_exhaustion_is_one_error_event(self) -> None:
        def disconnect_before_data() -> Iterable[bytes]:
            raise ConnectionError("provider reset before first event")
            yield b""  # pragma: no cover - makes this a generator

        transport = ScriptedTransport(
            HttpResponse(200, {}, disconnect_before_data()),
            HttpResponse(200, {}, disconnect_before_data()),
        )
        events = list(
            self.gateway(transport, max_retries=1).stream(
                ModelRequest(
                    request_id="stream-previsible-exhaustion",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        )
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "error")
        self.assertEqual(events[0].error["code"], "MODEL_UNAVAILABLE")
        self.assertTrue(events[0].error["retryable"])
        self.assertFalse(events[0].error["details"]["output_visible"])
        self.assertTrue(events[0].error["details"]["retry_exhausted"])
        self.assertEqual(events[0].error["details"]["attempts"], 2)

    def test_stream_tool_parse_error_has_no_tool_or_completed_event(self) -> None:
        payload = {
            "model": "qwen-private",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "bad-call",
                                "function": {"name": "search_knowledge", "arguments": "{bad"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        body = [f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()]
        stream = self.gateway(ScriptedTransport(HttpResponse(200, {}, body))).stream(
            ModelRequest(
                request_id="stream-bad-tool",
                messages=(ChatMessage(role="user", content="search"),),
                tools=(SEARCH_TOOL,),
                stream=True,
            )
        )
        result_events = list(stream)
        self.assertEqual(len(result_events), 1)
        self.assertEqual(result_events[0].event_type, "error")
        self.assertEqual(result_events[0].error["code"], "TOOL_ARGUMENT_PARSE_ERROR")
        self.assertFalse(result_events[0].error["retryable"])

    def test_rejects_ambiguous_request_headers_and_output_shapes_before_execution(self) -> None:
        bad_header_transport = ScriptedTransport(completion())
        with self.assertRaises(ValueError):
            self.gateway(bad_header_transport).generate(
                ModelRequest(
                    request_id="safe\r\nX-Injected: true",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(len(bad_header_transport.calls), 0)

        ambiguous_transport = ScriptedTransport(completion())
        with self.assertRaises(ValueError):
            ModelRequest(
                request_id="ambiguous-output",
                messages=(ChatMessage(role="user", content="hello"),),
                tools=(SEARCH_TOOL,),
                response_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            )
        self.assertEqual(len(ambiguous_transport.calls), 0)

    def test_serializes_assistant_tool_call_history_for_followup_turn(self) -> None:
        from puncture_agent.model_gateway import ToolCall

        previous = ToolCall(
            call_id="previous-call",
            name="search_knowledge",
            arguments={"query": "spacing", "top_k": 1},
        )
        transport = ScriptedTransport(completion(content="follow-up complete"))
        result = self.gateway(transport).generate(
            ModelRequest(
                request_id="history",
                messages=(
                    ChatMessage(role="user", content="search"),
                    ChatMessage(role="assistant", content="", tool_calls=(previous,)),
                    ChatMessage(role="tool", content='{"hits":1}', tool_call_id="previous-call"),
                ),
                tools=(SEARCH_TOOL,),
            )
        )
        history_call = transport.calls[0]["json_body"]["messages"][1]["tool_calls"][0]
        self.assertEqual(history_call["id"], "previous-call")
        self.assertEqual(history_call["function"]["name"], "search_knowledge")
        self.assertEqual(
            json.loads(history_call["function"]["arguments"]),
            {"query": "spacing", "top_k": 1},
        )
        self.assertEqual(result.text, "follow-up complete")

    def test_rejects_model_mismatch_multiple_choices_and_duplicate_call_ids(self) -> None:
        mismatch_payload = {
            "id": "wrong-model",
            "model": "untrusted-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }
        multiple_payload = {
            "id": "multiple",
            "model": "qwen-private",
            "choices": [
                {"index": 0, "message": {"content": "first"}, "finish_reason": "stop"},
                {"index": 1, "message": {"content": "second"}, "finish_reason": "stop"},
            ],
        }
        duplicate_calls = [
            {
                "id": "duplicate",
                "function": {"name": "search_knowledge", "arguments": '{"query":"one"}'},
            },
            {
                "id": "duplicate",
                "function": {"name": "search_knowledge", "arguments": '{"query":"two"}'},
            },
        ]
        requests_and_responses = [
            (
                json_response(mismatch_payload),
                ModelRequest(
                    request_id="model-mismatch",
                    messages=(ChatMessage(role="user", content="hello"),),
                ),
            ),
            (
                json_response(multiple_payload),
                ModelRequest(
                    request_id="multiple-choices",
                    messages=(ChatMessage(role="user", content="hello"),),
                ),
            ),
            (
                completion(content=None, tool_calls=duplicate_calls, finish_reason="tool_calls"),
                ModelRequest(
                    request_id="duplicate-calls",
                    messages=(ChatMessage(role="user", content="hello"),),
                    tools=(SEARCH_TOOL,),
                ),
            ),
        ]
        for provider_response, request in requests_and_responses:
            with self.subTest(request=request.request_id):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(provider_response)).generate(request)
                self.assertEqual(context.exception.code, "MODEL_PROTOCOL_ERROR")
                self.assertFalse(context.exception.retryable)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        duplicate = HttpResponse(
            200,
            {},
            b'{"id":"one","id":"two","model":"qwen-private","choices":[]}',
        )
        nonfinite = HttpResponse(
            200,
            {},
            b'{"id":"one","model":"qwen-private","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":NaN,"total_tokens":1}}',
        )
        for label, provider_response in [("duplicate", duplicate), ("nonfinite", nonfinite)]:
            with self.subTest(label):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(provider_response)).generate(
                        ModelRequest(
                            request_id=f"strict-{label}",
                            messages=(ChatMessage(role="user", content="hello"),),
                        )
                    )
                self.assertEqual(context.exception.code, "MODEL_PROTOCOL_ERROR")

    def test_tool_and_structured_payloads_reject_duplicate_and_nonfinite_json(self) -> None:
        bad_tool_arguments = (
            '{"query":"one","query":"two"}',
            '{"query":"spacing","top_k":NaN}',
        )
        for index, arguments in enumerate(bad_tool_arguments):
            provider_response = completion(
                content=None,
                tool_calls=[
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge",
                            "arguments": arguments,
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
            with self.subTest(tool_arguments=arguments):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(provider_response)).generate(
                        ModelRequest(
                            request_id=f"strict-tool-{index}",
                            messages=(ChatMessage(role="user", content="search"),),
                            tools=(SEARCH_TOOL,),
                        )
                    )
                self.assertEqual(context.exception.code, "TOOL_ARGUMENT_PARSE_ERROR")

        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }
        bad_structured_values = (
            '{"intent":"PLAN","intent":"SEARCH","confidence":0.9}',
            '{"intent":"PLAN","confidence":NaN}',
        )
        for index, content in enumerate(bad_structured_values):
            with self.subTest(structured=content):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(
                        ScriptedTransport(completion(content=content))
                    ).generate(
                        ModelRequest(
                            request_id=f"strict-structured-{index}",
                            messages=(ChatMessage(role="user", content="classify"),),
                            response_schema=schema,
                        )
                    )
                self.assertEqual(
                    context.exception.code,
                    "STRUCTURED_OUTPUT_SCHEMA_ERROR",
                )

    def test_tls_failures_are_nonretryable_for_generate_and_structured_for_stream(self) -> None:
        transport = ScriptedTransport(ssl.SSLError("certificate verify failed"))
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(transport, max_retries=4).generate(
                ModelRequest(
                    request_id="tls-generate",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(context.exception.code, "MODEL_TLS_ERROR")
        self.assertFalse(context.exception.retryable)
        self.assertEqual(len(transport.calls), 1)

        stream_events = list(
            self.gateway(
                ScriptedTransport(
                    TransportSecurityError("cross-origin redirect", code="MODEL_SECURITY_ERROR")
                )
            ).stream(
                ModelRequest(
                    request_id="security-stream",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        )
        self.assertEqual(len(stream_events), 1)
        self.assertEqual(stream_events[0].event_type, "error")
        self.assertEqual(stream_events[0].error["code"], "MODEL_SECURITY_ERROR")
        self.assertFalse(stream_events[0].error["retryable"])

    def test_http_408_is_a_retryable_timeout(self) -> None:
        sleeps: list[float] = []
        transport = ScriptedTransport(HttpResponse(408, {}, b"timeout"), completion(content="ok"))
        result = self.gateway(transport, sleeps=sleeps).generate(
            ModelRequest(
                request_id="http-408",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(sleeps, [0.25])

    def test_invalid_base_urls_are_rejected_even_with_an_injected_transport(self) -> None:
        invalid_urls = [
            "file:///tmp/model",
            "http://user:password@vllm.internal/v1",
            "http://vllm.internal/v1?token=secret",
            "http://vllm.internal/v1#fragment",
        ]
        for base_url in invalid_urls:
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    VllmModelGateway(
                        VllmGatewayConfig(base_url=base_url, model="qwen-private"),
                        transport=ScriptedTransport(completion()),
                    )

    def test_oversized_sse_is_a_terminal_protocol_error(self) -> None:
        original_limit = VllmModelGateway._MAX_SSE_BYTES
        VllmModelGateway._MAX_SSE_BYTES = 16
        try:
            events = list(
                self.gateway(
                    ScriptedTransport(HttpResponse(200, {}, [b"data: " + b"x" * 32]))
                ).stream(
                    ModelRequest(
                        request_id="oversized-sse",
                        messages=(ChatMessage(role="user", content="hello"),),
                        stream=True,
                    )
                )
            )
        finally:
            VllmModelGateway._MAX_SSE_BYTES = original_limit
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "error")
        self.assertEqual(events[0].error["code"], "MODEL_PROTOCOL_ERROR")

    def test_oversized_nonstream_body_is_a_protocol_error(self) -> None:
        chunks = (
            b"x" * (4 * 1024 * 1024),
            b"x" * (4 * 1024 * 1024 + 1),
        )
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(
                ScriptedTransport(HttpResponse(200, {}, chunks))
            ).generate(
                ModelRequest(
                    request_id="oversized-nonstream",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(context.exception.code, "MODEL_PROTOCOL_ERROR")
        self.assertFalse(context.exception.retryable)

    def test_generate_and_stream_enforce_the_request_stream_flag(self) -> None:
        generate_transport = ScriptedTransport(completion())
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(generate_transport).generate(
                ModelRequest(
                    request_id="wrong-generate-mode",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        self.assertEqual(context.exception.code, "MODEL_REQUEST_REJECTED")
        self.assertEqual(len(generate_transport.calls), 0)

        stream_transport = ScriptedTransport(HttpResponse(200, {}, []))
        events = list(
            self.gateway(stream_transport).stream(
                ModelRequest(
                    request_id="wrong-stream-mode",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=False,
                )
            )
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "error")
        self.assertEqual(events[0].error["code"], "MODEL_REQUEST_REJECTED")
        self.assertEqual(len(stream_transport.calls), 0)

    def test_total_deadline_reduces_attempt_timeout_and_never_oversleeps(self) -> None:
        sleeps: list[float] = []
        transport = ScriptedTransport(HttpResponse(503, {}, b"busy"), completion(content="ok"))
        result = self.gateway(
            transport,
            sleeps=sleeps,
            clock=Clock(0.0, 1.0, 1.0, 2.0, 2.0),
        ).generate(
            ModelRequest(
                request_id="deadline-remaining",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(sleeps, [0.25])
        self.assertEqual([call["timeout"] for call in transport.calls], [11.0, 10.0])

        sleeps = []
        too_long = ScriptedTransport(
            HttpResponse(429, {"Retry-After": "30"}, b"busy"),
            completion(content="must-not-run"),
        )
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway(
                too_long,
                sleeps=sleeps,
                clock=Clock(0.0, 1.0, 1.0),
            ).generate(
                ModelRequest(
                    request_id="deadline-no-oversleep",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(context.exception.code, "MODEL_TIMEOUT")
        self.assertTrue(context.exception.details["deadline_exceeded"])
        self.assertEqual(sleeps, [])
        self.assertEqual(len(too_long.calls), 1)

        late_success = ScriptedTransport(completion(content="too late"))
        with self.assertRaises(ModelGatewayError) as late_context:
            self.gateway(
                late_success,
                clock=Clock(0.0, 1.0, 13.0, 13.0),
            ).generate(
                ModelRequest(
                    request_id="deadline-late-success",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(late_context.exception.code, "MODEL_TIMEOUT")
        self.assertTrue(late_context.exception.details["deadline_exceeded"])

    def test_stream_total_deadline_becomes_one_terminal_error_event(self) -> None:
        chunks = [
            b'data: {"model":"qwen-private","choices":[{"index":0,"delta":{"content":"late"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        events = list(
            self.gateway(
                ScriptedTransport(HttpResponse(200, {}, chunks)),
                clock=Clock(0.0, 0.0, 0.0, 13.0),
            ).stream(
                ModelRequest(
                    request_id="stream-deadline",
                    messages=(ChatMessage(role="user", content="hello"),),
                    stream=True,
                )
            )
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "error")
        self.assertEqual(events[0].error["code"], "MODEL_TIMEOUT")
        self.assertTrue(events[0].error["details"]["deadline_exceeded"])

    def test_negative_retry_after_is_ignored_in_favor_of_backoff(self) -> None:
        sleeps: list[float] = []
        transport = ScriptedTransport(
            HttpResponse(429, {"Retry-After": "-5"}, b"busy"),
            completion(content="ok"),
        )
        result = self.gateway(transport, sleeps=sleeps).generate(
            ModelRequest(
                request_id="negative-retry-after",
                messages=(ChatMessage(role="user", content="hello"),),
            )
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(sleeps, [0.25])

    def test_provider_requires_model_and_exactly_one_index_zero_choice(self) -> None:
        missing_model = {
            "id": "missing-model",
            "choices": [
                {"index": 0, "message": {"content": "hello"}, "finish_reason": "stop"}
            ],
        }
        missing_index = {
            "id": "missing-index",
            "model": "qwen-private",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        }
        for label, payload in [("model", missing_model), ("index", missing_index)]:
            with self.subTest(label):
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(ScriptedTransport(json_response(payload))).generate(
                        ModelRequest(
                            request_id=f"required-{label}",
                            messages=(ChatMessage(role="user", content="hello"),),
                        )
                    )
                self.assertEqual(context.exception.code, "MODEL_PROTOCOL_ERROR")

    def test_stream_requires_model_finish_reason_done_and_usage_only_empty_choice(self) -> None:
        cases = {
            "missing-model": [
                b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            "missing-finish": [
                b'data: {"model":"qwen-private","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            "missing-done": [
                b'data: {"model":"qwen-private","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
            ],
            "empty-choice-without-usage": [
                b'data: {"model":"qwen-private","choices":[]}\n\n',
                b"data: [DONE]\n\n",
            ],
        }
        for label, chunks in cases.items():
            with self.subTest(label):
                events = list(
                    self.gateway(ScriptedTransport(HttpResponse(200, {}, chunks))).stream(
                        ModelRequest(
                            request_id=f"stream-{label}",
                            messages=(ChatMessage(role="user", content="hello"),),
                            stream=True,
                        )
                    )
                )
                self.assertEqual(events[-1].event_type, "error")
                self.assertEqual(events[-1].error["code"], "MODEL_PROTOCOL_ERROR")
                self.assertEqual(sum(event.event_type == "completed" for event in events), 0)

    def test_json_depth_and_node_limits_are_enforced(self) -> None:
        normal = {
            "id": "limited",
            "model": "qwen-private",
            "choices": [
                {"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}
            ],
        }
        deep = dict(normal)
        nested: Any = "leaf"
        for _ in range(70):
            nested = [nested]
        deep["extra"] = nested

        with self.assertRaises(ModelGatewayError) as depth_context:
            self.gateway(ScriptedTransport(json_response(deep))).generate(
                ModelRequest(
                    request_id="json-depth",
                    messages=(ChatMessage(role="user", content="hello"),),
                )
            )
        self.assertEqual(depth_context.exception.code, "MODEL_PROTOCOL_ERROR")

        original_nodes = VllmModelGateway._MAX_JSON_NODES
        VllmModelGateway._MAX_JSON_NODES = 5
        try:
            with self.assertRaises(ModelGatewayError) as node_context:
                self.gateway(ScriptedTransport(json_response(normal))).generate(
                    ModelRequest(
                        request_id="json-nodes",
                        messages=(ChatMessage(role="user", content="hello"),),
                    )
                )
        finally:
            VllmModelGateway._MAX_JSON_NODES = original_nodes
        self.assertEqual(node_context.exception.code, "MODEL_PROTOCOL_ERROR")

    def test_api_key_rejects_every_ascii_control_character(self) -> None:
        for character in ("\t", "\x1f", "\x7f"):
            with self.subTest(codepoint=ord(character)):
                transport = ScriptedTransport(completion())
                with self.assertRaises(ModelGatewayError) as context:
                    self.gateway(transport, api_key=f"token{character}injected").generate(
                        ModelRequest(
                            request_id=f"header-control-{ord(character)}",
                            messages=(ChatMessage(role="user", content="hello"),),
                        )
                    )
                self.assertEqual(context.exception.code, "MODEL_REQUEST_REJECTED")
                self.assertEqual(len(transport.calls), 0)


if __name__ == "__main__":
    unittest.main()
