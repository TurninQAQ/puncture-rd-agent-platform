"""Provider-neutral model gateway and the vLLM production adapter."""

from __future__ import annotations

import codecs
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import math
import random
import ssl
import time
from typing import Any, Protocol

from .http_transport import (
    create_default_transport,
    HttpResponse,
    HttpTransport,
    TransportSecurityError,
    validate_base_url,
)
from .models import (
    GatewayHealth,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    TokenUsage,
    ToolCall,
    VllmGatewayConfig,
    validate_json_schema_subset,
)


class ModelGatewayError(RuntimeError):
    """Normalized model provider failure consumed by Agent Runtime retry logic."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class ModelGateway(Protocol):
    """Contract implemented by both MockQwenGateway and VllmModelGateway."""

    def health(self) -> GatewayHealth: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]: ...


@dataclass
class _StreamingToolBuffer:
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class VllmModelGateway:
    """OpenAI-compatible adapter for a privately deployed Qwen/vLLM server.

    The optional collaborators are intentionally keyword-only.  Production code
    uses the standard-library transport and clocks; tests can inject deterministic
    transports, sleep functions, and clocks without starting vLLM or using a GPU.
    """

    _PROVIDER = "vllm"
    _MAX_BACKOFF_SECONDS = 30.0
    _BASE_BACKOFF_SECONDS = 0.25
    _MAX_SSE_BYTES = 64 * 1024 * 1024
    _MAX_SSE_EVENT_CHARS = 2 * 1024 * 1024
    _MAX_JSON_DEPTH = 64
    _MAX_JSON_NODES = 100_000

    def __init__(
        self,
        config: VllmGatewayConfig,
        *,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        jitter: Callable[[], float] | None = None,
        wall_time: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        validate_base_url(config.base_url)
        self._transport = transport or create_default_transport(
            config.base_url,
            config.timeout_seconds,
            config.ca_bundle_path,
        )
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._jitter = jitter or random.random
        self._wall_time = wall_time or time.time

    def close(self) -> None:
        """Release pooled transport resources during service shutdown."""

        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "VllmModelGateway":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def health(self) -> GatewayHealth:
        """Probe the lightweight model-list endpoint without generating tokens."""

        timeout = min(self.config.timeout_seconds, 5.0)
        deadline = self._monotonic() + timeout
        try:
            response, attempts = self._request_with_retry(
                "GET",
                "/models",
                headers=self._headers(request_id="health", accept="application/json"),
                json_body=None,
                timeout=timeout,
                stream=False,
                max_retries=0,
                deadline=deadline,
            )
        except ModelGatewayError as exc:
            return GatewayHealth(
                status="DOWN",
                model=self.config.model,
                provider=self._PROVIDER,
                details={
                    "endpoint_reachable": False,
                    "error_code": exc.code,
                    "attempts": int(exc.details.get("attempts", 1)),
                },
            )

        try:
            payload = self._decode_json_response(response)
            raw_models = payload.get("data")
            if not isinstance(raw_models, list):
                raise ValueError("models response data must be a list")
            model_ids = [
                item.get("id")
                for item in raw_models
                if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            ]
        except (ModelGatewayError, ValueError):
            return GatewayHealth(
                status="DEGRADED",
                model=self.config.model,
                provider=self._PROVIDER,
                details={
                    "endpoint_reachable": True,
                    "model_check": "invalid_models_response",
                    "attempts": attempts,
                },
            )

        if self.config.model not in model_ids:
            return GatewayHealth(
                status="DOWN",
                model=self.config.model,
                provider=self._PROVIDER,
                details={
                    "endpoint_reachable": True,
                    "model_check": "configured_model_not_served",
                    "available_model_count": len(model_ids),
                    "attempts": attempts,
                },
            )
        return GatewayHealth(
            status="UP",
            model=self.config.model,
            provider=self._PROVIDER,
            details={
                "endpoint_reachable": True,
                "model_check": "configured_model_served",
                "available_model_count": len(model_ids),
                "attempts": attempts,
            },
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Perform one bounded-retry, non-streaming chat completion."""

        if request.stream:
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "generate requires request.stream to be false",
                details={"attempts": 0},
            )
        started = self._monotonic()
        response, attempts = self._request_with_retry(
            "POST",
            "/chat/completions",
            headers=self._headers(request.request_id, "application/json"),
            json_body=self._serialize_request(request, stream=False),
            timeout=self.config.timeout_seconds,
            stream=False,
            deadline=started + self.config.timeout_seconds,
        )
        try:
            payload = self._decode_json_response(response)
            latency_ms = max(0.0, (self._monotonic() - started) * 1000.0)
            return self._parse_completion(payload, request, latency_ms)
        except ModelGatewayError as exc:
            self._add_error_context(exc, attempts=attempts, provider_status=response.status)
            raise

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        """Emit normalized data events followed by one terminal success/error.

        Streaming failures are represented only as a structured terminal error
        event.  They are not also raised to the caller, avoiding ambiguous retry
        behavior after partial text has already been observed.
        """

        next_sequence = 0
        output_visible = False
        started = self._monotonic()
        stream_deadline = started + self.config.timeout_seconds
        attempts_consumed = 0
        maximum_attempts = 1 + self.config.max_retries
        try:
            if not request.stream:
                raise ModelGatewayError(
                    "MODEL_REQUEST_REJECTED",
                    "stream requires request.stream to be true",
                    details={"attempts": 0},
                )
            while True:
                remaining_attempts = maximum_attempts - attempts_consumed
                try:
                    for event in self._stream_success(
                        request,
                        started=started,
                        stream_deadline=stream_deadline,
                        max_retries=max(0, remaining_attempts - 1),
                    ):
                        next_sequence = event.sequence + 1
                        if event.event_type in {"text_delta", "tool_call"}:
                            output_visible = True
                        yield event
                    return
                except ModelGatewayError as exc:
                    local_attempts = int(exc.details.get("attempts", 0))
                    total_attempts = attempts_consumed + max(0, local_attempts)
                    exc.details["attempts"] = total_attempts
                    if (
                        exc.retryable
                        and not output_visible
                        and total_attempts < maximum_attempts
                    ):
                        delay = self._backoff_seconds(max(1, total_attempts))
                        time_left = stream_deadline - self._monotonic()
                        if time_left > 0 and delay < time_left:
                            self._sleep(delay)
                            attempts_consumed = total_attempts
                            continue
                        exc = ModelGatewayError(
                            "MODEL_TIMEOUT",
                            "model stream exceeded its total deadline",
                            retryable=True,
                            details={
                                **exc.details,
                                "retry_exhausted": True,
                                "deadline_exceeded": True,
                            },
                        )
                    elif exc.retryable and not output_visible:
                        exc.details["retry_exhausted"] = True
                    raise exc
        except ModelGatewayError as exc:
            details = dict(exc.details)
            details["output_visible"] = output_visible
            retryable = exc.retryable
            if output_visible and retryable:
                details["upstream_retryable"] = True
                retryable = False
            normalized = ModelGatewayError(
                exc.code,
                "model streaming request failed",
                retryable=retryable,
                details=details,
            )
            yield ModelStreamEvent(
                request_id=request.request_id,
                event_type="error",
                sequence=next_sequence,
                error=self._stream_error_payload(normalized),
            )
        except Exception:
            # Provider/transport objects must never escape this boundary, and raw
            # exception text may contain URLs or credentials.
            error = ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "unexpected model streaming failure",
                retryable=False,
                details={"output_visible": output_visible},
            )
            yield ModelStreamEvent(
                request_id=request.request_id,
                event_type="error",
                sequence=next_sequence,
                error=self._stream_error_payload(error),
            )

    def _stream_success(
        self,
        request: ModelRequest,
        *,
        started: float,
        stream_deadline: float,
        max_retries: int,
    ) -> Iterator[ModelStreamEvent]:
        """Internal success iterator; the public wrapper normalizes failures."""

        response, attempts = self._request_with_retry(
            "POST",
            "/chat/completions",
            headers=self._headers(request.request_id, "text/event-stream"),
            json_body=self._serialize_request(request, stream=True),
            timeout=self.config.timeout_seconds,
            stream=True,
            max_retries=max_retries,
            deadline=stream_deadline,
        )

        sequence = 0
        text_parts: list[str] = []
        tool_buffers: dict[int, _StreamingToolBuffer] = {}
        emitted_tools: tuple[ToolCall, ...] | None = None
        provider_response_id: str | None = None
        actual_model: str | None = None
        finish_reason: str | None = None
        usage = TokenUsage(0, 0, 0, usage_known=False)
        done_seen = False

        try:
            for raw_data in self._iter_sse_data(response.iter_bytes()):
                if self._monotonic() >= stream_deadline:
                    raise ModelGatewayError(
                        "MODEL_TIMEOUT",
                        "model stream exceeded its total deadline",
                        retryable=True,
                        details={"deadline_exceeded": True},
                    )
                if raw_data == "[DONE]":
                    done_seen = True
                    break
                payload = self._decode_sse_payload(raw_data)
                if "error" in payload:
                    raise ModelGatewayError(
                        "MODEL_PROTOCOL_ERROR",
                        "provider returned an error event during model streaming",
                    )

                raw_id = payload.get("id")
                if isinstance(raw_id, str) and raw_id:
                    provider_response_id = raw_id
                raw_model = payload.get("model")
                if not isinstance(raw_model, str) or not raw_model:
                    raise self._protocol_error("provider stream frame is missing its model")
                if raw_model != self.config.model:
                    raise self._protocol_error("provider response model does not match configuration")
                actual_model = raw_model
                raw_usage = payload.get("usage")
                if raw_usage is not None:
                    usage = self._parse_usage(raw_usage)

                choices = payload.get("choices", [])
                if not isinstance(choices, list):
                    raise self._protocol_error("stream choices must be a list")
                if len(choices) > 1:
                    raise self._protocol_error("provider returned multiple stream choices")
                if not choices and raw_usage is None:
                    raise self._protocol_error("a zero-choice stream frame must contain token usage")
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        raise self._protocol_error("stream choice must be an object")
                    choice_index = choice.get("index")
                    if not isinstance(choice_index, int) or isinstance(choice_index, bool) or choice_index != 0:
                        raise self._protocol_error("provider stream choice must have index zero")
                    delta = choice.get("delta", {})
                    if delta is None:
                        delta = {}
                    if not isinstance(delta, Mapping):
                        raise self._protocol_error("stream delta must be an object")

                    content = self._content_text(delta.get("content"), allow_none=True)
                    if content:
                        text_parts.append(content)
                        yield ModelStreamEvent(
                            request_id=request.request_id,
                            event_type="text_delta",
                            sequence=sequence,
                            delta=content,
                        )
                        sequence += 1

                    self._accumulate_stream_tool_calls(tool_buffers, delta.get("tool_calls"))
                    raw_finish = choice.get("finish_reason")
                    if raw_finish is not None:
                        parsed_finish = self._normalize_finish_reason(raw_finish)
                        if finish_reason is not None and finish_reason != parsed_finish:
                            raise self._protocol_error("provider returned conflicting finish reasons")
                        finish_reason = parsed_finish

            if not done_seen:
                raise self._protocol_error("provider stream ended without a DONE event")
            if finish_reason is None:
                raise self._protocol_error("provider stream ended without a finish reason")
            if actual_model is None:
                raise self._protocol_error("provider stream ended without a model identifier")
            if tool_buffers and finish_reason != "tool_calls":
                raise self._protocol_error(
                    "provider stream emitted tool calls with an inconsistent finish reason"
                )
            if finish_reason == "tool_calls" and not tool_buffers:
                raise self._protocol_error(
                    "provider stream declared tool calls without tool call data"
                )
            if tool_buffers and emitted_tools is None:
                emitted_tools = self._finish_stream_tool_calls(tool_buffers, request)
                for tool_call in emitted_tools:
                    yield ModelStreamEvent(
                        request_id=request.request_id,
                        event_type="tool_call",
                        sequence=sequence,
                        tool_call=tool_call,
                    )
                    sequence += 1
            tool_calls = emitted_tools or ()
            text = "".join(text_parts)
            structured_output: Mapping[str, Any] | None = None
            normalized_text: str | None = text if text else None
            if request.response_schema is not None and not tool_calls:
                structured_output = self._parse_structured_output(text, request.response_schema)
                normalized_text = None

            if normalized_text is None and not tool_calls and structured_output is None:
                raise ModelGatewayError(
                    "EMPTY_MODEL_RESPONSE",
                    "provider returned no usable model output",
                )

            latency_ms = max(0.0, (self._monotonic() - started) * 1000.0)
            final_response = ModelResponse(
                request_id=request.request_id,
                model=actual_model,
                finish_reason=finish_reason,
                text=normalized_text,
                tool_calls=tool_calls,
                structured_output=structured_output,
                usage=usage,
                latency_ms=latency_ms,
                raw_response_id=provider_response_id,
            )
            yield ModelStreamEvent(
                request_id=request.request_id,
                event_type="completed",
                sequence=sequence,
                response=final_response,
            )
        except ModelGatewayError as exc:
            self._add_error_context(exc, attempts=attempts, provider_status=response.status)
            raise
        except TransportSecurityError as exc:
            error = ModelGatewayError(
                exc.code,
                "model stream transport security validation failed",
                retryable=False,
                details={"attempts": attempts, "provider_status": response.status},
            )
            raise error from exc
        except (ssl.SSLError, ssl.CertificateError) as exc:
            error = ModelGatewayError(
                "MODEL_TLS_ERROR",
                "model stream TLS validation failed",
                retryable=False,
                details={"attempts": attempts, "provider_status": response.status},
            )
            raise error from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            code = "MODEL_TIMEOUT" if isinstance(exc, TimeoutError) else "MODEL_UNAVAILABLE"
            error = ModelGatewayError(
                code,
                "model stream timed out" if code == "MODEL_TIMEOUT" else "model stream disconnected",
                retryable=True,
                details={"attempts": attempts, "provider_status": response.status},
            )
            raise error from exc
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            error = self._protocol_error("malformed provider SSE stream")
            self._add_error_context(error, attempts=attempts, provider_status=response.status)
            raise error from exc

    def _serialize_request(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        if request.tools and request.response_schema is not None:
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "tools and response_schema cannot be requested in the same completion",
                details={"attempts": 0},
            )
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            serialized: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name is not None:
                serialized["name"] = message.name
            if message.tool_call_id is not None:
                serialized["tool_call_id"] = message.tool_call_id
            assistant_calls = getattr(message, "assistant_tool_calls", None)
            if assistant_calls is None:
                assistant_calls = getattr(message, "tool_calls", None)
            if assistant_calls:
                serialized["tool_calls"] = [self._serialize_history_tool_call(call) for call in assistant_calls]
            messages.append(serialized)

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "n": 1,
            "stream": stream,
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._json_copy(tool.input_schema),
                    },
                }
                for tool in request.tools
            ]
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_response",
                    "strict": True,
                    "schema": self._json_copy(request.response_schema),
                },
            }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _parse_completion(
        self,
        payload: Mapping[str, Any],
        request: ModelRequest,
        latency_ms: float,
    ) -> ModelResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise self._protocol_error("completion response requires a choice object")
        if len(choices) != 1:
            raise self._protocol_error("provider must return exactly one completion choice")
        choice = choices[0]
        choice_index = choice.get("index")
        if not isinstance(choice_index, int) or isinstance(choice_index, bool) or choice_index != 0:
            raise self._protocol_error("completion choice must have index zero")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise self._protocol_error("completion choice requires an assistant message")
        if message.get("role") != "assistant":
            raise self._protocol_error("completion message role must be assistant")

        tool_calls = self._parse_tool_calls(message.get("tool_calls"), request)
        content = self._content_text(message.get("content"), allow_none=True)
        structured_output: Mapping[str, Any] | None = None
        text: str | None = content if content else None
        if request.response_schema is not None and not tool_calls:
            structured_output = self._parse_structured_output(content, request.response_schema)
            text = None

        if text is None and not tool_calls and structured_output is None:
            raise ModelGatewayError(
                "EMPTY_MODEL_RESPONSE",
                "provider returned no usable model output",
            )

        finish_reason = self._normalize_finish_reason(choice.get("finish_reason"))
        if tool_calls and finish_reason != "tool_calls":
            raise self._protocol_error(
                "provider returned tool calls with an inconsistent finish reason"
            )
        if finish_reason == "tool_calls" and not tool_calls:
            raise self._protocol_error(
                "provider declared tool calls without tool call data"
            )
        raw_model = payload.get("model")
        if not isinstance(raw_model, str) or not raw_model:
            raise self._protocol_error("provider response model is required")
        if raw_model != self.config.model:
            raise self._protocol_error("provider response model does not match configuration")
        actual_model = raw_model
        raw_response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
        return ModelResponse(
            request_id=request.request_id,
            model=actual_model,
            finish_reason=finish_reason,
            text=text,
            tool_calls=tool_calls,
            structured_output=structured_output,
            usage=self._parse_usage(payload.get("usage")),
            latency_ms=latency_ms,
            raw_response_id=raw_response_id,
        )

    def _parse_tool_calls(self, raw_calls: Any, request: ModelRequest) -> tuple[ToolCall, ...]:
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, list):
            raise self._protocol_error("assistant tool_calls must be a list")
        tools = {tool.name: tool for tool in request.tools}
        result: list[ToolCall] = []
        seen_call_ids: set[str] = set()
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise self._protocol_error("assistant tool call must be an object")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id or not isinstance(function, Mapping):
                raise self._protocol_error("assistant tool call requires id and function")
            if raw_call.get("type", "function") != "function":
                raise self._protocol_error("assistant tool call type must be function")
            if call_id in seen_call_ids:
                raise self._protocol_error("provider returned duplicate tool call IDs")
            seen_call_ids.add(call_id)
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise self._protocol_error("assistant tool call requires a function name")
            if name not in tools:
                raise ModelGatewayError(
                    "UNKNOWN_TOOL",
                    "provider selected a tool that was not offered",
                    details={"available_tools": sorted(tools)},
                )
            arguments = self._parse_tool_arguments(function.get("arguments"))
            try:
                validate_json_schema_subset(arguments, tools[name].input_schema)
            except ValueError as exc:
                raise ModelGatewayError(
                    "TOOL_ARGUMENT_SCHEMA_ERROR",
                    "provider tool arguments do not match the offered schema",
                ) from exc
            result.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
        return tuple(result)

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        if not isinstance(raw_arguments, str):
            raise ModelGatewayError(
                "TOOL_ARGUMENT_PARSE_ERROR",
                "provider tool arguments must be a JSON object string",
            )
        try:
            parsed = VllmModelGateway._strict_json_loads(raw_arguments)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ModelGatewayError(
                "TOOL_ARGUMENT_PARSE_ERROR",
                "provider tool arguments are not valid JSON",
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ModelGatewayError(
                "TOOL_ARGUMENT_PARSE_ERROR",
                "provider tool arguments must decode to an object",
            )
        return dict(parsed)

    def _parse_structured_output(
        self,
        content: Any,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if isinstance(content, Mapping):
            candidate: Any = dict(content)
        elif isinstance(content, str) and content.strip():
            try:
                candidate = self._strict_json_loads(content)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ModelGatewayError(
                    "STRUCTURED_OUTPUT_SCHEMA_ERROR",
                    "provider structured output is not valid JSON",
                ) from exc
        else:
            raise ModelGatewayError(
                "STRUCTURED_OUTPUT_SCHEMA_ERROR",
                "provider did not return the required structured output",
            )
        try:
            validate_json_schema_subset(candidate, schema)
        except ValueError as exc:
            raise ModelGatewayError(
                "STRUCTURED_OUTPUT_SCHEMA_ERROR",
                "provider structured output does not match the requested schema",
            ) from exc
        if not isinstance(candidate, Mapping):
            # ModelRequest currently constrains top-level response schemas to an
            # object, but this check keeps the adapter fail-closed if that changes.
            raise ModelGatewayError(
                "STRUCTURED_OUTPUT_SCHEMA_ERROR",
                "provider structured output must be an object",
            )
        return dict(candidate)

    @staticmethod
    def _parse_usage(raw_usage: Any) -> TokenUsage:
        if raw_usage is None:
            return TokenUsage(0, 0, 0, usage_known=False)
        if not isinstance(raw_usage, Mapping):
            raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider usage must be an object")
        required_fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
        if not required_fields.issubset(raw_usage):
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "provider usage is missing required token counts",
            )
        prompt_value = raw_usage["prompt_tokens"]
        completion_value = raw_usage["completion_tokens"]
        total_value = raw_usage["total_tokens"]
        if not isinstance(prompt_value, int) or isinstance(prompt_value, bool):
            raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider usage is malformed")
        if not isinstance(completion_value, int) or isinstance(completion_value, bool):
            raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider usage is malformed")
        if not isinstance(total_value, int) or isinstance(total_value, bool):
            raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider usage is malformed")
        prompt = prompt_value
        completion = completion_value
        total = total_value
        if prompt < 0 or completion < 0 or total < 0 or total != prompt + completion:
            raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider token usage is inconsistent")
        return TokenUsage(prompt, completion, total, usage_known=True)

    def _accumulate_stream_tool_calls(
        self,
        buffers: dict[int, _StreamingToolBuffer],
        raw_calls: Any,
    ) -> None:
        if raw_calls is None:
            return
        if not isinstance(raw_calls, list):
            raise self._protocol_error("stream tool_calls must be a list")
        for position, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                raise self._protocol_error("stream tool call delta must be an object")
            raw_index = raw_call.get("index", position)
            if not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0:
                raise self._protocol_error("stream tool call index must be a non-negative integer")
            buffer = buffers.setdefault(raw_index, _StreamingToolBuffer(index=raw_index))
            call_id = raw_call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise self._protocol_error("stream tool call id must be a string")
                if not buffer.call_id:
                    buffer.call_id = call_id
                elif call_id != buffer.call_id:
                    buffer.call_id += call_id
            function = raw_call.get("function")
            if function is not None:
                if not isinstance(function, Mapping):
                    raise self._protocol_error("stream tool function must be an object")
                name = function.get("name")
                if name is not None:
                    if not isinstance(name, str):
                        raise self._protocol_error("stream tool name fragment must be a string")
                    buffer.name += name
                arguments = function.get("arguments")
                if arguments is not None:
                    if not isinstance(arguments, str):
                        raise self._protocol_error("stream tool arguments fragment must be a string")
                    buffer.arguments += arguments

    def _finish_stream_tool_calls(
        self,
        buffers: Mapping[int, _StreamingToolBuffer],
        request: ModelRequest,
    ) -> tuple[ToolCall, ...]:
        raw_calls = [
            {
                "id": buffer.call_id,
                "type": "function",
                "function": {"name": buffer.name, "arguments": buffer.arguments},
            }
            for _, buffer in sorted(buffers.items())
        ]
        return self._parse_tool_calls(raw_calls, request)

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None,
        timeout: float,
        stream: bool,
        max_retries: int | None = None,
        deadline: float | None = None,
    ) -> tuple[HttpResponse, int]:
        retry_limit = self.config.max_retries if max_retries is None else max_retries
        if deadline is None:
            deadline = self._monotonic() + timeout
        attempt = 0
        while True:
            remaining_timeout = deadline - self._monotonic()
            if remaining_timeout <= 0:
                raise ModelGatewayError(
                    "MODEL_TIMEOUT",
                    "model provider request exceeded its total deadline",
                    retryable=True,
                    details={
                        "attempts": attempt,
                        "retry_exhausted": True,
                        "deadline_exceeded": True,
                    },
                )
            attempt += 1
            retry_after: float | None = None
            try:
                response = self._transport.request(
                    method,
                    path,
                    headers=headers,
                    json_body=json_body,
                    timeout=min(timeout, remaining_timeout),
                    stream=stream,
                )
                if self._monotonic() >= deadline:
                    raise ModelGatewayError(
                        "MODEL_TIMEOUT",
                        "model provider request exceeded its total deadline",
                        retryable=True,
                        details={"deadline_exceeded": True},
                    )
                if 200 <= response.status < 300:
                    return response, attempt
                error = self._error_for_status(response.status)
                retry_after = self._retry_after_seconds(response.headers)
                error.details["provider_status"] = response.status
            except TimeoutError as exc:
                error = ModelGatewayError(
                    "MODEL_TIMEOUT",
                    "model provider request timed out",
                    retryable=True,
                )
                cause: BaseException | None = exc
            except TransportSecurityError as exc:
                error = ModelGatewayError(
                    exc.code,
                    "model provider transport security validation failed",
                    retryable=False,
                )
                cause = exc
            except (ssl.SSLError, ssl.CertificateError) as exc:
                error = ModelGatewayError(
                    "MODEL_TLS_ERROR",
                    "model provider TLS validation failed",
                    retryable=False,
                )
                cause = exc
            except (ConnectionError, OSError) as exc:
                error = ModelGatewayError(
                    "MODEL_UNAVAILABLE",
                    "model provider is unavailable",
                    retryable=True,
                )
                cause = exc
            except ModelGatewayError as exc:
                error = exc
                cause = exc
            else:
                cause = None

            error.details["attempts"] = attempt
            if not error.retryable or attempt > retry_limit:
                error.details["retry_exhausted"] = bool(error.retryable and attempt > retry_limit)
                if cause is not None and cause is not error:
                    raise error from cause
                raise error
            delay = retry_after if retry_after is not None else self._backoff_seconds(attempt)
            error.details["retry_delay_seconds"] = delay
            time_left_before_sleep = deadline - self._monotonic()
            if time_left_before_sleep <= 0 or delay >= time_left_before_sleep:
                details = dict(error.details)
                details.update(
                    {
                        "retry_exhausted": True,
                        "deadline_exceeded": True,
                    }
                )
                timeout_error = ModelGatewayError(
                    "MODEL_TIMEOUT",
                    "model provider request exceeded its total deadline",
                    retryable=True,
                    details=details,
                )
                if cause is not None and cause is not error:
                    raise timeout_error from cause
                raise timeout_error from error
            self._sleep(delay)

    @staticmethod
    def _error_for_status(status: int) -> ModelGatewayError:
        if 300 <= status < 400:
            return ModelGatewayError(
                "MODEL_SECURITY_ERROR",
                "model provider redirect was rejected",
                retryable=False,
            )
        if status == 408:
            return ModelGatewayError(
                "MODEL_TIMEOUT",
                "model provider request timed out",
                retryable=True,
            )
        if status == 429:
            return ModelGatewayError(
                "MODEL_RATE_LIMITED",
                "model provider rate limit exceeded",
                retryable=True,
            )
        if status in {500, 502, 503, 504}:
            return ModelGatewayError(
                "MODEL_UNAVAILABLE",
                "model provider is temporarily unavailable",
                retryable=True,
            )
        if status in {401, 403}:
            return ModelGatewayError(
                "MODEL_PERMISSION_DENIED",
                "model provider rejected the gateway credentials",
            )
        return ModelGatewayError(
            "MODEL_REQUEST_REJECTED",
            "model provider rejected the request",
        )

    def _backoff_seconds(self, failed_attempt: int) -> float:
        base = min(
            self._MAX_BACKOFF_SECONDS,
            self._BASE_BACKOFF_SECONDS * (2 ** max(0, failed_attempt - 1)),
        )
        try:
            jitter = min(1.0, max(0.0, float(self._jitter())))
        except (TypeError, ValueError):
            jitter = 0.0
        return min(self._MAX_BACKOFF_SECONDS, base * (1.0 + 0.2 * jitter))

    def _retry_after_seconds(self, headers: Mapping[str, str]) -> float | None:
        raw_value = self._header(headers, "retry-after")
        if raw_value is None:
            return None
        try:
            seconds = float(raw_value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw_value).timestamp()
                seconds = retry_at - self._wall_time()
            except (TypeError, ValueError, OverflowError):
                return None
        if seconds < 0:
            return None
        return min(self._MAX_BACKOFF_SECONDS, seconds)

    def _headers(self, request_id: str, accept: str) -> dict[str, str]:
        self._validate_header_value(request_id, "request_id")
        if self.config.api_key is not None:
            self._validate_header_value(self.config.api_key, "api_key")
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
            "User-Agent": "puncture-rd-agent-platform/1",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _validate_header_value(value: str, field_name: str) -> None:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                f"{field_name} contains prohibited header characters",
                details={"attempts": 0},
            )

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        lowered = name.lower()
        for key, value in headers.items():
            if key.lower() == lowered:
                return str(value)
        return None

    @staticmethod
    def _decode_json_response(response: HttpResponse) -> Mapping[str, Any]:
        try:
            raw = response.read()
            payload = VllmModelGateway._strict_json_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "model provider returned malformed JSON",
            ) from exc
        if not isinstance(payload, Mapping):
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "model provider JSON response must be an object",
            )
        return payload

    @staticmethod
    def _decode_sse_payload(raw_data: str) -> Mapping[str, Any]:
        try:
            payload = VllmModelGateway._strict_json_loads(raw_data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "model provider returned malformed SSE JSON",
            ) from exc
        if not isinstance(payload, Mapping):
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "model provider SSE payload must be an object",
            )
        return payload

    @staticmethod
    def _iter_sse_data(chunks: Iterator[bytes]) -> Iterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        data_lines: list[str] = []
        total_bytes = 0

        def process_line(line: str) -> str | None:
            nonlocal data_lines
            if line == "":
                if not data_lines:
                    return None
                value = "\n".join(data_lines)
                data_lines = []
                return value
            if line.startswith(":"):
                return None
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            return None

        for chunk in chunks:
            total_bytes += len(chunk)
            if total_bytes > VllmModelGateway._MAX_SSE_BYTES:
                raise ValueError("provider SSE stream exceeds the size limit")
            buffer += decoder.decode(chunk)
            if len(buffer) > VllmModelGateway._MAX_SSE_EVENT_CHARS:
                raise ValueError("provider SSE event exceeds the size limit")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                value = process_line(line.rstrip("\r"))
                if value is not None:
                    if len(value) > VllmModelGateway._MAX_SSE_EVENT_CHARS:
                        raise ValueError("provider SSE event exceeds the size limit")
                    yield value
        buffer += decoder.decode(b"", final=True)
        if buffer:
            value = process_line(buffer.rstrip("\r"))
            if value is not None:
                yield value
        if data_lines:
            value = "\n".join(data_lines)
            if len(value) > VllmModelGateway._MAX_SSE_EVENT_CHARS:
                raise ValueError("provider SSE event exceeds the size limit")
            yield value

    @staticmethod
    def _content_text(raw_content: Any, *, allow_none: bool) -> str:
        if raw_content is None and allow_none:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for part in raw_content:
                if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                    raise ModelGatewayError(
                        "MODEL_PROTOCOL_ERROR",
                        "provider message content parts are malformed",
                    )
                parts.append(part["text"])
            return "".join(parts)
        if isinstance(raw_content, Mapping):
            # Structured-output backends may return an already-decoded object.
            try:
                return json.dumps(
                    raw_content,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ModelGatewayError(
                    "MODEL_PROTOCOL_ERROR",
                    "provider structured content contains non-JSON values",
                ) from exc
        raise ModelGatewayError("MODEL_PROTOCOL_ERROR", "provider message content is malformed")

    @staticmethod
    def _normalize_finish_reason(raw_reason: Any) -> str:
        if raw_reason is None:
            return "stop"
        mapping = {
            "stop": "stop",
            "eos": "stop",
            "eos_token": "stop",
            "tool_calls": "tool_calls",
            "function_call": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
            "error": "error",
        }
        normalized = mapping.get(str(raw_reason))
        if normalized is None:
            raise ModelGatewayError(
                "MODEL_PROTOCOL_ERROR",
                "provider returned an unsupported finish reason",
            )
        return normalized

    @staticmethod
    def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
        # Also proves at request time that schemas contain JSON-serializable data.
        try:
            copied = VllmModelGateway._strict_json_loads(
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "request schema or tool history is not valid finite JSON",
                details={"attempts": 0},
            ) from exc
        if not isinstance(copied, dict):
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "request schema must serialize to a JSON object",
                details={"attempts": 0},
            )
        return copied

    @staticmethod
    def _serialize_history_tool_call(call: Any) -> dict[str, Any]:
        """Serialize either the repository ToolCall or a compatible mapping."""

        if isinstance(call, Mapping):
            if "function" in call:
                return VllmModelGateway._json_copy(call)
            call_id = call.get("call_id", call.get("id"))
            name = call.get("name")
            arguments = call.get("arguments")
        else:
            call_id = getattr(call, "call_id", None)
            name = getattr(call, "name", None)
            arguments = getattr(call, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("assistant history tool call requires a call_id")
        if not isinstance(name, str) or not name:
            raise ValueError("assistant history tool call requires a name")
        if not isinstance(arguments, Mapping):
            raise ValueError("assistant history tool call arguments must be an object")
        safe_arguments = VllmModelGateway._json_copy(arguments)
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    safe_arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        }

    @staticmethod
    def _strict_json_loads(raw: str) -> Any:
        """Decode JSON while rejecting duplicate keys and non-finite numbers."""

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("JSON object contains a duplicate key")
                result[key] = value
            return result

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("JSON number is not finite")
            return parsed

        def reject_constant(_: str) -> Any:
            raise ValueError("JSON constants must be finite")

        try:
            parsed = json.loads(
                raw,
                object_pairs_hook=unique_object,
                parse_float=finite_float,
                parse_constant=reject_constant,
            )
        except RecursionError as exc:
            raise ValueError("JSON nesting exceeds the parser limit") from exc
        VllmModelGateway._validate_json_complexity(parsed)
        return parsed

    @staticmethod
    def _validate_json_complexity(value: Any) -> None:
        stack: list[tuple[Any, int]] = [(value, 0)]
        node_count = 0
        while stack:
            node, depth = stack.pop()
            node_count += 1
            if node_count > VllmModelGateway._MAX_JSON_NODES:
                raise ValueError("JSON document exceeds the node limit")
            if depth > VllmModelGateway._MAX_JSON_DEPTH:
                raise ValueError("JSON document exceeds the nesting limit")
            if isinstance(node, Mapping):
                stack.extend((item, depth + 1) for item in node.values())
            elif isinstance(node, list):
                stack.extend((item, depth + 1) for item in node)

    @staticmethod
    def _protocol_error(message: str) -> ModelGatewayError:
        return ModelGatewayError("MODEL_PROTOCOL_ERROR", message)

    @staticmethod
    def _stream_error_payload(error: ModelGatewayError) -> dict[str, Any]:
        safe_detail_keys = {
            "attempts",
            "provider_status",
            "retry_exhausted",
            "retry_delay_seconds",
            "deadline_exceeded",
            "output_visible",
            "upstream_retryable",
            "available_tools",
        }
        details = {
            key: value
            for key, value in error.details.items()
            if key in safe_detail_keys
        }
        messages = {
            "MODEL_TIMEOUT": "model stream timed out",
            "MODEL_UNAVAILABLE": "model stream is unavailable",
            "MODEL_RATE_LIMITED": "model stream was rate limited",
            "MODEL_PERMISSION_DENIED": "model stream permission was denied",
            "MODEL_REQUEST_REJECTED": "model stream request was rejected",
            "MODEL_TLS_ERROR": "model stream TLS validation failed",
            "MODEL_SECURITY_ERROR": "model stream security validation failed",
            "UNKNOWN_TOOL": "model selected an unavailable tool",
            "TOOL_ARGUMENT_PARSE_ERROR": "model tool arguments were invalid JSON",
            "TOOL_ARGUMENT_SCHEMA_ERROR": "model tool arguments failed schema validation",
            "STRUCTURED_OUTPUT_SCHEMA_ERROR": "model structured output failed validation",
            "EMPTY_MODEL_RESPONSE": "model stream returned no usable output",
            "MODEL_PROTOCOL_ERROR": "model stream violated the provider protocol",
        }
        return {
            "code": error.code,
            "message": messages.get(error.code, "model stream failed"),
            "retryable": error.retryable,
            "details": details,
        }

    @staticmethod
    def _add_error_context(error: ModelGatewayError, *, attempts: int, provider_status: int) -> None:
        error.details.setdefault("attempts", attempts)
        error.details.setdefault("provider_status", provider_status)
