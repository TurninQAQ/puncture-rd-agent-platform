"""Deterministic Qwen development double for offline graph and contract tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .client import ModelGatewayError
from .models import (
    GatewayHealth,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    TokenUsage,
    ToolCall,
    example_from_json_schema,
    validate_json_schema_subset,
)


class MockQwenGateway:
    """A predictable model gateway controlled through request.metadata.

    Supported controls:
      mock_tool_call: mapping or sequence of {name, arguments, call_id?}
      mock_structured_output: exact object returned after schema validation
      mock_text: exact text returned for ordinary generation
      force_error: {code, message?, retryable?}

    These controls belong only to the mock and must never be forwarded to vLLM.
    """

    def __init__(self, model: str = "mock-qwen-instruct") -> None:
        self.model = model

    def health(self) -> GatewayHealth:
        return GatewayHealth(
            status="UP",
            model=self.model,
            provider="mock",
            details={"deterministic": True, "gpu_required": False},
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.stream:
            raise ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "generate requires request.stream to be false",
                details={"attempts": 0},
            )
        return self._generate_response(request)

    def _generate_response(self, request: ModelRequest) -> ModelResponse:
        force_error = request.metadata.get("force_error")
        if isinstance(force_error, Mapping):
            raise ModelGatewayError(
                str(force_error.get("code", "MOCK_MODEL_ERROR")),
                str(force_error.get("message", "forced mock model failure")),
                retryable=bool(force_error.get("retryable", False)),
            )

        tool_calls = self._build_tool_calls(request)
        structured_output = None
        text: str | None = None

        if tool_calls:
            finish_reason = "tool_calls"
        elif request.response_schema is not None:
            candidate = request.metadata.get("mock_structured_output")
            if candidate is None:
                candidate = example_from_json_schema(request.response_schema)
            validate_json_schema_subset(candidate, request.response_schema)
            structured_output = dict(candidate)
            finish_reason = "stop"
        else:
            last_user_text = next(
                (message.content for message in reversed(request.messages) if message.role == "user"),
                "",
            )
            text = str(request.metadata.get("mock_text", f"MOCK_QWEN_RESPONSE: {last_user_text}"))
            finish_reason = "stop"

        prompt_text = "\n".join(message.content for message in request.messages)
        completion_text = text or str(structured_output or [call.arguments for call in tool_calls])
        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(completion_text)
        return ModelResponse(
            request_id=request.request_id,
            model=self.model,
            finish_reason=finish_reason,
            text=text,
            tool_calls=tool_calls,
            structured_output=structured_output,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=1.0,
            raw_response_id=f"mock-response-{request.request_id}",
        )

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        if not request.stream:
            error = ModelGatewayError(
                "MODEL_REQUEST_REJECTED",
                "stream requires request.stream to be true",
                details={"attempts": 0},
            )
            yield self._error_event(request, error, sequence=0)
            return
        try:
            response = self._generate_response(request)
        except ModelGatewayError as error:
            yield self._error_event(request, error, sequence=0)
            return
        sequence = 0
        if response.text:
            words = response.text.split(" ")
            for index, word in enumerate(words):
                delta = word if index == 0 else f" {word}"
                yield ModelStreamEvent(
                    request_id=request.request_id,
                    event_type="text_delta",
                    sequence=sequence,
                    delta=delta,
                )
                sequence += 1
        for tool_call in response.tool_calls:
            yield ModelStreamEvent(
                request_id=request.request_id,
                event_type="tool_call",
                sequence=sequence,
                tool_call=tool_call,
            )
            sequence += 1
        yield ModelStreamEvent(
            request_id=request.request_id,
            event_type="completed",
            sequence=sequence,
            response=response,
        )

    @staticmethod
    def _error_event(
        request: ModelRequest,
        error: ModelGatewayError,
        *,
        sequence: int,
    ) -> ModelStreamEvent:
        safe_details = {
            key: value
            for key, value in error.details.items()
            if key in {"attempts", "available_tools"}
        }
        safe_details["output_visible"] = False
        return ModelStreamEvent(
            request_id=request.request_id,
            event_type="error",
            sequence=sequence,
            error={
                "code": error.code,
                "message": "mock model stream failed",
                "retryable": error.retryable,
                "details": safe_details,
            },
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _build_tool_calls(request: ModelRequest) -> tuple[ToolCall, ...]:
        raw_calls = request.metadata.get("mock_tool_call")
        if raw_calls is None:
            return ()
        if isinstance(raw_calls, Mapping):
            normalized_calls: Sequence[Mapping[str, Any]] = [raw_calls]
        elif isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
            normalized_calls = raw_calls
        else:
            raise ModelGatewayError(
                "INVALID_MOCK_CONTROL",
                "mock_tool_call must be an object or sequence of objects",
            )

        tools_by_name = {tool.name: tool for tool in request.tools}
        result: list[ToolCall] = []
        for index, raw_call in enumerate(normalized_calls):
            if not isinstance(raw_call, Mapping):
                raise ModelGatewayError(
                    "INVALID_MOCK_CONTROL",
                    f"mock_tool_call item {index} must be an object",
                )
            name = str(raw_call.get("name", ""))
            if name not in tools_by_name:
                raise ModelGatewayError(
                    "UNKNOWN_TOOL",
                    f"mock requested unavailable tool {name!r}",
                    details={"available_tools": sorted(tools_by_name)},
                )
            arguments = raw_call.get("arguments", {})
            try:
                validate_json_schema_subset(arguments, tools_by_name[name].input_schema)
            except ValueError as exc:
                raise ModelGatewayError("TOOL_ARGUMENT_SCHEMA_ERROR", str(exc)) from exc
            result.append(
                ToolCall(
                    call_id=str(raw_call.get("call_id", f"mock-call-{index + 1}")),
                    name=name,
                    arguments=dict(arguments),
                )
            )
        return tuple(result)
