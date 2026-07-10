"""Stable input/output contracts for a Qwen-compatible model gateway.

Only Python's standard library is used so the scaffold and contract tests can run
before vLLM, an OpenAI client, or a GPU runtime is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


JsonObject = Mapping[str, Any]
MODEL_GATEWAY_CONTRACT_VERSION = "2"


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class ChatMessage:
    """One OpenAI-compatible chat message."""

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {self.role}")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role == "assistant" and self.tool_call_id is not None:
            raise ValueError("assistant messages must not contain tool_call_id")
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise ValueError("assistant tool_calls must contain ToolCall values")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool_call IDs must be unique")


@dataclass(frozen=True)
class ToolDefinition:
    """A model-visible tool definition using a JSON Schema input contract."""

    name: str
    description: str
    input_schema: JsonObject

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "tool name")
        _require_non_empty(self.description, "tool description")
        if not isinstance(self.input_schema, Mapping):
            raise ValueError("tool input_schema must be an object")
        if self.input_schema.get("type") != "object":
            raise ValueError("tool input_schema must be a JSON object schema")


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral request passed from the Agent Runtime to the model."""

    request_id: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    response_schema: JsonObject | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    stream: bool = False
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        if any(ord(character) < 32 or ord(character) == 127 for character in self.request_id):
            raise ValueError("request_id must not contain control characters")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, ChatMessage) for message in self.messages):
            raise ValueError("messages must contain ChatMessage values")
        if any(not isinstance(tool, ToolDefinition) for tool in self.tools):
            raise ValueError("tools must contain ToolDefinition values")
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ValueError("temperature must be numeric")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        if self.response_schema is not None:
            if not isinstance(self.response_schema, Mapping):
                raise ValueError("response_schema must be an object")
            if self.response_schema.get("type") != "object":
                raise ValueError("response_schema must be a JSON object schema")
        if self.tools and self.response_schema is not None:
            raise ValueError("tools and response_schema are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: JsonObject

    def __post_init__(self) -> None:
        _require_non_empty(self.call_id, "call_id")
        _require_non_empty(self.name, "tool call name")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool call arguments must be an object")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_known: bool = True

    def __post_init__(self) -> None:
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("token counts must be integers")
        if not isinstance(self.usage_known, bool):
            raise ValueError("usage_known must be a boolean")
        if min(self.prompt_tokens, self.completion_tokens, self.total_tokens) < 0:
            raise ValueError("token counts must not be negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        if not self.usage_known and any(values):
            raise ValueError("unknown token usage must use zero-count sentinels")


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response independent of the concrete vLLM/OpenAI client."""

    request_id: str
    model: str
    finish_reason: str
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    structured_output: JsonObject | None
    usage: TokenUsage
    latency_ms: float
    raw_response_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        _require_non_empty(self.model, "model")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.finish_reason not in {"stop", "tool_calls", "length", "content_filter", "error"}:
            raise ValueError(f"unsupported finish_reason: {self.finish_reason}")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")
        if self.finish_reason == "tool_calls" and not self.tool_calls:
            raise ValueError("tool_calls finish reason requires at least one tool call")
        if self.text is None and not self.tool_calls and self.structured_output is None:
            raise ValueError("response must contain text, tool_calls, or structured_output")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelStreamEvent:
    """One normalized streaming event.

    event_type is one of: text_delta, tool_call, completed, error.
    The completed event must carry the final normalized response.
    """

    request_id: str
    event_type: str
    sequence: int
    delta: str | None = None
    tool_call: ToolCall | None = None
    response: ModelResponse | None = None
    error: JsonObject | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        if self.event_type not in {"text_delta", "tool_call", "completed", "error"}:
            raise ValueError(f"unsupported stream event_type: {self.event_type}")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise ValueError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        if self.event_type == "text_delta" and self.delta is None:
            raise ValueError("text_delta event requires delta")
        if self.event_type == "tool_call" and self.tool_call is None:
            raise ValueError("tool_call event requires tool_call")
        if self.event_type == "completed" and self.response is None:
            raise ValueError("completed event requires response")
        if self.event_type == "error" and self.error is None:
            raise ValueError("error event requires structured error")
        if self.event_type == "error":
            if not isinstance(self.error, Mapping):
                raise ValueError("error payload must be an object")
            _require_non_empty(self.error.get("code"), "error code")
            _require_non_empty(self.error.get("message"), "error message")
            if not isinstance(self.error.get("retryable"), bool):
                raise ValueError("error retryable must be a boolean")
            if not isinstance(self.error.get("details"), Mapping):
                raise ValueError("error details must be an object")
        payloads = {
            "text_delta": self.delta,
            "tool_call": self.tool_call,
            "completed": self.response,
            "error": self.error,
        }
        unexpected = [
            name
            for name, value in payloads.items()
            if name != self.event_type and value is not None
        ]
        if unexpected:
            raise ValueError(
                f"{self.event_type} event contains unexpected payloads: {unexpected!r}"
            )


@dataclass(frozen=True)
class GatewayHealth:
    status: str
    model: str
    provider: str
    details: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"UP", "DEGRADED", "DOWN"}:
            raise ValueError("health status must be UP, DEGRADED, or DOWN")


@dataclass(frozen=True)
class VllmGatewayConfig:
    """Configuration consumed by the production vLLM adapter."""

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 60.0
    max_retries: int = 2
    ca_bundle_path: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.base_url, "base_url")
        _require_non_empty(self.model, "model")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string or None")
        if self.ca_bundle_path is not None:
            _require_non_empty(self.ca_bundle_path, "ca_bundle_path")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("timeout_seconds must be a finite number")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")


def validate_json_schema_subset(value: Any, schema: JsonObject, path: str = "$") -> None:
    """Validate the JSON Schema subset used by tool and structured outputs.

    Supported keywords: type, properties, required, additionalProperties, items,
    enum. Production code may replace this helper with a full JSON Schema library,
    but must preserve the observable validation behavior covered by tests.
    """

    expected_type = schema.get("type")
    valid = True
    if expected_type == "object":
        valid = isinstance(value, Mapping)
    elif expected_type == "array":
        valid = isinstance(value, (list, tuple))
    elif expected_type == "string":
        valid = isinstance(value, str)
    elif expected_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected_type == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    elif expected_type == "boolean":
        valid = isinstance(value, bool)
    elif expected_type == "null":
        valid = value is None
    elif expected_type is not None:
        raise ValueError(f"{path}: unsupported schema type {expected_type!r}")

    if not valid:
        raise ValueError(f"{path}: expected {expected_type}, got {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in enum {schema['enum']!r}")

    if expected_type == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ValueError(f"{path}: unexpected properties {sorted(unknown)!r}")
        for key, item in value.items():
            if key in properties:
                validate_json_schema_subset(item, properties[key], f"{path}.{key}")

    if expected_type == "array" and "items" in schema:
        for index, item in enumerate(value):
            validate_json_schema_subset(item, schema["items"], f"{path}[{index}]")


def example_from_json_schema(schema: JsonObject) -> Any:
    """Create a deterministic minimal example for mock structured generation."""

    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        return {
            key: example_from_json_schema(properties[key])
            for key in schema.get("required", [])
            if key in properties
        }
    if schema_type == "array":
        return []
    if schema_type == "string":
        return "mock"
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    raise ValueError(f"cannot create mock example for schema: {schema!r}")
