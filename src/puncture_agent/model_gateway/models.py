"""Stable input/output contracts for a Qwen-compatible model gateway.

Only Python's standard library is used so the scaffold and contract tests can run
before vLLM, an OpenAI client, or a GPU runtime is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


JsonObject = Mapping[str, Any]


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

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {self.role}")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")


@dataclass(frozen=True)
class ToolDefinition:
    """A model-visible tool definition using a JSON Schema input contract."""

    name: str
    description: str
    input_schema: JsonObject

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "tool name")
        _require_non_empty(self.description, "tool description")
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
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("messages must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.response_schema is not None and self.response_schema.get("type") != "object":
            raise ValueError("response_schema must be a JSON object schema")

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

    def __post_init__(self) -> None:
        if min(self.prompt_tokens, self.completion_tokens, self.total_tokens) < 0:
            raise ValueError("token counts must not be negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")


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

    def __post_init__(self) -> None:
        if self.event_type not in {"text_delta", "tool_call", "completed", "error"}:
            raise ValueError(f"unsupported stream event_type: {self.event_type}")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        if self.event_type == "text_delta" and self.delta is None:
            raise ValueError("text_delta event requires delta")
        if self.event_type == "tool_call" and self.tool_call is None:
            raise ValueError("tool_call event requires tool_call")
        if self.event_type == "completed" and self.response is None:
            raise ValueError("completed event requires response")


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
    """Configuration consumed by the future real vLLM adapter."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        _require_non_empty(self.base_url, "base_url")
        _require_non_empty(self.model, "model")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
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
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
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
