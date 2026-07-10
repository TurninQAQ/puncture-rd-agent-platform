"""Model gateway contracts and deterministic development doubles."""

from .client import ModelGateway, ModelGatewayError, VllmModelGateway
from .mock_qwen import MockQwenGateway
from .models import (
    ChatMessage,
    GatewayHealth,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    ToolDefinition,
    TokenUsage,
    VllmGatewayConfig,
    validate_json_schema_subset,
)

__all__ = [
    "ChatMessage",
    "GatewayHealth",
    "MockQwenGateway",
    "ModelGateway",
    "ModelGatewayError",
    "ModelRequest",
    "ModelResponse",
    "ModelStreamEvent",
    "ToolCall",
    "ToolDefinition",
    "TokenUsage",
    "VllmGatewayConfig",
    "VllmModelGateway",
    "validate_json_schema_subset",
]
