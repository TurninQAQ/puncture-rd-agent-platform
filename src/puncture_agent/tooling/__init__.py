"""Tool catalog, stubs, and registry used by the Agent runtime."""

from .catalog import TOOL_DEFINITIONS
from .registry import ToolDefinition, ToolRegistry


def build_mock_registry() -> ToolRegistry:
    """Return a registry containing deterministic, dependency-free mocks."""

    from puncture_agent.mocks.tool_mocks import MOCK_HANDLERS

    registry = ToolRegistry()
    for name, definition in TOOL_DEFINITIONS.items():
        registry.register(definition, MOCK_HANDLERS[name])
    return registry


__all__ = ["TOOL_DEFINITIONS", "ToolDefinition", "ToolRegistry", "build_mock_registry"]
