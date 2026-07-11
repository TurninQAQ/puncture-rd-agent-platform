"""Tool catalog, stubs, and registry used by the Agent runtime."""

from .catalog import TOOL_DEFINITIONS
from .factory import AdapterRegistryBundle, bind_tool_handlers, build_adapter_registry
from .registry import ToolDefinition, ToolRegistry


def build_mock_registry() -> ToolRegistry:
    """Return a registry containing deterministic, dependency-free mocks."""

    from puncture_agent.mocks.tool_mocks import MOCK_HANDLERS

    registry = ToolRegistry()
    for name, definition in TOOL_DEFINITIONS.items():
        registry.register(definition, MOCK_HANDLERS[name])
    return registry


__all__ = [
    "AdapterRegistryBundle",
    "TOOL_DEFINITIONS",
    "ToolDefinition",
    "ToolRegistry",
    "bind_tool_handlers",
    "build_adapter_registry",
    "build_mock_registry",
]
