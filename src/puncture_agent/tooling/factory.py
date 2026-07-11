"""Factories that bind the ten stable tool contracts to injectable adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from contracts.common import ToolResponseEnvelope

from .case_data import CaseDataBackendPort, CaseDataToolAdapter
from .catalog import TOOL_DEFINITIONS
from .planning import (
    DeterministicPlanningBackend,
    PlanningKernelPort,
    PlanningToolAdapters,
    build_planning_handlers,
)
from .registry import ToolRegistry
from .segmentation import (
    SegmentationBackendPort,
    SegmentationToolAdapter,
    build_segmentation_handlers,
)

ToolHandler = Callable[[Any], ToolResponseEnvelope[Any]]


@dataclass(frozen=True, slots=True)
class AdapterRegistryBundle:
    registry: ToolRegistry
    handlers: Mapping[str, ToolHandler]
    case_data: CaseDataToolAdapter
    segmentation: SegmentationToolAdapter
    planning: PlanningToolAdapters


def bind_tool_handlers(handlers: Mapping[str, ToolHandler]) -> ToolRegistry:
    """Build a registry only when handlers exactly match the frozen catalog."""

    supplied = set(handlers)
    expected = set(TOOL_DEFINITIONS)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("handler catalog mismatch: " + "; ".join(details))
    registry = ToolRegistry()
    for name in TOOL_DEFINITIONS:
        handler = handlers[name]
        if not callable(handler):
            raise TypeError(f"handler is not callable: {name}")
        registry.register(TOOL_DEFINITIONS[name], handler)
    return registry


def build_adapter_registry(
    *,
    case_data_backend: CaseDataBackendPort | None = None,
    segmentation_backend: SegmentationBackendPort | None = None,
    planning_backend: PlanningKernelPort | None = None,
) -> AdapterRegistryBundle:
    """Build all three local adapter groups behind one internal registry.

    The defaults are deterministic manifest backends.  A company deployment
    replaces only the three narrow ports while keeping MCP schemas, Agent graph
    and tests unchanged.
    """

    case_data = CaseDataToolAdapter(case_data_backend)
    segmentation, segmentation_handlers = build_segmentation_handlers(
        segmentation_backend
    )
    planning, planning_handlers = build_planning_handlers(
        planning_backend or DeterministicPlanningBackend()
    )
    combined: dict[str, ToolHandler] = {}
    for group in (case_data.handlers(), segmentation_handlers, planning_handlers):
        overlap = set(combined).intersection(group)
        if overlap:
            raise ValueError("duplicate tool handlers: " + ", ".join(sorted(overlap)))
        combined.update(group)
    registry = bind_tool_handlers(combined)
    return AdapterRegistryBundle(
        registry=registry,
        handlers=MappingProxyType(dict(combined)),
        case_data=case_data,
        segmentation=segmentation,
        planning=planning,
    )


__all__ = [
    "AdapterRegistryBundle",
    "ToolHandler",
    "bind_tool_handlers",
    "build_adapter_registry",
]
