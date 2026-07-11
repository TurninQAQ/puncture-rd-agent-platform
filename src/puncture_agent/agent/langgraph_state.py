"""Safe conversion between :class:`AgentState` and LangGraph mappings.

LangGraph works naturally with ``TypedDict`` state, while the repository's
stable public/checkpoint contract is the dataclass in :mod:`.state`.  This
module keeps that boundary explicit and rejects values that must never enter a
checkpoint (notably raw image bytes).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any, TypedDict

from .state import AgentState


DEFAULT_MAX_STATE_BYTES = 1024 * 1024
"""Default maximum UTF-8 JSON checkpoint size (one mebibyte)."""

# A shorter alias is useful to callers configuring checkpoint limits.
MAX_STATE_BYTES = DEFAULT_MAX_STATE_BYTES


class StateConversionError(ValueError):
    """Base class for failures at the AgentState/LangGraph boundary."""


class StateValidationError(StateConversionError):
    """The state contains a value forbidden by the checkpoint contract."""


class UnknownStateFieldError(StateValidationError):
    """A mapping contains a field not defined by :class:`AgentState`."""


class RawBytesStateError(StateValidationError):
    """Raw binary data was found anywhere in state."""


class NonStringMappingKeyError(StateValidationError):
    """A state mapping uses a key that is not a string."""


class NonFiniteFloatError(StateValidationError):
    """State contains NaN or positive/negative infinity."""


class StateSerializationError(StateConversionError):
    """State cannot be represented as JSON."""


class StateSizeLimitError(StateConversionError):
    """The serialized state exceeds the configured checkpoint limit."""

    def __init__(self, actual_bytes: int, max_bytes: int) -> None:
        self.actual_bytes = actual_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"serialized state is {actual_bytes} bytes; limit is {max_bytes} bytes"
        )


class ProductionAgentState(TypedDict, total=False):
    """LangGraph state with field names and values compatible with AgentState.

    ``total=False`` permits LangGraph nodes to return partial updates.  A full
    graph checkpoint still has the exact shape produced by
    :meth:`AgentState.to_dict`.
    """

    user_query: str
    session_id: str
    task_type: str
    case_id: str | None
    status: str

    artifacts: dict[str, Any]
    planning_constraints: dict[str, Any]

    retrieved_documents: list[dict[str, Any]]
    citations: list[dict[str, Any]]

    tool_plan: list[str]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    candidate_paths: list[dict[str, Any]]
    safety_result: dict[str, Any]
    risk_flags: dict[str, Any]
    skin_penetration_result: dict[str, Any]

    subgraph_result: dict[str, Any]
    verification_status: str
    retry_count: int
    max_retries: int
    errors: list[dict[str, Any]]

    current_node: str | None
    visited_nodes: list[str]
    node_outputs: dict[str, Any]
    metadata: dict[str, Any]
    final_report: dict[str, Any]


# Both names describe the same internal state contract.  ProductionAgentState
# matches the terminology used in the runtime specification; the longer alias
# is convenient for imports that emphasize the framework boundary.
LangGraphAgentState = ProductionAgentState
LangGraphState = ProductionAgentState


_AGENT_STATE_FIELDS = frozenset(item.name for item in fields(AgentState))
_TYPED_DICT_FIELDS = frozenset(ProductionAgentState.__annotations__)

if _TYPED_DICT_FIELDS != _AGENT_STATE_FIELDS:  # pragma: no cover - import guard
    missing = sorted(_AGENT_STATE_FIELDS - _TYPED_DICT_FIELDS)
    extra = sorted(_TYPED_DICT_FIELDS - _AGENT_STATE_FIELDS)
    raise RuntimeError(
        "ProductionAgentState fields do not match AgentState: "
        f"missing={missing!r}, extra={extra!r}"
    )


def state_to_mapping(
    state: AgentState,
    *,
    max_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> ProductionAgentState:
    """Return an isolated, validated LangGraph mapping for ``state``.

    Size is measured from compact UTF-8 JSON.  The returned mapping shares no
    mutable nested values with the input dataclass.
    """

    if not isinstance(state, AgentState):
        raise StateValidationError("state_to_mapping requires an AgentState instance")
    _validate_max_bytes(max_bytes)

    # Validate the live fields before AgentState.to_dict() calls deepcopy.  In
    # particular, deepcopy(memoryview(...)) raises a generic TypeError and
    # would otherwise bypass the boundary's explicit RawBytesStateError.
    live_values = {item.name: getattr(state, item.name) for item in fields(state)}
    _validate_value(live_values, path="$", active=set())

    try:
        payload = state.to_dict()
    except Exception as exc:  # deepcopy/asdict failures become boundary errors
        raise StateSerializationError("AgentState could not be copied") from exc

    _validate_mapping(payload, max_bytes=max_bytes)
    return payload


def state_from_mapping(
    payload: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> AgentState:
    """Build an isolated :class:`AgentState` from a LangGraph/checkpoint map."""

    if not isinstance(payload, Mapping):
        raise StateValidationError("state_from_mapping requires a mapping")
    _validate_max_bytes(max_bytes)
    _validate_value(payload, path="$", active=set())

    unknown = set(payload) - _AGENT_STATE_FIELDS
    if unknown:
        raise UnknownStateFieldError(f"unknown AgentState fields: {sorted(unknown)!r}")

    try:
        copied = deepcopy(dict(payload))
    except Exception as exc:
        raise StateSerializationError("state mapping could not be copied") from exc

    _validate_mapping(copied, max_bytes=max_bytes)
    try:
        return AgentState.from_dict(copied)
    except (TypeError, ValueError) as exc:
        raise StateValidationError(f"invalid AgentState mapping: {exc}") from exc


def _validate_mapping(payload: Mapping[str, Any], *, max_bytes: int) -> None:
    # Callers validate the live/source object before copying.  Re-walking the
    # copied mapping here doubles every node transition cost without adding a
    # new trust boundary; JSON encoding remains the final representation check.
    serialized = _encode_json(payload)
    actual_bytes = len(serialized)
    if actual_bytes > max_bytes:
        raise StateSizeLimitError(actual_bytes, max_bytes)


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return encoded.encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise StateSerializationError(f"state is not JSON serializable: {exc}") from exc


def _validate_value(value: Any, *, path: str, active: set[int]) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise RawBytesStateError(
            f"raw {type(value).__name__} payload is forbidden at {path}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteFloatError(f"non-finite float is forbidden at {path}")

    if isinstance(value, Mapping):
        marker = _enter_container(value, path=path, active=active)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise NonStringMappingKeyError(
                        f"mapping key at {path} must be str, got {type(key).__name__}"
                    )
                _validate_value(
                    item,
                    path=_mapping_path(path, key),
                    active=active,
                )
        finally:
            active.remove(marker)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        marker = _enter_container(value, path=path, active=active)
        try:
            for index, item in enumerate(value):
                _validate_value(item, path=f"{path}[{index}]", active=active)
        finally:
            active.remove(marker)
        return

    if is_dataclass(value) and not isinstance(value, type):
        marker = _enter_container(value, path=path, active=active)
        try:
            for item in fields(value):
                _validate_value(
                    getattr(value, item.name),
                    path=_mapping_path(path, item.name),
                    active=active,
                )
        finally:
            active.remove(marker)


def _enter_container(value: Any, *, path: str, active: set[int]) -> int:
    marker = id(value)
    if marker in active:
        raise StateSerializationError(f"circular reference in state at {path}")
    active.add(marker)
    return marker


def _mapping_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _validate_max_bytes(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")


__all__ = [
    "DEFAULT_MAX_STATE_BYTES",
    "MAX_STATE_BYTES",
    "LangGraphAgentState",
    "LangGraphState",
    "NonFiniteFloatError",
    "NonStringMappingKeyError",
    "ProductionAgentState",
    "RawBytesStateError",
    "StateConversionError",
    "StateSerializationError",
    "StateSizeLimitError",
    "StateValidationError",
    "UnknownStateFieldError",
    "state_from_mapping",
    "state_to_mapping",
]
