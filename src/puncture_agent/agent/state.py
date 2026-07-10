"""Stable agent-state contract shared by mock and future LangGraph runtimes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from uuid import uuid4


class TaskType:
    """String constants are used so serialized state is framework-neutral."""

    UNKNOWN = "UNKNOWN"
    DATA_MODEL_VALIDATION = "DATA_MODEL_VALIDATION"
    PLANNING_SAFETY = "PLANNING_SAFETY"


class VerificationStatus:
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    NEED_RETRY = "NEED_RETRY"
    NEED_REPLAN = "NEED_REPLAN"
    MISSING_INPUT = "MISSING_INPUT"
    NO_FEASIBLE_PATH = "NO_FEASIBLE_PATH"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"


class AgentStatus:
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    COMPLETED_WITH_NO_RESULT = "COMPLETED_WITH_NO_RESULT"
    AWAITING_INPUT = "AWAITING_INPUT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"


@dataclass
class AgentState:
    """Serializable state passed between every graph node.

    Real implementations must preserve these field names. New optional fields
    may be added, but existing fields must not be removed or change meaning.
    Large CT/Mask payloads are never stored here; ``artifacts`` contains IDs and
    compact metadata only.
    """

    user_query: str
    session_id: str = field(default_factory=lambda: f"session-{uuid4().hex[:12]}")
    task_type: str = TaskType.UNKNOWN
    case_id: str | None = None
    status: str = AgentStatus.CREATED

    artifacts: dict[str, Any] = field(default_factory=dict)
    planning_constraints: dict[str, Any] = field(default_factory=dict)

    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    tool_plan: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    candidate_paths: list[dict[str, Any]] = field(default_factory=list)
    safety_result: dict[str, Any] = field(default_factory=dict)
    risk_flags: dict[str, Any] = field(default_factory=dict)
    skin_penetration_result: dict[str, Any] = field(default_factory=dict)

    subgraph_result: dict[str, Any] = field(default_factory=dict)
    verification_status: str = VerificationStatus.NOT_RUN
    retry_count: int = 0
    max_retries: int = 1
    errors: list[dict[str, Any]] = field(default_factory=list)

    current_node: str | None = None
    visited_nodes: list[str] = field(default_factory=list)
    node_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-ready mapping suitable for checkpointing."""

        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentState":
        """Rebuild state from a checkpoint while rejecting unknown fields."""

        known_fields = cls.__dataclass_fields__
        unknown = set(payload) - set(known_fields)
        if unknown:
            raise ValueError(f"Unknown AgentState fields: {sorted(unknown)}")
        return cls(**deepcopy(dict(payload)))

    def get_path(self, path: str, default: Any = None) -> Any:
        """Read ``metadata.foo`` style paths from attributes and dictionaries."""

        if not path:
            return default
        parts = path.split(".")
        current: Any = self
        for part in parts:
            if isinstance(current, Mapping):
                if part not in current:
                    return default
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        return current

    def set_path(self, path: str, value: Any) -> None:
        """Set an attribute or nested dictionary path used by node updates."""

        parts = path.split(".")
        if len(parts) == 1:
            if parts[0] not in self.__dataclass_fields__:
                raise KeyError(f"Unknown AgentState field: {parts[0]}")
            setattr(self, parts[0], value)
            return

        root = parts[0]
        if root not in self.__dataclass_fields__:
            raise KeyError(f"Unknown AgentState field: {root}")
        current = getattr(self, root)
        if not isinstance(current, dict):
            raise TypeError(f"Nested update requires a dict field: {root}")
        for part in parts[1:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise TypeError(f"Cannot descend through non-dict path: {path}")
        current[parts[-1]] = value

    def apply_updates(self, updates: Mapping[str, Any]) -> None:
        for path, value in updates.items():
            self.set_path(path, deepcopy(value))

    def add_error(
        self,
        code: str,
        message: str,
        *,
        node_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            {
                "code": code,
                "message": message,
                "node_id": node_id or self.current_node,
                "retryable": retryable,
                "details": dict(details or {}),
            }
        )
