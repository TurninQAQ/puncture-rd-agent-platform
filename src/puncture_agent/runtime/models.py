"""Framework-neutral HTTP/runtime data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EventType(str, Enum):
    RUN_CREATED = "RUN_CREATED"
    RUN_STARTED = "RUN_STARTED"
    NODE_STARTED = "NODE_STARTED"
    NODE_COMPLETED = "NODE_COMPLETED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_RESULT = "TOOL_RESULT"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"


@dataclass(frozen=True, slots=True)
class RunRequest:
    case_id: str
    user_query: str
    task_type: str
    idempotency_key: str
    tenant_id: str = "default"
    principal_id: str = "local-user"
    artifact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (self.case_id, self.user_query, self.task_type, self.idempotency_key)
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("case_id, user_query, task_type, and idempotency_key are required")
        if self.task_type not in {"DATA_MODEL_VALIDATION", "PLANNING_SAFETY"}:
            raise ValueError("unsupported task_type")
        if len(self.idempotency_key) > 256:
            raise ValueError("idempotency_key is too long")
        object.__setattr__(self, "artifact_ids", tuple(self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approval_id: str
    approved: bool
    principal_id: str
    comment: str = ""

    def __post_init__(self) -> None:
        if not self.approval_id.strip() or not self.principal_id.strip():
            raise ValueError("approval_id and principal_id are required")


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: RunStatus
    final_report: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    approval_id: str | None = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status is RunStatus.WAITING_APPROVAL and not self.approval_id:
            raise ValueError("WAITING_APPROVAL requires approval_id")
        if self.status is RunStatus.FAILED and self.error is None:
            raise ValueError("FAILED outcome requires error")
        object.__setattr__(self, "final_report", dict(self.final_report))
        object.__setattr__(self, "checkpoint", dict(self.checkpoint))
        if self.error is not None:
            object.__setattr__(self, "error", dict(self.error))


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    event_type: EventType
    node_name: str | None
    timestamp: str
    payload: Mapping[str, Any]
    trace_id: str

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("event sequence starts at 1")
        if not self.timestamp.endswith("Z"):
            raise ValueError("timestamp must be UTC and end in Z")
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: str
    request: RunRequest
    status: RunStatus
    trace_id: str
    created_at: str
    updated_at: str
    final_report: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    approval_id: str | None
    error: Mapping[str, Any] | None

