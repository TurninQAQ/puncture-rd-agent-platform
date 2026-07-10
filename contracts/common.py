"""Tool call context, response envelope, and JSON serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar

from .artifacts import ArtifactRef
from .enums import ToolExecutionStatus
from .errors import ErrorDetail

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    request_id: str
    trace_id: str
    case_id: str
    caller: str
    idempotency_key: str
    requested_at: str
    deadline_epoch_ms: int | None = None

    def __post_init__(self) -> None:
        required = (self.request_id, self.trace_id, self.case_id, self.caller, self.idempotency_key)
        if any(not value.strip() for value in required):
            raise ValueError("request_id, trace_id, case_id, caller, and idempotency_key are required")
        if not self.requested_at.endswith("Z"):
            raise ValueError("requested_at must be an ISO-8601 UTC string ending in Z")
        if self.deadline_epoch_ms is not None and self.deadline_epoch_ms <= 0:
            raise ValueError("deadline_epoch_ms must be positive when provided")


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("metric name and unit are required")


@dataclass(frozen=True, slots=True)
class ToolResponseEnvelope(Generic[T]):
    request_id: str
    trace_id: str
    tool_name: str
    tool_version: str
    status: ToolExecutionStatus
    result: T | None
    artifacts: tuple[ArtifactRef, ...]
    metrics: tuple[MetricValue, ...]
    warnings: tuple[str, ...]
    error: ErrorDetail | None
    started_at: str
    finished_at: str

    def __post_init__(self) -> None:
        if not all((self.request_id, self.trace_id, self.tool_name, self.tool_version)):
            raise ValueError("response identity fields must not be empty")
        if self.status is ToolExecutionStatus.SUCCESS:
            if self.result is None:
                raise ValueError("SUCCESS response requires a result")
            if self.error is not None:
                raise ValueError("SUCCESS response must not contain an error")
        if self.status is ToolExecutionStatus.FAILED and self.error is None:
            raise ValueError("FAILED response requires an error")
        if not self.started_at.endswith("Z") or not self.finished_at.endswith("Z"):
            raise ValueError("response timestamps must be ISO-8601 UTC strings ending in Z")

    @property
    def ok(self) -> bool:
        return self.status in (ToolExecutionStatus.SUCCESS, ToolExecutionStatus.PARTIAL)


def to_primitive(value: Any) -> Any:
    """Convert nested contracts into JSON-compatible Python values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [to_primitive(item) for item in value]
    return value


def to_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=False, indent=indent, sort_keys=True)
