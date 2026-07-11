"""Atomic Run/event persistence protocol and dependency-free memory backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .errors import (
    ExecutionSuperseded,
    RunRepositoryIdempotencyConflict,
    RunRepositoryNotFound,
    RunRepositoryTransitionError,
    RunRepositoryVersionConflict,
)
from .json_boundary import copy_json_mapping
from .models import EventType, RunEvent, RunRequest, RunSnapshot, RunStatus


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class RunEventDraft:
    event_type: EventType
    node_name: str | None
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError("event_type must be an EventType")
        if self.node_name is not None and not isinstance(self.node_name, str):
            raise TypeError("node_name must be a string or None")
        if not isinstance(self.payload, Mapping):
            raise TypeError("event payload must be a mapping")
        object.__setattr__(self, "payload", copy_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class VersionedRun:
    snapshot: RunSnapshot
    version: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError("run version starts at 1")


@dataclass(frozen=True, slots=True)
class CreateRunResult:
    run: VersionedRun
    created: bool


@dataclass(slots=True)
class _StoredRun:
    snapshot: RunSnapshot
    version: int
    events: list[RunEvent]


_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_APPROVAL,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.WAITING_APPROVAL: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLED}
    ),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}
_TRANSITION_EVENT: dict[RunStatus, EventType | None] = {
    RunStatus.RUNNING: None,
    RunStatus.WAITING_APPROVAL: EventType.APPROVAL_REQUESTED,
    RunStatus.SUCCEEDED: EventType.RUN_COMPLETED,
    RunStatus.FAILED: EventType.RUN_FAILED,
    RunStatus.CANCELLED: EventType.RUN_CANCELLED,
    RunStatus.PENDING: None,
}
_RUNNING_STREAM_EVENTS = frozenset(
    {
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
        EventType.TOOL_CALLED,
        EventType.TOOL_RESULT,
    }
)


@runtime_checkable
class RunRepository(Protocol):
    def create_or_get_started(
        self,
        snapshot: RunSnapshot,
        initial_events: tuple[RunEventDraft, ...],
    ) -> CreateRunResult: ...

    def get(self, run_id: str, *, tenant_id: str) -> VersionedRun: ...

    def get_events(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]: ...

    def assert_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
    ) -> None: ...

    def append_if_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        event: RunEventDraft,
    ) -> RunEvent: ...

    def compare_and_swap(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun: ...


class InMemoryRunRepository:
    """Thread-safe reference backend with version fencing and detached reads."""

    def __init__(self, *, clock: Callable[[], str] = _utc_now) -> None:
        self._clock = clock
        self._records: dict[str, _StoredRun] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def create_or_get_started(
        self,
        snapshot: RunSnapshot,
        initial_events: tuple[RunEventDraft, ...],
    ) -> CreateRunResult:
        normalized_snapshot = self._copy_snapshot(snapshot)
        normalized_events = tuple(
            self._copy_event(event) for event in initial_events
        )
        self._validate_snapshot_state(normalized_snapshot)
        if normalized_snapshot.status is not RunStatus.RUNNING:
            raise RunRepositoryTransitionError(
                "new runs must enter the repository in RUNNING state"
            )
        if tuple(event.event_type for event in normalized_events) != (
            EventType.RUN_CREATED,
            EventType.RUN_STARTED,
        ):
            raise RunRepositoryTransitionError(
                "new runs require RUN_CREATED then RUN_STARTED"
            )
        key = (
            normalized_snapshot.request.tenant_id,
            normalized_snapshot.request.idempotency_key,
        )
        with self._lock:
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                existing = self._records[existing_id]
                if existing.snapshot.request != normalized_snapshot.request:
                    raise RunRepositoryIdempotencyConflict()
                return CreateRunResult(self._view(existing), created=False)
            if normalized_snapshot.run_id in self._records:
                raise RunRepositoryIdempotencyConflict()

            stored = _StoredRun(
                snapshot=normalized_snapshot,
                version=1,
                events=[],
            )
            for event in normalized_events:
                self._append_event_locked(stored, event)
            self._records[normalized_snapshot.run_id] = stored
            self._idempotency[key] = normalized_snapshot.run_id
            return CreateRunResult(self._view(stored), created=True)

    def get(self, run_id: str, *, tenant_id: str) -> VersionedRun:
        with self._lock:
            return self._view(self._require(run_id, tenant_id))

    def get_events(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("after_sequence must be an integer")
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        with self._lock:
            stored = self._require(run_id, tenant_id)
            return tuple(
                deepcopy(event)
                for event in stored.events
                if event.sequence > after_sequence
            )

    def assert_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
    ) -> None:
        self._validate_version(expected_version)
        with self._lock:
            stored = self._require(run_id, tenant_id)
            if (
                stored.version != expected_version
                or stored.snapshot.status is not RunStatus.RUNNING
            ):
                raise ExecutionSuperseded("execution no longer owns this run")

    def append_if_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        event: RunEventDraft,
    ) -> RunEvent:
        self._validate_version(expected_version)
        normalized_event = self._copy_event(event)
        if normalized_event.event_type not in _RUNNING_STREAM_EVENTS:
            raise RunRepositoryTransitionError(
                "state events must be committed atomically with a state transition"
            )
        with self._lock:
            stored = self._require(run_id, tenant_id)
            if (
                stored.version != expected_version
                or stored.snapshot.status is not RunStatus.RUNNING
            ):
                raise ExecutionSuperseded("execution no longer owns this run")
            return deepcopy(self._append_event_locked(stored, normalized_event))

    def compare_and_swap(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        self._validate_version(expected_version)
        normalized_snapshot = self._copy_snapshot(snapshot)
        normalized_events = tuple(self._copy_event(event) for event in events)
        with self._lock:
            stored = self._require(run_id, tenant_id)
            if stored.version != expected_version:
                raise RunRepositoryVersionConflict()
            self._validate_replacement(stored.snapshot, normalized_snapshot)
            self._validate_transition_events(normalized_snapshot, normalized_events)
            candidate = _StoredRun(
                snapshot=normalized_snapshot,
                version=stored.version + 1,
                events=deepcopy(stored.events),
            )
            for event in normalized_events:
                self._append_event_locked(candidate, event)
            stored.snapshot = candidate.snapshot
            stored.version = candidate.version
            stored.events = candidate.events
            return self._view(stored)

    def _require(self, run_id: str, tenant_id: str) -> _StoredRun:
        stored = self._records.get(run_id)
        if stored is None or stored.snapshot.request.tenant_id != tenant_id:
            raise RunRepositoryNotFound()
        return stored

    def _append_event_locked(
        self,
        stored: _StoredRun,
        draft: RunEventDraft,
    ) -> RunEvent:
        event = RunEvent(
            run_id=stored.snapshot.run_id,
            sequence=len(stored.events) + 1,
            event_type=draft.event_type,
            node_name=draft.node_name,
            timestamp=self._clock(),
            payload=copy_json_mapping(draft.payload),
            trace_id=stored.snapshot.trace_id,
        )
        stored.events.append(event)
        return event

    @staticmethod
    def _validate_replacement(current: RunSnapshot, replacement: RunSnapshot) -> None:
        InMemoryRunRepository._validate_snapshot_state(replacement)
        if replacement.run_id != current.run_id:
            raise RunRepositoryTransitionError("run_id is immutable")
        if replacement.request != current.request:
            raise RunRepositoryTransitionError("run request is immutable")
        if replacement.trace_id != current.trace_id:
            raise RunRepositoryTransitionError("trace_id is immutable")
        if replacement.created_at != current.created_at:
            raise RunRepositoryTransitionError("created_at is immutable")
        if replacement.status not in _ALLOWED_TRANSITIONS[current.status]:
            raise RunRepositoryTransitionError()
        if (
            current.status is RunStatus.FAILED
            and replacement.status is RunStatus.RUNNING
            and current.checkpoint.get("recoverable") is not True
        ):
            raise RunRepositoryTransitionError(
                "FAILED run checkpoint is not recoverable"
            )

    @staticmethod
    def _validate_snapshot_state(snapshot: RunSnapshot) -> None:
        if not isinstance(snapshot.status, RunStatus):
            raise RunRepositoryTransitionError("run status is invalid")
        if snapshot.status is RunStatus.WAITING_APPROVAL:
            if (
                not isinstance(snapshot.approval_id, str)
                or not snapshot.approval_id.strip()
            ):
                raise RunRepositoryTransitionError(
                    "WAITING_APPROVAL requires an approval_id"
                )
        elif snapshot.approval_id is not None:
            raise RunRepositoryTransitionError(
                "approval_id is only valid while waiting for approval"
            )
        if snapshot.status is RunStatus.FAILED:
            if snapshot.error is None:
                raise RunRepositoryTransitionError("FAILED requires an error")
        elif snapshot.error is not None:
            raise RunRepositoryTransitionError(
                "error is only valid for a FAILED run"
            )
        if snapshot.status is not RunStatus.SUCCEEDED and snapshot.final_report:
            raise RunRepositoryTransitionError(
                "final_report is only valid for a SUCCEEDED run"
            )

    @staticmethod
    def _validate_transition_events(
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...],
    ) -> None:
        target_status = snapshot.status
        expected = _TRANSITION_EVENT[target_status]
        actual = tuple(event.event_type for event in events)
        if expected is None:
            if actual:
                raise RunRepositoryTransitionError(
                    "this transition must not append a state event"
                )
            return
        if actual != (expected,):
            raise RunRepositoryTransitionError(
                f"transition to {target_status.value} requires one {expected.value} event"
            )
        if (
            target_status is RunStatus.WAITING_APPROVAL
            and events[0].payload.get("approval_id") != snapshot.approval_id
        ):
            raise RunRepositoryTransitionError(
                "approval event must match the waiting approval_id"
            )

    @staticmethod
    def _validate_version(version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("expected_version must be a positive integer")

    @staticmethod
    def _copy_event(event: RunEventDraft) -> RunEventDraft:
        if not isinstance(event, RunEventDraft):
            raise TypeError("repository events must be RunEventDraft values")
        return RunEventDraft(
            event_type=event.event_type,
            node_name=event.node_name,
            payload=event.payload,
        )

    @staticmethod
    def _copy_snapshot(snapshot: RunSnapshot) -> RunSnapshot:
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("repository snapshots must be RunSnapshot values")
        if not isinstance(snapshot.request, RunRequest):
            raise TypeError("snapshot request must be a RunRequest")
        request = replace(
            snapshot.request,
            metadata=copy_json_mapping(snapshot.request.metadata),
        )
        normalized = replace(
            snapshot,
            request=request,
            final_report=copy_json_mapping(snapshot.final_report),
            checkpoint=copy_json_mapping(snapshot.checkpoint),
            error=(
                copy_json_mapping(snapshot.error)
                if snapshot.error is not None
                else None
            ),
        )
        return deepcopy(normalized)

    @staticmethod
    def _view(stored: _StoredRun) -> VersionedRun:
        return VersionedRun(
            snapshot=deepcopy(stored.snapshot),
            version=stored.version,
        )


__all__ = [
    "CreateRunResult",
    "InMemoryRunRepository",
    "RunEventDraft",
    "RunRepository",
    "VersionedRun",
]
