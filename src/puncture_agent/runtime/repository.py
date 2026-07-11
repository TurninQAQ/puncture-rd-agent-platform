"""Atomic Run/event persistence protocol and dependency-free memory backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .errors import (
    ExecutionSuperseded,
    RunRepositoryEventConflict,
    RunRepositoryIdempotencyConflict,
    RunRepositoryNotFound,
    RunRepositoryTransitionError,
    RunRepositoryVersionConflict,
)
from .json_boundary import copy_json_mapping, copy_json_value
from .models import EventType, RunEvent, RunRequest, RunSnapshot, RunStatus


_UTC_MILLISECOND_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _canonical_utc_millisecond(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _UTC_MILLISECOND_TIMESTAMP.fullmatch(value):
        raise ValueError(
            f"{field_name} must use canonical UTC millisecond format"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid canonical UTC timestamp"
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class RunEventDraft:
    event_type: EventType
    node_name: str | None
    payload: Mapping[str, Any]
    event_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError("event_type must be an EventType")
        if self.node_name is not None and not isinstance(self.node_name, str):
            raise TypeError("node_name must be a string or None")
        if self.node_name is not None:
            try:
                encoded_node_name = self.node_name.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("node_name must be valid UTF-8") from exc
            if "\x00" in self.node_name or len(encoded_node_name) > 512:
                raise ValueError(
                    "node_name must be a bounded UTF-8 string without NUL"
                )
        if not isinstance(self.payload, Mapping):
            raise TypeError("event payload must be a mapping")
        if self.event_key is not None:
            if (
                not isinstance(self.event_key, str)
                or not self.event_key.strip()
                or self.event_key != self.event_key.strip()
            ):
                raise ValueError("event_key must be a non-empty bounded string")
            try:
                encoded_event_key = self.event_key.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("event_key must be valid UTF-8") from exc
            if len(encoded_event_key) > 256 or any(
                character in self.event_key
                for character in ("\r", "\n", "\x00")
            ):
                raise ValueError("event_key must be a non-empty bounded string")
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
class _StoredEventClaim:
    execution_version: int
    fingerprint: str
    event: RunEvent


@dataclass(slots=True)
class _StoredMutationClaim:
    snapshot_fingerprint: str
    transition_fingerprint: str


@dataclass(slots=True)
class _StoredRun:
    snapshot: RunSnapshot
    request_fingerprint: str
    version: int
    events: list[RunEvent]
    event_claims: dict[str, _StoredEventClaim]
    mutation_claims: dict[int, _StoredMutationClaim]


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


def _canonical_json(value: Any) -> str:
    normalized = copy_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _request_document(request: RunRequest) -> dict[str, Any]:
    document = copy_json_value(
        {
            "case_id": request.case_id,
            "user_query": request.user_query,
            "task_type": request.task_type,
            "idempotency_key": request.idempotency_key,
            "tenant_id": request.tenant_id,
            "principal_id": request.principal_id,
            "artifact_ids": list(request.artifact_ids),
            "metadata": request.metadata,
        }
    )
    if not isinstance(document, dict):
        raise TypeError("request document must remain a mapping")
    return document


def _request_fingerprint(request: RunRequest) -> str:
    canonical = _canonical_json(_request_document(request)).encode("utf-8")
    return sha256(canonical).hexdigest()


def _event_fingerprint(event: RunEventDraft, execution_version: int) -> str:
    document = {
        "execution_version": execution_version,
        "event_type": event.event_type.value,
        "node_name": event.node_name,
        "payload": event.payload,
    }
    return sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def _snapshot_fingerprint(snapshot: RunSnapshot) -> str:
    document = {
        "run_id": snapshot.run_id,
        "request": _request_document(snapshot.request),
        "status": snapshot.status.value,
        "trace_id": snapshot.trace_id,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "final_report": snapshot.final_report,
        "checkpoint": snapshot.checkpoint,
        "approval_id": snapshot.approval_id,
        "error": snapshot.error,
    }
    return sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def _transition_fingerprint(
    events: tuple[RunEventDraft, ...],
    expected_version: int,
) -> str:
    document = {
        "expected_version": expected_version,
        "events": [
            {
                "event_type": event.event_type.value,
                "node_name": event.node_name,
                "payload": event.payload,
            }
            for event in events
        ],
    }
    return sha256(_canonical_json(document).encode("utf-8")).hexdigest()


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
        if any(event.event_key is not None for event in normalized_events):
            raise RunRepositoryTransitionError(
                "initial lifecycle events must not define event_key"
            )
        request_fingerprint = _request_fingerprint(normalized_snapshot.request)
        key = (
            normalized_snapshot.request.tenant_id,
            normalized_snapshot.request.idempotency_key,
        )
        with self._lock:
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                existing = self._records[existing_id]
                if existing.request_fingerprint != request_fingerprint:
                    raise RunRepositoryIdempotencyConflict()
                return CreateRunResult(self._view(existing), created=False)
            if normalized_snapshot.run_id in self._records:
                raise RunRepositoryIdempotencyConflict()

            stored = _StoredRun(
                snapshot=normalized_snapshot,
                request_fingerprint=request_fingerprint,
                version=1,
                events=[],
                event_claims={},
                mutation_claims={},
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
        if normalized_event.event_key is None:
            raise RunRepositoryTransitionError(
                "running stream events require a stable event_key"
            )
        fingerprint = _event_fingerprint(normalized_event, expected_version)
        with self._lock:
            stored = self._require(run_id, tenant_id)
            existing = stored.event_claims.get(normalized_event.event_key)
            if existing is not None:
                if (
                    existing.execution_version != expected_version
                    or existing.fingerprint != fingerprint
                ):
                    raise RunRepositoryEventConflict()
                return deepcopy(existing.event)
            if (
                stored.version != expected_version
                or stored.snapshot.status is not RunStatus.RUNNING
            ):
                raise ExecutionSuperseded("execution no longer owns this run")
            appended = self._append_event_locked(stored, normalized_event)
            stored.event_claims[normalized_event.event_key] = _StoredEventClaim(
                execution_version=expected_version,
                fingerprint=fingerprint,
                event=appended,
            )
            return deepcopy(appended)

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
        self._validate_snapshot_state(normalized_snapshot)
        if any(event.event_key is not None for event in normalized_events):
            raise RunRepositoryTransitionError(
                "lifecycle transition events must not define event_key"
            )
        target_fingerprint = _snapshot_fingerprint(normalized_snapshot)
        transition_fingerprint = _transition_fingerprint(
            normalized_events,
            expected_version,
        )
        with self._lock:
            stored = self._require(run_id, tenant_id)
            if stored.version != expected_version:
                claim = stored.mutation_claims.get(expected_version + 1)
                if claim is not None and (
                    claim.snapshot_fingerprint == target_fingerprint
                    and claim.transition_fingerprint == transition_fingerprint
                ):
                    return VersionedRun(
                        snapshot=deepcopy(normalized_snapshot),
                        version=expected_version + 1,
                    )
                raise RunRepositoryVersionConflict()
            self._validate_replacement(stored.snapshot, normalized_snapshot)
            self._validate_transition_events(normalized_snapshot, normalized_events)
            candidate = _StoredRun(
                snapshot=normalized_snapshot,
                request_fingerprint=stored.request_fingerprint,
                version=stored.version + 1,
                events=deepcopy(stored.events),
                event_claims=deepcopy(stored.event_claims),
                mutation_claims=deepcopy(stored.mutation_claims),
            )
            for event in normalized_events:
                self._append_event_locked(candidate, event)
            candidate.mutation_claims[candidate.version] = _StoredMutationClaim(
                snapshot_fingerprint=target_fingerprint,
                transition_fingerprint=transition_fingerprint,
            )
            stored.snapshot = candidate.snapshot
            stored.version = candidate.version
            stored.events = candidate.events
            stored.event_claims = candidate.event_claims
            stored.mutation_claims = candidate.mutation_claims
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
            timestamp=_canonical_utc_millisecond(
                self._clock(),
                field_name="event timestamp",
            ),
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
        if _request_fingerprint(replacement.request) != _request_fingerprint(
            current.request
        ):
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
            event_key=event.event_key,
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
        created_at = _canonical_utc_millisecond(
            snapshot.created_at,
            field_name="created_at",
        )
        updated_at = _canonical_utc_millisecond(
            snapshot.updated_at,
            field_name="updated_at",
        )
        if datetime.strptime(
            updated_at,
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ) < datetime.strptime(
            created_at,
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ):
            raise ValueError("updated_at must not precede created_at")
        normalized = replace(
            snapshot,
            request=request,
            created_at=created_at,
            updated_at=updated_at,
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
