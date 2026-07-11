"""Atomic Run/event persistence protocol and dependency-free memory backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
import math
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
from .models import (
    ApprovalDecision,
    EventType,
    RunEvent,
    RunRequest,
    RunSnapshot,
    RunStatus,
)


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


class RunExecutionIntentKind(str, Enum):
    CREATE = "CREATE"
    APPROVAL = "APPROVAL"
    RESUME = "RESUME"


@dataclass(frozen=True, slots=True)
class RunExecutionIntent:
    kind: RunExecutionIntentKind
    approval: ApprovalDecision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RunExecutionIntentKind):
            raise TypeError("execution intent kind is invalid")
        if self.kind is RunExecutionIntentKind.APPROVAL:
            if not isinstance(self.approval, ApprovalDecision):
                raise ValueError("approval execution intent requires a decision")
            approval = self.approval
            if type(approval.approved) is not bool:
                raise ValueError("approval decision must use a boolean")
            for field_name, value, maximum in (
                ("approval_id", approval.approval_id, 128),
                ("principal_id", approval.principal_id, 128),
                ("comment", approval.comment, 4096),
            ):
                if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
                    raise ValueError(f"approval {field_name} is invalid")
                if field_name != "comment" and (
                    not value.strip() or value != value.strip()
                ):
                    raise ValueError(f"approval {field_name} is invalid")
                if any(character in value for character in ("\x00",)):
                    raise ValueError(f"approval {field_name} is invalid")
            object.__setattr__(self, "approval", deepcopy(self.approval))
        elif self.approval is not None:
            raise ValueError("only approval execution intent accepts a decision")


def _execution_intent_document(intent: RunExecutionIntent) -> dict[str, Any]:
    if not isinstance(intent, RunExecutionIntent):
        raise TypeError("execution intent is invalid")
    approval = intent.approval
    return copy_json_value(
        {
            "kind": intent.kind.value,
            "approval": (
                {
                    "approval_id": approval.approval_id,
                    "approved": approval.approved,
                    "principal_id": approval.principal_id,
                    "comment": approval.comment,
                }
                if approval is not None
                else None
            ),
        }
    )


def _execution_intent_fingerprint(intent: RunExecutionIntent) -> str:
    return sha256(
        _canonical_json(_execution_intent_document(intent)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RunExecutionClaim:
    run: VersionedRun
    intent: RunExecutionIntent
    generation: int
    owner_token: str
    worker_id: str
    claimed_at: str
    heartbeat_at: str
    lease_expires_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, VersionedRun):
            raise TypeError("execution claim run is invalid")
        if not isinstance(self.intent, RunExecutionIntent):
            raise TypeError("execution claim intent is invalid")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ValueError("execution claim generation starts at 1")
        for field_name in ("owner_token", "worker_id"):
            _validate_execution_identifier(field_name, getattr(self, field_name))
        for field_name in ("claimed_at", "heartbeat_at", "lease_expires_at"):
            _canonical_utc_millisecond(getattr(self, field_name), field_name=field_name)


def _validate_execution_identifier(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    if len(encoded) > 128 or any(
        character in value for character in ("\r", "\n", "\x00")
    ):
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    return value


def _validate_execution_lease_seconds(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.05
        or float(value) > 3600.0
    ):
        raise ValueError("execution lease must be between 0.05 and 3600 seconds")
    return float(value)


def _parse_canonical_timestamp(value: str) -> datetime:
    _canonical_utc_millisecond(value, field_name="execution timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _format_canonical_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class RunEventPage:
    events: tuple[RunEvent, ...]
    high_water_sequence: int
    status: RunStatus

    def __post_init__(self) -> None:
        if (
            isinstance(self.high_water_sequence, bool)
            or not isinstance(self.high_water_sequence, int)
            or self.high_water_sequence < 0
        ):
            raise ValueError("high_water_sequence must be non-negative")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        events = tuple(self.events)
        if any(not isinstance(event, RunEvent) for event in events):
            raise TypeError("events must contain RunEvent values")
        if events and events[-1].sequence > self.high_water_sequence:
            raise ValueError("event page exceeds its high-water sequence")
        object.__setattr__(self, "events", events)

    @property
    def has_more(self) -> bool:
        return bool(
            self.events
            and self.events[-1].sequence < self.high_water_sequence
        )


def _validate_event_page_request(after_sequence: int, limit: int) -> None:
    if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
        raise ValueError("after_sequence must be an integer")
    if after_sequence < 0:
        raise ValueError("after_sequence must be non-negative")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > 512
    ):
        raise ValueError("event page limit must be between 1 and 512")


@runtime_checkable
class RunEventPager(Protocol):
    def get_event_page(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
        limit: int = 128,
    ) -> RunEventPage: ...


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
class _StoredExecutionJob:
    intent: RunExecutionIntent
    intent_fingerprint: str
    execution_version: int
    generation: int = 0
    owner_token: str | None = None
    worker_id: str | None = None
    claimed_at: str | None = None
    heartbeat_at: str | None = None
    lease_expires_at: str | None = None
    released_at: str | None = None
    completed_at: str | None = None


@dataclass(slots=True)
class _StoredRun:
    snapshot: RunSnapshot
    request_fingerprint: str
    version: int
    events: list[RunEvent]
    event_claims: dict[str, _StoredEventClaim]
    mutation_claims: dict[int, _StoredMutationClaim]
    execution_jobs: dict[int, _StoredExecutionJob]


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
    def check_health(self) -> None: ...

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


@runtime_checkable
class RunExecutionRepository(Protocol):
    def compare_and_swap_and_enqueue(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        intent: RunExecutionIntent,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun: ...

    def claim_next_execution(
        self,
        *,
        worker_id: str,
        owner_token: str,
        lease_seconds: float,
    ) -> RunExecutionClaim | None: ...

    def heartbeat_execution_claim(
        self,
        claim: RunExecutionClaim,
        *,
        lease_seconds: float,
    ) -> RunExecutionClaim: ...

    def release_execution_claim(self, claim: RunExecutionClaim) -> None: ...

    def abandon_execution_claim(self, claim: RunExecutionClaim) -> None: ...

    def assert_execution_claim(self, claim: RunExecutionClaim) -> None: ...

    def append_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        event: RunEventDraft,
    ) -> RunEvent: ...

    def compare_and_swap_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
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

    def check_health(self) -> None:
        """The in-process reference repository is ready while reachable."""

        return None

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

            initial_intent = RunExecutionIntent(RunExecutionIntentKind.CREATE)
            stored = _StoredRun(
                snapshot=normalized_snapshot,
                request_fingerprint=request_fingerprint,
                version=1,
                events=[],
                event_claims={},
                mutation_claims={},
                execution_jobs={
                    1: _StoredExecutionJob(
                        intent=initial_intent,
                        intent_fingerprint=_execution_intent_fingerprint(
                            initial_intent
                        ),
                        execution_version=1,
                    )
                },
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

    def get_event_page(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
        limit: int = 128,
    ) -> RunEventPage:
        _validate_event_page_request(after_sequence, limit)
        with self._lock:
            stored = self._require(run_id, tenant_id)
            high_water = len(stored.events)
            start = min(after_sequence, high_water)
            events = tuple(
                deepcopy(event)
                for event in stored.events[start : start + limit]
            )
            return RunEventPage(
                events=events,
                high_water_sequence=high_water,
                status=stored.snapshot.status,
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
        return self._compare_and_swap(
            run_id,
            tenant_id=tenant_id,
            expected_version=expected_version,
            snapshot=snapshot,
            events=events,
            intent=None,
            execution_claim=None,
        )

    def compare_and_swap_and_enqueue(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        intent: RunExecutionIntent,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        if not isinstance(intent, RunExecutionIntent):
            raise TypeError("execution intent is invalid")
        return self._compare_and_swap(
            run_id,
            tenant_id=tenant_id,
            expected_version=expected_version,
            snapshot=snapshot,
            events=events,
            intent=deepcopy(intent),
            execution_claim=None,
        )

    def compare_and_swap_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        return self._compare_and_swap(
            claim.run.snapshot.run_id,
            tenant_id=claim.run.snapshot.request.tenant_id,
            expected_version=claim.run.version,
            snapshot=snapshot,
            events=events,
            intent=None,
            execution_claim=claim,
        )

    def _compare_and_swap(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...],
        intent: RunExecutionIntent | None,
        execution_claim: RunExecutionClaim | None,
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
        intent_fingerprint = (
            _execution_intent_fingerprint(intent) if intent is not None else None
        )
        with self._lock:
            stored = self._require(run_id, tenant_id)
            if stored.version != expected_version:
                mutation = stored.mutation_claims.get(expected_version + 1)
                replay_matches = mutation is not None and (
                    mutation.snapshot_fingerprint == target_fingerprint
                    and mutation.transition_fingerprint == transition_fingerprint
                )
                if replay_matches and intent_fingerprint is not None:
                    target_job = stored.execution_jobs.get(expected_version + 1)
                    replay_matches = bool(
                        target_job is not None
                        and target_job.intent_fingerprint == intent_fingerprint
                    )
                if replay_matches and execution_claim is not None:
                    source_job = stored.execution_jobs.get(expected_version)
                    replay_matches = bool(
                        source_job is not None
                        and self._claim_identity_matches_locked(
                            source_job,
                            execution_claim,
                        )
                    )
                if replay_matches:
                    return VersionedRun(
                        snapshot=deepcopy(normalized_snapshot),
                        version=expected_version + 1,
                    )
                raise RunRepositoryVersionConflict()
            if execution_claim is not None:
                self._require_active_claim_locked(stored, execution_claim)
            self._validate_replacement(stored.snapshot, normalized_snapshot)
            self._validate_transition_events(normalized_snapshot, normalized_events)
            if intent is not None and normalized_snapshot.status is not RunStatus.RUNNING:
                raise RunRepositoryTransitionError(
                    "execution intent requires a RUNNING target"
                )
            if intent is not None:
                expected_intent = {
                    RunStatus.WAITING_APPROVAL: RunExecutionIntentKind.APPROVAL,
                    RunStatus.FAILED: RunExecutionIntentKind.RESUME,
                }.get(stored.snapshot.status)
                if intent.kind is not expected_intent:
                    raise RunRepositoryTransitionError(
                        "execution intent does not match the lifecycle transition"
                    )
            candidate = _StoredRun(
                snapshot=normalized_snapshot,
                request_fingerprint=stored.request_fingerprint,
                version=stored.version + 1,
                events=deepcopy(stored.events),
                event_claims=deepcopy(stored.event_claims),
                mutation_claims=deepcopy(stored.mutation_claims),
                execution_jobs=deepcopy(stored.execution_jobs),
            )
            completed_at = self._canonical_now()
            current_job = candidate.execution_jobs.get(expected_version)
            if stored.snapshot.status is RunStatus.RUNNING and current_job is not None:
                current_job.completed_at = completed_at
                current_job.released_at = completed_at
            for event in normalized_events:
                self._append_event_locked(candidate, event)
            candidate.mutation_claims[candidate.version] = _StoredMutationClaim(
                snapshot_fingerprint=target_fingerprint,
                transition_fingerprint=transition_fingerprint,
            )
            if intent is not None:
                if candidate.version in candidate.execution_jobs:
                    raise RunRepositoryTransitionError(
                        "execution job already exists for target version"
                    )
                candidate.execution_jobs[candidate.version] = _StoredExecutionJob(
                    intent=deepcopy(intent),
                    intent_fingerprint=_execution_intent_fingerprint(intent),
                    execution_version=candidate.version,
                )
            stored.snapshot = candidate.snapshot
            stored.version = candidate.version
            stored.events = candidate.events
            stored.event_claims = candidate.event_claims
            stored.mutation_claims = candidate.mutation_claims
            stored.execution_jobs = candidate.execution_jobs
            return self._view(stored)

    def claim_next_execution(
        self,
        *,
        worker_id: str,
        owner_token: str,
        lease_seconds: float,
    ) -> RunExecutionClaim | None:
        _validate_execution_identifier("worker_id", worker_id)
        _validate_execution_identifier("owner_token", owner_token)
        lease = _validate_execution_lease_seconds(lease_seconds)
        with self._lock:
            now = self._canonical_now()
            now_value = _parse_canonical_timestamp(now)
            for stored in self._records.values():
                for job in stored.execution_jobs.values():
                    if job.owner_token != owner_token:
                        continue
                    if (
                        job.worker_id == worker_id
                        and job.execution_version == stored.version
                        and self._job_is_active_at(job, now_value)
                    ):
                        return self._execution_claim_view(stored, job)
                    raise RunRepositoryTransitionError(
                        "execution owner token is already in use"
                    )
            ordered = sorted(
                self._records.values(),
                key=lambda stored: (
                    stored.snapshot.updated_at,
                    stored.snapshot.run_id,
                ),
            )
            for stored in ordered:
                if stored.snapshot.status is not RunStatus.RUNNING:
                    continue
                job = stored.execution_jobs.get(stored.version)
                if job is None or job.completed_at is not None:
                    continue
                if self._job_is_active_at(job, now_value):
                    continue
                job.generation += 1
                job.owner_token = owner_token
                job.worker_id = worker_id
                job.claimed_at = now
                job.heartbeat_at = now
                job.lease_expires_at = _format_canonical_timestamp(
                    now_value + timedelta(seconds=lease)
                )
                job.released_at = None
                return self._execution_claim_view(stored, job)
            return None

    def heartbeat_execution_claim(
        self,
        claim: RunExecutionClaim,
        *,
        lease_seconds: float,
    ) -> RunExecutionClaim:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        lease = _validate_execution_lease_seconds(lease_seconds)
        with self._lock:
            stored = self._require(
                claim.run.snapshot.run_id,
                claim.run.snapshot.request.tenant_id,
            )
            now = self._canonical_now()
            now_value = _parse_canonical_timestamp(now)
            job = self._require_active_claim_locked(
                stored,
                claim,
                now=now_value,
            )
            job.heartbeat_at = now
            job.lease_expires_at = _format_canonical_timestamp(
                now_value + timedelta(seconds=lease)
            )
            return self._execution_claim_view(stored, job)

    def release_execution_claim(self, claim: RunExecutionClaim) -> None:
        self._finish_execution_claim(claim)

    def abandon_execution_claim(self, claim: RunExecutionClaim) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        with self._lock:
            stored = self._require(
                claim.run.snapshot.run_id,
                claim.run.snapshot.request.tenant_id,
            )
            job = stored.execution_jobs.get(claim.run.version)
            if job is None or not self._claim_identity_matches_locked(job, claim):
                return

    def _finish_execution_claim(
        self,
        claim: RunExecutionClaim,
    ) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        with self._lock:
            stored = self._require(
                claim.run.snapshot.run_id,
                claim.run.snapshot.request.tenant_id,
            )
            job = stored.execution_jobs.get(claim.run.version)
            if job is None or not self._claim_identity_matches_locked(job, claim):
                return
            now = self._canonical_now()
            job.released_at = now
            job.lease_expires_at = now

    def assert_execution_claim(self, claim: RunExecutionClaim) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        with self._lock:
            stored = self._require(
                claim.run.snapshot.run_id,
                claim.run.snapshot.request.tenant_id,
            )
            self._require_active_claim_locked(stored, claim)

    def append_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        event: RunEventDraft,
    ) -> RunEvent:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        normalized_event = self._copy_event(event)
        expected_version = claim.run.version
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
            stored = self._require(
                claim.run.snapshot.run_id,
                claim.run.snapshot.request.tenant_id,
            )
            existing = stored.event_claims.get(normalized_event.event_key)
            if existing is not None:
                if (
                    existing.execution_version != expected_version
                    or existing.fingerprint != fingerprint
                ):
                    raise RunRepositoryEventConflict()
                job = stored.execution_jobs.get(expected_version)
                if job is None or not self._claim_identity_matches_locked(job, claim):
                    raise ExecutionSuperseded("execution claim was superseded")
                return deepcopy(existing.event)
            self._require_active_claim_locked(stored, claim)
            return self.append_if_running(
                claim.run.snapshot.run_id,
                tenant_id=claim.run.snapshot.request.tenant_id,
                expected_version=expected_version,
                event=normalized_event,
            )

    def _canonical_now(self) -> str:
        return _canonical_utc_millisecond(
            self._clock(),
            field_name="execution clock",
        )

    @staticmethod
    def _job_is_active_at(job: _StoredExecutionJob, now: datetime) -> bool:
        return bool(
            job.owner_token is not None
            and job.worker_id is not None
            and job.released_at is None
            and job.completed_at is None
            and job.lease_expires_at is not None
            and _parse_canonical_timestamp(job.lease_expires_at) > now
        )

    @staticmethod
    def _claim_identity_matches_locked(
        job: _StoredExecutionJob,
        claim: RunExecutionClaim,
    ) -> bool:
        return bool(
            job.execution_version == claim.run.version
            and job.generation == claim.generation
            and job.owner_token == claim.owner_token
            and job.worker_id == claim.worker_id
            and job.intent_fingerprint
            == _execution_intent_fingerprint(claim.intent)
        )

    def _require_active_claim_locked(
        self,
        stored: _StoredRun,
        claim: RunExecutionClaim,
        *,
        now: datetime | None = None,
    ) -> _StoredExecutionJob:
        job = stored.execution_jobs.get(claim.run.version)
        current_time = now or _parse_canonical_timestamp(self._canonical_now())
        if (
            stored.version != claim.run.version
            or stored.snapshot.status is not RunStatus.RUNNING
            or job is None
            or not self._claim_identity_matches_locked(job, claim)
            or not self._job_is_active_at(job, current_time)
        ):
            raise ExecutionSuperseded("execution claim is no longer active")
        return job

    @staticmethod
    def _execution_claim_view(
        stored: _StoredRun,
        job: _StoredExecutionJob,
    ) -> RunExecutionClaim:
        if any(
            value is None
            for value in (
                job.owner_token,
                job.worker_id,
                job.claimed_at,
                job.heartbeat_at,
                job.lease_expires_at,
            )
        ):
            raise RunRepositoryTransitionError("execution claim is incomplete")
        return RunExecutionClaim(
            run=InMemoryRunRepository._view(stored),
            intent=deepcopy(job.intent),
            generation=job.generation,
            owner_token=job.owner_token or "",
            worker_id=job.worker_id or "",
            claimed_at=job.claimed_at or "",
            heartbeat_at=job.heartbeat_at or "",
            lease_expires_at=job.lease_expires_at or "",
        )

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
    "RunEventPage",
    "RunEventPager",
    "RunExecutionClaim",
    "RunExecutionIntent",
    "RunExecutionIntentKind",
    "RunExecutionRepository",
    "RunRepository",
    "VersionedRun",
    "_execution_intent_document",
    "_execution_intent_fingerprint",
    "_validate_execution_identifier",
    "_validate_execution_lease_seconds",
]
