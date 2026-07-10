"""In-memory run service and deterministic executor for API development."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .models import (
    ApprovalDecision,
    EventType,
    ExecutionOutcome,
    RunEvent,
    RunRequest,
    RunSnapshot,
    RunStatus,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RunServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


Emit = Callable[[EventType, str | None, Mapping[str, Any]], None]


class RunExecutor(Protocol):
    def execute(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome: ...


class ScenarioExecutor:
    """Predictable executor controlled by request metadata.

    `requires_approval`, `force_failure`, and `dependency_timeout` are mock-only
    controls and must never become production request parameters.
    """

    def __init__(self) -> None:
        self.execution_count = 0

    def execute(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome:
        self.execution_count += 1
        start_node = "resume_from_checkpoint" if checkpoint else "parse_request"
        emit(EventType.NODE_STARTED, start_node, {"task_type": request.task_type})
        emit(EventType.NODE_COMPLETED, start_node, {"ok": True})

        if request.metadata.get("dependency_timeout"):
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint={"last_completed_node": start_node, "recoverable": True},
                error={"code": "TIMEOUT", "message": "mock dependency timeout", "retryable": True},
            )
        if request.metadata.get("force_failure"):
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint={"last_completed_node": start_node, "recoverable": False},
                error={"code": "MOCK_FAILURE", "message": "forced failure", "retryable": False},
            )
        if request.metadata.get("requires_approval") and approval is None:
            approval_id = str(request.metadata.get("approval_id", "approval-1"))
            emit(EventType.APPROVAL_REQUESTED, "human_approval", {"approval_id": approval_id})
            return ExecutionOutcome(
                status=RunStatus.WAITING_APPROVAL,
                checkpoint={"last_completed_node": start_node, "waiting_at": "human_approval"},
                approval_id=approval_id,
            )
        if approval is not None and not approval.approved:
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint=dict(checkpoint or {}),
                error={"code": "APPROVAL_REJECTED", "message": approval.comment or "rejected", "retryable": False},
            )

        emit(EventType.NODE_STARTED, "report_generator", {})
        report = {
            "case_id": request.case_id,
            "task_type": request.task_type,
            "summary": "Mock workflow completed; replace modules according to task cards.",
            "artifact_ids": list(request.artifact_ids),
        }
        emit(EventType.NODE_COMPLETED, "report_generator", {"verified": True})
        return ExecutionOutcome(status=RunStatus.SUCCEEDED, final_report=report)


@dataclass(slots=True)
class _RunRecord:
    snapshot: RunSnapshot
    events: list[RunEvent] = field(default_factory=list)


class InMemoryRunService:
    def __init__(self, executor: RunExecutor | None = None) -> None:
        self.executor = executor or ScenarioExecutor()
        self._records: dict[str, _RunRecord] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def create_run(self, request: RunRequest) -> RunSnapshot:
        key = (request.tenant_id, request.idempotency_key)
        with self._lock:
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                existing = self._records[existing_id].snapshot
                if existing.request != request:
                    raise RunServiceError("IDEMPOTENCY_CONFLICT", "key was used for a different request")
                return existing

            now = _utc_now()
            run_id = f"run-{uuid4().hex}"
            trace_id = f"trace-{uuid4().hex}"
            snapshot = RunSnapshot(
                run_id=run_id,
                request=request,
                status=RunStatus.PENDING,
                trace_id=trace_id,
                created_at=now,
                updated_at=now,
                final_report={},
                checkpoint={},
                approval_id=None,
                error=None,
            )
            self._records[run_id] = _RunRecord(snapshot)
            self._idempotency[key] = run_id
            self._emit(run_id, EventType.RUN_CREATED, None, {"case_id": request.case_id})
            self._set_status(run_id, RunStatus.RUNNING)
            self._emit(run_id, EventType.RUN_STARTED, None, {})

        self._execute(run_id)
        return self.get_run(run_id, tenant_id=request.tenant_id)

    def get_run(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        with self._lock:
            record = self._require_authorized(run_id, tenant_id)
            return record.snapshot

    def get_events(self, run_id: str, *, tenant_id: str, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        if after_sequence < 0:
            raise RunServiceError("INVALID_ARGUMENT", "after_sequence must be non-negative")
        with self._lock:
            record = self._require_authorized(run_id, tenant_id)
            return tuple(event for event in record.events if event.sequence > after_sequence)

    def approve(self, run_id: str, decision: ApprovalDecision, *, tenant_id: str) -> RunSnapshot:
        with self._lock:
            record = self._require_authorized(run_id, tenant_id)
            snapshot = record.snapshot
            if snapshot.status is not RunStatus.WAITING_APPROVAL:
                raise RunServiceError("CONFLICT", "run is not waiting for approval")
            if decision.approval_id != snapshot.approval_id:
                raise RunServiceError("CONFLICT", "approval_id does not match current checkpoint")
            self._set_status(run_id, RunStatus.RUNNING)
        self._execute(run_id, approval=decision)
        return self.get_run(run_id, tenant_id=tenant_id)

    def cancel(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        with self._lock:
            record = self._require_authorized(run_id, tenant_id)
            if record.snapshot.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
                raise RunServiceError("CONFLICT", "terminal run cannot be cancelled")
            self._set_status(run_id, RunStatus.CANCELLED)
            self._emit(run_id, EventType.RUN_CANCELLED, None, {})
            return self._records[run_id].snapshot

    def resume(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        with self._lock:
            record = self._require_authorized(run_id, tenant_id)
            if record.snapshot.status is not RunStatus.FAILED:
                raise RunServiceError("CONFLICT", "only FAILED runs can be resumed")
            if not record.snapshot.checkpoint.get("recoverable"):
                raise RunServiceError("CONFLICT", "run has no recoverable checkpoint")
            self._set_status(run_id, RunStatus.RUNNING)
        self._execute(run_id)
        return self.get_run(run_id, tenant_id=tenant_id)

    def _execute(self, run_id: str, approval: ApprovalDecision | None = None) -> None:
        with self._lock:
            record = self._records[run_id]
            request = record.snapshot.request
            checkpoint = record.snapshot.checkpoint or None

        try:
            outcome = self.executor.execute(
                request,
                lambda event_type, node_name, payload: self._emit(run_id, event_type, node_name, payload),
                checkpoint=checkpoint,
                approval=approval,
            )
        except Exception as exc:
            outcome = ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint=dict(checkpoint or {}),
                error={"code": "EXECUTOR_ERROR", "message": str(exc), "retryable": False},
            )

        with self._lock:
            record = self._records[run_id]
            record.snapshot = replace(
                record.snapshot,
                status=outcome.status,
                updated_at=_utc_now(),
                final_report=dict(outcome.final_report),
                checkpoint=dict(outcome.checkpoint),
                approval_id=outcome.approval_id,
                error=(dict(outcome.error) if outcome.error is not None else None),
            )
            if outcome.status is RunStatus.SUCCEEDED:
                self._emit(run_id, EventType.RUN_COMPLETED, None, {"status": outcome.status.value})
            elif outcome.status is RunStatus.FAILED:
                self._emit(run_id, EventType.RUN_FAILED, None, dict(outcome.error or {}))

    def _set_status(self, run_id: str, status: RunStatus) -> None:
        record = self._records[run_id]
        record.snapshot = replace(record.snapshot, status=status, updated_at=_utc_now())

    def _emit(
        self,
        run_id: str,
        event_type: EventType,
        node_name: str | None,
        payload: Mapping[str, Any],
    ) -> None:
        with self._lock:
            record = self._records[run_id]
            redacted = self._redact(payload)
            record.events.append(
                RunEvent(
                    run_id=run_id,
                    sequence=len(record.events) + 1,
                    event_type=event_type,
                    node_name=node_name,
                    timestamp=_utc_now(),
                    payload=redacted,
                    trace_id=record.snapshot.trace_id,
                )
            )

    def _require_authorized(self, run_id: str, tenant_id: str) -> _RunRecord:
        record = self._records.get(run_id)
        if record is None or record.snapshot.request.tenant_id != tenant_id:
            # Same response avoids confirming that another tenant's run exists.
            raise RunServiceError("NOT_FOUND", "run was not found")
        return record

    @classmethod
    def _redact(cls, value: Any) -> Any:
        sensitive = {"internal_uri", "uri", "authorization", "token", "api_key", "patient_name"}
        if isinstance(value, Mapping):
            return {
                str(key): ("[REDACTED]" if str(key).lower() in sensitive else cls._redact(item))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._redact(item) for item in value]
        return value
