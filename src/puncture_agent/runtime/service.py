"""Repository-backed Run service and deterministic executor for API development."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .errors import (
    ExecutionSuperseded,
    RunRepositoryError,
    RunRepositoryVersionConflict,
    RunServiceError,
)
from .models import (
    ApprovalDecision,
    EventType,
    ExecutionOutcome,
    RunEvent,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from .json_boundary import RuntimeJsonBoundaryError, copy_json_mapping
from .repository import (
    InMemoryRunRepository,
    RunEventDraft,
    RunRepository,
    VersionedRun,
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


Emit = Callable[[EventType, str | None, Mapping[str, Any]], None]
_EXECUTION_STREAM_EVENTS = frozenset(
    {
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
        EventType.TOOL_CALLED,
        EventType.TOOL_RESULT,
    }
)


class _ExecutorContractViolation(RuntimeError):
    pass


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
                error={
                    "code": "TIMEOUT",
                    "message": "mock dependency timeout",
                    "retryable": True,
                },
            )
        if request.metadata.get("force_failure"):
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint={"last_completed_node": start_node, "recoverable": False},
                error={
                    "code": "MOCK_FAILURE",
                    "message": "forced failure",
                    "retryable": False,
                },
            )
        if request.metadata.get("requires_approval") and approval is None:
            approval_id = str(request.metadata.get("approval_id", "approval-1"))
            emit(
                EventType.APPROVAL_REQUESTED,
                "human_approval",
                {"approval_id": approval_id},
            )
            return ExecutionOutcome(
                status=RunStatus.WAITING_APPROVAL,
                checkpoint={
                    "last_completed_node": start_node,
                    "waiting_at": "human_approval",
                },
                approval_id=approval_id,
            )
        if approval is not None and not approval.approved:
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint=dict(checkpoint or {}),
                error={
                    "code": "APPROVAL_REJECTED",
                    "message": approval.comment or "rejected",
                    "retryable": False,
                },
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


class InMemoryRunService:
    def __init__(
        self,
        executor: RunExecutor | None = None,
        *,
        repository: RunRepository | None = None,
    ) -> None:
        self.executor = executor if executor is not None else ScenarioExecutor()
        self.repository = (
            repository if repository is not None else InMemoryRunRepository()
        )

    def create_run(self, request: RunRequest) -> RunSnapshot:
        now = _utc_now()
        snapshot = RunSnapshot(
            run_id=f"run-{uuid4().hex}",
            request=request,
            status=RunStatus.RUNNING,
            trace_id=f"trace-{uuid4().hex}",
            created_at=now,
            updated_at=now,
            final_report={},
            checkpoint={},
            approval_id=None,
            error=None,
        )
        initial_events = (
            RunEventDraft(
                EventType.RUN_CREATED,
                None,
                self._redact({"case_id": request.case_id}),
            ),
            RunEventDraft(EventType.RUN_STARTED, None, {}),
        )
        try:
            created = self.repository.create_or_get_started(
                snapshot,
                initial_events,
            )
        except RuntimeJsonBoundaryError as exc:
            raise RunServiceError(
                "INVALID_REQUEST",
                "run request cannot cross the durable JSON boundary",
            ) from exc
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc
        if not created.created:
            return created.run.snapshot

        self._execute(created.run)
        return self.get_run(snapshot.run_id, tenant_id=request.tenant_id)

    def get_run(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        return self._get_versioned(run_id, tenant_id=tenant_id).snapshot

    def get_events(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise RunServiceError(
                "INVALID_ARGUMENT",
                "after_sequence must be an integer",
            )
        if after_sequence < 0:
            raise RunServiceError(
                "INVALID_ARGUMENT",
                "after_sequence must be non-negative",
            )
        try:
            return self.repository.get_events(
                run_id,
                tenant_id=tenant_id,
                after_sequence=after_sequence,
            )
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc

    def approve(
        self,
        run_id: str,
        decision: ApprovalDecision,
        *,
        tenant_id: str,
    ) -> RunSnapshot:
        current = self._get_versioned(run_id, tenant_id=tenant_id)
        snapshot = current.snapshot
        if snapshot.status is not RunStatus.WAITING_APPROVAL:
            raise RunServiceError("CONFLICT", "run is not waiting for approval")
        if decision.approval_id != snapshot.approval_id:
            raise RunServiceError(
                "CONFLICT",
                "approval_id does not match current checkpoint",
            )
        replacement = replace(
            snapshot,
            status=RunStatus.RUNNING,
            updated_at=_utc_now(),
            final_report={},
            approval_id=None,
            error=None,
        )
        resumed = self._compare_and_swap(
            current,
            replacement,
            events=(),
        )
        self._execute(resumed, approval=decision)
        return self.get_run(run_id, tenant_id=tenant_id)

    def cancel(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        current = self._get_versioned(run_id, tenant_id=tenant_id)
        if current.snapshot.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise RunServiceError("CONFLICT", "terminal run cannot be cancelled")
        replacement = replace(
            current.snapshot,
            status=RunStatus.CANCELLED,
            updated_at=_utc_now(),
            approval_id=None,
            error=None,
        )
        cancelled = self._compare_and_swap(
            current,
            replacement,
            events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
        )
        return cancelled.snapshot

    def resume(self, run_id: str, *, tenant_id: str) -> RunSnapshot:
        current = self._get_versioned(run_id, tenant_id=tenant_id)
        snapshot = current.snapshot
        if snapshot.status is not RunStatus.FAILED:
            raise RunServiceError("CONFLICT", "only FAILED runs can be resumed")
        if not snapshot.checkpoint.get("recoverable"):
            raise RunServiceError(
                "CONFLICT",
                "run has no recoverable checkpoint",
            )
        replacement = replace(
            snapshot,
            status=RunStatus.RUNNING,
            updated_at=_utc_now(),
            final_report={},
            approval_id=None,
            error=None,
        )
        resumed = self._compare_and_swap(current, replacement, events=())
        self._execute(resumed)
        return self.get_run(run_id, tenant_id=tenant_id)

    def _execute(
        self,
        run: VersionedRun,
        approval: ApprovalDecision | None = None,
    ) -> None:
        snapshot = run.snapshot
        tenant_id = snapshot.request.tenant_id
        checkpoint = copy_json_mapping(snapshot.checkpoint) or None
        request = deepcopy(snapshot.request)
        pending_transition_events: list[RunEventDraft] = []

        def emit(
            event_type: EventType,
            node_name: str | None,
            payload: Mapping[str, Any],
        ) -> None:
            if pending_transition_events:
                raise _ExecutorContractViolation(
                    "executor emitted an event after requesting approval"
                )
            if not isinstance(event_type, EventType):
                raise _ExecutorContractViolation("executor emitted an invalid event")
            if event_type not in _EXECUTION_STREAM_EVENTS and (
                event_type is not EventType.APPROVAL_REQUESTED
            ):
                raise _ExecutorContractViolation(
                    "executor emitted a state event outside a state transition"
                )
            try:
                durable_payload = copy_json_mapping(payload)
                draft = RunEventDraft(
                    event_type,
                    node_name,
                    self._redact(durable_payload),
                )
            except (RuntimeJsonBoundaryError, TypeError, ValueError) as exc:
                raise _ExecutorContractViolation(
                    "executor event crossed the JSON boundary"
                ) from exc
            if event_type is EventType.APPROVAL_REQUESTED:
                self.repository.assert_running(
                    snapshot.run_id,
                    tenant_id=tenant_id,
                    expected_version=run.version,
                )
                pending_transition_events.append(draft)
                return
            self.repository.append_if_running(
                snapshot.run_id,
                tenant_id=tenant_id,
                expected_version=run.version,
                event=draft,
            )

        try:
            outcome = self.executor.execute(
                request,
                emit,
                checkpoint=checkpoint,
                approval=approval,
            )
        except ExecutionSuperseded:
            return
        except _ExecutorContractViolation:
            outcome = self._failed_outcome(
                checkpoint=checkpoint,
                code="EXECUTOR_CONTRACT_ERROR",
                message="executor violated the event contract",
            )
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc
        except Exception:
            outcome = self._failed_outcome(
                checkpoint=checkpoint,
                code="EXECUTOR_ERROR",
                message="executor failed",
            )

        outcome = self._normalize_outcome(outcome, fallback_checkpoint=checkpoint)
        if outcome.error and outcome.error.get("code") == "EXECUTOR_CONTRACT_ERROR":
            pending_transition_events.clear()

        if outcome.status is RunStatus.WAITING_APPROVAL:
            if not pending_transition_events:
                pending_transition_events.append(
                    RunEventDraft(
                        EventType.APPROVAL_REQUESTED,
                        "human_approval",
                        {"approval_id": outcome.approval_id},
                    )
                )
            elif len(pending_transition_events) != 1:
                outcome = self._failed_outcome(
                    checkpoint=outcome.checkpoint,
                    code="EXECUTOR_CONTRACT_ERROR",
                    message="executor emitted multiple approval requests",
                )
                pending_transition_events.clear()
            elif (
                pending_transition_events[0].payload.get("approval_id")
                != outcome.approval_id
            ):
                outcome = self._failed_outcome(
                    checkpoint=outcome.checkpoint,
                    code="EXECUTOR_CONTRACT_ERROR",
                    message="approval event did not match the waiting approval_id",
                )
                pending_transition_events.clear()
        elif pending_transition_events:
            outcome = self._failed_outcome(
                checkpoint=outcome.checkpoint,
                code="EXECUTOR_CONTRACT_ERROR",
                message="approval event did not match the executor outcome",
            )
            pending_transition_events.clear()

        transition_events = list(pending_transition_events)
        if outcome.status is RunStatus.SUCCEEDED:
            transition_events.append(
                RunEventDraft(
                    EventType.RUN_COMPLETED,
                    None,
                    {"status": outcome.status.value},
                )
            )
        elif outcome.status is RunStatus.FAILED:
            transition_events.append(
                RunEventDraft(
                    EventType.RUN_FAILED,
                    None,
                    self._redact(outcome.error or {}),
                )
            )
        elif outcome.status is RunStatus.CANCELLED:
            transition_events.append(
                RunEventDraft(EventType.RUN_CANCELLED, None, {})
            )

        replacement = replace(
            snapshot,
            status=outcome.status,
            updated_at=_utc_now(),
            final_report=dict(outcome.final_report),
            checkpoint=dict(outcome.checkpoint),
            approval_id=outcome.approval_id,
            error=(deepcopy(dict(outcome.error)) if outcome.error is not None else None),
        )
        try:
            self.repository.compare_and_swap(
                snapshot.run_id,
                tenant_id=tenant_id,
                expected_version=run.version,
                snapshot=replacement,
                events=tuple(transition_events),
            )
        except RunRepositoryVersionConflict:
            return
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc

    def _get_versioned(self, run_id: str, *, tenant_id: str) -> VersionedRun:
        try:
            return self.repository.get(run_id, tenant_id=tenant_id)
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc

    def _compare_and_swap(
        self,
        current: VersionedRun,
        replacement: RunSnapshot,
        *,
        events: tuple[RunEventDraft, ...],
    ) -> VersionedRun:
        try:
            return self.repository.compare_and_swap(
                current.snapshot.run_id,
                tenant_id=current.snapshot.request.tenant_id,
                expected_version=current.version,
                snapshot=replacement,
                events=events,
            )
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc

    @staticmethod
    def _failed_outcome(
        *,
        checkpoint: Mapping[str, Any] | None,
        code: str,
        message: str,
    ) -> ExecutionOutcome:
        return ExecutionOutcome(
            status=RunStatus.FAILED,
            checkpoint=dict(checkpoint or {}),
            error={
                "code": code,
                "message": message,
                "retryable": False,
            },
        )

    @classmethod
    def _normalize_outcome(
        cls,
        outcome: Any,
        *,
        fallback_checkpoint: Mapping[str, Any] | None,
    ) -> ExecutionOutcome:
        if not isinstance(outcome, ExecutionOutcome) or outcome.status not in {
            RunStatus.WAITING_APPROVAL,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return cls._failed_outcome(
                checkpoint=fallback_checkpoint,
                code="EXECUTOR_CONTRACT_ERROR",
                message="executor returned an invalid outcome",
            )
        try:
            if outcome.status is RunStatus.WAITING_APPROVAL:
                if outcome.error is not None:
                    raise RuntimeJsonBoundaryError(
                        "WAITING_APPROVAL cannot contain an error"
                    )
            elif outcome.approval_id is not None:
                raise RuntimeJsonBoundaryError(
                    "approval_id is only valid while waiting for approval"
                )
            if outcome.status is RunStatus.FAILED:
                if outcome.error is None:
                    raise RuntimeJsonBoundaryError("FAILED requires an error")
            elif outcome.error is not None:
                raise RuntimeJsonBoundaryError(
                    "error is only valid for a FAILED outcome"
                )
            if outcome.status is not RunStatus.SUCCEEDED and outcome.final_report:
                raise RuntimeJsonBoundaryError(
                    "final_report is only valid for a SUCCEEDED outcome"
                )
            if outcome.approval_id is not None and (
                not isinstance(outcome.approval_id, str)
                or not outcome.approval_id.strip()
                or len(outcome.approval_id) > 128
            ):
                raise RuntimeJsonBoundaryError("approval_id is invalid")
            return ExecutionOutcome(
                status=outcome.status,
                final_report=copy_json_mapping(outcome.final_report),
                checkpoint=copy_json_mapping(outcome.checkpoint),
                approval_id=outcome.approval_id,
                error=(
                    copy_json_mapping(outcome.error)
                    if outcome.error is not None
                    else None
                ),
            )
        except (RuntimeJsonBoundaryError, TypeError, ValueError):
            safe_checkpoint: Mapping[str, Any] | None = fallback_checkpoint
            try:
                safe_checkpoint = copy_json_mapping(outcome.checkpoint)
            except (RuntimeJsonBoundaryError, TypeError, ValueError):
                pass
            return cls._failed_outcome(
                checkpoint=safe_checkpoint,
                code="EXECUTOR_CONTRACT_ERROR",
                message="executor outcome crossed the JSON boundary",
            )

    @staticmethod
    def _service_error(exc: RunRepositoryError) -> RunServiceError:
        return RunServiceError(exc.code, str(exc), retryable=exc.retryable)

    @classmethod
    def _redact(cls, value: Any) -> Any:
        sensitive = {
            "internal_uri",
            "uri",
            "authorization",
            "token",
            "api_key",
            "patient_name",
        }
        if isinstance(value, Mapping):
            return {
                str(key): (
                    "[REDACTED]"
                    if str(key).lower() in sensitive
                    else cls._redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._redact(item) for item in value]
        return value


__all__ = [
    "Emit",
    "InMemoryRunService",
    "RunExecutor",
    "RunServiceError",
    "ScenarioExecutor",
]
