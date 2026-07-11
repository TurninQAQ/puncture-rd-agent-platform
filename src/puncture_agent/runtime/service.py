"""Repository-backed Run service and deterministic executor for API development."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, cast, runtime_checkable
from uuid import uuid4

from .errors import (
    ExecutionSuperseded,
    RunRepositoryEventConflict,
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
    RunEventPage,
    RunExecutionClaim,
    RunExecutionIntent,
    RunExecutionIntentKind,
    RunExecutionRepository,
    RunRepository,
    VersionedRun,
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class Emit(Protocol):
    def __call__(
        self,
        event_type: EventType,
        node_name: str | None,
        payload: Mapping[str, Any],
        *,
        event_key: str | None = None,
    ) -> None: ...


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


@runtime_checkable
class RunExecutor(Protocol):
    def execute(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome: ...


@dataclass(frozen=True, slots=True)
class RunExecutionContext:
    """Fenced execution identity exposed to a recovery-safe executor."""

    run_id: str
    trace_id: str
    version: int
    generation: int
    recovering: bool
    assert_active: Callable[[], None]
    stop_requested: Callable[[], bool]

    def __post_init__(self) -> None:
        for field_name in ("run_id", "trace_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("version", "generation"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.recovering, bool):
            raise TypeError("recovering must be a boolean")
        if not callable(self.assert_active) or not callable(self.stop_requested):
            raise TypeError("execution context callbacks must be callable")


@runtime_checkable
class RecoverableRunExecutor(Protocol):
    """Executor that can safely replay one fenced durable execution claim."""

    recovery_safe: bool

    def execute_claimed(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        context: RunExecutionContext,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome: ...


class ScenarioExecutor:
    """Predictable executor controlled by request metadata.

    `requires_approval`, `force_failure`, and `dependency_timeout` are mock-only
    controls and must never become production request parameters.
    """

    recovery_safe = True

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

    def execute_claimed(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        context: RunExecutionContext,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome:
        context.assert_active()
        outcome = self.execute(
            request,
            emit,
            checkpoint=checkpoint,
            approval=approval,
        )
        context.assert_active()
        return outcome


class InMemoryRunService:
    def __init__(
        self,
        executor: RunExecutor | RecoverableRunExecutor | None = None,
        *,
        repository: RunRepository | None = None,
        deferred_execution: bool = False,
        execution_notifier: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(deferred_execution, bool):
            raise TypeError("deferred_execution must be a boolean")
        if execution_notifier is not None and not callable(execution_notifier):
            raise TypeError("execution_notifier must be callable")
        self.executor = executor if executor is not None else ScenarioExecutor()
        self.repository = (
            repository if repository is not None else InMemoryRunRepository()
        )
        self.deferred_execution = deferred_execution
        self.execution_notifier = execution_notifier
        if deferred_execution:
            if execution_notifier is None:
                raise ValueError(
                    "deferred execution requires an execution_notifier"
                )
            if not isinstance(self.repository, RunExecutionRepository):
                raise ValueError(
                    "deferred execution requires a RunExecutionRepository"
                )
            if (
                not isinstance(self.executor, RecoverableRunExecutor)
                or self.executor.recovery_safe is not True
            ):
                raise ValueError(
                    "deferred execution requires a recovery-safe executor"
                )
        elif not isinstance(self.executor, RunExecutor):
            raise ValueError("inline execution requires a RunExecutor")

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
            if (
                self.deferred_execution
                and created.run.snapshot.status is RunStatus.RUNNING
            ):
                self._notify_execution()
            return created.run.snapshot

        if self.deferred_execution:
            self._notify_execution()
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

    def get_event_page(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
        limit: int = 128,
    ) -> RunEventPage:
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
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 512
        ):
            raise RunServiceError(
                "INVALID_ARGUMENT",
                "event page limit must be between 1 and 512",
            )
        pager = getattr(self.repository, "get_event_page", None)
        if not callable(pager):
            raise RunServiceError(
                "EVENT_PAGING_UNAVAILABLE",
                "run repository does not support event paging",
                retryable=True,
            )
        try:
            return pager(
                run_id,
                tenant_id=tenant_id,
                after_sequence=after_sequence,
                limit=limit,
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
        if self.deferred_execution:
            resumed = self._compare_and_swap_and_enqueue(
                current,
                replacement,
                intent=RunExecutionIntent(
                    RunExecutionIntentKind.APPROVAL,
                    approval=decision,
                ),
                events=(),
            )
            self._notify_execution()
            return resumed.snapshot
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
        if self.deferred_execution:
            resumed = self._compare_and_swap_and_enqueue(
                current,
                replacement,
                intent=RunExecutionIntent(RunExecutionIntentKind.RESUME),
                events=(),
            )
            self._notify_execution()
            return resumed.snapshot
        resumed = self._compare_and_swap(current, replacement, events=())
        self._execute(resumed)
        return self.get_run(run_id, tenant_id=tenant_id)

    def execute_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        """Execute one durable job while its repository claim remains active."""

        if not self.deferred_execution:
            raise RunServiceError(
                "EXECUTION_MODE_CONFLICT",
                "claimed execution requires deferred execution mode",
            )
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("claim must be a RunExecutionClaim")
        if not callable(stop_requested):
            raise TypeError("stop_requested must be callable")
        approval = (
            claim.intent.approval
            if claim.intent.kind is RunExecutionIntentKind.APPROVAL
            else None
        )
        self._execute(
            claim.run,
            approval=approval,
            claim=claim,
            stop_requested=stop_requested,
        )

    def _execute(
        self,
        run: VersionedRun,
        approval: ApprovalDecision | None = None,
        *,
        claim: RunExecutionClaim | None = None,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        snapshot = run.snapshot
        tenant_id = snapshot.request.tenant_id
        checkpoint = copy_json_mapping(snapshot.checkpoint) or None
        request = deepcopy(snapshot.request)
        pending_transition_events: list[RunEventDraft] = []
        stream_event_ordinal = 0
        execution_repository: RunExecutionRepository | None = None
        context: RunExecutionContext | None = None

        if claim is not None:
            if not isinstance(self.repository, RunExecutionRepository):
                raise RunServiceError(
                    "EXECUTION_REPOSITORY_UNAVAILABLE",
                    "run repository does not support claimed execution",
                )
            execution_repository = cast(RunExecutionRepository, self.repository)

            def assert_execution_active() -> None:
                if stop_requested():
                    raise ExecutionSuperseded("execution stop was requested")
                execution_repository.assert_execution_claim(claim)

            context = RunExecutionContext(
                run_id=snapshot.run_id,
                trace_id=snapshot.trace_id,
                version=run.version,
                generation=claim.generation,
                recovering=claim.generation > 1,
                assert_active=assert_execution_active,
                stop_requested=stop_requested,
            )
        else:

            def assert_execution_active() -> None:
                self.repository.assert_running(
                    snapshot.run_id,
                    tenant_id=tenant_id,
                    expected_version=run.version,
                )

        def emit(
            event_type: EventType,
            node_name: str | None,
            payload: Mapping[str, Any],
            *,
            event_key: str | None = None,
        ) -> None:
            nonlocal stream_event_ordinal
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
                if event_type is EventType.APPROVAL_REQUESTED:
                    if event_key is not None:
                        raise _ExecutorContractViolation(
                            "approval lifecycle events cannot define event_key"
                        )
                    resolved_event_key = None
                else:
                    stream_event_ordinal += 1
                    if event_key is not None:
                        if (
                            not isinstance(event_key, str)
                            or not event_key
                            or event_key != event_key.strip()
                            or any(
                                character in event_key
                                for character in ("\r", "\n", "\x00")
                            )
                        ):
                            raise _ExecutorContractViolation(
                                "executor event_key is invalid"
                            )
                        try:
                            encoded_event_key = event_key.encode("utf-8")
                        except UnicodeError as exc:
                            raise _ExecutorContractViolation(
                                "executor event_key is invalid"
                            ) from exc
                        if len(encoded_event_key) > 256:
                            raise _ExecutorContractViolation(
                                "executor event_key is invalid"
                            )
                    resolved_event_key = (
                        f"execution-v{run.version}:{event_key}"
                        if event_key is not None
                        else f"execution-v{run.version}-event-{stream_event_ordinal}"
                    )
                draft = RunEventDraft(
                    event_type,
                    node_name,
                    self._redact(durable_payload),
                    event_key=resolved_event_key,
                )
            except _ExecutorContractViolation:
                raise
            except (RuntimeJsonBoundaryError, TypeError, ValueError) as exc:
                raise _ExecutorContractViolation(
                    "executor event crossed the JSON boundary"
                ) from exc
            if event_type is EventType.APPROVAL_REQUESTED:
                assert_execution_active()
                pending_transition_events.append(draft)
                return
            try:
                if execution_repository is not None and claim is not None:
                    assert_execution_active()
                    execution_repository.append_if_claimed(claim, event=draft)
                else:
                    self.repository.append_if_running(
                        snapshot.run_id,
                        tenant_id=tenant_id,
                        expected_version=run.version,
                        event=draft,
                    )
            except RunRepositoryEventConflict as exc:
                raise _ExecutorContractViolation(
                    "executor reused event_key with different content"
                ) from exc
            assert_execution_active()

        try:
            if context is not None:
                assert_execution_active()
                recoverable_executor = cast(RecoverableRunExecutor, self.executor)
                outcome = recoverable_executor.execute_claimed(
                    request,
                    emit,
                    context=context,
                    checkpoint=checkpoint,
                    approval=approval,
                )
                assert_execution_active()
            else:
                inline_executor = cast(RunExecutor, self.executor)
                outcome = inline_executor.execute(
                    request,
                    emit,
                    checkpoint=checkpoint,
                    approval=approval,
                )
        except ExecutionSuperseded:
            if claim is not None:
                raise
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
            if execution_repository is not None and claim is not None:
                assert_execution_active()
                execution_repository.compare_and_swap_if_claimed(
                    claim,
                    snapshot=replacement,
                    events=tuple(transition_events),
                )
            else:
                self.repository.compare_and_swap(
                    snapshot.run_id,
                    tenant_id=tenant_id,
                    expected_version=run.version,
                    snapshot=replacement,
                    events=tuple(transition_events),
                )
        except ExecutionSuperseded:
            if claim is not None:
                raise
            return
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

    def _compare_and_swap_and_enqueue(
        self,
        current: VersionedRun,
        replacement: RunSnapshot,
        *,
        intent: RunExecutionIntent,
        events: tuple[RunEventDraft, ...],
    ) -> VersionedRun:
        if not isinstance(self.repository, RunExecutionRepository):
            raise RunServiceError(
                "EXECUTION_REPOSITORY_UNAVAILABLE",
                "run repository does not support durable execution jobs",
            )
        try:
            return self.repository.compare_and_swap_and_enqueue(
                current.snapshot.run_id,
                tenant_id=current.snapshot.request.tenant_id,
                expected_version=current.version,
                snapshot=replacement,
                intent=intent,
                events=events,
            )
        except RunRepositoryError as exc:
            raise self._service_error(exc) from exc

    def _notify_execution(self) -> None:
        notifier = self.execution_notifier
        if notifier is None:
            return
        try:
            notifier()
        except Exception:
            # The durable job is authoritative. Notification only reduces poll
            # latency and must never turn a committed transition into an API
            # failure or encourage a non-idempotent caller retry.
            return

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
    "RecoverableRunExecutor",
    "RunExecutor",
    "RunExecutionContext",
    "RunServiceError",
    "ScenarioExecutor",
]
