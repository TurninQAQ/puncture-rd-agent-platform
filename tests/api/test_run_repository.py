from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock
import unittest

from puncture_agent.runtime import (
    ApprovalDecision,
    EventType,
    ExecutionOutcome,
    InMemoryRunRepository,
    InMemoryRunService,
    RunEventDraft,
    RunRequest,
    RunServiceError,
    RunSnapshot,
    RunStatus,
    ScenarioExecutor,
)
from puncture_agent.runtime.errors import (
    ExecutionSuperseded,
    RunRepositoryIdempotencyConflict,
    RunRepositoryNotFound,
    RunRepositoryTransitionError,
    RunRepositoryVersionConflict,
)
from puncture_agent.runtime.json_boundary import RuntimeJsonBoundaryError


FIXED_TIME = "2026-07-11T12:00:00.000Z"


def request(
    *,
    case_id: str = "case-001",
    tenant_id: str = "tenant-a",
    idempotency_key: str = "key-001",
    **metadata: object,
) -> RunRequest:
    return RunRequest(
        case_id=case_id,
        user_query="validate this case",
        task_type="DATA_MODEL_VALIDATION",
        idempotency_key=idempotency_key,
        tenant_id=tenant_id,
        metadata=metadata,
    )


def started_snapshot(
    run_request: RunRequest,
    *,
    run_id: str = "run-001",
) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id,
        request=run_request,
        status=RunStatus.RUNNING,
        trace_id=f"trace-{run_id}",
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
        final_report={},
        checkpoint={},
        approval_id=None,
        error=None,
    )


def initial_events() -> tuple[RunEventDraft, ...]:
    return (
        RunEventDraft(EventType.RUN_CREATED, None, {"nested": {"value": 1}}),
        RunEventDraft(EventType.RUN_STARTED, None, {}),
    )


class InMemoryRunRepositoryTests(unittest.TestCase):
    def test_idempotency_tenant_scope_and_detached_reads(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        original_request = request(metadata={"nested": {"value": 1}})
        created = repository.create_or_get_started(
            started_snapshot(original_request),
            initial_events(),
        )

        self.assertTrue(created.created)
        self.assertEqual(1, created.run.version)
        self.assertEqual(
            [EventType.RUN_CREATED, EventType.RUN_STARTED],
            [
                event.event_type
                for event in repository.get_events("run-001", tenant_id="tenant-a")
            ],
        )

        created.run.snapshot.request.metadata["metadata"]["nested"]["value"] = 99
        public_events = repository.get_events("run-001", tenant_id="tenant-a")
        public_events[0].payload["nested"]["value"] = 99
        restored = repository.get("run-001", tenant_id="tenant-a")
        restored_events = repository.get_events("run-001", tenant_id="tenant-a")
        self.assertEqual(
            1,
            restored.snapshot.request.metadata["metadata"]["nested"]["value"],
        )
        self.assertEqual(1, restored_events[0].payload["nested"]["value"])

        duplicate = repository.create_or_get_started(
            started_snapshot(original_request, run_id="run-duplicate"),
            initial_events(),
        )
        self.assertFalse(duplicate.created)
        self.assertEqual("run-001", duplicate.run.snapshot.run_id)

        with self.assertRaises(RunRepositoryIdempotencyConflict):
            repository.create_or_get_started(
                started_snapshot(
                    request(case_id="case-other"),
                    run_id="run-conflict",
                ),
                initial_events(),
            )

        other_tenant = repository.create_or_get_started(
            started_snapshot(
                request(tenant_id="tenant-b"),
                run_id="run-tenant-b",
            ),
            initial_events(),
        )
        self.assertTrue(other_tenant.created)
        with self.assertRaises(RunRepositoryNotFound):
            repository.get("run-001", tenant_id="tenant-b")

    def test_version_fences_events_and_terminal_transitions(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )
        repository.append_if_running(
            "run-001",
            tenant_id="tenant-a",
            expected_version=created.run.version,
            event=RunEventDraft(EventType.NODE_STARTED, "parse_request", {}),
        )
        self.assertEqual(
            1,
            repository.get("run-001", tenant_id="tenant-a").version,
        )

        cancelled_snapshot = replace(
            created.run.snapshot,
            status=RunStatus.CANCELLED,
            updated_at="2026-07-11T12:00:01.000Z",
        )
        cancelled = repository.compare_and_swap(
            "run-001",
            tenant_id="tenant-a",
            expected_version=1,
            snapshot=cancelled_snapshot,
            events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
        )
        self.assertEqual(2, cancelled.version)

        with self.assertRaises(ExecutionSuperseded):
            repository.append_if_running(
                "run-001",
                tenant_id="tenant-a",
                expected_version=1,
                event=RunEventDraft(EventType.NODE_COMPLETED, "parse_request", {}),
            )
        with self.assertRaises(RunRepositoryVersionConflict):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=1,
                snapshot=cancelled_snapshot,
            )
        with self.assertRaises(RunRepositoryTransitionError):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=2,
                snapshot=replace(
                    cancelled.snapshot,
                    status=RunStatus.RUNNING,
                    updated_at="2026-07-11T12:00:02.000Z",
                ),
            )
        with self.assertRaisesRegex(RunRepositoryTransitionError, "immutable"):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=2,
                snapshot=replace(
                    cancelled.snapshot,
                    request=request(case_id="case-mutated"),
                    status=RunStatus.RUNNING,
                    updated_at="2026-07-11T12:00:02.000Z",
                ),
            )

        events = repository.get_events("run-001", tenant_id="tenant-a")
        self.assertEqual(list(range(1, 5)), [event.sequence for event in events])
        self.assertEqual(EventType.RUN_CANCELLED, events[-1].event_type)

    def test_state_events_cannot_bypass_atomic_transition(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )

        with self.assertRaisesRegex(
            RunRepositoryTransitionError,
            "atomically",
        ):
            repository.append_if_running(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                event=RunEventDraft(EventType.RUN_COMPLETED, None, {}),
            )

        events = repository.get_events("run-001", tenant_id="tenant-a")
        self.assertEqual(
            [EventType.RUN_CREATED, EventType.RUN_STARTED],
            [event.event_type for event in events],
        )
        self.assertEqual(
            RunStatus.RUNNING,
            repository.get("run-001", tenant_id="tenant-a").snapshot.status,
        )

    def test_all_write_paths_enforce_the_durable_json_boundary(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        with self.assertRaises(RuntimeJsonBoundaryError):
            repository.create_or_get_started(
                started_snapshot(request(non_finite=float("nan"))),
                initial_events(),
            )
        with self.assertRaises(RunRepositoryNotFound):
            repository.get("run-001", tenant_id="tenant-a")

        with self.assertRaises(RuntimeJsonBoundaryError):
            RunEventDraft(EventType.NODE_STARTED, "bad-key", {1: "integer-key"})

        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )
        mutated_event = RunEventDraft(EventType.NODE_STARTED, "mutated", {})
        mutated_event.payload["raw"] = b"not-json"
        with self.assertRaises(RuntimeJsonBoundaryError):
            repository.append_if_running(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                event=mutated_event,
            )

        failed_snapshot = replace(
            created.run.snapshot,
            status=RunStatus.FAILED,
            updated_at="2026-07-11T12:00:01.000Z",
            checkpoint={"non_finite": float("nan")},
            error={"code": "FAILED", "retryable": False},
        )
        with self.assertRaises(RuntimeJsonBoundaryError):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                snapshot=failed_snapshot,
                events=(RunEventDraft(EventType.RUN_FAILED, None, {}),),
            )

        cancelled_snapshot = replace(
            created.run.snapshot,
            status=RunStatus.CANCELLED,
            updated_at="2026-07-11T12:00:02.000Z",
        )
        mutated_transition = RunEventDraft(EventType.RUN_CANCELLED, None, {})
        mutated_transition.payload["raw"] = b"not-json"
        with self.assertRaises(RuntimeJsonBoundaryError):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                snapshot=cancelled_snapshot,
                events=(mutated_transition,),
            )

        restored = repository.get("run-001", tenant_id="tenant-a")
        self.assertEqual(RunStatus.RUNNING, restored.snapshot.status)
        self.assertEqual(created.run.version, restored.version)
        self.assertEqual(
            2,
            len(repository.get_events("run-001", tenant_id="tenant-a")),
        )

    def test_repository_rejects_inconsistent_snapshot_fields(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )
        cases = (
            (
                replace(
                    created.run.snapshot,
                    status=RunStatus.WAITING_APPROVAL,
                    approval_id="approval-1",
                    error={"code": "STALE"},
                ),
                RunEventDraft(
                    EventType.APPROVAL_REQUESTED,
                    None,
                    {"approval_id": "approval-1"},
                ),
            ),
            (
                replace(
                    created.run.snapshot,
                    status=RunStatus.SUCCEEDED,
                    approval_id="stale-approval",
                ),
                RunEventDraft(EventType.RUN_COMPLETED, None, {}),
            ),
            (
                replace(
                    created.run.snapshot,
                    status=RunStatus.CANCELLED,
                    error={"code": "STALE"},
                ),
                RunEventDraft(EventType.RUN_CANCELLED, None, {}),
            ),
            (
                replace(
                    created.run.snapshot,
                    status=RunStatus.FAILED,
                    error=None,
                ),
                RunEventDraft(EventType.RUN_FAILED, None, {}),
            ),
        )

        for replacement, event in cases:
            with self.subTest(status=replacement.status, event=event.event_type):
                with self.assertRaises(RunRepositoryTransitionError):
                    repository.compare_and_swap(
                        "run-001",
                        tenant_id="tenant-a",
                        expected_version=created.run.version,
                        snapshot=replacement,
                        events=(event,),
                    )

        restored = repository.get("run-001", tenant_id="tenant-a")
        self.assertEqual(RunStatus.RUNNING, restored.snapshot.status)
        self.assertEqual(created.run.version, restored.version)

    def test_concurrent_event_sequences_are_unique_and_contiguous(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )

        def append(index: int) -> None:
            repository.append_if_running(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                event=RunEventDraft(
                    EventType.NODE_STARTED,
                    f"node-{index}",
                    {"index": index},
                ),
            )

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(append, range(100)))

        events = repository.get_events("run-001", tenant_id="tenant-a")
        self.assertEqual(list(range(1, 103)), [event.sequence for event in events])
        self.assertEqual(
            set(range(100)),
            {event.payload["index"] for event in events[2:]},
        )
        self.assertEqual(
            1,
            repository.get("run-001", tenant_id="tenant-a").version,
        )

    def test_failed_terminal_event_build_rolls_back_state_and_version(self) -> None:
        class FailingClock:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> str:
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("clock unavailable")
                return FIXED_TIME

        repository = InMemoryRunRepository(clock=FailingClock())
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )
        with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=created.run.version,
                snapshot=replace(
                    created.run.snapshot,
                    status=RunStatus.CANCELLED,
                    updated_at="2026-07-11T12:00:01.000Z",
                ),
                events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
            )

        restored = repository.get("run-001", tenant_id="tenant-a")
        self.assertEqual(RunStatus.RUNNING, restored.snapshot.status)
        self.assertEqual(1, restored.version)
        self.assertEqual(
            2,
            len(repository.get_events("run-001", tenant_id="tenant-a")),
        )

    def test_failed_initial_events_leave_no_record_or_idempotency_claim(self) -> None:
        class FailingClock:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> str:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("initial event clock unavailable")
                return FIXED_TIME

        repository = InMemoryRunRepository(clock=FailingClock())
        snapshot = started_snapshot(request())
        with self.assertRaisesRegex(RuntimeError, "initial event clock unavailable"):
            repository.create_or_get_started(snapshot, initial_events())
        with self.assertRaises(RunRepositoryNotFound):
            repository.get("run-001", tenant_id="tenant-a")

        retried = repository.create_or_get_started(snapshot, initial_events())
        self.assertTrue(retried.created)
        self.assertEqual(2, len(repository.get_events("run-001", tenant_id="tenant-a")))

    def test_repository_rejects_resume_of_unrecoverable_failure(self) -> None:
        repository = InMemoryRunRepository(clock=lambda: FIXED_TIME)
        created = repository.create_or_get_started(
            started_snapshot(request()),
            initial_events(),
        )
        failed = repository.compare_and_swap(
            "run-001",
            tenant_id="tenant-a",
            expected_version=created.run.version,
            snapshot=replace(
                created.run.snapshot,
                status=RunStatus.FAILED,
                updated_at="2026-07-11T12:00:01.000Z",
                checkpoint={"recoverable": False},
                error={"code": "PERMANENT", "retryable": False},
            ),
            events=(RunEventDraft(EventType.RUN_FAILED, None, {}),),
        )

        with self.assertRaisesRegex(
            RunRepositoryTransitionError,
            "not recoverable",
        ):
            repository.compare_and_swap(
                "run-001",
                tenant_id="tenant-a",
                expected_version=failed.version,
                snapshot=replace(
                    failed.snapshot,
                    status=RunStatus.RUNNING,
                    updated_at="2026-07-11T12:00:02.000Z",
                    error=None,
                ),
            )

        restored = repository.get("run-001", tenant_id="tenant-a")
        self.assertEqual(RunStatus.FAILED, restored.snapshot.status)
        self.assertEqual(failed.version, restored.version)


class RunServiceConcurrencyTests(unittest.TestCase):
    def test_executor_event_payload_must_be_durable_json(self) -> None:
        class NonJsonEventExecutor:
            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, checkpoint, approval
                emit(
                    EventType.NODE_STARTED,
                    "unsafe-event",
                    {"raw": b"not-json", "non_finite": float("nan")},
                )
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"unexpected": True},
                )

        service = InMemoryRunService(NonJsonEventExecutor())
        failed = service.create_run(
            request(idempotency_key="non-json-event")
        )

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual("EXECUTOR_CONTRACT_ERROR", failed.error["code"])
        events = service.get_events(failed.run_id, tenant_id="tenant-a")
        self.assertNotIn("unsafe-event", [event.node_name for event in events])
        self.assertEqual(EventType.RUN_FAILED, events[-1].event_type)

    def test_executor_cannot_emit_terminal_event_directly(self) -> None:
        class ForgedTerminalEventExecutor:
            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, checkpoint, approval
                emit(EventType.RUN_COMPLETED, None, {"forged": True})
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"ok": True},
                )

        service = InMemoryRunService(ForgedTerminalEventExecutor())
        failed = service.create_run(
            request(idempotency_key="forged-terminal-event")
        )

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual("EXECUTOR_CONTRACT_ERROR", failed.error["code"])
        events = service.get_events(failed.run_id, tenant_id="tenant-a")
        event_types = [event.event_type for event in events]
        self.assertNotIn(EventType.RUN_COMPLETED, event_types)
        self.assertEqual(1, event_types.count(EventType.RUN_FAILED))

    def test_executor_outcome_fields_must_match_status(self) -> None:
        outcomes = (
            ExecutionOutcome(
                status=RunStatus.SUCCEEDED,
                approval_id="stale-approval",
            ),
            ExecutionOutcome(
                status=RunStatus.SUCCEEDED,
                error={"code": "STALE"},
            ),
            ExecutionOutcome(
                status=RunStatus.WAITING_APPROVAL,
                approval_id="approval-1",
                error={"code": "STALE"},
            ),
            ExecutionOutcome(
                status=RunStatus.FAILED,
                final_report={"must_not_survive": True},
                approval_id="stale-approval",
                error={"code": "FAILED"},
            ),
            ExecutionOutcome(
                status=RunStatus.CANCELLED,
                approval_id="stale-approval",
                error={"code": "STALE"},
            ),
        )

        for index, invalid_outcome in enumerate(outcomes):
            class InvalidOutcomeExecutor:
                def execute(
                    self,
                    run_request,
                    emit,
                    *,
                    checkpoint=None,
                    approval=None,
                ):
                    del run_request, emit, checkpoint, approval
                    return invalid_outcome

            with self.subTest(index=index, status=invalid_outcome.status):
                service = InMemoryRunService(InvalidOutcomeExecutor())
                failed = service.create_run(
                    request(idempotency_key=f"invalid-outcome-{index}")
                )
                self.assertEqual(RunStatus.FAILED, failed.status)
                self.assertIsNone(failed.approval_id)
                self.assertEqual(
                    "EXECUTOR_CONTRACT_ERROR",
                    failed.error["code"],
                )
                self.assertEqual({}, failed.final_report)
                events = service.get_events(failed.run_id, tenant_id="tenant-a")
                self.assertEqual(EventType.RUN_FAILED, events[-1].event_type)

    def test_non_json_outcome_becomes_terminal_contract_failure(self) -> None:
        class NonJsonExecutor:
            def __init__(self) -> None:
                self.calls = 0

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, emit, checkpoint, approval
                self.calls += 1
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"lock": Lock()},
                )

        executor = NonJsonExecutor()
        service = InMemoryRunService(executor)
        run_request = request(idempotency_key="non-json-outcome")

        failed = service.create_run(run_request)
        duplicate = service.create_run(run_request)

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual("EXECUTOR_CONTRACT_ERROR", failed.error["code"])
        self.assertEqual(failed.run_id, duplicate.run_id)
        self.assertEqual(RunStatus.FAILED, duplicate.status)
        self.assertEqual(1, executor.calls)
        events = service.get_events(failed.run_id, tenant_id="tenant-a")
        self.assertEqual(EventType.RUN_FAILED, events[-1].event_type)

    def test_cancel_fences_buffered_approval_request(self) -> None:
        class BlockingApprovalRequestExecutor:
            def __init__(self) -> None:
                self.entered = Event()
                self.release = Event()

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, checkpoint, approval
                self.entered.set()
                if not self.release.wait(timeout=5.0):
                    raise RuntimeError("test release timeout")
                emit(
                    EventType.APPROVAL_REQUESTED,
                    "human_approval",
                    {"approval_id": "approval-after-cancel"},
                )
                return ExecutionOutcome(
                    status=RunStatus.WAITING_APPROVAL,
                    checkpoint={"waiting_at": "human_approval"},
                    approval_id="approval-after-cancel",
                )

        executor = BlockingApprovalRequestExecutor()
        service = InMemoryRunService(executor)
        run_request = request(idempotency_key="approval-cancel-race")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service.create_run, run_request)
            self.assertTrue(executor.entered.wait(timeout=5.0))
            running = service.create_run(run_request)
            cancelled = service.cancel(running.run_id, tenant_id="tenant-a")
            executor.release.set()
            completed_call = future.result(timeout=5.0)

        self.assertEqual(RunStatus.CANCELLED, cancelled.status)
        self.assertEqual(RunStatus.CANCELLED, completed_call.status)
        events = service.get_events(cancelled.run_id, tenant_id="tenant-a")
        self.assertEqual(EventType.RUN_CANCELLED, events[-1].event_type)
        self.assertNotIn(
            EventType.APPROVAL_REQUESTED,
            [event.event_type for event in events],
        )

    def test_event_after_approval_request_fails_without_reordering(self) -> None:
        class LateEventExecutor:
            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, checkpoint, approval
                emit(
                    EventType.APPROVAL_REQUESTED,
                    "human_approval",
                    {"approval_id": "approval-ordered"},
                )
                emit(EventType.NODE_STARTED, "must-not-persist", {})
                return ExecutionOutcome(
                    status=RunStatus.WAITING_APPROVAL,
                    checkpoint={"waiting_at": "human_approval"},
                    approval_id="approval-ordered",
                )

        service = InMemoryRunService(LateEventExecutor())
        failed = service.create_run(
            request(idempotency_key="approval-event-order")
        )

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual("EXECUTOR_CONTRACT_ERROR", failed.error["code"])
        events = service.get_events(failed.run_id, tenant_id="tenant-a")
        event_types = [event.event_type for event in events]
        self.assertNotIn(EventType.APPROVAL_REQUESTED, event_types)
        self.assertNotIn(
            "must-not-persist",
            [event.node_name for event in events],
        )
        self.assertEqual(EventType.RUN_FAILED, events[-1].event_type)

    def test_mismatched_approval_event_fails_atomically(self) -> None:
        class MismatchedApprovalExecutor:
            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, checkpoint, approval
                emit(
                    EventType.APPROVAL_REQUESTED,
                    "human_approval",
                    {"approval_id": "approval-event"},
                )
                return ExecutionOutcome(
                    status=RunStatus.WAITING_APPROVAL,
                    checkpoint={"waiting_at": "human_approval"},
                    approval_id="approval-outcome",
                )

        service = InMemoryRunService(MismatchedApprovalExecutor())
        failed = service.create_run(request(idempotency_key="approval-mismatch"))

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual("EXECUTOR_CONTRACT_ERROR", failed.error["code"])
        events = service.get_events(failed.run_id, tenant_id="tenant-a")
        self.assertNotIn(
            EventType.APPROVAL_REQUESTED,
            [event.event_type for event in events],
        )
        self.assertEqual(EventType.RUN_FAILED, events[-1].event_type)

    def test_one_hundred_idempotent_creates_execute_once(self) -> None:
        executor = ScenarioExecutor()
        service = InMemoryRunService(executor)
        run_request = request()

        with ThreadPoolExecutor(max_workers=20) as pool:
            snapshots = list(pool.map(lambda _: service.create_run(run_request), range(100)))

        self.assertEqual(1, executor.execution_count)
        self.assertEqual(1, len({snapshot.run_id for snapshot in snapshots}))
        final = service.get_run(snapshots[0].run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.SUCCEEDED, final.status)

    def test_cancel_fences_late_events_and_outcome(self) -> None:
        class BlockingExecutor:
            def __init__(self) -> None:
                self.entered = Event()
                self.release = Event()
                self.execution_count = 0
                self.published_after_release = False

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del checkpoint, approval
                self.execution_count += 1
                emit(EventType.NODE_STARTED, "blocking_node", {})
                self.entered.set()
                if not self.release.wait(timeout=5.0):
                    raise RuntimeError("test release timeout")
                emit(EventType.NODE_COMPLETED, "blocking_node", {})
                self.published_after_release = True
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"case_id": run_request.case_id},
                )

        executor = BlockingExecutor()
        service = InMemoryRunService(executor)
        run_request = request()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service.create_run, run_request)
            self.assertTrue(executor.entered.wait(timeout=5.0))
            running = service.create_run(run_request)
            self.assertEqual(RunStatus.RUNNING, running.status)
            cancelled = service.cancel(running.run_id, tenant_id="tenant-a")
            self.assertEqual(RunStatus.CANCELLED, cancelled.status)
            executor.release.set()
            completed_call = future.result(timeout=5.0)

        self.assertEqual(RunStatus.CANCELLED, completed_call.status)
        self.assertEqual(1, executor.execution_count)
        self.assertFalse(executor.published_after_release)
        events = service.get_events(cancelled.run_id, tenant_id="tenant-a")
        self.assertEqual(EventType.RUN_CANCELLED, events[-1].event_type)
        self.assertNotIn(EventType.RUN_COMPLETED, [event.event_type for event in events])
        self.assertNotIn(
            EventType.NODE_COMPLETED,
            [event.event_type for event in events],
        )

    def test_cancel_wins_after_handler_returns_before_commit(self) -> None:
        class BlockingCommitRepository(InMemoryRunRepository):
            def __init__(self) -> None:
                super().__init__()
                self.entered = Event()
                self.release = Event()
                self._blocked = False
                self._gate_lock = Lock()

            def compare_and_swap(self, run_id, **kwargs):
                target = kwargs["snapshot"]
                should_block = False
                with self._gate_lock:
                    if target.status is RunStatus.SUCCEEDED and not self._blocked:
                        self._blocked = True
                        should_block = True
                if should_block:
                    self.entered.set()
                    if not self.release.wait(timeout=5.0):
                        raise RuntimeError("test release timeout")
                return super().compare_and_swap(run_id, **kwargs)

        repository = BlockingCommitRepository()
        service = InMemoryRunService(repository=repository)
        run_request = request()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service.create_run, run_request)
            self.assertTrue(repository.entered.wait(timeout=5.0))
            running = service.create_run(run_request)
            cancelled = service.cancel(running.run_id, tenant_id="tenant-a")
            repository.release.set()
            result = future.result(timeout=5.0)

        self.assertEqual(RunStatus.CANCELLED, cancelled.status)
        self.assertEqual(RunStatus.CANCELLED, result.status)
        events = service.get_events(result.run_id, tenant_id="tenant-a")
        self.assertEqual(EventType.RUN_CANCELLED, events[-1].event_type)
        self.assertNotIn(EventType.RUN_COMPLETED, [event.event_type for event in events])

    def test_concurrent_approval_and_resume_start_one_execution(self) -> None:
        class BlockingApprovalExecutor(ScenarioExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.entered = Event()
                self.release = Event()

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                if approval is not None:
                    self.entered.set()
                    if not self.release.wait(timeout=5.0):
                        raise RuntimeError("test release timeout")
                return super().execute(
                    run_request,
                    emit,
                    checkpoint=checkpoint,
                    approval=approval,
                )

        approval_executor = BlockingApprovalExecutor()
        approval_service = InMemoryRunService(approval_executor)
        waiting = approval_service.create_run(
            request(requires_approval=True, approval_id="approval-7")
        )
        decision = ApprovalDecision("approval-7", True, "reviewer-1")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                approval_service.approve,
                waiting.run_id,
                decision,
                tenant_id="tenant-a",
            )
            self.assertTrue(approval_executor.entered.wait(timeout=5.0))
            with self.assertRaises(RunServiceError):
                approval_service.approve(
                    waiting.run_id,
                    decision,
                    tenant_id="tenant-a",
                )
            approval_executor.release.set()
            approved = future.result(timeout=5.0)
        self.assertEqual(RunStatus.SUCCEEDED, approved.status)
        self.assertEqual(2, approval_executor.execution_count)

        class BlockingResumeExecutor:
            def __init__(self) -> None:
                self.calls = 0
                self.lock = Lock()
                self.entered = Event()
                self.release = Event()

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del approval
                with self.lock:
                    self.calls += 1
                    call = self.calls
                if call == 1:
                    return ExecutionOutcome(
                        status=RunStatus.FAILED,
                        checkpoint={"recoverable": True, "case_id": run_request.case_id},
                        error={
                            "code": "TIMEOUT",
                            "message": "dependency unavailable",
                            "retryable": True,
                        },
                    )
                emit(EventType.NODE_STARTED, "resume_node", {})
                self.entered.set()
                if not self.release.wait(timeout=5.0):
                    raise RuntimeError("test release timeout")
                emit(EventType.NODE_COMPLETED, "resume_node", {})
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"resumed": bool(checkpoint)},
                )

        resume_executor = BlockingResumeExecutor()
        resume_service = InMemoryRunService(resume_executor)
        failed = resume_service.create_run(
            request(idempotency_key="resume-key")
        )
        self.assertEqual(RunStatus.FAILED, failed.status)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                resume_service.resume,
                failed.run_id,
                tenant_id="tenant-a",
            )
            self.assertTrue(resume_executor.entered.wait(timeout=5.0))
            with self.assertRaises(RunServiceError):
                resume_service.resume(failed.run_id, tenant_id="tenant-a")
            resume_executor.release.set()
            resumed = future.result(timeout=5.0)
        self.assertEqual(RunStatus.SUCCEEDED, resumed.status)
        self.assertEqual(2, resume_executor.calls)


if __name__ == "__main__":
    unittest.main()
