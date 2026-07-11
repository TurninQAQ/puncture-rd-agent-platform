from __future__ import annotations

import unittest

from puncture_agent.runtime.errors import ExecutionSuperseded
from puncture_agent.runtime import (
    ApprovalDecision,
    EventType,
    ExecutionOutcome,
    InMemoryRunRepository,
    InMemoryRunService,
    RunRequest,
    RunExecutionIntentKind,
    RunServiceError,
    RunStatus,
    ScenarioExecutor,
)


def request(**metadata: object) -> RunRequest:
    return RunRequest(
        case_id="case-001",
        user_query="validate this case",
        task_type="DATA_MODEL_VALIDATION",
        idempotency_key="key-001",
        tenant_id="tenant-a",
        metadata=metadata,
    )


class RunServiceTests(unittest.TestCase):
    def test_happy_path_has_ordered_events_and_report(self) -> None:
        service = InMemoryRunService()
        snapshot = service.create_run(request())

        self.assertEqual(RunStatus.SUCCEEDED, snapshot.status)
        self.assertEqual("case-001", snapshot.final_report["case_id"])
        events = service.get_events(snapshot.run_id, tenant_id="tenant-a")
        self.assertEqual(list(range(1, len(events) + 1)), [event.sequence for event in events])
        self.assertEqual("RUN_CREATED", events[0].event_type.value)
        self.assertEqual("RUN_COMPLETED", events[-1].event_type.value)

    def test_duplicate_create_is_idempotent(self) -> None:
        executor = ScenarioExecutor()
        service = InMemoryRunService(executor)
        first = service.create_run(request())
        second = service.create_run(request())

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(1, executor.execution_count)

    def test_same_key_with_different_payload_conflicts(self) -> None:
        service = InMemoryRunService()
        service.create_run(request())
        changed = RunRequest(
            case_id="case-002",
            user_query="different",
            task_type="DATA_MODEL_VALIDATION",
            idempotency_key="key-001",
            tenant_id="tenant-a",
        )
        with self.assertRaises(RunServiceError) as raised:
            service.create_run(changed)
        self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)

    def test_approval_resumes_from_checkpoint(self) -> None:
        service = InMemoryRunService()
        waiting = service.create_run(request(requires_approval=True, approval_id="approval-7"))
        self.assertEqual(RunStatus.WAITING_APPROVAL, waiting.status)

        completed = service.approve(
            waiting.run_id,
            ApprovalDecision("approval-7", True, "reviewer-1"),
            tenant_id="tenant-a",
        )
        self.assertEqual(RunStatus.SUCCEEDED, completed.status)
        nodes = [event.node_name for event in service.get_events(waiting.run_id, tenant_id="tenant-a")]
        self.assertIn("resume_from_checkpoint", nodes)

    def test_wrong_or_repeated_approval_is_rejected(self) -> None:
        service = InMemoryRunService()
        waiting = service.create_run(request(requires_approval=True, approval_id="approval-7"))
        with self.assertRaises(RunServiceError):
            service.approve(
                waiting.run_id,
                ApprovalDecision("wrong", True, "reviewer-1"),
                tenant_id="tenant-a",
            )
        service.approve(
            waiting.run_id,
            ApprovalDecision("approval-7", True, "reviewer-1"),
            tenant_id="tenant-a",
        )
        with self.assertRaises(RunServiceError):
            service.approve(
                waiting.run_id,
                ApprovalDecision("approval-7", True, "reviewer-1"),
                tenant_id="tenant-a",
            )

    def test_waiting_run_can_be_cancelled_but_terminal_run_cannot(self) -> None:
        service = InMemoryRunService()
        waiting = service.create_run(request(requires_approval=True))
        cancelled = service.cancel(waiting.run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.CANCELLED, cancelled.status)
        with self.assertRaises(RunServiceError):
            service.cancel(waiting.run_id, tenant_id="tenant-a")

    def test_event_cursor_returns_only_new_events(self) -> None:
        service = InMemoryRunService()
        snapshot = service.create_run(request())
        events = service.get_events(snapshot.run_id, tenant_id="tenant-a")
        tail = service.get_events(snapshot.run_id, tenant_id="tenant-a", after_sequence=2)
        first_page = service.get_event_page(
            snapshot.run_id,
            tenant_id="tenant-a",
            limit=2,
        )
        second_page = service.get_event_page(
            snapshot.run_id,
            tenant_id="tenant-a",
            after_sequence=2,
            limit=2,
        )
        final_page = service.get_event_page(
            snapshot.run_id,
            tenant_id="tenant-a",
            after_sequence=max(0, len(events) - 2),
            limit=2,
        )
        self.assertEqual(events[2:], tail)
        self.assertEqual(events[:2], first_page.events)
        self.assertEqual(events[2:4], second_page.events)
        self.assertEqual(len(events), first_page.high_water_sequence)
        self.assertTrue(first_page.has_more)
        self.assertEqual(events[-2:], final_page.events)
        self.assertFalse(final_page.has_more)
        with self.assertRaises(RunServiceError):
            service.get_event_page(
                snapshot.run_id,
                tenant_id="tenant-a",
                limit=513,
            )

    def test_cross_tenant_lookup_does_not_disclose_run(self) -> None:
        service = InMemoryRunService()
        snapshot = service.create_run(request())
        with self.assertRaises(RunServiceError) as raised:
            service.get_run(snapshot.run_id, tenant_id="tenant-b")
        self.assertEqual("NOT_FOUND", raised.exception.code)

    def test_dependency_timeout_is_structured_and_recoverable(self) -> None:
        service = InMemoryRunService()
        snapshot = service.create_run(request(dependency_timeout=True))
        self.assertEqual(RunStatus.FAILED, snapshot.status)
        self.assertEqual("TIMEOUT", snapshot.error["code"])
        self.assertTrue(snapshot.error["retryable"])
        self.assertTrue(snapshot.checkpoint["recoverable"])

    def test_invalid_request_fails_before_run_creation(self) -> None:
        with self.assertRaises(ValueError):
            RunRequest(
                case_id="",
                user_query="query",
                task_type="DATA_MODEL_VALIDATION",
                idempotency_key="key",
            )


class DeferredRunServiceTests(unittest.TestCase):
    @staticmethod
    def _claim(repository: InMemoryRunRepository, suffix: str):
        claim = repository.claim_next_execution(
            worker_id=f"worker-{suffix}",
            owner_token=f"owner-{suffix}",
            lease_seconds=5.0,
        )
        if claim is None:
            raise AssertionError("expected a durable execution claim")
        return claim

    def test_deferred_mode_is_fail_closed_and_keeps_inline_default(self) -> None:
        class UnsafeExecutor:
            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                del run_request, emit, checkpoint, approval
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"unsafe": True},
                )

        class ClaimedOnlyExecutor:
            recovery_safe = True

            def execute_claimed(
                self,
                run_request,
                emit,
                *,
                context,
                checkpoint=None,
                approval=None,
            ):
                del run_request, emit, context, checkpoint, approval
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"claimed": True},
                )

        inline = InMemoryRunService(UnsafeExecutor())
        self.assertEqual(RunStatus.SUCCEEDED, inline.create_run(request()).status)

        with self.assertRaisesRegex(ValueError, "execution_notifier"):
            InMemoryRunService(deferred_execution=True)
        with self.assertRaisesRegex(ValueError, "recovery-safe"):
            InMemoryRunService(
                UnsafeExecutor(),
                deferred_execution=True,
                execution_notifier=lambda: None,
            )
        deferred = InMemoryRunService(
            ClaimedOnlyExecutor(),
            deferred_execution=True,
            execution_notifier=lambda: None,
        )
        self.assertTrue(deferred.deferred_execution)

    def test_deferred_create_only_persists_and_notifies_before_claim(self) -> None:
        repository = InMemoryRunRepository()
        executor = ScenarioExecutor()
        notifications: list[str] = []
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=lambda: notifications.append("ready"),
        )

        running = service.create_run(request())
        duplicate = service.create_run(request())

        self.assertEqual(RunStatus.RUNNING, running.status)
        self.assertEqual(running.run_id, duplicate.run_id)
        self.assertEqual(0, executor.execution_count)
        self.assertEqual(["ready", "ready"], notifications)

        claim = self._claim(repository, "create")
        self.assertIs(RunExecutionIntentKind.CREATE, claim.intent.kind)
        service.execute_claimed(claim)

        completed = service.get_run(running.run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.SUCCEEDED, completed.status)
        self.assertEqual(1, executor.execution_count)

    def test_notifier_failure_never_changes_the_committed_job_result(self) -> None:
        repository = InMemoryRunRepository()

        def unavailable_notifier() -> None:
            raise RuntimeError("best-effort wakeup failed")

        service = InMemoryRunService(
            repository=repository,
            deferred_execution=True,
            execution_notifier=unavailable_notifier,
        )

        running = service.create_run(request())

        self.assertEqual(RunStatus.RUNNING, running.status)
        self.assertIsNotNone(self._claim(repository, "notifier"))

    def test_approval_decision_is_atomically_preserved_in_claimed_intent(self) -> None:
        class ApprovalRecorder:
            recovery_safe = True

            def __init__(self) -> None:
                self.approvals: list[ApprovalDecision | None] = []

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                raise AssertionError("deferred mode must not use inline execute")

            def execute_claimed(
                self,
                run_request,
                emit,
                *,
                context,
                checkpoint=None,
                approval=None,
            ):
                del run_request, emit, checkpoint
                context.assert_active()
                self.approvals.append(approval)
                if approval is None:
                    return ExecutionOutcome(
                        status=RunStatus.WAITING_APPROVAL,
                        checkpoint={"waiting_at": "review"},
                        approval_id="approval-7",
                    )
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"approved_by": approval.principal_id},
                )

        repository = InMemoryRunRepository()
        executor = ApprovalRecorder()
        notifications: list[None] = []
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=lambda: notifications.append(None),
        )
        running = service.create_run(request())
        service.execute_claimed(self._claim(repository, "approval-create"))
        waiting = service.get_run(running.run_id, tenant_id="tenant-a")
        decision = ApprovalDecision(
            "approval-7",
            True,
            "reviewer-17",
            "reviewed against source evidence",
        )

        resumed = service.approve(
            waiting.run_id,
            decision,
            tenant_id="tenant-a",
        )
        approval_claim = self._claim(repository, "approval-resume")

        self.assertEqual(RunStatus.RUNNING, resumed.status)
        self.assertIs(RunExecutionIntentKind.APPROVAL, approval_claim.intent.kind)
        self.assertEqual(decision, approval_claim.intent.approval)
        service.execute_claimed(approval_claim)
        self.assertIs(executor.approvals[-1], approval_claim.intent.approval)
        self.assertEqual(2, len(notifications))
        self.assertEqual(
            RunStatus.SUCCEEDED,
            service.get_run(running.run_id, tenant_id="tenant-a").status,
        )

    def test_resume_is_enqueued_and_restores_the_checkpoint(self) -> None:
        class ResumeRecorder:
            recovery_safe = True

            def __init__(self) -> None:
                self.checkpoints: list[dict[str, object]] = []

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                raise AssertionError("deferred mode must not use inline execute")

            def execute_claimed(
                self,
                run_request,
                emit,
                *,
                context,
                checkpoint=None,
                approval=None,
            ):
                del run_request, emit, approval
                context.assert_active()
                durable_checkpoint = dict(checkpoint or {})
                self.checkpoints.append(durable_checkpoint)
                if not durable_checkpoint:
                    return ExecutionOutcome(
                        status=RunStatus.FAILED,
                        checkpoint={"recoverable": True, "cursor": "node-3"},
                        error={
                            "code": "DEPENDENCY_TIMEOUT",
                            "message": "retry later",
                            "retryable": True,
                        },
                    )
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"restored": durable_checkpoint["cursor"]},
                )

        repository = InMemoryRunRepository()
        executor = ResumeRecorder()
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=lambda: None,
        )
        running = service.create_run(request())
        service.execute_claimed(self._claim(repository, "resume-create"))
        failed = service.get_run(running.run_id, tenant_id="tenant-a")

        resumed = service.resume(failed.run_id, tenant_id="tenant-a")
        resume_claim = self._claim(repository, "resume-job")

        self.assertEqual(RunStatus.RUNNING, resumed.status)
        self.assertIs(RunExecutionIntentKind.RESUME, resume_claim.intent.kind)
        service.execute_claimed(resume_claim)
        completed = service.get_run(running.run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.SUCCEEDED, completed.status)
        self.assertEqual("node-3", completed.final_report["restored"])
        self.assertEqual(
            {"recoverable": True, "cursor": "node-3"},
            executor.checkpoints[-1],
        )

    def test_reclaimed_execution_replays_events_without_generation_in_key(self) -> None:
        stopped = [False]

        class ReplayExecutor:
            recovery_safe = True

            def __init__(self) -> None:
                self.contexts = []

            def execute(self, run_request, emit, *, checkpoint=None, approval=None):
                raise AssertionError("deferred mode must not use inline execute")

            def execute_claimed(
                self,
                run_request,
                emit,
                *,
                context,
                checkpoint=None,
                approval=None,
            ):
                del checkpoint, approval
                self.contexts.append(context)
                emit(
                    EventType.NODE_STARTED,
                    "recoverable-node",
                    {"case_id": run_request.case_id},
                )
                if not context.recovering:
                    stopped[0] = True
                    context.assert_active()
                return ExecutionOutcome(
                    status=RunStatus.SUCCEEDED,
                    final_report={"recovered": context.recovering},
                )

        repository = InMemoryRunRepository()
        executor = ReplayExecutor()
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=lambda: None,
        )
        running = service.create_run(request())
        first_claim = self._claim(repository, "generation-one")

        with self.assertRaises(ExecutionSuperseded):
            service.execute_claimed(
                first_claim,
                stop_requested=lambda: stopped[0],
            )
        repository.release_execution_claim(first_claim)
        stopped[0] = False
        second_claim = self._claim(repository, "generation-two")
        service.execute_claimed(
            second_claim,
            stop_requested=lambda: stopped[0],
        )

        contexts = executor.contexts
        self.assertEqual([1, 2], [context.generation for context in contexts])
        self.assertEqual([False, True], [context.recovering for context in contexts])
        self.assertEqual(running.run_id, contexts[1].run_id)
        self.assertEqual(running.trace_id, contexts[1].trace_id)
        self.assertEqual(1, contexts[1].version)
        events = service.get_events(running.run_id, tenant_id="tenant-a")
        self.assertEqual(
            1,
            [event.node_name for event in events].count("recoverable-node"),
        )
        completed = service.get_run(running.run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.SUCCEEDED, completed.status)
        self.assertTrue(completed.final_report["recovered"])

    def test_lost_claim_and_requested_stop_are_fenced_without_state_change(self) -> None:
        repository = InMemoryRunRepository()
        executor = ScenarioExecutor()
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=lambda: None,
        )
        running = service.create_run(request())
        lost_claim = self._claim(repository, "lost")
        repository.release_execution_claim(lost_claim)

        with self.assertRaises(ExecutionSuperseded):
            service.execute_claimed(lost_claim)
        fresh_claim = self._claim(repository, "stopped")
        with self.assertRaises(ExecutionSuperseded):
            service.execute_claimed(fresh_claim, stop_requested=lambda: True)

        self.assertEqual(0, executor.execution_count)
        self.assertEqual(
            RunStatus.RUNNING,
            service.get_run(running.run_id, tenant_id="tenant-a").status,
        )


if __name__ == "__main__":
    unittest.main()
