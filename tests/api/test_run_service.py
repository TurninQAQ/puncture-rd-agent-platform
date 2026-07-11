from __future__ import annotations

import unittest

from puncture_agent.runtime import (
    ApprovalDecision,
    InMemoryRunService,
    RunRequest,
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


if __name__ == "__main__":
    unittest.main()
