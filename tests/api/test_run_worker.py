from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import count
from threading import Event, Lock, Thread
import time
import unittest
from unittest import mock

from puncture_agent.runtime.errors import ExecutionSuperseded
from puncture_agent.runtime.models import (
    ApprovalDecision,
    EventType,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from puncture_agent.runtime.repository import (
    InMemoryRunRepository,
    RunEventDraft,
    RunExecutionClaim,
    RunExecutionIntent,
    RunExecutionIntentKind,
)
from puncture_agent.runtime.worker import (
    RunWorker,
    WorkerConfig,
    WorkerState,
)


class ManualClock:
    def __init__(self) -> None:
        self._lock = Lock()
        self._value = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> str:
        with self._lock:
            value = self._value
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += timedelta(seconds=seconds)


def _request(*, idempotency_key: str = "worker-key") -> RunRequest:
    return RunRequest(
        case_id="case-worker",
        user_query="execute in background",
        task_type="DATA_MODEL_VALIDATION",
        idempotency_key=idempotency_key,
        tenant_id="tenant-worker",
    )


def _seed(repository: InMemoryRunRepository, clock: ManualClock) -> None:
    request = _request()
    snapshot = RunSnapshot(
        run_id="run-worker",
        request=request,
        status=RunStatus.RUNNING,
        trace_id="trace-worker",
        created_at=clock(),
        updated_at=clock(),
        final_report={},
        checkpoint={},
        approval_id=None,
        error=None,
    )
    repository.create_or_get_started(
        snapshot,
        (
            RunEventDraft(EventType.RUN_CREATED, None, {}),
            RunEventDraft(EventType.RUN_STARTED, None, {}),
        ),
    )


def _complete(
    repository: InMemoryRunRepository,
    clock: ManualClock,
    claim: RunExecutionClaim,
) -> None:
    repository.compare_and_swap_if_claimed(
        claim,
        snapshot=replace(
            claim.run.snapshot,
            status=RunStatus.SUCCEEDED,
            updated_at=clock(),
            final_report={"completed": True},
        ),
        events=(
            RunEventDraft(
                EventType.RUN_COMPLETED,
                None,
                {"status": RunStatus.SUCCEEDED.value},
            ),
        ),
    )


def _tokens(prefix: str):
    sequence = count(1)
    return lambda: f"{prefix}-{next(sequence)}"


def _wait_until(predicate, *, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


class CompletingService:
    def __init__(
        self,
        repository: InMemoryRunRepository,
        clock: ManualClock,
        *,
        gate: Event | None = None,
    ) -> None:
        self.repository = repository
        self.clock = clock
        self.gate = gate
        self.entered = Event()
        self.completed = Event()
        self.claims: list[RunExecutionClaim] = []
        self._lock = Lock()

    def execute_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        stop_requested,
    ) -> None:
        del stop_requested
        with self._lock:
            self.claims.append(claim)
        self.entered.set()
        if self.gate is not None and not self.gate.wait(1.5):
            raise RuntimeError("test gate timed out")
        _complete(self.repository, self.clock, claim)
        self.completed.set()


class CooperativeShutdownService:
    def __init__(
        self,
        repository: InMemoryRunRepository,
        clock: ManualClock,
    ) -> None:
        self.repository = repository
        self.clock = clock
        self.entered = Event()
        self.stop_seen = Event()
        self.allow_finish = Event()
        self.completed = Event()

    def execute_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        stop_requested,
    ) -> None:
        self.entered.set()
        deadline = time.monotonic() + 1.5
        while not stop_requested():
            if time.monotonic() >= deadline:
                raise RuntimeError("stop signal was not delivered")
            time.sleep(0.002)
        self.stop_seen.set()
        if not self.allow_finish.wait(1.5):
            raise RuntimeError("shutdown completion gate timed out")
        _complete(self.repository, self.clock, claim)
        self.completed.set()


class UncooperativeService:
    def __init__(self, repository: InMemoryRunRepository) -> None:
        self.repository = repository
        self.entered = Event()
        self.release = Event()
        self.old_write_rejected = Event()

    def execute_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        stop_requested,
    ) -> None:
        del stop_requested
        self.entered.set()
        if not self.release.wait(2.0):
            raise RuntimeError("uncooperative test service timed out")
        try:
            self.repository.append_if_claimed(
                claim,
                event=RunEventDraft(
                    EventType.NODE_STARTED,
                    "stale-execution",
                    {},
                    event_key="stale-execution-started",
                ),
            )
        except ExecutionSuperseded:
            self.old_write_rejected.set()
            raise


class RunWorkerTests(unittest.TestCase):
    def test_claims_one_durable_job_and_exposes_low_cardinality_status(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        service = CompletingService(repository, clock)
        wakeup = RunWorker.create_wakeup_event()
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-single",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.09,
                shutdown_grace_seconds=0.2,
            ),
            wakeup_event=wakeup,
            owner_token_factory=_tokens("owner-single"),
        )

        self.assertIs(wakeup, worker.wakeup_event)
        self.assertEqual(WorkerState.NEW, worker.status.state)
        worker.start()
        self.assertTrue(service.completed.wait(1.0))
        _wait_until(lambda: worker.status.active_executions == 0)
        worker.stop()

        self.assertEqual(1, len(service.claims))
        self.assertEqual(RunStatus.SUCCEEDED, repository.get(
            "run-worker", tenant_id="tenant-worker"
        ).snapshot.status)
        self.assertEqual(WorkerState.STOPPED, worker.status.state)
        self.assertFalse(worker.status.accepting_claims)
        snapshot = worker.metrics.snapshot()
        self.assertEqual(1, snapshot.claims_started_total)
        self.assertEqual(1, snapshot.executions_completed_total)
        rendered = worker.metrics.render()
        self.assertNotIn("run-worker", rendered)
        self.assertNotIn("worker-single", rendered)

    def test_independent_heartbeat_prevents_reclaim_past_original_lease(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        finish = Event()
        service = CompletingService(repository, clock, gate=finish)
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-heartbeat",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.015,
                lease_seconds=0.06,
                shutdown_grace_seconds=0.2,
            ),
            owner_token_factory=_tokens("owner-heartbeat"),
        )
        worker.start()
        self.assertTrue(service.entered.wait(1.0))

        clock.advance(0.04)
        before = worker.metrics.snapshot().heartbeats_total
        _wait_until(
            lambda: worker.metrics.snapshot().heartbeats_total > before
        )
        clock.advance(0.03)
        self.assertIsNone(
            repository.claim_next_execution(
                worker_id="competing-worker",
                owner_token="competing-owner",
                lease_seconds=0.06,
            )
        )

        finish.set()
        self.assertTrue(service.completed.wait(1.0))
        worker.stop()
        self.assertGreaterEqual(worker.metrics.snapshot().heartbeats_total, 1)

    def test_expired_claim_is_reclaimed_with_new_generation_and_fences_old_write(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        old_claim = repository.claim_next_execution(
            worker_id="worker-old",
            owner_token="owner-old",
            lease_seconds=0.05,
        )
        self.assertIsNotNone(old_claim)
        clock.advance(0.051)
        new_claim = repository.claim_next_execution(
            worker_id="worker-new",
            owner_token="owner-new",
            lease_seconds=0.05,
        )
        self.assertIsNotNone(new_claim)
        assert old_claim is not None and new_claim is not None
        self.assertEqual(old_claim.generation + 1, new_claim.generation)
        with self.assertRaises(ExecutionSuperseded):
            repository.append_if_claimed(
                old_claim,
                event=RunEventDraft(
                    EventType.NODE_STARTED,
                    "old-owner",
                    {},
                    event_key="old-owner-started",
                ),
            )
        repository.abandon_execution_claim(new_claim)

    def test_stop_keeps_heartbeating_during_grace_and_signals_execution(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        service = CooperativeShutdownService(repository, clock)
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-grace",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.015,
                lease_seconds=0.06,
                shutdown_grace_seconds=0.5,
            ),
            owner_token_factory=_tokens("owner-grace"),
        )
        worker.start()
        self.assertTrue(service.entered.wait(1.0))

        stopper = Thread(target=worker.stop)
        stopper.start()
        self.assertTrue(service.stop_seen.wait(1.0))
        before = worker.metrics.snapshot().heartbeats_total
        clock.advance(0.04)
        _wait_until(
            lambda: worker.metrics.snapshot().heartbeats_total > before
        )
        service.allow_finish.set()
        stopper.join(1.0)

        self.assertFalse(stopper.is_alive())
        self.assertTrue(service.completed.is_set())
        self.assertEqual(WorkerState.STOPPED, worker.status.state)
        self.assertEqual(0, worker.metrics.snapshot().shutdown_timeouts_total)
        self.assertEqual(
            RunStatus.SUCCEEDED,
            repository.get("run-worker", tenant_id="tenant-worker").snapshot.status,
        )

    def test_shutdown_timeout_stops_heartbeat_and_leaves_ttl_takeover(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        service = UncooperativeService(repository)
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-timeout",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.01,
                lease_seconds=0.05,
                shutdown_grace_seconds=0.04,
            ),
            owner_token_factory=_tokens("owner-timeout"),
        )
        worker.start()
        self.assertTrue(service.entered.wait(1.0))

        started = time.monotonic()
        worker.stop()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.4)
        self.assertEqual(WorkerState.STOPPED, worker.status.state)
        self.assertEqual(1, worker.metrics.snapshot().shutdown_timeouts_total)
        self.assertFalse(service.old_write_rejected.is_set())
        self.assertIsNone(
            repository.claim_next_execution(
                worker_id="worker-too-early",
                owner_token="owner-too-early",
                lease_seconds=0.05,
            )
        )

        clock.advance(0.2)
        takeover = repository.claim_next_execution(
            worker_id="worker-takeover",
            owner_token="owner-takeover",
            lease_seconds=0.05,
        )
        self.assertIsNotNone(takeover)
        service.release.set()
        self.assertTrue(service.old_write_rejected.wait(1.0))
        _wait_until(lambda: worker.status.active_executions == 0)
        self.assertEqual(1, worker.metrics.snapshot().claims_lost_total)
        assert takeover is not None
        repository.abandon_execution_claim(takeover)

    def test_supervisor_failure_is_visible_and_stop_remains_safe(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)

        def fail_token() -> str:
            raise RuntimeError("owner token source failed")

        worker = RunWorker(
            repository,
            CompletingService(repository, clock),
            config=WorkerConfig(
                worker_id="worker-supervisor-failure",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.09,
                shutdown_grace_seconds=0.2,
            ),
            owner_token_factory=fail_token,
        )

        worker.start()
        _wait_until(lambda: worker.status.state is WorkerState.STOPPING)
        self.assertFalse(worker.status.accepting_claims)
        self.assertEqual(
            1,
            worker.metrics.snapshot().supervisor_failures_total,
        )
        worker.stop()
        self.assertEqual(WorkerState.STOPPED, worker.status.state)

        start_repository = InMemoryRunRepository(clock=clock)
        start_worker = RunWorker(
            start_repository,
            CompletingService(start_repository, clock),
            config=WorkerConfig(
                worker_id="worker-thread-start-failure",
                poll_interval_seconds=0.5,
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.09,
                shutdown_grace_seconds=0.2,
            ),
            owner_token_factory=_tokens("owner-thread-start-failure"),
        )
        start_worker.start()
        time.sleep(0.02)
        with mock.patch.object(
            Thread,
            "start",
            side_effect=RuntimeError("execution thread start failed"),
        ):
            _seed(start_repository, clock)
            start_worker.notify()
            _wait_until(
                lambda: start_worker.status.state is WorkerState.STOPPING
            )
        self.assertEqual(
            1,
            start_worker.metrics.snapshot().supervisor_failures_total,
        )
        self.assertEqual(
            1,
            start_worker.metrics.snapshot().executions_failed_total,
        )
        reclaimed = start_repository.claim_next_execution(
            worker_id="worker-thread-start-reclaim",
            owner_token="owner-thread-start-reclaim",
            lease_seconds=0.09,
        )
        self.assertIsNotNone(reclaimed)
        start_worker.stop()

    def test_duplicate_owner_token_never_starts_duplicate_execution(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        finish = Event()
        service = CompletingService(repository, clock, gate=finish)
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-duplicate-token",
                concurrency=2,
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.09,
                shutdown_grace_seconds=0.2,
            ),
            owner_token_factory=lambda: "owner-duplicate",
        )

        worker.start()
        self.assertTrue(service.entered.wait(1.0))
        time.sleep(0.05)
        self.assertEqual(1, len(service.claims))
        self.assertEqual(1, worker.status.active_executions)
        finish.set()
        self.assertTrue(service.completed.wait(1.0))
        worker.stop()

    def test_approval_execution_intent_is_passed_through_unchanged(self) -> None:
        clock = ManualClock()
        repository = InMemoryRunRepository(clock=clock)
        _seed(repository, clock)
        create_claim = repository.claim_next_execution(
            worker_id="setup-worker",
            owner_token="setup-owner",
            lease_seconds=0.1,
        )
        self.assertIsNotNone(create_claim)
        assert create_claim is not None
        waiting = repository.compare_and_swap_if_claimed(
            create_claim,
            snapshot=replace(
                create_claim.run.snapshot,
                status=RunStatus.WAITING_APPROVAL,
                updated_at=clock(),
                checkpoint={"waiting_at": "human_approval"},
                approval_id="approval-worker",
            ),
            events=(
                RunEventDraft(
                    EventType.APPROVAL_REQUESTED,
                    "human_approval",
                    {"approval_id": "approval-worker"},
                ),
            ),
        )
        decision = ApprovalDecision(
            approval_id="approval-worker",
            approved=True,
            principal_id="reviewer-worker",
            comment="approved for worker",
        )
        repository.compare_and_swap_and_enqueue(
            "run-worker",
            tenant_id="tenant-worker",
            expected_version=waiting.version,
            snapshot=replace(
                waiting.snapshot,
                status=RunStatus.RUNNING,
                updated_at=clock(),
                approval_id=None,
            ),
            intent=RunExecutionIntent(
                RunExecutionIntentKind.APPROVAL,
                approval=decision,
            ),
        )
        service = CompletingService(repository, clock)
        worker = RunWorker(
            repository,
            service,
            config=WorkerConfig(
                worker_id="worker-approval",
                poll_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.09,
                shutdown_grace_seconds=0.2,
            ),
            owner_token_factory=_tokens("owner-approval"),
        )
        worker.start()
        self.assertTrue(service.completed.wait(1.0))
        worker.stop()

        self.assertEqual(1, len(service.claims))
        approval_claim = service.claims[0]
        self.assertIs(RunExecutionIntentKind.APPROVAL, approval_claim.intent.kind)
        self.assertEqual(decision, approval_claim.intent.approval)

    def test_config_requires_lease_safety_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "three heartbeat"):
            WorkerConfig(
                worker_id="unsafe-worker",
                heartbeat_interval_seconds=0.02,
                lease_seconds=0.05,
            )


if __name__ == "__main__":
    unittest.main()
