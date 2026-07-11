"""Durable-claim worker supervisor for background Run execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from threading import Condition, Event, Lock, RLock, Thread, current_thread
import time
from typing import Callable, Protocol
from uuid import uuid4

from .errors import ExecutionSuperseded
from .repository import (
    RunExecutionClaim,
    RunExecutionRepository,
    _validate_execution_identifier,
    _validate_execution_lease_seconds,
)


class ClaimedRunService(Protocol):
    """Structural service boundary used by the worker.

    The service owns all fenced writes.  The worker only owns durable claim
    acquisition, lease renewal, and cooperative shutdown signalling.
    """

    def execute_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        stop_requested: Callable[[], bool],
    ) -> None: ...


def _duration(
    name: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
    allow_zero: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if allow_zero and normalized == 0.0:
        return normalized
    if normalized < minimum or normalized > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum} seconds"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: str
    concurrency: int = 1
    poll_interval_seconds: float = 0.5
    heartbeat_interval_seconds: float = 5.0
    lease_seconds: float = 30.0
    shutdown_grace_seconds: float = 30.0

    def __post_init__(self) -> None:
        _validate_execution_identifier("worker_id", self.worker_id)
        if (
            isinstance(self.concurrency, bool)
            or not isinstance(self.concurrency, int)
            or self.concurrency < 1
            or self.concurrency > 256
        ):
            raise ValueError("worker concurrency must be between 1 and 256")
        poll = _duration(
            "worker poll interval",
            self.poll_interval_seconds,
            minimum=0.001,
            maximum=60.0,
        )
        heartbeat = _duration(
            "worker heartbeat interval",
            self.heartbeat_interval_seconds,
            minimum=0.001,
            maximum=1200.0,
        )
        lease = _validate_execution_lease_seconds(self.lease_seconds)
        grace = _duration(
            "worker shutdown grace",
            self.shutdown_grace_seconds,
            minimum=0.001,
            maximum=3600.0,
            allow_zero=True,
        )
        if lease < 3.0 * heartbeat:
            raise ValueError(
                "worker lease_seconds must be at least three heartbeat intervals"
            )
        object.__setattr__(self, "poll_interval_seconds", poll)
        object.__setattr__(self, "heartbeat_interval_seconds", heartbeat)
        object.__setattr__(self, "lease_seconds", lease)
        object.__setattr__(self, "shutdown_grace_seconds", grace)


class WorkerState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Low-cardinality worker status without tenant or Run identifiers."""

    state: WorkerState
    accepting_claims: bool
    active_executions: int


@dataclass(frozen=True, slots=True)
class WorkerMetricsSnapshot:
    active_executions: int
    claims_started_total: int
    heartbeats_total: int
    repository_errors_total: int
    executions_completed_total: int
    executions_failed_total: int
    executions_superseded_total: int
    claims_lost_total: int
    shutdown_timeouts_total: int
    supervisor_failures_total: int


class WorkerMetrics:
    """Thread-safe counters with fixed, bounded dimensions only."""

    _OUTCOMES = ("completed", "failed", "superseded")

    def __init__(self) -> None:
        self._lock = RLock()
        self._active = 0
        self._claims_started = 0
        self._heartbeats = 0
        self._repository_errors = 0
        self._outcomes = {outcome: 0 for outcome in self._OUTCOMES}
        self._claims_lost = 0
        self._shutdown_timeouts = 0
        self._supervisor_failures = 0

    def claim_started(self) -> None:
        with self._lock:
            self._active += 1
            self._claims_started += 1

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeats += 1

    def repository_error(self) -> None:
        with self._lock:
            self._repository_errors += 1

    def execution_finished(self, outcome: str) -> None:
        normalized = outcome if outcome in self._outcomes else "failed"
        with self._lock:
            self._active = max(0, self._active - 1)
            self._outcomes[normalized] += 1

    def claim_lost(self) -> None:
        with self._lock:
            self._claims_lost += 1

    def shutdown_timeout(self) -> None:
        with self._lock:
            self._shutdown_timeouts += 1

    def supervisor_failure(self) -> None:
        with self._lock:
            self._supervisor_failures += 1

    def snapshot(self) -> WorkerMetricsSnapshot:
        with self._lock:
            return WorkerMetricsSnapshot(
                active_executions=self._active,
                claims_started_total=self._claims_started,
                heartbeats_total=self._heartbeats,
                repository_errors_total=self._repository_errors,
                executions_completed_total=self._outcomes["completed"],
                executions_failed_total=self._outcomes["failed"],
                executions_superseded_total=self._outcomes["superseded"],
                claims_lost_total=self._claims_lost,
                shutdown_timeouts_total=self._shutdown_timeouts,
                supervisor_failures_total=self._supervisor_failures,
            )

    def render(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP puncture_worker_active_executions Current claimed executions.",
            "# TYPE puncture_worker_active_executions gauge",
            f"puncture_worker_active_executions {snapshot.active_executions}",
            "# HELP puncture_worker_claims_started_total Durable claims started.",
            "# TYPE puncture_worker_claims_started_total counter",
            f"puncture_worker_claims_started_total {snapshot.claims_started_total}",
            "# HELP puncture_worker_heartbeats_total Successful lease renewals.",
            "# TYPE puncture_worker_heartbeats_total counter",
            f"puncture_worker_heartbeats_total {snapshot.heartbeats_total}",
            "# HELP puncture_worker_repository_errors_total Repository operation errors.",
            "# TYPE puncture_worker_repository_errors_total counter",
            (
                "puncture_worker_repository_errors_total "
                f"{snapshot.repository_errors_total}"
            ),
            "# HELP puncture_worker_execution_outcomes_total Executions by fixed outcome.",
            "# TYPE puncture_worker_execution_outcomes_total counter",
            (
                'puncture_worker_execution_outcomes_total{outcome="completed"} '
                f"{snapshot.executions_completed_total}"
            ),
            (
                'puncture_worker_execution_outcomes_total{outcome="failed"} '
                f"{snapshot.executions_failed_total}"
            ),
            (
                'puncture_worker_execution_outcomes_total{outcome="superseded"} '
                f"{snapshot.executions_superseded_total}"
            ),
            "# HELP puncture_worker_claims_lost_total Claims lost while executing.",
            "# TYPE puncture_worker_claims_lost_total counter",
            f"puncture_worker_claims_lost_total {snapshot.claims_lost_total}",
            (
                "# HELP puncture_worker_shutdown_timeouts_total "
                "Executions exceeding shutdown grace."
            ),
            "# TYPE puncture_worker_shutdown_timeouts_total counter",
            (
                "puncture_worker_shutdown_timeouts_total "
                f"{snapshot.shutdown_timeouts_total}"
            ),
            (
                "# HELP puncture_worker_supervisor_failures_total "
                "Fatal supervisor failures."
            ),
            "# TYPE puncture_worker_supervisor_failures_total counter",
            (
                "puncture_worker_supervisor_failures_total "
                f"{snapshot.supervisor_failures_total}"
            ),
        ]
        return "\n".join(lines) + "\n"


@dataclass(slots=True)
class _ActiveExecution:
    claim: RunExecutionClaim
    stop_requested: Event = field(default_factory=Event)
    heartbeat_stop: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    execution_thread: Thread | None = None
    heartbeat_thread: Thread | None = None
    release_on_finish: bool = True
    claim_lost: bool = False
    shutdown_timed_out: bool = False
    execution_returned: bool = False

    def current_claim(self) -> RunExecutionClaim:
        with self.lock:
            return self.claim

    def update_claim(self, claim: RunExecutionClaim) -> None:
        with self.lock:
            self.claim = claim

    def mark_claim_lost(self) -> bool:
        with self.lock:
            if self.claim_lost:
                return False
            self.claim_lost = True
            self.release_on_finish = False
            return True

    def mark_shutdown_timeout(self) -> bool:
        with self.lock:
            if self.shutdown_timed_out or self.execution_returned:
                return False
            self.shutdown_timed_out = True
            self.release_on_finish = False
            return True

    def begin_finish(self) -> bool:
        with self.lock:
            self.execution_returned = True
            return self.release_on_finish and not self.claim_lost


class RunWorker:
    """Supervise bounded background execution from durable repository claims."""

    def __init__(
        self,
        repository: RunExecutionRepository,
        service: ClaimedRunService,
        *,
        config: WorkerConfig,
        metrics: WorkerMetrics | None = None,
        wakeup_event: Event | None = None,
        owner_token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.repository = repository
        self.service = service
        self.config = config
        self.metrics = metrics if metrics is not None else WorkerMetrics()
        self._wakeup = wakeup_event if wakeup_event is not None else Event()
        self._owner_token_factory = owner_token_factory or (lambda: uuid4().hex)
        self._lock = RLock()
        self._changed = Condition(self._lock)
        self._state = WorkerState.NEW
        self._stop_supervisor = Event()
        self._supervisor_thread: Thread | None = None
        self._active: dict[str, _ActiveExecution] = {}

    @staticmethod
    def create_wakeup_event() -> Event:
        """Create an Event that can be shared with a deferred API notifier."""

        return Event()

    @property
    def wakeup_event(self) -> Event:
        return self._wakeup

    @property
    def status(self) -> WorkerStatus:
        with self._lock:
            state = self._state
            return WorkerStatus(
                state=state,
                accepting_claims=state is WorkerState.RUNNING,
                active_executions=len(self._active),
            )

    def start(self) -> None:
        with self._lock:
            if self._state is WorkerState.RUNNING:
                return
            if self._state is not WorkerState.NEW:
                raise RuntimeError("a stopped RunWorker cannot be restarted")
            self._state = WorkerState.RUNNING
            supervisor = Thread(
                target=self._supervise,
                name="puncture-run-worker-supervisor",
                daemon=True,
            )
            self._supervisor_thread = supervisor
            try:
                supervisor.start()
            except Exception:
                self._supervisor_thread = None
                self._state = WorkerState.NEW
                raise
        self.notify()

    def notify(self) -> None:
        """Wake claim polling after a durable job is enqueued."""

        self._wakeup.set()

    def stop(self) -> None:
        """Cooperatively stop, then leave over-grace claims to lease expiry."""

        with self._changed:
            if self._state is WorkerState.NEW:
                self._state = WorkerState.STOPPED
                self._changed.notify_all()
                return
            if self._state is WorkerState.STOPPED:
                return
            self._state = WorkerState.STOPPING
            self._stop_supervisor.set()
            for active in self._active.values():
                active.stop_requested.set()
            self._changed.notify_all()
        self.notify()

        deadline = time.monotonic() + self.config.shutdown_grace_seconds
        with self._changed:
            while self._active:
                for active in self._active.values():
                    active.stop_requested.set()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(timeout=remaining)
            overdue = tuple(self._active.values())
            for active in overdue:
                active.stop_requested.set()
                if active.mark_shutdown_timeout():
                    self.metrics.shutdown_timeout()
                active.heartbeat_stop.set()

        supervisor = self._supervisor_thread
        if supervisor is not None and supervisor is not current_thread():
            supervisor.join(timeout=max(0.25, deadline - time.monotonic()))
        with self._changed:
            self._state = WorkerState.STOPPED
            self._changed.notify_all()

    def _supervise(self) -> None:
        while not self._stop_supervisor.is_set():
            self._wakeup.clear()
            try:
                self._fill_capacity()
            except Exception:
                self.metrics.supervisor_failure()
                with self._changed:
                    if self._state is WorkerState.RUNNING:
                        self._state = WorkerState.STOPPING
                    self._stop_supervisor.set()
                    self._changed.notify_all()
                return
            if self._stop_supervisor.is_set():
                break
            self._wakeup.wait(self.config.poll_interval_seconds)

    def _fill_capacity(self) -> None:
        while True:
            with self._lock:
                if (
                    self._state is not WorkerState.RUNNING
                    or len(self._active) >= self.config.concurrency
                ):
                    return
            owner_token = self._owner_token_factory()
            try:
                claim = self.repository.claim_next_execution(
                    worker_id=self.config.worker_id,
                    owner_token=owner_token,
                    lease_seconds=self.config.lease_seconds,
                )
            except Exception:
                self.metrics.repository_error()
                return
            if claim is None:
                return
            active = _ActiveExecution(claim=claim)
            with self._changed:
                if self._state is not WorkerState.RUNNING:
                    should_start = False
                    already_active = False
                elif claim.owner_token in self._active:
                    should_start = False
                    already_active = True
                else:
                    self._active[claim.owner_token] = active
                    should_start = True
                    already_active = False
                    self._changed.notify_all()
            if not should_start:
                if not already_active:
                    try:
                        self.repository.release_execution_claim(claim)
                    except Exception:
                        self.metrics.repository_error()
                return
            self.metrics.claim_started()
            self._start_execution(active)

    def _start_execution(self, active: _ActiveExecution) -> None:
        heartbeat_thread = Thread(
            target=self._heartbeat,
            args=(active,),
            name="puncture-run-worker-heartbeat",
            daemon=True,
        )
        execution_thread = Thread(
            target=self._execute,
            args=(active,),
            name="puncture-run-worker-execution",
            daemon=True,
        )
        active.heartbeat_thread = heartbeat_thread
        active.execution_thread = execution_thread
        try:
            heartbeat_thread.start()
            execution_thread.start()
        except Exception:
            active.heartbeat_stop.set()
            active.stop_requested.set()
            try:
                self.repository.release_execution_claim(active.current_claim())
            except Exception:
                self.metrics.repository_error()
            self.metrics.execution_finished("failed")
            with self._changed:
                self._active.pop(active.claim.owner_token, None)
                active.finished.set()
                self._changed.notify_all()
            self.notify()
            raise

    def _heartbeat(self, active: _ActiveExecution) -> None:
        while not active.heartbeat_stop.wait(
            self.config.heartbeat_interval_seconds
        ):
            if active.finished.is_set():
                return
            try:
                refreshed = self.repository.heartbeat_execution_claim(
                    active.current_claim(),
                    lease_seconds=self.config.lease_seconds,
                )
            except ExecutionSuperseded:
                if active.mark_claim_lost():
                    self.metrics.claim_lost()
                active.stop_requested.set()
                return
            except Exception:
                self.metrics.repository_error()
                continue
            active.update_claim(refreshed)
            self.metrics.heartbeat()

    def _execute(self, active: _ActiveExecution) -> None:
        outcome = "completed"
        try:
            self.service.execute_claimed(
                active.claim,
                stop_requested=active.stop_requested.is_set,
            )
        except ExecutionSuperseded:
            outcome = "superseded"
            if active.mark_claim_lost():
                self.metrics.claim_lost()
            active.stop_requested.set()
        except Exception:
            outcome = "failed"
        finally:
            active.heartbeat_stop.set()
            if active.begin_finish():
                try:
                    if outcome == "completed":
                        self.repository.release_execution_claim(
                            active.current_claim()
                        )
                    else:
                        self.repository.abandon_execution_claim(
                            active.current_claim()
                        )
                except Exception:
                    self.metrics.repository_error()
            active.finished.set()
            self.metrics.execution_finished(outcome)
            with self._changed:
                self._active.pop(active.claim.owner_token, None)
                self._changed.notify_all()
            self.notify()


__all__ = [
    "ClaimedRunService",
    "RunWorker",
    "WorkerConfig",
    "WorkerMetrics",
    "WorkerMetricsSnapshot",
    "WorkerState",
    "WorkerStatus",
]
