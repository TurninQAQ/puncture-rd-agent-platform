from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import math
import os
import sys
from threading import Barrier, Event, Lock
import time
import unittest
from unittest import mock
from uuid import uuid4

from puncture_agent.runtime import (
    ApprovalDecision,
    EventType,
    PostgresRunRepository,
    RunEventDraft,
    RunExecutionIntent,
    RunExecutionIntentKind,
    RunExecutionRepository,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from puncture_agent.runtime.errors import (
    ExecutionSuperseded,
    RunRepositoryConfigurationError,
    RunRepositoryIdempotencyConflict,
    RunRepositoryIntegrityError,
    RunRepositoryNotFound,
    RunRepositoryTransitionError,
    RunRepositoryUnavailable,
    RunRepositoryVersionConflict,
)


POSTGRES_DSN = os.environ.get("PUNCTURE_TEST_POSTGRES_DSN", "")


class _CommitAcknowledgementLoss:
    def __init__(self, *, pause_after_commit: bool = False) -> None:
        self._lock = Lock()
        self._armed = True
        self.raised_count = 0
        self.committed = Event()
        self.release = Event()
        if not pause_after_commit:
            self.release.set()

    def raise_once_after_commit(self) -> None:
        with self._lock:
            if not self._armed:
                return
            self._armed = False
            self.raised_count += 1
        self.committed.set()
        if not self.release.wait(timeout=30):
            raise RuntimeError("test did not release the post-commit fault")
        raise OSError("simulated commit acknowledgement loss")


class _CommitAcknowledgementLossConnection:
    def __init__(self, delegate, fault: _CommitAcknowledgementLoss) -> None:
        self._delegate = delegate
        self._fault = fault

    @property
    def autocommit(self):
        return self._delegate.autocommit

    @autocommit.setter
    def autocommit(self, value) -> None:
        self._delegate.autocommit = value

    @property
    def closed(self):
        return self._delegate.closed

    def commit(self):
        result = self._delegate.commit()
        self._fault.raise_once_after_commit()
        return result

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _commit_acknowledgement_loss_factory(
    fault: _CommitAcknowledgementLoss,
):
    import psycopg

    def connect(dsn, **kwargs):
        return _CommitAcknowledgementLossConnection(
            psycopg.connect(dsn, **kwargs),
            fault,
        )

    return connect


def _psycopg_available() -> bool:
    try:
        import psycopg  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def request(
    *,
    case_id: str = "case-postgres",
    tenant_id: str = "tenant-postgres",
    idempotency_key: str = "key-postgres",
    marker: object = "default",
) -> RunRequest:
    return RunRequest(
        case_id=case_id,
        user_query="validate the PostgreSQL run repository",
        task_type="DATA_MODEL_VALIDATION",
        idempotency_key=idempotency_key,
        tenant_id=tenant_id,
        principal_id="postgres-test",
        metadata={"marker": marker},
    )


def started_snapshot(
    run_request: RunRequest,
    *,
    run_id: str | None = None,
) -> RunSnapshot:
    suffix = run_id or f"run-{uuid4().hex}"
    return RunSnapshot(
        run_id=suffix,
        request=run_request,
        status=RunStatus.RUNNING,
        trace_id=f"trace-{suffix}",
        created_at="2026-07-11T12:00:00.000Z",
        updated_at="2026-07-11T12:00:00.000Z",
        final_report={},
        checkpoint={},
        approval_id=None,
        error=None,
    )


def initial_events() -> tuple[RunEventDraft, ...]:
    return (
        RunEventDraft(EventType.RUN_CREATED, None, {"source": "postgres-test"}),
        RunEventDraft(EventType.RUN_STARTED, None, {}),
    )


class PostgresRunRepositoryContractTests(unittest.TestCase):
    def test_constructor_rejects_unsafe_connection_configuration(self) -> None:
        with self.assertRaises(ValueError):
            PostgresRunRepository("https://database.example.test/agent")
        with self.assertRaises(ValueError):
            PostgresRunRepository(
                "postgresql://database.example.test/agent",
                schema="unsafe-schema",
            )
        with self.assertRaises(ValueError):
            PostgresRunRepository(
                "postgresql://database.example.test/agent",
                statement_timeout_ms=0,
            )
        with self.assertRaises(ValueError):
            PostgresRunRepository(
                "postgresql://database.example.test/agent",
                application_name="x" * 64,
            )
        with self.assertRaises(TypeError):
            PostgresRunRepository(
                "postgresql://database.example.test/agent",
                connection_factory="not-callable",
            )

    def test_dependency_and_connection_failures_are_sanitized(self) -> None:
        repository = PostgresRunRepository(
            "postgresql://user:secret@database.example.test/agent"
        )
        with mock.patch.dict(sys.modules, {"psycopg": None}):
            with self.assertRaises(RunRepositoryConfigurationError) as missing:
                repository.get("run-1", tenant_id="tenant-1")
        self.assertNotIn("secret", str(missing.exception))

        calls = []

        def failing_factory(dsn, **kwargs):
            calls.append((dsn, kwargs))
            raise OSError("network path exposed")

        unavailable_repository = PostgresRunRepository(
            "postgresql://user:secret@database.example.test/agent",
            connection_factory=failing_factory,
        )
        with self.assertRaises(RunRepositoryUnavailable) as unavailable:
            unavailable_repository.get("run-1", tenant_id="tenant-1")
        self.assertTrue(unavailable.exception.retryable)
        self.assertNotIn("secret", str(unavailable.exception))
        self.assertEqual(1, len(calls))
        self.assertEqual(False, calls[0][1]["autocommit"])
        self.assertEqual(5, calls[0][1]["connect_timeout"])

    def test_healthcheck_commits_success_and_rolls_back_invalid_migration(self) -> None:
        class Cursor:
            def __init__(self, row=None) -> None:
                self.row = row
                self.closed = False

            def fetchone(self):
                return self.row

            def close(self) -> None:
                self.closed = True

        class Connection:
            def __init__(self) -> None:
                self.closed = False
                self.autocommit = False
                self.health_rows = {}
                self.commits = 0
                self.rollbacks = 0
                self.closes = 0

            def execute(self, statement, parameters=()):
                if "SELECT name, checksum_sha256" in statement:
                    return Cursor(self.health_rows.get(parameters[0]))
                return Cursor()

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                self.rollbacks += 1

            def close(self) -> None:
                self.closes += 1
                self.closed = True

        healthy_connection = Connection()
        healthy = PostgresRunRepository(
            "postgresql://database.example.test/agent",
            connection_factory=lambda *args, **kwargs: healthy_connection,
        )
        healthy_connection.health_rows = {
            version: (
                name,
                sha256("\n".join(statements).encode("utf-8")).hexdigest(),
            )
            for version, name, statements in healthy._migration_versions()
        }

        healthy.check_health()

        self.assertEqual(1, healthy_connection.commits)
        self.assertEqual(0, healthy_connection.rollbacks)
        self.assertEqual(1, healthy_connection.closes)

        invalid_connection = Connection()
        invalid = PostgresRunRepository(
            "postgresql://database.example.test/agent",
            connection_factory=lambda *args, **kwargs: invalid_connection,
        )
        with self.assertRaises(RunRepositoryConfigurationError):
            invalid.check_health()
        self.assertEqual(0, invalid_connection.commits)
        self.assertEqual(1, invalid_connection.rollbacks)
        self.assertEqual(1, invalid_connection.closes)

    def test_execution_contract_preserves_v1_and_exposes_v2(self) -> None:
        repository = PostgresRunRepository(
            "postgresql://database.example.test/agent"
        )
        self.assertIsInstance(repository, RunExecutionRepository)
        self.assertEqual(
            "c358594722a0a42311981f4547110077cf6d389cc7758cd6a2c84c65a710ce00",
            sha256(
                "\n".join(repository._migration_statements()).encode("utf-8")
            ).hexdigest(),
        )
        versions = repository._migration_versions()
        self.assertEqual([1, 2], [item[0] for item in versions])
        self.assertEqual("initial-run-event-store", versions[0][1])
        self.assertEqual("durable-run-execution-jobs", versions[1][1])


@unittest.skipUnless(
    POSTGRES_DSN and _psycopg_available(),
    "PostgreSQL Run repository environment is not configured",
)
class PostgresRunRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = f"run_repo_{uuid4().hex[:20]}"
        self.application_name = f"run-repo-{uuid4().hex[:20]}"
        self.repository = self._repository()
        self.repository.migrate()

    def tearDown(self) -> None:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            connection.execute("SET lock_timeout = '5s'")
            connection.execute("SET statement_timeout = '10s'")
            connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    def _repository(
        self,
        *,
        application_name: str | None = None,
        connection_factory=None,
    ):
        return PostgresRunRepository(
            POSTGRES_DSN,
            schema=self.schema,
            connect_timeout_seconds=5,
            statement_timeout_ms=10_000,
            lock_timeout_ms=5_000,
            application_name=application_name or self.application_name,
            connection_factory=connection_factory,
        )

    def _execute_sql(self, statement: str, parameters=()):
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            cursor = connection.execute(statement, parameters)
            return cursor.fetchall() if cursor.description is not None else []

    def _install_reject_trigger(self, *, event_type: EventType) -> None:
        self._execute_sql(
            f"""
            CREATE OR REPLACE FUNCTION "{self.schema}".reject_event()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.event_type = '{event_type.value}' THEN
                    RAISE EXCEPTION 'test event rejection';
                END IF;
                RETURN NEW;
            END
            $$
            """
        )
        self._execute_sql(
            f"""
            CREATE TRIGGER reject_event
            BEFORE INSERT ON "{self.schema}".run_events
            FOR EACH ROW EXECUTE FUNCTION "{self.schema}".reject_event()
            """
        )

    def _drop_reject_trigger(self) -> None:
        self._execute_sql(
            f'DROP TRIGGER IF EXISTS reject_event ON "{self.schema}".run_events'
        )
        self._execute_sql(
            f'DROP FUNCTION IF EXISTS "{self.schema}".reject_event()'
        )

    def test_pg_run_01_persists_across_instances_and_cursor(self) -> None:
        self.repository.migrate()
        numeric = {
            "negative_zero": -0.0,
            "large_float": 1e20,
            "unit_float": 1.0,
        }
        run_request = request(
            idempotency_key=f"persist-{uuid4().hex}",
            marker=numeric,
        )
        initial_snapshot = replace(
            started_snapshot(run_request),
            created_at="2026-07-11T12:00:00.123Z",
            updated_at="2026-07-11T12:00:00.123Z",
            checkpoint={"numeric": numeric},
        )
        created = self.repository.create_or_get_started(
            initial_snapshot,
            (
                RunEventDraft(
                    EventType.RUN_CREATED,
                    None,
                    {"numeric": numeric},
                ),
                RunEventDraft(EventType.RUN_STARTED, None, {}),
            ),
        )
        node_event = self.repository.append_if_running(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            event=RunEventDraft(
                EventType.NODE_STARTED,
                "postgres-node",
                {"typed": True, "numeric": numeric},
                event_key="execution-v1:postgres-node-started",
            ),
        )
        succeeded_snapshot = replace(
            created.run.snapshot,
            status=RunStatus.SUCCEEDED,
            updated_at="2026-07-11T12:00:01.456Z",
            final_report={"stored": True, "numeric": numeric},
            checkpoint={"numeric": numeric, "stage": "complete"},
        )
        succeeded = self.repository.compare_and_swap(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            snapshot=succeeded_snapshot,
            events=(
                RunEventDraft(
                    EventType.RUN_COMPLETED,
                    None,
                    {"numeric": numeric},
                ),
            ),
        )
        replayed_cas = self.repository.compare_and_swap(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            snapshot=succeeded_snapshot,
            events=(
                RunEventDraft(
                    EventType.RUN_COMPLETED,
                    None,
                    {"numeric": numeric},
                ),
            ),
        )

        restarted = self._repository()
        restored = restarted.get(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
        )
        events = restarted.get_events(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
        )
        tail = restarted.get_events(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            after_sequence=node_event.sequence,
        )
        first_page = restarted.get_event_page(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            limit=2,
        )
        second_page = restarted.get_event_page(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            after_sequence=2,
            limit=2,
        )

        self.assertEqual(succeeded, replayed_cas)
        self.assertEqual(succeeded, restored)
        self.assertEqual("2026-07-11T12:00:00.123Z", restored.snapshot.created_at)
        self.assertEqual("2026-07-11T12:00:01.456Z", restored.snapshot.updated_at)
        self.assertEqual(list(range(1, 5)), [event.sequence for event in events])
        self.assertEqual([EventType.RUN_COMPLETED], [event.event_type for event in tail])
        self.assertEqual(events[:2], first_page.events)
        self.assertEqual(events[2:], second_page.events)
        self.assertEqual(4, first_page.high_water_sequence)
        self.assertEqual(RunStatus.SUCCEEDED, second_page.status)
        self.assertTrue(first_page.has_more)
        self.assertFalse(second_page.has_more)
        numeric_values = (
            restored.snapshot.request.metadata["marker"],
            restored.snapshot.checkpoint["numeric"],
            restored.snapshot.final_report["numeric"],
            events[0].payload["numeric"],
            events[2].payload["numeric"],
            events[3].payload["numeric"],
        )
        for restored_numeric in numeric_values:
            self.assertIs(type(restored_numeric["large_float"]), float)
            self.assertIs(type(restored_numeric["unit_float"]), float)
            self.assertEqual(1e20, restored_numeric["large_float"])
            self.assertEqual(
                -1.0,
                math.copysign(1.0, restored_numeric["negative_zero"]),
            )

        rows = self._execute_sql(
            f'SELECT checksum_sha256 FROM "{self.schema}".schema_migrations '
            "WHERE version = 1"
        )
        checksum = rows[0][0]
        self._execute_sql(
            f'UPDATE "{self.schema}".schema_migrations '
            "SET checksum_sha256 = %s WHERE version = 1",
            ("0" * 64,),
        )
        with self.assertRaises(RunRepositoryConfigurationError):
            restarted.migrate()
        self._execute_sql(
            f'UPDATE "{self.schema}".schema_migrations '
            "SET checksum_sha256 = %s WHERE version = 1",
            (checksum,),
        )

    def test_pg_run_02_concurrent_idempotent_create_has_one_winner(self) -> None:
        workers = 20
        barrier = Barrier(workers)
        run_request = request(idempotency_key=f"concurrent-{uuid4().hex}")

        def create(index: int):
            repository = self._repository()
            snapshot = started_snapshot(run_request, run_id=f"run-{index}-{uuid4().hex}")
            barrier.wait(timeout=30)
            return repository.create_or_get_started(snapshot, initial_events())

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(create, range(workers)))

        self.assertEqual(1, sum(result.created for result in results))
        self.assertEqual(1, len({result.run.snapshot.run_id for result in results}))
        canonical = results[0].run.snapshot.run_id
        events = self.repository.get_events(
            canonical,
            tenant_id=run_request.tenant_id,
        )
        self.assertEqual(
            [EventType.RUN_CREATED, EventType.RUN_STARTED],
            [event.event_type for event in events],
        )

        with self.assertRaises(RunRepositoryIdempotencyConflict):
            self.repository.create_or_get_started(
                started_snapshot(
                    request(
                        case_id="different-case",
                        idempotency_key=run_request.idempotency_key,
                    )
                ),
                initial_events(),
            )

    def test_pg_run_03_tenant_scope_and_not_found_isolation(self) -> None:
        idempotency_key = f"tenant-scope-{uuid4().hex}"
        request_a = request(
            tenant_id="tenant-a",
            idempotency_key=idempotency_key,
        )
        request_b = request(
            tenant_id="tenant-b",
            idempotency_key=idempotency_key,
        )
        run_a = self.repository.create_or_get_started(
            started_snapshot(request_a),
            initial_events(),
        ).run
        run_b = self.repository.create_or_get_started(
            started_snapshot(request_b),
            initial_events(),
        ).run
        self.assertNotEqual(run_a.snapshot.run_id, run_b.snapshot.run_id)

        with self.assertRaises(RunRepositoryNotFound):
            self.repository.get(run_a.snapshot.run_id, tenant_id="tenant-b")
        with self.assertRaises(RunRepositoryNotFound):
            self.repository.get_events(run_a.snapshot.run_id, tenant_id="tenant-b")
        with self.assertRaises(RunRepositoryNotFound):
            self.repository.assert_running(
                run_a.snapshot.run_id,
                tenant_id="tenant-b",
                expected_version=1,
            )
        with self.assertRaises(RunRepositoryNotFound):
            self.repository.append_if_running(
                run_a.snapshot.run_id,
                tenant_id="tenant-b",
                expected_version=1,
                event=RunEventDraft(
                    EventType.NODE_STARTED,
                    "hidden",
                    {},
                    event_key="execution-v1:hidden",
                ),
            )
        with self.assertRaises(RunRepositoryNotFound):
            self.repository.compare_and_swap(
                run_a.snapshot.run_id,
                tenant_id="tenant-b",
                expected_version=1,
                snapshot=replace(
                    run_a.snapshot,
                    status=RunStatus.CANCELLED,
                    updated_at="2026-07-11T12:00:01.000Z",
                ),
                events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
            )

        restored_a = self.repository.get(run_a.snapshot.run_id, tenant_id="tenant-a")
        self.assertEqual(RunStatus.RUNNING, restored_a.snapshot.status)
        self.assertEqual(1, restored_a.version)

    def test_pg_run_04_concurrent_events_are_contiguous_across_instances(self) -> None:
        run_request = request(idempotency_key=f"events-{uuid4().hex}")
        created = self.repository.create_or_get_started(
            started_snapshot(run_request),
            initial_events(),
        )
        workers = 20
        barrier = Barrier(workers)

        def append(index: int) -> None:
            if index < workers:
                barrier.wait(timeout=30)
            self._repository().append_if_running(
                created.run.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=1,
                event=RunEventDraft(
                    EventType.NODE_STARTED,
                    f"node-{index}",
                    {"index": index},
                    event_key=f"execution-v1:node-{index}",
                ),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(append, range(100)))

        events = self.repository.get_events(
            created.run.snapshot.run_id,
            tenant_id=run_request.tenant_id,
        )
        self.assertEqual(list(range(1, 103)), [event.sequence for event in events])
        self.assertEqual(set(range(100)), {event.payload["index"] for event in events[2:]})
        self.assertEqual(
            1,
            self.repository.get(
                created.run.snapshot.run_id,
                tenant_id=run_request.tenant_id,
            ).version,
        )

    def test_pg_run_05_cas_has_one_winner_and_fences_old_version(self) -> None:
        run_request = request(idempotency_key=f"cas-{uuid4().hex}")
        created = self.repository.create_or_get_started(
            started_snapshot(run_request),
            initial_events(),
        ).run
        stream_event = RunEventDraft(
            EventType.TOOL_CALLED,
            "tool-a",
            {"call_id": "call-1"},
            event_key="execution-v1:call-1",
        )
        first_event = self.repository.append_if_running(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            event=stream_event,
        )
        barrier = Barrier(2)

        def cancel():
            barrier.wait(timeout=30)
            return self._repository().compare_and_swap(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=1,
                snapshot=replace(
                    created.snapshot,
                    status=RunStatus.CANCELLED,
                    updated_at="2026-07-11T12:00:01.000Z",
                ),
                events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
            )

        def fail():
            barrier.wait(timeout=30)
            return self._repository().compare_and_swap(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=1,
                snapshot=replace(
                    created.snapshot,
                    status=RunStatus.FAILED,
                    updated_at="2026-07-11T12:00:02.000Z",
                    error={"code": "FAILED", "retryable": False},
                ),
                events=(RunEventDraft(EventType.RUN_FAILED, None, {}),),
            )

        outcomes = []
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (pool.submit(cancel), pool.submit(fail))
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=30))
                except BaseException as exc:
                    errors.append(exc)

        self.assertEqual(1, len(outcomes))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], RunRepositoryVersionConflict)
        final = self.repository.get(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
        )
        self.assertEqual(2, final.version)
        replayed_event = self.repository.append_if_running(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            event=stream_event,
        )
        self.assertEqual(first_event, replayed_event)
        with self.assertRaises(ExecutionSuperseded):
            self.repository.append_if_running(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=1,
                event=RunEventDraft(
                    EventType.TOOL_RESULT,
                    "tool-a",
                    {"call_id": "call-2"},
                    event_key="execution-v1:call-2",
                ),
            )
        events = self.repository.get_events(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
        )
        terminal = [
            event
            for event in events
            if event.event_type in {EventType.RUN_CANCELLED, EventType.RUN_FAILED}
        ]
        self.assertEqual(1, len(terminal))

        history_request = request(
            idempotency_key=f"cas-history-{uuid4().hex}"
        )
        history_created = self.repository.create_or_get_started(
            started_snapshot(history_request),
            initial_events(),
        ).run
        waiting_snapshot = replace(
            history_created.snapshot,
            status=RunStatus.WAITING_APPROVAL,
            updated_at="2026-07-11T12:00:01.000Z",
            checkpoint={"stage": "approval"},
            approval_id="approval-history",
        )
        waiting_events = (
            RunEventDraft(
                EventType.APPROVAL_REQUESTED,
                None,
                {"approval_id": "approval-history"},
            ),
        )
        waiting = self.repository.compare_and_swap(
            history_created.snapshot.run_id,
            tenant_id=history_request.tenant_id,
            expected_version=history_created.version,
            snapshot=waiting_snapshot,
            events=waiting_events,
        )
        resumed_snapshot = replace(
            waiting.snapshot,
            status=RunStatus.RUNNING,
            updated_at="2026-07-11T12:00:02.000Z",
            checkpoint={"stage": "resumed"},
            approval_id=None,
        )
        resumed = self.repository.compare_and_swap(
            history_created.snapshot.run_id,
            tenant_id=history_request.tenant_id,
            expected_version=waiting.version,
            snapshot=resumed_snapshot,
        )
        replayed_waiting = self._repository().compare_and_swap(
            history_created.snapshot.run_id,
            tenant_id=history_request.tenant_id,
            expected_version=history_created.version,
            snapshot=waiting_snapshot,
            events=waiting_events,
        )

        self.assertEqual(waiting, replayed_waiting)
        self.assertEqual(
            resumed,
            self.repository.get(
                history_created.snapshot.run_id,
                tenant_id=history_request.tenant_id,
            ),
        )
        history_events = self.repository.get_events(
            history_created.snapshot.run_id,
            tenant_id=history_request.tenant_id,
        )
        self.assertEqual(
            1,
            sum(
                event.event_type is EventType.APPROVAL_REQUESTED
                for event in history_events
            ),
        )
        mutation_rows = self._execute_sql(
            f'SELECT target_version FROM "{self.schema}".run_mutations '
            "WHERE tenant_id = %s AND run_id = %s ORDER BY target_version",
            (history_request.tenant_id, history_created.snapshot.run_id),
        )
        self.assertEqual([(2,), (3,)], mutation_rows)

    def test_pg_run_06_create_transaction_rolls_back_claim_and_events(self) -> None:
        self._install_reject_trigger(event_type=EventType.RUN_STARTED)
        run_request = request(idempotency_key=f"create-rollback-{uuid4().hex}")
        snapshot = started_snapshot(run_request)
        try:
            with self.assertRaises(RunRepositoryIntegrityError):
                self.repository.create_or_get_started(snapshot, initial_events())

            rows = self._execute_sql(
                f'SELECT count(*) FROM "{self.schema}".runs '
                "WHERE tenant_id = %s AND idempotency_key = %s",
                (run_request.tenant_id, run_request.idempotency_key),
            )
            self.assertEqual(0, rows[0][0])
        finally:
            self._drop_reject_trigger()
        retried = self.repository.create_or_get_started(snapshot, initial_events())
        self.assertTrue(retried.created)

    def test_pg_run_07_terminal_event_failure_rolls_back_version_and_state(self) -> None:
        run_request = request(idempotency_key=f"terminal-rollback-{uuid4().hex}")
        created = self.repository.create_or_get_started(
            started_snapshot(run_request),
            initial_events(),
        ).run
        self._install_reject_trigger(event_type=EventType.RUN_CANCELLED)
        cancelled_snapshot = replace(
            created.snapshot,
            status=RunStatus.CANCELLED,
            updated_at="2026-07-11T12:00:01.000Z",
        )
        try:
            with self.assertRaises(RunRepositoryIntegrityError):
                self.repository.compare_and_swap(
                    created.snapshot.run_id,
                    tenant_id=run_request.tenant_id,
                    expected_version=1,
                    snapshot=cancelled_snapshot,
                    events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
                )

            restored = self.repository.get(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
            )
            self.assertEqual(RunStatus.RUNNING, restored.snapshot.status)
            self.assertEqual(1, restored.version)
            self.assertEqual(
                2,
                len(
                    self.repository.get_events(
                        created.snapshot.run_id,
                        tenant_id=run_request.tenant_id,
                    )
                ),
            )
        finally:
            self._drop_reject_trigger()
        cancelled = self.repository.compare_and_swap(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            snapshot=cancelled_snapshot,
            events=(RunEventDraft(EventType.RUN_CANCELLED, None, {}),),
        )
        self.assertEqual(RunStatus.CANCELLED, cancelled.snapshot.status)

    def test_pg_run_08_transient_failures_reconcile_or_roll_back(self) -> None:
        self._execute_sql(
            f"""
            CREATE OR REPLACE FUNCTION "{self.schema}".block_event()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.payload ->> 'block_backend' = 'true' THEN
                    PERFORM pg_sleep(30);
                END IF;
                RETURN NEW;
            END
            $$
            """
        )
        self._execute_sql(
            f"""
            CREATE TRIGGER block_event
            BEFORE INSERT ON "{self.schema}".run_events
            FOR EACH ROW EXECUTE FUNCTION "{self.schema}".block_event()
            """
        )
        run_request = request(idempotency_key=f"terminate-{uuid4().hex}")
        created = self.repository.create_or_get_started(
            started_snapshot(run_request),
            initial_events(),
        ).run
        application_name = f"terminate-{uuid4().hex[:20]}"
        repository = self._repository(application_name=application_name)
        event = RunEventDraft(
            EventType.NODE_STARTED,
            "blocked-node",
            {"block_backend": True},
            event_key="execution-v1:blocked-node",
        )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    repository.append_if_running,
                    created.snapshot.run_id,
                    tenant_id=run_request.tenant_id,
                    expected_version=1,
                    event=event,
                )
                deadline = time.monotonic() + 20
                backend_pid = None
                while time.monotonic() < deadline and backend_pid is None:
                    rows = self._execute_sql(
                        """
                        SELECT pid
                        FROM pg_stat_activity
                        WHERE application_name = %s
                          AND state = 'active'
                          AND query LIKE '%%INSERT INTO%%run_events%%'
                        """,
                        (application_name,),
                    )
                    if rows:
                        backend_pid = rows[0][0]
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(
                    backend_pid,
                    "blocked repository backend not found",
                )
                terminated = self._execute_sql(
                    "SELECT pg_terminate_backend(%s::integer, %s::bigint)",
                    (backend_pid, 5_000),
                )
                self.assertEqual([(True,)], terminated)
                with self.assertRaises(RunRepositoryUnavailable) as raised:
                    future.result(timeout=30)

            self.assertTrue(raised.exception.retryable)
            self.assertNotIn("postgresql://", str(raised.exception))
            restored = self.repository.get(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
            )
            self.assertEqual(1, restored.version)
            self.assertEqual(
                2,
                len(
                    self.repository.get_events(
                        created.snapshot.run_id,
                        tenant_id=run_request.tenant_id,
                    )
                ),
            )
        finally:
            self._execute_sql(
                f'DROP TRIGGER IF EXISTS block_event ON "{self.schema}".run_events'
            )
            self._execute_sql(
                f'DROP FUNCTION IF EXISTS "{self.schema}".block_event()'
            )
        retried = self.repository.append_if_running(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=1,
            event=event,
        )
        self.assertEqual(3, retried.sequence)

        with self.subTest(failure="create commit acknowledgement loss"):
            create_fault = _CommitAcknowledgementLoss()
            create_repository = self._repository(
                connection_factory=_commit_acknowledgement_loss_factory(
                    create_fault
                )
            )
            create_request = request(
                idempotency_key=f"ack-create-{uuid4().hex}"
            )
            create_snapshot = started_snapshot(create_request)
            create_result = create_repository.create_or_get_started(
                create_snapshot,
                initial_events(),
            )
            self.assertEqual(1, create_fault.raised_count)
            self.assertTrue(create_result.created)
            self.assertEqual(
                create_result.run,
                self.repository.get(
                    create_snapshot.run_id,
                    tenant_id=create_request.tenant_id,
                ),
            )
            create_events = self.repository.get_events(
                create_snapshot.run_id,
                tenant_id=create_request.tenant_id,
            )
            self.assertEqual([1, 2], [item.sequence for item in create_events])
            duplicate = self.repository.create_or_get_started(
                create_snapshot,
                initial_events(),
            )
            self.assertFalse(duplicate.created)
            self.assertEqual(create_result.run, duplicate.run)

        with self.subTest(failure="append commit acknowledgement loss"):
            append_request = request(
                idempotency_key=f"ack-append-{uuid4().hex}"
            )
            append_created = self.repository.create_or_get_started(
                started_snapshot(append_request),
                initial_events(),
            ).run
            append_fault = _CommitAcknowledgementLoss()
            append_repository = self._repository(
                connection_factory=_commit_acknowledgement_loss_factory(
                    append_fault
                )
            )
            append_draft = RunEventDraft(
                EventType.NODE_STARTED,
                "ack-node",
                {"step": 1},
                event_key="execution-v1:ack-node",
            )
            appended = append_repository.append_if_running(
                append_created.snapshot.run_id,
                tenant_id=append_request.tenant_id,
                expected_version=append_created.version,
                event=append_draft,
            )
            replayed_append = append_repository.append_if_running(
                append_created.snapshot.run_id,
                tenant_id=append_request.tenant_id,
                expected_version=append_created.version,
                event=append_draft,
            )
            self.assertEqual(1, append_fault.raised_count)
            self.assertEqual(appended, replayed_append)
            self.assertEqual(3, appended.sequence)
            append_rows = self._execute_sql(
                f'SELECT count(*) FROM "{self.schema}".run_events '
                "WHERE tenant_id = %s AND run_id = %s AND event_key = %s",
                (
                    append_request.tenant_id,
                    append_created.snapshot.run_id,
                    append_draft.event_key,
                ),
            )
            self.assertEqual(1, append_rows[0][0])

        with self.subTest(failure="CAS commit acknowledgement loss"):
            cas_request = request(idempotency_key=f"ack-cas-{uuid4().hex}")
            cas_created = self.repository.create_or_get_started(
                started_snapshot(cas_request),
                initial_events(),
            ).run
            cas_target = replace(
                cas_created.snapshot,
                status=RunStatus.CANCELLED,
                updated_at="2026-07-11T12:00:01.000Z",
            )
            cas_events = (RunEventDraft(EventType.RUN_CANCELLED, None, {}),)
            cas_fault = _CommitAcknowledgementLoss()
            cas_repository = self._repository(
                connection_factory=_commit_acknowledgement_loss_factory(cas_fault)
            )
            committed = cas_repository.compare_and_swap(
                cas_created.snapshot.run_id,
                tenant_id=cas_request.tenant_id,
                expected_version=cas_created.version,
                snapshot=cas_target,
                events=cas_events,
            )
            replayed_cas = self.repository.compare_and_swap(
                cas_created.snapshot.run_id,
                tenant_id=cas_request.tenant_id,
                expected_version=cas_created.version,
                snapshot=cas_target,
                events=cas_events,
            )
            self.assertEqual(1, cas_fault.raised_count)
            self.assertEqual(committed, replayed_cas)
            self.assertEqual(2, committed.version)
            terminal_events = self.repository.get_events(
                cas_created.snapshot.run_id,
                tenant_id=cas_request.tenant_id,
            )
            self.assertEqual(
                1,
                sum(
                    item.event_type is EventType.RUN_CANCELLED
                    for item in terminal_events
                ),
            )
            mutation_rows = self._execute_sql(
                f'SELECT count(*) FROM "{self.schema}".run_mutations '
                "WHERE tenant_id = %s AND run_id = %s",
                (cas_request.tenant_id, cas_created.snapshot.run_id),
            )
            self.assertEqual(1, mutation_rows[0][0])

        with self.subTest(
            failure="CAS acknowledgement loss after later state progression"
        ):
            progressed_request = request(
                idempotency_key=f"ack-cas-progressed-{uuid4().hex}"
            )
            progressed_created = self.repository.create_or_get_started(
                started_snapshot(progressed_request),
                initial_events(),
            ).run
            failed_target = replace(
                progressed_created.snapshot,
                status=RunStatus.FAILED,
                updated_at="2026-07-11T12:00:01.000Z",
                checkpoint={"recoverable": True, "stage": "failed"},
                error={"code": "TRANSIENT", "retryable": True},
            )
            failed_events = (RunEventDraft(EventType.RUN_FAILED, None, {}),)
            progressed_fault = _CommitAcknowledgementLoss(
                pause_after_commit=True
            )
            progressed_repository = self._repository(
                connection_factory=_commit_acknowledgement_loss_factory(
                    progressed_fault
                )
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    progressed_repository.compare_and_swap,
                    progressed_created.snapshot.run_id,
                    tenant_id=progressed_request.tenant_id,
                    expected_version=progressed_created.version,
                    snapshot=failed_target,
                    events=failed_events,
                )
                self.assertTrue(
                    progressed_fault.committed.wait(timeout=30),
                    "faulted CAS did not commit",
                )
                try:
                    resumed_target = replace(
                        failed_target,
                        status=RunStatus.RUNNING,
                        updated_at="2026-07-11T12:00:02.000Z",
                        checkpoint={"stage": "resumed"},
                        error=None,
                    )
                    resumed = self.repository.compare_and_swap(
                        progressed_created.snapshot.run_id,
                        tenant_id=progressed_request.tenant_id,
                        expected_version=2,
                        snapshot=resumed_target,
                    )
                finally:
                    progressed_fault.release.set()
                reconciled_failed = future.result(timeout=30)

            self.assertEqual(1, progressed_fault.raised_count)
            self.assertEqual(2, reconciled_failed.version)
            self.assertEqual(failed_target, reconciled_failed.snapshot)
            self.assertEqual(
                resumed,
                self.repository.get(
                    progressed_created.snapshot.run_id,
                    tenant_id=progressed_request.tenant_id,
                ),
            )
            progressed_events = self.repository.get_events(
                progressed_created.snapshot.run_id,
                tenant_id=progressed_request.tenant_id,
            )
            self.assertEqual(
                1,
                sum(
                    item.event_type is EventType.RUN_FAILED
                    for item in progressed_events
                ),
            )


@unittest.skipUnless(
    POSTGRES_DSN and _psycopg_available(),
    "PostgreSQL execution repository environment is not configured",
)
class PostgresRunExecutionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = f"run_exec_{uuid4().hex[:20]}"
        self.repository = self._repository()
        self.repository.migrate()

    def tearDown(self) -> None:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            connection.execute("SET lock_timeout = '5s'")
            connection.execute("SET statement_timeout = '10s'")
            connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    def _repository(self, *, connection_factory=None):
        return PostgresRunRepository(
            POSTGRES_DSN,
            schema=self.schema,
            connect_timeout_seconds=5,
            statement_timeout_ms=10_000,
            lock_timeout_ms=5_000,
            application_name=f"run-exec-{uuid4().hex[:20]}",
            connection_factory=connection_factory,
        )

    def _execute_sql(self, statement: str, parameters=()):
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            cursor = connection.execute(statement, parameters)
            return cursor.fetchall() if cursor.description is not None else []

    def _create(self, marker: str):
        run_request = request(idempotency_key=f"{marker}-{uuid4().hex}")
        run = self.repository.create_or_get_started(
            started_snapshot(run_request),
            initial_events(),
        ).run
        return run_request, run

    def test_v1_schema_upgrade_backfills_running_create_job(self) -> None:
        run_request, created = self._create("v1-upgrade")
        terminal_request, terminal_created = self._create("v1-terminal")
        terminal = self.repository.compare_and_swap(
            terminal_created.snapshot.run_id,
            tenant_id=terminal_request.tenant_id,
            expected_version=terminal_created.version,
            snapshot=replace(
                terminal_created.snapshot,
                status=RunStatus.SUCCEEDED,
                updated_at="2026-07-11T12:00:01.000Z",
                final_report={"legacy": "terminal"},
            ),
            events=(
                RunEventDraft(
                    EventType.RUN_COMPLETED,
                    None,
                    {"status": RunStatus.SUCCEEDED.value},
                ),
            ),
        )
        resume_request, resume_created = self._create("v1-resume")
        failed = self.repository.compare_and_swap(
            resume_created.snapshot.run_id,
            tenant_id=resume_request.tenant_id,
            expected_version=resume_created.version,
            snapshot=replace(
                resume_created.snapshot,
                status=RunStatus.FAILED,
                updated_at="2026-07-11T12:00:01.000Z",
                checkpoint={"recoverable": True},
                error={"code": "LEGACY_TIMEOUT"},
            ),
            events=(
                RunEventDraft(
                    EventType.RUN_FAILED,
                    None,
                    {"code": "LEGACY_TIMEOUT"},
                ),
            ),
        )
        resumed = self.repository.compare_and_swap(
            resume_created.snapshot.run_id,
            tenant_id=resume_request.tenant_id,
            expected_version=failed.version,
            snapshot=replace(
                failed.snapshot,
                status=RunStatus.RUNNING,
                updated_at="2026-07-11T12:00:02.000Z",
                error=None,
            ),
        )
        v1_checksum = self._execute_sql(
            f'SELECT checksum_sha256 FROM "{self.schema}".schema_migrations '
            "WHERE version = 1"
        )[0][0]
        self._execute_sql(
            f'DROP TABLE "{self.schema}".run_execution_jobs'
        )
        self._execute_sql(
            f'DELETE FROM "{self.schema}".schema_migrations WHERE version = 2'
        )

        self.repository.migrate()
        self.repository.check_health()

        self.assertEqual(
            v1_checksum,
            self._execute_sql(
                f'SELECT checksum_sha256 FROM "{self.schema}".schema_migrations '
                "WHERE version = 1"
            )[0][0],
        )
        replayed = self.repository.create_or_get_started(
            started_snapshot(terminal_request),
            initial_events(),
        )
        self.assertFalse(replayed.created)
        self.assertEqual(terminal, replayed.run)
        self.assertEqual(
            [(1, "CREATE", True)],
            self._execute_sql(
                f'SELECT execution_version, intent_kind, '
                f'completed_at IS NOT NULL FROM "{self.schema}".'
                "run_execution_jobs WHERE tenant_id = %s AND run_id = %s "
                "ORDER BY execution_version",
                (terminal_request.tenant_id, terminal.snapshot.run_id),
            ),
        )
        self.assertEqual(
            [(1, "CREATE", True), (3, "RESUME", False)],
            self._execute_sql(
                f'SELECT execution_version, intent_kind, '
                f'completed_at IS NOT NULL FROM "{self.schema}".'
                "run_execution_jobs WHERE tenant_id = %s AND run_id = %s "
                "ORDER BY execution_version",
                (resume_request.tenant_id, resumed.snapshot.run_id),
            ),
        )
        claim = self.repository.claim_next_execution(
            worker_id="worker-v1-upgrade",
            owner_token=f"owner-v1-upgrade-{uuid4().hex}",
            lease_seconds=1.0,
        )
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(created.snapshot.run_id, claim.run.snapshot.run_id)
        self.assertEqual(run_request.tenant_id, claim.run.snapshot.request.tenant_id)
        self.assertEqual(RunExecutionIntentKind.CREATE, claim.intent.kind)
        self.assertEqual(1, claim.generation)

    def test_execution_claim_lifecycle_enqueue_and_db_ttl_fencing(self) -> None:
        run_request, created = self._create("claim-lifecycle")
        claim = self.repository.claim_next_execution(
            worker_id="worker-a",
            owner_token=f"owner-a-{uuid4().hex}",
            lease_seconds=1.0,
        )
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(RunExecutionIntentKind.CREATE, claim.intent.kind)
        self.assertEqual(
            claim,
            self.repository.claim_next_execution(
                worker_id=claim.worker_id,
                owner_token=claim.owner_token,
                lease_seconds=1.0,
            ),
        )
        with self.assertRaises(RunRepositoryTransitionError):
            self.repository.claim_next_execution(
                worker_id="different-worker",
                owner_token=claim.owner_token,
                lease_seconds=1.0,
            )

        draft = RunEventDraft(
            EventType.NODE_STARTED,
            "claimed-node",
            {"claimed": True},
            event_key="execution-v1:claimed-node",
        )
        appended = self.repository.append_if_claimed(claim, event=draft)
        self.assertEqual(
            appended,
            self._repository().append_if_claimed(claim, event=draft),
        )
        heartbeat = self.repository.heartbeat_execution_claim(
            claim,
            lease_seconds=1.0,
        )
        self.assertGreaterEqual(heartbeat.heartbeat_at, claim.heartbeat_at)
        lease_before_abandon = self._execute_sql(
            f'SELECT lease_expires_at, released_at FROM "{self.schema}".'
            "run_execution_jobs WHERE tenant_id = %s AND run_id = %s "
            "AND execution_version = 1",
            (run_request.tenant_id, created.snapshot.run_id),
        )[0]
        self.repository.abandon_execution_claim(heartbeat)
        lease_after_abandon = self._execute_sql(
            f'SELECT lease_expires_at, released_at FROM "{self.schema}".'
            "run_execution_jobs WHERE tenant_id = %s AND run_id = %s "
            "AND execution_version = 1",
            (run_request.tenant_id, created.snapshot.run_id),
        )[0]
        self.assertEqual(lease_before_abandon, lease_after_abandon)

        self.repository.release_execution_claim(heartbeat)
        with self.assertRaises(ExecutionSuperseded):
            self.repository.assert_execution_claim(heartbeat)
        reclaimed = self.repository.claim_next_execution(
            worker_id="worker-b",
            owner_token=f"owner-b-{uuid4().hex}",
            lease_seconds=1.0,
        )
        self.assertIsNotNone(reclaimed)
        assert reclaimed is not None
        self.assertEqual(2, reclaimed.generation)
        with self.assertRaises(ExecutionSuperseded):
            self.repository.append_if_claimed(
                heartbeat,
                event=RunEventDraft(
                    EventType.NODE_COMPLETED,
                    "stale-node",
                    {},
                    event_key="execution-v1:stale-node",
                ),
            )

        waiting_snapshot = replace(
            reclaimed.run.snapshot,
            status=RunStatus.WAITING_APPROVAL,
            updated_at="2026-07-11T12:00:01.000Z",
            checkpoint={"stage": "approval"},
            approval_id="approval-pg-execution",
        )
        waiting = self.repository.compare_and_swap_if_claimed(
            reclaimed,
            snapshot=waiting_snapshot,
            events=(
                RunEventDraft(
                    EventType.APPROVAL_REQUESTED,
                    None,
                    {"approval_id": "approval-pg-execution"},
                ),
            ),
        )
        approval_intent = RunExecutionIntent(
            RunExecutionIntentKind.APPROVAL,
            ApprovalDecision(
                approval_id="approval-pg-execution",
                approved=True,
                principal_id="approver-pg",
                comment="approved",
            ),
        )
        resumed_snapshot = replace(
            waiting.snapshot,
            status=RunStatus.RUNNING,
            updated_at="2026-07-11T12:00:02.000Z",
            checkpoint={"stage": "resumed"},
            approval_id=None,
        )
        with self.assertRaises(RunRepositoryTransitionError):
            self.repository.compare_and_swap_and_enqueue(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=waiting.version,
                snapshot=resumed_snapshot,
                intent=RunExecutionIntent(RunExecutionIntentKind.RESUME),
            )
        resumed = self.repository.compare_and_swap_and_enqueue(
            created.snapshot.run_id,
            tenant_id=run_request.tenant_id,
            expected_version=waiting.version,
            snapshot=resumed_snapshot,
            intent=approval_intent,
        )
        self.assertEqual(
            resumed,
            self._repository().compare_and_swap_and_enqueue(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=waiting.version,
                snapshot=resumed_snapshot,
                intent=approval_intent,
            ),
        )
        with self.assertRaises(RunRepositoryVersionConflict):
            self.repository.compare_and_swap_and_enqueue(
                created.snapshot.run_id,
                tenant_id=run_request.tenant_id,
                expected_version=waiting.version,
                snapshot=resumed_snapshot,
                intent=RunExecutionIntent(RunExecutionIntentKind.RESUME),
            )
        approval_claim = self.repository.claim_next_execution(
            worker_id="worker-c",
            owner_token=f"owner-c-{uuid4().hex}",
            lease_seconds=1.0,
        )
        self.assertIsNotNone(approval_claim)
        assert approval_claim is not None
        self.assertEqual(approval_intent, approval_claim.intent)
        succeeded_snapshot = replace(
            approval_claim.run.snapshot,
            status=RunStatus.SUCCEEDED,
            updated_at="2026-07-11T12:00:03.000Z",
            final_report={"ok": True},
        )
        succeeded = self.repository.compare_and_swap_if_claimed(
            approval_claim,
            snapshot=succeeded_snapshot,
            events=(RunEventDraft(EventType.RUN_COMPLETED, None, {}),),
        )
        self.assertEqual(
            succeeded,
            self.repository.compare_and_swap_if_claimed(
                approval_claim,
                snapshot=succeeded_snapshot,
                events=(RunEventDraft(EventType.RUN_COMPLETED, None, {}),),
            ),
        )
        job_rows = self._execute_sql(
            f'SELECT execution_version, intent_kind, completed_at IS NOT NULL '
            f'FROM "{self.schema}".run_execution_jobs '
            "WHERE tenant_id = %s AND run_id = %s ORDER BY execution_version",
            (run_request.tenant_id, created.snapshot.run_id),
        )
        self.assertEqual([(1, "CREATE", True), (3, "APPROVAL", True)], job_rows)

        ttl_request, ttl_run = self._create("claim-ttl")
        ttl_claim = self.repository.claim_next_execution(
            worker_id="worker-ttl-a",
            owner_token=f"owner-ttl-a-{uuid4().hex}",
            lease_seconds=0.1,
        )
        self.assertIsNotNone(ttl_claim)
        assert ttl_claim is not None
        self.repository.abandon_execution_claim(ttl_claim)
        self.assertIsNone(
            self.repository.claim_next_execution(
                worker_id="worker-ttl-b",
                owner_token=f"owner-ttl-probe-{uuid4().hex}",
                lease_seconds=1.0,
            )
        )
        time.sleep(0.2)
        ttl_reclaimed = self.repository.claim_next_execution(
            worker_id="worker-ttl-b",
            owner_token=f"owner-ttl-b-{uuid4().hex}",
            lease_seconds=1.0,
        )
        self.assertIsNotNone(ttl_reclaimed)
        assert ttl_reclaimed is not None
        self.assertEqual(ttl_run.snapshot.run_id, ttl_reclaimed.run.snapshot.run_id)
        self.assertEqual(ttl_request.tenant_id, ttl_reclaimed.run.snapshot.request.tenant_id)
        self.assertEqual(2, ttl_reclaimed.generation)
        with self.assertRaises(ExecutionSuperseded):
            self.repository.assert_execution_claim(ttl_claim)

    def test_claimed_writes_reconcile_commit_acknowledgement_loss(self) -> None:
        run_request, created = self._create("claim-ack")
        claim_fault = _CommitAcknowledgementLoss()
        claim = self._repository(
            connection_factory=_commit_acknowledgement_loss_factory(claim_fault)
        ).claim_next_execution(
            worker_id="worker-ack",
            owner_token=f"owner-ack-{uuid4().hex}",
            lease_seconds=2.0,
        )
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(1, claim_fault.raised_count)

        append_fault = _CommitAcknowledgementLoss()
        draft = RunEventDraft(
            EventType.TOOL_CALLED,
            "ack-tool",
            {"call_id": "ack-call"},
            event_key="execution-v1:ack-call",
        )
        appended = self._repository(
            connection_factory=_commit_acknowledgement_loss_factory(append_fault)
        ).append_if_claimed(claim, event=draft)
        self.assertEqual(1, append_fault.raised_count)
        self.assertEqual(
            appended,
            self.repository.append_if_claimed(claim, event=draft),
        )

        target = replace(
            created.snapshot,
            status=RunStatus.SUCCEEDED,
            updated_at="2026-07-11T12:00:01.000Z",
            final_report={"ack": True},
        )
        cas_fault = _CommitAcknowledgementLoss()
        committed = self._repository(
            connection_factory=_commit_acknowledgement_loss_factory(cas_fault)
        ).compare_and_swap_if_claimed(
            claim,
            snapshot=target,
            events=(RunEventDraft(EventType.RUN_COMPLETED, None, {}),),
        )
        self.assertEqual(1, cas_fault.raised_count)
        self.assertEqual(2, committed.version)
        self.assertEqual(
            committed,
            self.repository.compare_and_swap_if_claimed(
                claim,
                snapshot=target,
                events=(RunEventDraft(EventType.RUN_COMPLETED, None, {}),),
            ),
        )
        rows = self._execute_sql(
            f'SELECT count(*) FROM "{self.schema}".run_events '
            "WHERE tenant_id = %s AND run_id = %s AND event_key = %s",
            (run_request.tenant_id, created.snapshot.run_id, draft.event_key),
        )
        self.assertEqual(1, rows[0][0])

    def test_skip_locked_claimers_take_distinct_runs(self) -> None:
        run_ids = {self._create(f"parallel-{index}")[1].snapshot.run_id for index in range(8)}
        barrier = Barrier(8)

        def claim(index: int):
            barrier.wait(timeout=30)
            return self._repository().claim_next_execution(
                worker_id=f"worker-{index}",
                owner_token=f"owner-{index}-{uuid4().hex}",
                lease_seconds=2.0,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(claim, range(8)))
        self.assertNotIn(None, claims)
        claimed_ids = {item.run.snapshot.run_id for item in claims if item is not None}
        self.assertEqual(run_ids, claimed_ids)


if __name__ == "__main__":
    unittest.main()
