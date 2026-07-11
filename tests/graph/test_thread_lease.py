"""Tests for cross-runtime thread execution lease backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from typing import Any

from puncture_agent.agent.thread_lease import (
    PostgresAdvisoryThreadExecutionLeaseManager,
    SQLiteThreadExecutionLeaseManager,
    ThreadLeaseBusy,
    ThreadLeaseLost,
    ThreadLeaseUnavailable,
    thread_advisory_lock_key,
)


class _MutableClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeCursor:
    def __init__(self, value: Any) -> None:
        self._value = value
        self.closed = False

    def fetchone(self) -> tuple[Any]:
        return (self._value,)

    def close(self) -> None:
        self.closed = True


class _FakePostgresConnection:
    def __init__(
        self,
        *,
        backend_pid: int = 4242,
        try_lock: bool = True,
        unlock: bool = True,
        fail_statement: str | None = None,
    ) -> None:
        self.autocommit = False
        self.closed = False
        self.backend_pid = backend_pid
        self.try_lock = try_lock
        self.unlock = unlock
        self.fail_statement = fail_statement
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.close_calls = 0

    def execute(
        self,
        statement: str,
        parameters: tuple[Any, ...] = (),
    ) -> _FakeCursor:
        self.calls.append((statement, parameters))
        if self.fail_statement is not None and self.fail_statement in statement:
            raise OSError("simulated PostgreSQL connection failure")
        if statement == "SELECT pg_backend_pid()":
            return _FakeCursor(self.backend_pid)
        if statement == "SELECT pg_try_advisory_lock(%s::bigint)":
            return _FakeCursor(self.try_lock)
        if statement == "SELECT pg_advisory_unlock(%s::bigint)":
            return _FakeCursor(self.unlock)
        raise AssertionError(f"unexpected SQL: {statement!r}")

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _GatedSQLiteConnection:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.begin_entered = Event()
        self.allow_begin = Event()
        self.gate_next_write = True

    def execute(self, statement: str, *args: Any, **kwargs: Any) -> Any:
        if statement == "BEGIN IMMEDIATE" and self.gate_next_write:
            self.gate_next_write = False
            self.begin_entered.set()
            if not self.allow_begin.wait(timeout=2.0):
                raise RuntimeError("test synchronization timeout")
        return self._delegate.execute(statement, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class SQLiteThreadExecutionLeaseTests(unittest.TestCase):
    def test_two_managers_reject_same_live_thread(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            with (
                SQLiteThreadExecutionLeaseManager(path) as first_manager,
                SQLiteThreadExecutionLeaseManager(path) as second_manager,
            ):
                first = first_manager.acquire("thread-1", operation="run")
                self.assertEqual(first.backend, "sqlite-ttl")
                self.assertEqual(first.thread_id, "thread-1")
                self.assertEqual(first.operation, "run")

                with self.assertRaises(ThreadLeaseBusy):
                    second_manager.acquire("thread-1", operation="resume")

                first.release()
                replacement = second_manager.acquire(
                    "thread-1",
                    operation="resume",
                )
                replacement.assert_valid()
                replacement.release()

    def test_different_threads_can_be_held_by_different_managers(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            with (
                SQLiteThreadExecutionLeaseManager(path) as first_manager,
                SQLiteThreadExecutionLeaseManager(path) as second_manager,
            ):
                first = first_manager.acquire("thread-a", operation="stream")
                second = second_manager.acquire("thread-b", operation="run")

                first.assert_valid()
                second.assert_valid()
                self.assertNotEqual(first.owner_token, second.owner_token)

                first.release()
                second.release()

    def test_renewal_expiry_and_reclaim_fence_stale_owner(self) -> None:
        clock = _MutableClock(1_000.0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            with (
                SQLiteThreadExecutionLeaseManager(
                    path,
                    lease_seconds=5.0,
                    clock=clock,
                ) as first_manager,
                SQLiteThreadExecutionLeaseManager(
                    path,
                    lease_seconds=5.0,
                    clock=clock,
                ) as second_manager,
            ):
                stale = first_manager.acquire("thread-1", operation="run")
                self.assertEqual(stale.generation, 1)

                clock.advance(4.0)
                stale.renew()
                clock.advance(4.0)
                stale.assert_valid()

                clock.advance(2.0)
                with self.assertRaises(ThreadLeaseLost):
                    stale.assert_valid()

                replacement = second_manager.acquire(
                    "thread-1",
                    operation="resume",
                )
                self.assertEqual(replacement.generation, 2)
                self.assertNotEqual(stale.owner_token, replacement.owner_token)

                with self.assertRaises(ThreadLeaseLost):
                    stale.renew()
                with self.assertRaises(ThreadLeaseLost):
                    stale.release()

                replacement.assert_valid()
                replacement.release()

                blocked = first_manager.acquire(
                    "thread-blocked-renewal",
                    operation="run",
                )
                clock.advance(4.0)
                gated = _GatedSQLiteConnection(first_manager._connection)
                first_manager._connection = gated
                with ThreadPoolExecutor(max_workers=1) as pool:
                    renewal = pool.submit(blocked.renew)
                    self.assertTrue(gated.begin_entered.wait(timeout=2.0))
                    clock.advance(2.0)
                    gated.allow_begin.set()
                    with self.assertRaises(ThreadLeaseLost):
                        renewal.result(timeout=2.0)


class PostgresAdvisoryThreadExecutionLeaseTests(unittest.TestCase):
    def test_advisory_key_is_stable_namespaced_signed_bigint(self) -> None:
        namespace = "test/thread-leases/v1"
        expected_digest = sha256(
            namespace.encode("utf-8") + b"\x00" + b"thread-1"
        ).digest()
        expected = int.from_bytes(
            expected_digest[:8],
            byteorder="big",
            signed=True,
        )

        first = thread_advisory_lock_key("thread-1", namespace=namespace)
        second = thread_advisory_lock_key("thread-1", namespace=namespace)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertGreaterEqual(first, -(2**63))
        self.assertLessEqual(first, 2**63 - 1)
        self.assertNotEqual(
            first,
            thread_advisory_lock_key("thread-1", namespace="other"),
        )

    def test_acquire_assert_and_release_use_parameterized_sql(self) -> None:
        connection = _FakePostgresConnection()
        factory_calls: list[tuple[str, dict[str, Any]]] = []

        def connection_factory(dsn: str, **kwargs: Any) -> Any:
            factory_calls.append((dsn, kwargs))
            return connection

        namespace = "test/thread-leases/v1"
        manager = PostgresAdvisoryThreadExecutionLeaseManager(
            "postgresql://lease.example.test/agent",
            namespace=namespace,
            connection_factory=connection_factory,
        )
        lease = manager.acquire("thread-1", operation="stream")
        expected_key = thread_advisory_lock_key(
            "thread-1",
            namespace=namespace,
        )

        self.assertEqual(
            factory_calls,
            [
                (
                    "postgresql://lease.example.test/agent",
                    {
                        "autocommit": True,
                        "connect_timeout": 5,
                        "application_name": "puncture-langgraph-thread-lease",
                    },
                )
            ],
        )
        self.assertTrue(connection.autocommit)
        self.assertEqual(lease.advisory_lock_key, expected_key)
        self.assertEqual(lease.backend_pid, 4242)
        self.assertEqual(lease.backend, "postgres-advisory")
        self.assertEqual(
            connection.calls,
            [
                ("SELECT pg_backend_pid()", ()),
                ("SELECT pg_try_advisory_lock(%s::bigint)", (expected_key,)),
            ],
        )

        lease.assert_valid()
        lease.release()
        lease.release()

        self.assertEqual(
            connection.calls,
            [
                ("SELECT pg_backend_pid()", ()),
                ("SELECT pg_try_advisory_lock(%s::bigint)", (expected_key,)),
                ("SELECT pg_backend_pid()", ()),
                ("SELECT pg_advisory_unlock(%s::bigint)", (expected_key,)),
            ],
        )
        self.assertEqual(connection.close_calls, 1)

    def test_busy_lock_closes_dedicated_connection(self) -> None:
        connection = _FakePostgresConnection(try_lock=False)
        manager = PostgresAdvisoryThreadExecutionLeaseManager(
            "postgresql://lease.example.test/agent",
            connection_factory=lambda *args, **kwargs: connection,
        )

        with self.assertRaises(ThreadLeaseBusy):
            manager.acquire("thread-1", operation="run")

        expected_key = thread_advisory_lock_key("thread-1")
        self.assertEqual(
            connection.calls,
            [
                ("SELECT pg_backend_pid()", ()),
                ("SELECT pg_try_advisory_lock(%s::bigint)", (expected_key,)),
            ],
        )
        self.assertTrue(connection.closed)
        self.assertEqual(connection.close_calls, 1)

    def test_connection_loss_is_reported_as_lost(self) -> None:
        connection = _FakePostgresConnection()
        manager = PostgresAdvisoryThreadExecutionLeaseManager(
            "postgresql://lease.example.test/agent",
            connection_factory=lambda *args, **kwargs: connection,
        )
        lease = manager.acquire("thread-1", operation="resume")
        connection.closed = True

        with self.assertRaises(ThreadLeaseLost):
            lease.assert_valid()
        with self.assertRaises(ThreadLeaseLost):
            lease.release()
        self.assertEqual(connection.close_calls, 1)

    def test_connection_failure_is_unavailable_and_closed(self) -> None:
        connection = _FakePostgresConnection(
            fail_statement="pg_try_advisory_lock",
        )
        manager = PostgresAdvisoryThreadExecutionLeaseManager(
            "postgresql://lease.example.test/agent",
            connection_factory=lambda *args, **kwargs: connection,
        )

        with self.assertRaises(ThreadLeaseUnavailable):
            manager.acquire("thread-1", operation="run")

        self.assertTrue(connection.closed)
        self.assertEqual(connection.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
