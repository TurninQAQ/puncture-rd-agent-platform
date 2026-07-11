"""Cross-runtime single-flight leases for LangGraph thread execution.

The production implementation uses a PostgreSQL *session* advisory lock held
on one dedicated connection for the complete graph invocation.  The SQLite
implementation is a deterministic TTL/CAS test double: its transactions are
short and only protect lease-row changes, so different thread IDs are not
serialized for the duration of a graph run.

Neither backend fences arbitrary writes made by a stale worker.  Callers must
therefore validate ownership before and after side-effecting work and stop
immediately when :class:`ThreadLeaseLost` is raised.  A production deployment
that permits lease expiry or connection loss while arbitrary side effects can
continue additionally needs checkpoint/tool fencing or durable reconciliation.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import math
import os
from pathlib import Path
import sqlite3
import stat
from threading import RLock
import time
from types import TracebackType
from typing import Any, Callable, Iterator, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit
from uuid import uuid4


ThreadLeaseOperation = Literal["run", "resume", "stream"]

DEFAULT_LEASE_NAMESPACE = "puncture-rd-agent-platform/langgraph-thread/v1"
_VALID_OPERATIONS = frozenset(("run", "resume", "stream"))


class ThreadLeaseError(RuntimeError):
    """Base error for distributed thread-execution coordination."""


class ThreadLeaseBusy(ThreadLeaseError):
    """Another live owner already holds the requested thread lease."""


class ThreadLeaseUnavailable(ThreadLeaseError):
    """The lease backend cannot currently make a safe ownership decision."""


class ThreadLeaseLost(ThreadLeaseError):
    """A previously acquired lease is no longer owned by this execution."""


@runtime_checkable
class ThreadExecutionLease(Protocol):
    """One exclusive execution claim returned by a lease manager.

    ``assert_valid`` and ``renew`` succeed silently or raise a ``ThreadLease``
    exception.  ``release`` is idempotent after a successful release; it raises
    ``ThreadLeaseLost`` if first release discovers that ownership was lost.
    """

    @property
    def thread_id(self) -> str: ...

    @property
    def owner_token(self) -> str: ...

    @property
    def operation(self) -> ThreadLeaseOperation: ...

    @property
    def backend(self) -> str: ...

    def assert_valid(self) -> None: ...

    def renew(self) -> None: ...

    def release(self) -> None: ...

    def __enter__(self) -> "ThreadExecutionLease": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class ThreadExecutionLeaseManager(Protocol):
    """Factory for exclusive, context-managed thread execution claims."""

    def acquire(
        self,
        thread_id: str,
        *,
        operation: ThreadLeaseOperation,
    ) -> ThreadExecutionLease: ...


def thread_advisory_lock_key(
    thread_id: str,
    *,
    namespace: str = DEFAULT_LEASE_NAMESPACE,
) -> int:
    """Return a stable signed PostgreSQL ``bigint`` advisory-lock key.

    Python's process-randomized ``hash`` and PostgreSQL's 32-bit ``hashtext``
    are intentionally not used.  The namespace makes this lock family distinct
    from unrelated advisory-lock users in the same database.
    """

    normalized_thread = _validate_thread_id(thread_id)
    normalized_namespace = _validate_namespace(namespace)
    digest = sha256(
        normalized_namespace.encode("utf-8")
        + b"\x00"
        + normalized_thread.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


class SQLiteThreadExecutionLeaseManager:
    """SQLite TTL/CAS implementation intended for deterministic tests.

    Each manager owns one connection, while separate manager instances can
    coordinate through the same database file.  Lease acquisition, renewal and
    release use short ``BEGIN IMMEDIATE`` transactions.  No transaction remains
    open while the caller executes the graph.

    ``renew`` never resurrects an expired claim.  Reclaiming an expired row
    increments its generation, so the former owner can neither renew nor delete
    the replacement claim.  The injected clock permits deterministic expiry and
    lease-loss tests.
    """

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        lease_seconds: float = 300.0,
        busy_timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
        namespace: str = DEFAULT_LEASE_NAMESPACE,
    ) -> None:
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or not math.isfinite(float(lease_seconds))
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive finite number")
        if (
            isinstance(busy_timeout_seconds, bool)
            or not isinstance(busy_timeout_seconds, (int, float))
            or not math.isfinite(float(busy_timeout_seconds))
            or busy_timeout_seconds <= 0
        ):
            raise ValueError(
                "busy_timeout_seconds must be a positive finite number"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._namespace = _validate_namespace(namespace)
        self._database_path = _prepare_sqlite_path(database_path)
        self._lease_ms = max(1, int(float(lease_seconds) * 1000))
        self._busy_timeout_ms = max(1, int(float(busy_timeout_seconds) * 1000))
        self._clock = clock
        self._lock = RLock()
        self._closed = False
        try:
            self._connection = sqlite3.connect(
                self._database_path,
                timeout=float(busy_timeout_seconds),
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute(
                f"PRAGMA busy_timeout = {self._busy_timeout_ms}"
            )
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_schema()
            self._secure_database_files()
        except ThreadLeaseError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                _close_connection_quietly(connection)
            raise
        except sqlite3.Error as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                _close_connection_quietly(connection)
            raise ThreadLeaseUnavailable(
                "SQLite thread lease backend could not be initialized"
            ) from exc

    @property
    def lease_seconds(self) -> float:
        return self._lease_ms / 1000.0

    def __enter__(self) -> "SQLiteThreadExecutionLeaseManager":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close this manager connection without deleting active lease rows."""

        with self._lock:
            if self._closed:
                return
            if self._connection.in_transaction:
                self._connection.rollback()
            self._connection.close()
            self._closed = True

    def acquire(
        self,
        thread_id: str,
        *,
        operation: ThreadLeaseOperation,
    ) -> ThreadExecutionLease:
        normalized_thread = _validate_thread_id(thread_id)
        normalized_operation = _validate_operation(operation)
        scope_key = self._scope_key(normalized_thread)
        owner_token = uuid4().hex
        generation: int
        try:
            with self._transaction(write=True, busy_is_contention=True) as connection:
                now_ms = self._now_ms()
                row = connection.execute(
                    """
                    SELECT owner_token, generation, expires_at_ms
                    FROM thread_execution_leases
                    WHERE scope_key = ?
                    """,
                    (scope_key,),
                ).fetchone()
                if row is None:
                    generation = 1
                    connection.execute(
                        """
                        INSERT INTO thread_execution_leases (
                            scope_key, owner_token, operation, generation,
                            expires_at_ms, updated_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scope_key,
                            owner_token,
                            normalized_operation,
                            generation,
                            now_ms + self._lease_ms,
                            now_ms,
                        ),
                    )
                else:
                    if int(row["expires_at_ms"]) > now_ms:
                        raise ThreadLeaseBusy(
                            "the LangGraph thread is already executing"
                        )
                    generation = int(row["generation"]) + 1
                    updated = connection.execute(
                        """
                        UPDATE thread_execution_leases
                        SET owner_token = ?, operation = ?, generation = ?,
                            expires_at_ms = ?, updated_at_ms = ?
                        WHERE scope_key = ? AND owner_token = ?
                          AND generation = ? AND expires_at_ms <= ?
                        """,
                        (
                            owner_token,
                            normalized_operation,
                            generation,
                            now_ms + self._lease_ms,
                            now_ms,
                            scope_key,
                            row["owner_token"],
                            row["generation"],
                            now_ms,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise ThreadLeaseBusy(
                            "the LangGraph thread lease changed concurrently"
                        )
        except ThreadLeaseError:
            raise
        except sqlite3.Error as exc:
            raise ThreadLeaseUnavailable(
                "SQLite thread lease acquisition failed"
            ) from exc
        return _SQLiteThreadExecutionLease(
            manager=self,
            thread_id=normalized_thread,
            owner_token=owner_token,
            operation=normalized_operation,
            scope_key=scope_key,
            generation=generation,
        )

    def _assert_owned(
        self,
        *,
        scope_key: str,
        owner_token: str,
        generation: int,
    ) -> None:
        try:
            with self._transaction(write=False) as connection:
                now_ms = self._now_ms()
                row = connection.execute(
                    """
                    SELECT owner_token, generation, expires_at_ms
                    FROM thread_execution_leases
                    WHERE scope_key = ?
                    """,
                    (scope_key,),
                ).fetchone()
        except ThreadLeaseError:
            raise
        except sqlite3.Error as exc:
            raise ThreadLeaseUnavailable(
                "SQLite thread lease ownership could not be checked"
            ) from exc
        if (
            row is None
            or row["owner_token"] != owner_token
            or int(row["generation"]) != generation
            or int(row["expires_at_ms"]) <= now_ms
        ):
            raise ThreadLeaseLost("the SQLite thread execution lease was lost")

    def _renew(
        self,
        *,
        scope_key: str,
        owner_token: str,
        generation: int,
    ) -> None:
        try:
            with self._transaction(write=True) as connection:
                now_ms = self._now_ms()
                updated = connection.execute(
                    """
                    UPDATE thread_execution_leases
                    SET expires_at_ms = ?, updated_at_ms = ?
                    WHERE scope_key = ? AND owner_token = ?
                      AND generation = ? AND expires_at_ms > ?
                    """,
                    (
                        now_ms + self._lease_ms,
                        now_ms,
                        scope_key,
                        owner_token,
                        generation,
                        now_ms,
                    ),
                )
        except ThreadLeaseError:
            raise
        except sqlite3.Error as exc:
            raise ThreadLeaseUnavailable(
                "SQLite thread lease renewal failed"
            ) from exc
        if updated.rowcount != 1:
            raise ThreadLeaseLost("the SQLite thread execution lease was lost")

    def _release(
        self,
        *,
        scope_key: str,
        owner_token: str,
        generation: int,
    ) -> None:
        try:
            with self._transaction(write=True) as connection:
                now_ms = self._now_ms()
                deleted = connection.execute(
                    """
                    DELETE FROM thread_execution_leases
                    WHERE scope_key = ? AND owner_token = ?
                      AND generation = ? AND expires_at_ms > ?
                    """,
                    (scope_key, owner_token, generation, now_ms),
                )
        except ThreadLeaseError:
            raise
        except sqlite3.Error as exc:
            raise ThreadLeaseUnavailable(
                "SQLite thread lease release failed"
            ) from exc
        if deleted.rowcount != 1:
            raise ThreadLeaseLost("the SQLite thread execution lease was lost")

    def _scope_key(self, thread_id: str) -> str:
        return sha256(
            self._namespace.encode("utf-8")
            + b"\x00"
            + thread_id.encode("utf-8")
        ).hexdigest()

    def _initialize_schema(self) -> None:
        with self._transaction(write=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_execution_leases (
                    scope_key TEXT PRIMARY KEY,
                    owner_token TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK (
                        operation IN ('run', 'resume', 'stream')
                    ),
                    generation INTEGER NOT NULL CHECK (generation > 0),
                    expires_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )

    @contextmanager
    def _transaction(
        self,
        *,
        write: bool,
        busy_is_contention: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield self._connection
                self._secure_database_files()
                self._connection.commit()
            except ThreadLeaseError:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
            except sqlite3.OperationalError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                if busy_is_contention and _is_sqlite_busy(exc):
                    raise ThreadLeaseBusy(
                        "the SQLite thread lease database is busy"
                    ) from exc
                raise ThreadLeaseUnavailable(
                    "SQLite thread lease operation failed"
                ) from exc
            except sqlite3.DatabaseError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise ThreadLeaseUnavailable(
                    "SQLite thread lease operation failed"
                ) from exc
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise ThreadLeaseUnavailable("SQLite thread lease manager is closed")

    def _now_ms(self) -> int:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ThreadLeaseUnavailable("SQLite thread lease clock is invalid")
        return int(float(value) * 1000)

    def _secure_database_files(self) -> None:
        if self._database_path == ":memory:":
            return
        for suffix in ("", "-wal", "-shm"):
            path = Path(self._database_path + suffix)
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ThreadLeaseUnavailable(
                    "SQLite thread lease database path is unavailable"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ThreadLeaseUnavailable(
                    "SQLite thread lease database sidecar is unsafe"
                )
            try:
                path.chmod(0o600)
            except OSError as exc:
                raise ThreadLeaseUnavailable(
                    "SQLite thread lease database permissions could not be secured"
                ) from exc


class _SQLiteThreadExecutionLease:
    def __init__(
        self,
        *,
        manager: SQLiteThreadExecutionLeaseManager,
        thread_id: str,
        owner_token: str,
        operation: ThreadLeaseOperation,
        scope_key: str,
        generation: int,
    ) -> None:
        self._manager = manager
        self._thread_id = thread_id
        self._owner_token = owner_token
        self._operation = operation
        self._scope_key = scope_key
        self._generation = generation
        self._released = False
        self._lock = RLock()

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def owner_token(self) -> str:
        return self._owner_token

    @property
    def operation(self) -> ThreadLeaseOperation:
        return self._operation

    @property
    def backend(self) -> str:
        return "sqlite-ttl"

    @property
    def generation(self) -> int:
        return self._generation

    def assert_valid(self) -> None:
        with self._lock:
            self._require_active()
            self._manager._assert_owned(
                scope_key=self._scope_key,
                owner_token=self._owner_token,
                generation=self._generation,
            )

    def renew(self) -> None:
        with self._lock:
            self._require_active()
            self._manager._renew(
                scope_key=self._scope_key,
                owner_token=self._owner_token,
                generation=self._generation,
            )

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            try:
                self._manager._release(
                    scope_key=self._scope_key,
                    owner_token=self._owner_token,
                    generation=self._generation,
                )
            finally:
                self._released = True

    def __enter__(self) -> "_SQLiteThreadExecutionLease":
        self.assert_valid()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _require_active(self) -> None:
        if self._released:
            raise ThreadLeaseLost("the SQLite thread execution lease was released")


class PostgresAdvisoryThreadExecutionLeaseManager:
    """Acquire PostgreSQL session advisory locks on exclusive connections.

    The connection is never shared with LangGraph's ``PostgresSaver``.  Session
    scope is required because one graph run spans multiple checkpoint
    transactions; transaction-level advisory locks would be released too early.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        connect_timeout_seconds: float = 5.0,
        namespace: str = DEFAULT_LEASE_NAMESPACE,
        application_name: str = "puncture-langgraph-thread-lease",
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._connection_string = _validate_postgres_dsn(connection_string)
        if (
            isinstance(connect_timeout_seconds, bool)
            or not isinstance(connect_timeout_seconds, (int, float))
            or not math.isfinite(float(connect_timeout_seconds))
            or connect_timeout_seconds <= 0
        ):
            raise ValueError(
                "connect_timeout_seconds must be a positive finite number"
            )
        self._connect_timeout_seconds = max(
            1, math.ceil(float(connect_timeout_seconds))
        )
        self._namespace = _validate_namespace(namespace)
        self._application_name = _validate_application_name(application_name)
        if connection_factory is not None and not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory

    def acquire(
        self,
        thread_id: str,
        *,
        operation: ThreadLeaseOperation,
    ) -> ThreadExecutionLease:
        normalized_thread = _validate_thread_id(thread_id)
        normalized_operation = _validate_operation(operation)
        lock_key = thread_advisory_lock_key(
            normalized_thread,
            namespace=self._namespace,
        )
        connection: Any | None = None
        try:
            connection_factory = self._connection_factory
            if connection_factory is None:
                try:
                    import psycopg
                except (ImportError, ModuleNotFoundError) as exc:
                    raise ThreadLeaseUnavailable(
                        "PostgreSQL thread leases require psycopg"
                    ) from exc
                connection_factory = psycopg.connect
            connection = connection_factory(
                self._connection_string,
                autocommit=True,
                connect_timeout=self._connect_timeout_seconds,
                application_name=self._application_name,
            )
            if getattr(connection, "closed", False):
                raise ThreadLeaseUnavailable(
                    "PostgreSQL thread lease connection is closed"
                )
            # Do not rely only on a factory respecting the autocommit argument.
            # A fresh dedicated connection can safely be switched before SQL.
            if getattr(connection, "autocommit", None) is not True:
                connection.autocommit = True
            backend_pid = _postgres_scalar(connection, "SELECT pg_backend_pid()")
            if isinstance(backend_pid, bool) or not isinstance(backend_pid, int):
                raise ThreadLeaseUnavailable(
                    "PostgreSQL returned an invalid backend identity"
                )
            acquired = _postgres_scalar(
                connection,
                "SELECT pg_try_advisory_lock(%s::bigint)",
                (lock_key,),
            )
            if acquired is not True:
                _close_connection_quietly(connection)
                connection = None
                raise ThreadLeaseBusy(
                    "the LangGraph thread is already executing"
                )
            return _PostgresAdvisoryThreadExecutionLease(
                connection=connection,
                thread_id=normalized_thread,
                owner_token=uuid4().hex,
                operation=normalized_operation,
                lock_key=lock_key,
                backend_pid=backend_pid,
            )
        except ThreadLeaseError:
            if connection is not None:
                _close_connection_quietly(connection)
            raise
        except Exception as exc:
            if connection is not None:
                _close_connection_quietly(connection)
            raise ThreadLeaseUnavailable(
                "PostgreSQL thread lease acquisition failed"
            ) from exc


class _PostgresAdvisoryThreadExecutionLease:
    def __init__(
        self,
        *,
        connection: Any,
        thread_id: str,
        owner_token: str,
        operation: ThreadLeaseOperation,
        lock_key: int,
        backend_pid: int,
    ) -> None:
        self._connection = connection
        self._thread_id = thread_id
        self._owner_token = owner_token
        self._operation = operation
        self._lock_key = lock_key
        self._backend_pid = backend_pid
        self._released = False
        self._lock = RLock()

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def owner_token(self) -> str:
        return self._owner_token

    @property
    def operation(self) -> ThreadLeaseOperation:
        return self._operation

    @property
    def backend(self) -> str:
        return "postgres-advisory"

    @property
    def advisory_lock_key(self) -> int:
        return self._lock_key

    @property
    def backend_pid(self) -> int:
        return self._backend_pid

    def assert_valid(self) -> None:
        with self._lock:
            self._require_active()
            if getattr(self._connection, "closed", False):
                raise ThreadLeaseLost(
                    "the PostgreSQL advisory thread lease connection was lost"
                )
            try:
                backend_pid = _postgres_scalar(
                    self._connection,
                    "SELECT pg_backend_pid()",
                )
            except Exception as exc:
                raise ThreadLeaseLost(
                    "the PostgreSQL advisory thread lease cannot be verified"
                ) from exc
            if backend_pid != self._backend_pid:
                raise ThreadLeaseLost(
                    "the PostgreSQL advisory thread lease session changed"
                )

    def renew(self) -> None:
        # Session advisory locks have no TTL.  Verifying the unchanged backend
        # session is the corresponding liveness operation.
        self.assert_valid()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            release_error: ThreadLeaseError | None = None
            try:
                if getattr(self._connection, "closed", False):
                    release_error = ThreadLeaseLost(
                        "the PostgreSQL advisory thread lease connection was lost"
                    )
                else:
                    try:
                        unlocked = _postgres_scalar(
                            self._connection,
                            "SELECT pg_advisory_unlock(%s::bigint)",
                            (self._lock_key,),
                        )
                    except Exception as exc:
                        release_error = ThreadLeaseLost(
                            "the PostgreSQL advisory thread lease could not be released"
                        )
                        release_error.__cause__ = exc
                    else:
                        if unlocked is not True:
                            release_error = ThreadLeaseLost(
                                "the PostgreSQL advisory thread lease was not owned"
                            )
            finally:
                try:
                    self._connection.close()
                except Exception as exc:
                    if release_error is None:
                        release_error = ThreadLeaseUnavailable(
                            "the PostgreSQL advisory lease connection did not close"
                        )
                        release_error.__cause__ = exc
                self._released = True
            if release_error is not None:
                raise release_error

    def __enter__(self) -> "_PostgresAdvisoryThreadExecutionLease":
        self.assert_valid()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _require_active(self) -> None:
        if self._released:
            raise ThreadLeaseLost(
                "the PostgreSQL advisory thread lease was released"
            )


def _validate_thread_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("thread_id must be a non-empty string")
    if len(value) >= 255:
        raise ValueError("thread_id must contain fewer than 255 characters")
    if any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError("thread_id contains control characters")
    return value


def _validate_operation(value: str) -> ThreadLeaseOperation:
    if not isinstance(value, str) or value not in _VALID_OPERATIONS:
        raise ValueError("operation must be one of: run, resume, stream")
    return value  # type: ignore[return-value]


def _validate_namespace(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("namespace must be a non-empty bounded string")
    if any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError("namespace contains control characters")
    return value


def _validate_application_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 63
    ):
        raise ValueError("application_name must be a non-empty bounded string")
    if any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError("application_name contains control characters")
    return value


def _validate_postgres_dsn(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PostgreSQL connection string must be non-empty")
    if any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError("PostgreSQL connection string contains control characters")
    parsed = urlsplit(value)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError(
            "PostgreSQL connection string must use postgres:// or postgresql://"
        )
    return value


def _prepare_sqlite_path(database_path: str | os.PathLike[str]) -> str:
    try:
        raw_path = os.fspath(database_path)
    except TypeError as exc:
        raise ValueError("database_path must be a filesystem path") from exc
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("database_path must be a non-empty filesystem path")
    if raw_path == ":memory:":
        return raw_path
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise ThreadLeaseUnavailable(
            "SQLite thread lease database directory is unavailable"
        ) from exc
    if path.is_symlink():
        raise ThreadLeaseUnavailable(
            "SQLite thread lease database must not be a symlink"
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ThreadLeaseUnavailable(
                "SQLite thread lease database must be a regular file"
            )
        os.fchmod(descriptor, 0o600)
    except ThreadLeaseError:
        raise
    except OSError as exc:
        raise ThreadLeaseUnavailable(
            "SQLite thread lease database path is unsafe"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return os.fspath(path)


def _is_sqlite_busy(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _postgres_scalar(
    connection: Any,
    statement: str,
    parameters: tuple[Any, ...] = (),
) -> Any:
    cursor = connection.execute(statement, parameters)
    try:
        row = cursor.fetchone()
    finally:
        close_cursor = getattr(cursor, "close", None)
        if callable(close_cursor):
            close_cursor()
    if row is None:
        raise ThreadLeaseUnavailable("PostgreSQL lease query returned no row")
    try:
        return row[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ThreadLeaseUnavailable(
            "PostgreSQL lease query returned an invalid row"
        ) from exc


def _close_connection_quietly(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


__all__ = [
    "DEFAULT_LEASE_NAMESPACE",
    "PostgresAdvisoryThreadExecutionLeaseManager",
    "SQLiteThreadExecutionLeaseManager",
    "ThreadExecutionLease",
    "ThreadExecutionLeaseManager",
    "ThreadLeaseBusy",
    "ThreadLeaseError",
    "ThreadLeaseLost",
    "ThreadLeaseOperation",
    "ThreadLeaseUnavailable",
    "thread_advisory_lock_key",
]
