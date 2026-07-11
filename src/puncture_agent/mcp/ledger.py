"""Durable replay ledger for idempotent MCP tool executions.

The graph checkpoint and the external tool response cannot be committed in one
transaction.  A tool server therefore records a replayable terminal public response
before returning it to the graph.  If the graph process dies before its next
checkpoint, a new tool-server/runtime instance can replay the recorded response
without invoking the tool handler again.

Only MCP-safe structured content is stored.  Storage URIs, checksums and raw
artifact bytes never enter this database.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import stat
from threading import RLock
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable
from uuid import uuid4


MAX_LEDGER_RESPONSE_BYTES = 1024 * 1024


class ToolReplayLedgerError(RuntimeError):
    """Base error for unavailable or invalid durable replay state."""


class ToolReplayConflict(ToolReplayLedgerError):
    """An idempotency scope was reused with a different semantic request."""


class ToolReplayBusy(ToolReplayLedgerError):
    """Another live worker currently owns the same tool execution claim."""


class ToolReplayUncertain(ToolReplayLedgerError):
    """A write-like execution may have advanced without a durable response."""


@dataclass(frozen=True, slots=True)
class ToolReplayDecision:
    """Result of atomically inspecting/claiming one replay scope."""

    scope_key: str
    request_fingerprint: str
    owner_token: str | None
    response: Mapping[str, Any] | None

    @property
    def is_replay(self) -> bool:
        return self.response is not None


@runtime_checkable
class ToolReplayLedger(Protocol):
    """Persistence contract used by :class:`McpToolRuntime`."""

    @property
    def claim_ttl_seconds(self) -> float: ...

    def begin(
        self,
        scope_key: str,
        request_fingerprint: str,
        *,
        reclaim_expired: bool,
    ) -> ToolReplayDecision: ...

    def complete(
        self,
        decision: ToolReplayDecision,
        response: Mapping[str, Any],
    ) -> None: ...

    def abandon(self, decision: ToolReplayDecision) -> None: ...

    def mark_uncertain(self, decision: ToolReplayDecision) -> None: ...


class SQLiteToolReplayLedger:
    """SQLite-backed replay ledger safe across processes and restarts.

    A short lease prevents simultaneous workers from running one idempotency
    scope.  If a process dies before completing the record, another worker may
    reclaim it after the lease.  Replayable terminal responses (success,
    partial, and non-retryable failure) are retained until an explicit
    operational retention policy removes the database.
    """

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        lease_seconds: float = 300.0,
        busy_timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._database_path = self._prepare_database_path(database_path)
        self._lease_ms = int(lease_seconds * 1000)
        self._clock = clock
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self._database_path,
            timeout=busy_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}"
        )
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize_schema()
        self._secure_database_files()

    @property
    def claim_ttl_seconds(self) -> float:
        return self._lease_ms / 1000.0

    def __enter__(self) -> "SQLiteToolReplayLedger":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._connection.in_transaction:
                self._connection.rollback()
            self._connection.close()
            self._closed = True

    def begin(
        self,
        scope_key: str,
        request_fingerprint: str,
        *,
        reclaim_expired: bool,
    ) -> ToolReplayDecision:
        self._validate_identity(scope_key, "scope_key")
        self._validate_identity(request_fingerprint, "request_fingerprint")
        now_ms = self._now_ms()
        owner_token = uuid4().hex
        deferred_uncertain: str | None = None
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM tool_replay_ledger WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO tool_replay_ledger (
                        scope_key, request_fingerprint, status, owner_token,
                        lease_expires_ms, response_json, created_at_ms, updated_at_ms
                    ) VALUES (?, ?, 'PENDING', ?, ?, NULL, ?, ?)
                    """,
                    (
                        scope_key,
                        request_fingerprint,
                        owner_token,
                        now_ms + self._lease_ms,
                        now_ms,
                        now_ms,
                    ),
                )
                return ToolReplayDecision(
                    scope_key,
                    request_fingerprint,
                    owner_token,
                    None,
                )

            if row["request_fingerprint"] != request_fingerprint:
                raise ToolReplayConflict(
                    "idempotency scope is bound to a different request fingerprint"
                )
            if row["status"] == "COMPLETED":
                response = self._decode_response(
                    row["response_json"],
                    row["response_sha256"],
                )
                return ToolReplayDecision(
                    scope_key,
                    request_fingerprint,
                    None,
                    response,
                )
            if row["status"] == "UNCERTAIN":
                raise ToolReplayUncertain(
                    "tool execution requires manual replay reconciliation"
                )
            if row["status"] != "PENDING":
                raise ToolReplayLedgerError("replay ledger contains an invalid status")
            if int(row["lease_expires_ms"]) > now_ms:
                raise ToolReplayBusy("tool execution is already in progress")
            if not reclaim_expired:
                updated = connection.execute(
                    """
                    UPDATE tool_replay_ledger
                    SET status = 'UNCERTAIN', owner_token = NULL,
                        lease_expires_ms = NULL, updated_at_ms = ?
                    WHERE scope_key = ? AND status = 'PENDING'
                      AND request_fingerprint = ? AND lease_expires_ms <= ?
                    """,
                    (now_ms, scope_key, request_fingerprint, now_ms),
                )
                if updated.rowcount != 1:
                    raise ToolReplayBusy(
                        "tool execution claim changed concurrently"
                    )
                deferred_uncertain = (
                    "expired write execution claim requires manual reconciliation"
                )
            else:
                updated = connection.execute(
                    """
                    UPDATE tool_replay_ledger
                    SET owner_token = ?, lease_expires_ms = ?, updated_at_ms = ?
                    WHERE scope_key = ? AND status = 'PENDING'
                      AND request_fingerprint = ? AND lease_expires_ms <= ?
                    """,
                    (
                        owner_token,
                        now_ms + self._lease_ms,
                        now_ms,
                        scope_key,
                        request_fingerprint,
                        now_ms,
                    ),
                )
                if updated.rowcount != 1:
                    raise ToolReplayBusy("tool execution claim changed concurrently")
                return ToolReplayDecision(
                    scope_key,
                    request_fingerprint,
                    owner_token,
                    None,
                )
        if deferred_uncertain is not None:
            raise ToolReplayUncertain(deferred_uncertain)
        raise ToolReplayLedgerError("tool replay begin produced no decision")

    def complete(
        self,
        decision: ToolReplayDecision,
        response: Mapping[str, Any],
    ) -> None:
        self._require_owned_decision(decision)
        response_json = self._encode_response(response)
        response_sha256 = self._response_sha256(response_json)
        now_ms = self._now_ms()
        with self._transaction(write=True) as connection:
            updated = connection.execute(
                """
                UPDATE tool_replay_ledger
                SET status = 'COMPLETED', owner_token = NULL,
                    lease_expires_ms = NULL, response_json = ?,
                    response_sha256 = ?, updated_at_ms = ?
                WHERE scope_key = ? AND request_fingerprint = ?
                  AND status = 'PENDING' AND owner_token = ?
                """,
                (
                    response_json,
                    response_sha256,
                    now_ms,
                    decision.scope_key,
                    decision.request_fingerprint,
                    decision.owner_token,
                ),
            )
            if updated.rowcount != 1:
                raise ToolReplayLedgerError(
                    "tool execution claim was lost before completion"
                )

    def abandon(self, decision: ToolReplayDecision) -> None:
        if decision.is_replay or decision.owner_token is None:
            return
        with self._transaction(write=True) as connection:
            connection.execute(
                """
                DELETE FROM tool_replay_ledger
                WHERE scope_key = ? AND request_fingerprint = ?
                  AND status = 'PENDING' AND owner_token = ?
                """,
                (
                    decision.scope_key,
                    decision.request_fingerprint,
                    decision.owner_token,
                ),
            )

    def mark_uncertain(self, decision: ToolReplayDecision) -> None:
        if decision.is_replay or decision.owner_token is None:
            return
        now_ms = self._now_ms()
        with self._transaction(write=True) as connection:
            connection.execute(
                """
                UPDATE tool_replay_ledger
                SET status = 'UNCERTAIN', owner_token = NULL,
                    lease_expires_ms = NULL, updated_at_ms = ?
                WHERE scope_key = ? AND request_fingerprint = ?
                  AND status = 'PENDING' AND owner_token = ?
                """,
                (
                    now_ms,
                    decision.scope_key,
                    decision.request_fingerprint,
                    decision.owner_token,
                ),
            )

    def _initialize_schema(self) -> None:
        with self._transaction(write=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_replay_ledger (
                    scope_key TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('PENDING', 'COMPLETED', 'UNCERTAIN')
                    ),
                    owner_token TEXT,
                    lease_expires_ms INTEGER,
                    response_json TEXT,
                    response_sha256 TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    CHECK (
                        (status = 'PENDING' AND owner_token IS NOT NULL
                            AND lease_expires_ms IS NOT NULL
                            AND response_json IS NULL AND response_sha256 IS NULL)
                        OR
                        (status = 'COMPLETED' AND owner_token IS NULL
                            AND lease_expires_ms IS NULL AND response_json IS NOT NULL
                            AND response_sha256 IS NOT NULL)
                        OR
                        (status = 'UNCERTAIN' AND owner_token IS NULL
                            AND lease_expires_ms IS NULL
                            AND response_json IS NULL AND response_sha256 IS NULL)
                    )
                )
                """
            )

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield self._connection
                self._secure_database_files()
                self._connection.commit()
            except ToolReplayLedgerError:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
            except sqlite3.OperationalError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise ToolReplayBusy("tool replay ledger is busy") from exc
                raise ToolReplayLedgerError("tool replay ledger operation failed") from exc
            except sqlite3.DatabaseError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise ToolReplayLedgerError("tool replay ledger operation failed") from exc
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise ToolReplayLedgerError("tool replay ledger is closed")

    def _now_ms(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolReplayLedgerError("tool replay ledger clock is invalid")
        return int(float(value) * 1000)

    @staticmethod
    def _validate_identity(value: str, field: str) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError(f"{field} must be a non-empty bounded string")
        if any(character in value for character in ("\r", "\n", "\x00")):
            raise ValueError(f"{field} contains control characters")

    @staticmethod
    def _require_owned_decision(decision: ToolReplayDecision) -> None:
        if not isinstance(decision, ToolReplayDecision):
            raise TypeError("decision must be ToolReplayDecision")
        if decision.is_replay or decision.owner_token is None:
            raise ToolReplayLedgerError("replay decisions cannot be completed")

    @staticmethod
    def _encode_response(response: Mapping[str, Any]) -> str:
        if not isinstance(response, Mapping):
            raise ToolReplayLedgerError("tool replay response must be an object")
        try:
            encoded = json.dumps(
                response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ToolReplayLedgerError(
                "tool replay response must be JSON-compatible"
            ) from exc
        if len(encoded.encode("utf-8")) > MAX_LEDGER_RESPONSE_BYTES:
            raise ToolReplayLedgerError("tool replay response exceeds 1 MiB")
        return encoded

    @staticmethod
    def _decode_response(value: Any, expected_sha256: Any) -> dict[str, Any]:
        if not isinstance(value, str) or not isinstance(expected_sha256, str):
            raise ToolReplayLedgerError("completed replay record has no response")
        if SQLiteToolReplayLedger._response_sha256(value) != expected_sha256:
            raise ToolReplayLedgerError("stored replay response integrity check failed")
        try:
            response = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ToolReplayLedgerError("stored replay response is invalid") from exc
        if not isinstance(response, dict):
            raise ToolReplayLedgerError("stored replay response must be an object")
        return deepcopy(response)

    @staticmethod
    def _response_sha256(response_json: str) -> str:
        from hashlib import sha256

        return sha256(response_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _prepare_database_path(database_path: str | os.PathLike[str]) -> str:
        raw_path = os.fspath(database_path)
        if raw_path == ":memory:":
            return raw_path
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if path.is_symlink():
            raise ToolReplayLedgerError("tool replay database must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ToolReplayLedgerError(
                    "tool replay database must be a regular file"
                )
            os.fchmod(descriptor, 0o600)
        except ToolReplayLedgerError:
            raise
        except OSError as exc:
            raise ToolReplayLedgerError("tool replay database path is unsafe") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return os.fspath(path)

    def _secure_database_files(self) -> None:
        if self._database_path == ":memory:":
            return
        for suffix in ("", "-wal", "-shm"):
            path = Path(self._database_path + suffix)
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ToolReplayLedgerError(
                        "tool replay database sidecar is not a safe regular file"
                    )
                path.chmod(0o600)
            except FileNotFoundError:
                continue
            except ToolReplayLedgerError:
                raise
            except OSError as exc:
                raise ToolReplayLedgerError(
                    "tool replay database permissions could not be secured"
                ) from exc


__all__ = [
    "MAX_LEDGER_RESPONSE_BYTES",
    "SQLiteToolReplayLedger",
    "ToolReplayBusy",
    "ToolReplayConflict",
    "ToolReplayDecision",
    "ToolReplayLedger",
    "ToolReplayLedgerError",
    "ToolReplayUncertain",
]
