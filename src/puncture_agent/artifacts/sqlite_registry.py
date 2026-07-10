"""Durable SQLite implementation of the artifact registry contract.

The implementation intentionally mirrors :class:`InMemoryArtifactRegistry`:
large objects remain in object storage, SQLite stores only metadata, and an
``ArtifactPublicView`` is the only unauthorised projection.  Storage URIs are
returned only by the existing internal ``ArtifactRef`` methods or after an
explicit :class:`Principal` authorisation check in :meth:`resolve_uri`.

Every operation owns an explicit transaction.  Write operations use
``BEGIN IMMEDIATE`` so competing finalisations are serialised before the
partial unique index on active idempotency scopes is evaluated.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import sqlite3
import stat
from pathlib import Path
from threading import RLock
from typing import Iterator, Mapping
from uuid import uuid4

from contracts.artifacts import ArtifactPublicView, ArtifactRef
from contracts.enums import ArtifactStatus, ArtifactType, CoordinateSystem
from contracts.geometry import VolumeGeometry

from .identity import canonical_json
from .registry import ArtifactLineage, ArtifactRegistryError, Principal


@dataclass(frozen=True, slots=True)
class _StoredArtifact:
    artifact_id: str
    case_id: str
    artifact_type: ArtifactType
    status: ArtifactStatus
    internal_uri: str
    created_by: str
    idempotency_key: str
    producer_name: str
    producer_version: str
    geometry: VolumeGeometry | None
    metadata: dict[str, str]
    checksum_sha256: str | None
    size_bytes: int | None
    failure_reason: str | None
    parent_artifact_ids: tuple[str, ...]
    registration_fingerprint: str

    def to_public_view(self) -> ArtifactPublicView:
        return ArtifactPublicView(
            artifact_id=self.artifact_id,
            case_id=self.case_id,
            artifact_type=self.artifact_type,
            status=self.status,
            producer_name=self.producer_name,
            producer_version=self.producer_version,
            geometry_fingerprint=(
                self.geometry.geometry_fingerprint if self.geometry is not None else None
            ),
        )


class SQLiteArtifactRegistry:
    """Persistent registry with the same public behaviour as the memory double.

    Args:
        database_path: SQLite file path. ``":memory:"`` is also supported for
            tests, although a file path is required for restart durability.
        busy_timeout_seconds: Maximum time SQLite waits for another writer.

    One connection is retained per registry instance.  Calls on an instance
    are protected by an ``RLock``; separate instances coordinate through
    SQLite's write lock and database constraints.
    """

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        self._database_path = self._prepare_database_path(database_path)
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self._database_path,
            timeout=busy_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
        # WAL permits readers while a writer finalises an artifact. SQLite
        # harmlessly falls back to its in-memory journal for ':memory:'.
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()
        self._secure_database_files()

    def close(self) -> None:
        """Close the underlying connection; committed data remains durable."""

        with self._lock:
            if self._closed:
                return
            if self._connection.in_transaction:
                self._connection.rollback()
            self._connection.close()
            self._closed = True

    @property
    def coordination_key(self) -> str:
        if self._database_path == ":memory:":
            return f"sqlite-memory-registry:{id(self)}"
        return f"sqlite-registry:{Path(self._database_path).resolve()}"

    def __enter__(self) -> "SQLiteArtifactRegistry":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def begin_registration(
        self,
        *,
        case_id: str,
        artifact_type: ArtifactType,
        internal_uri: str,
        created_by: str,
        idempotency_key: str,
        producer_name: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...] = (),
        geometry: VolumeGeometry | None = None,
        metadata: Mapping[str, str] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactPublicView:
        required = (case_id, internal_uri, created_by, idempotency_key, producer_name, producer_version)
        if any(not value.strip() for value in required):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "required registration field is empty")
        if len(parent_artifact_ids) != len(set(parent_artifact_ids)):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "parent artifact IDs must be unique")
        if not isinstance(artifact_type, ArtifactType):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "artifact_type must be canonical ArtifactType")

        geometry_json = self._encode_geometry(geometry)
        metadata_value = dict(metadata or {})
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata_value.items()
        ):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "metadata keys and values must be strings")
        try:
            metadata_json = json.dumps(metadata_value, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ArtifactRegistryError("INVALID_ARGUMENT", "metadata must be JSON serializable") from exc
        registration_fingerprint = self._compute_registration_fingerprint(
            case_id=case_id,
            artifact_type=artifact_type,
            producer_name=producer_name,
            producer_version=producer_version,
            parent_artifact_ids=parent_artifact_ids,
            geometry=geometry,
            metadata=metadata_value,
        )

        with self._transaction(write=True) as connection:
            reusable = self._find_active_record(connection, case_id, idempotency_key)
            if reusable is not None:
                if reusable.registration_fingerprint != registration_fingerprint:
                    raise ArtifactRegistryError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency key is already bound to a different registration request",
                    )
                return reusable.to_public_view()

            new_id = artifact_id or f"art-{uuid4().hex}"
            if self._select_artifact_row(connection, new_id) is not None:
                raise ArtifactRegistryError("CONFLICT", f"artifact {new_id} already exists")
            if new_id in parent_artifact_ids:
                raise ArtifactRegistryError("LINEAGE_CYCLE", "artifact cannot be its own parent")

            for parent_id in parent_artifact_ids:
                parent_row = self._select_artifact_row(connection, parent_id)
                if parent_row is None:
                    raise ArtifactRegistryError(
                        "PARENT_NOT_FOUND", f"parent {parent_id} does not exist"
                    )
                if ArtifactStatus(parent_row["status"]) is not ArtifactStatus.AVAILABLE:
                    raise ArtifactRegistryError(
                        "PARENT_NOT_AVAILABLE", f"parent {parent_id} is not AVAILABLE"
                    )
                if parent_row["case_id"] != case_id:
                    raise ArtifactRegistryError(
                        "CASE_MISMATCH", "parent belongs to a different case"
                    )

            try:
                connection.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id, case_id, artifact_type, status, internal_uri,
                        created_by, idempotency_key, producer_name, producer_version,
                        geometry_json, metadata_json, registration_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id,
                        case_id,
                        artifact_type.value,
                        ArtifactStatus.PENDING.value,
                        internal_uri,
                        created_by,
                        idempotency_key,
                        producer_name,
                        producer_version,
                        geometry_json,
                        metadata_json,
                        registration_fingerprint,
                    ),
                )
                connection.executemany(
                    "INSERT INTO artifact_lineage (child_artifact_id, parent_artifact_id) VALUES (?, ?)",
                    ((new_id, parent_id) for parent_id in parent_artifact_ids),
                )
            except sqlite3.IntegrityError as exc:
                raise ArtifactRegistryError("CONFLICT", "artifact registration conflicts with stored data") from exc

            record = self._require_record(connection, new_id)
            return record.to_public_view()

    def finalize(self, artifact_id: str, checksum_sha256: str, size_bytes: int) -> ArtifactRef:
        checksum = checksum_sha256.lower()
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ArtifactRegistryError("INVALID_CHECKSUM", "checksum must be a SHA-256 hex digest")
        if size_bytes < 0:
            raise ArtifactRegistryError("INVALID_ARGUMENT", "size_bytes must be non-negative")

        deferred_error: ArtifactRegistryError | None = None
        finalized: ArtifactRef | None = None
        with self._transaction(write=True) as connection:
            record = self._require_record(connection, artifact_id)
            if record.status is ArtifactStatus.AVAILABLE:
                if record.checksum_sha256 == checksum and record.size_bytes == size_bytes:
                    return self._to_ref(record)
                raise ArtifactRegistryError("CONFLICT", "AVAILABLE artifact cannot be overwritten")
            if record.status is not ArtifactStatus.PENDING:
                raise ArtifactRegistryError(
                    "INVALID_STATE", f"cannot finalize {record.status.value} artifact"
                )

            existing = self._find_available_record(
                connection, record.idempotency_key, case_id=record.case_id
            )
            if existing is not None and existing.artifact_id != artifact_id:
                if existing.checksum_sha256 != checksum:
                    connection.execute(
                        """
                        UPDATE artifacts
                        SET status = ?, failure_reason = ?
                        WHERE artifact_id = ? AND status = ?
                        """,
                        (
                            ArtifactStatus.INVALID.value,
                            "IDEMPOTENCY_CHECKSUM_CONFLICT",
                            artifact_id,
                            ArtifactStatus.PENDING.value,
                        ),
                    )
                    deferred_error = ArtifactRegistryError(
                        "IDEMPOTENCY_CONFLICT", "idempotency key maps to different content"
                    )
                else:
                    connection.execute(
                        """
                        UPDATE artifacts
                        SET status = ?, failure_reason = ?
                        WHERE artifact_id = ? AND status = ?
                        """,
                        (
                            ArtifactStatus.INVALID.value,
                            "DUPLICATE_OUTPUT",
                            artifact_id,
                            ArtifactStatus.PENDING.value,
                        ),
                    )
                    finalized = self._to_ref(existing)
            else:
                try:
                    updated = connection.execute(
                        """
                        UPDATE artifacts
                        SET status = ?, checksum_sha256 = ?, size_bytes = ?
                        WHERE artifact_id = ? AND status = ?
                        """,
                        (
                            ArtifactStatus.AVAILABLE.value,
                            checksum,
                            size_bytes,
                            artifact_id,
                            ArtifactStatus.PENDING.value,
                        ),
                    )
                except sqlite3.IntegrityError:
                    # If a legacy/non-cooperating writer violates the active
                    # scope, persist INVALID before surfacing the conflict.
                    # This prevents a failed finalization from orphaning a
                    # PENDING row that can never become canonical.
                    connection.execute(
                        """
                        UPDATE artifacts
                        SET status = ?, failure_reason = ?
                        WHERE artifact_id = ? AND status = ?
                        """,
                        (
                            ArtifactStatus.INVALID.value,
                            "IDEMPOTENCY_CONSTRAINT_CONFLICT",
                            artifact_id,
                            ArtifactStatus.PENDING.value,
                        ),
                    )
                    deferred_error = ArtifactRegistryError(
                        "IDEMPOTENCY_CONFLICT", "idempotency key is already finalized"
                    )
                else:
                    if updated.rowcount != 1:
                        raise ArtifactRegistryError(
                            "INVALID_STATE", "artifact status changed concurrently"
                        )
                    finalized = self._to_ref(self._require_record(connection, artifact_id))

        if deferred_error is not None:
            raise deferred_error
        if finalized is None:
            raise ArtifactRegistryError("INTERNAL_ERROR", "finalization produced no result")
        return finalized

    def fail(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        """Mark generation or validation failure as canonical INVALID."""

        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "failure reason is required")
        return self._transition_from(
            artifact_id,
            expected=ArtifactStatus.PENDING,
            target=ArtifactStatus.INVALID,
            reason=reason,
            invalid_message="only PENDING artifacts can fail",
        )

    def invalidate(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "invalidation reason is required")
        return self._transition_from(
            artifact_id,
            expected=ArtifactStatus.AVAILABLE,
            target=ArtifactStatus.INVALID,
            reason=reason,
            invalid_message="only AVAILABLE artifacts can be invalidated",
        )

    def mark_missing(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        """Mark an AVAILABLE artifact whose object is no longer accessible."""

        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "missing reason is required")
        return self._transition_from(
            artifact_id,
            expected=ArtifactStatus.AVAILABLE,
            target=ArtifactStatus.MISSING,
            reason=reason,
            invalid_message="only AVAILABLE artifacts can become MISSING",
        )

    def get_metadata(self, artifact_id: str) -> ArtifactPublicView:
        with self._transaction(write=False) as connection:
            return self._require_record(connection, artifact_id).to_public_view()

    def resolve_uri(self, artifact_id: str, principal: Principal) -> str:
        with self._transaction(write=False) as connection:
            record = self._require_record(connection, artifact_id)
            if record.status is not ArtifactStatus.AVAILABLE:
                raise ArtifactRegistryError("ARTIFACT_NOT_AVAILABLE", "artifact is not AVAILABLE")
            if not principal.may_resolve_uri(record.case_id):
                raise ArtifactRegistryError(
                    "PERMISSION_DENIED", "principal cannot resolve artifact URI"
                )
            return record.internal_uri

    def find_available_by_idempotency_key(
        self,
        key: str,
        *,
        case_id: str | None = None,
    ) -> ArtifactRef | None:
        with self._transaction(write=False) as connection:
            record = self._find_available_record(connection, key, case_id=case_id)
            return self._to_ref(record) if record is not None else None

    # Backward-compatible name used by the written specification.
    def find_ready_by_idempotency_key(
        self,
        key: str,
        *,
        case_id: str | None = None,
    ) -> ArtifactRef | None:
        return self.find_available_by_idempotency_key(key, case_id=case_id)

    def get_lineage(self, artifact_id: str) -> ArtifactLineage:
        with self._transaction(write=False) as connection:
            self._require_record(connection, artifact_id)
            self._assert_acyclic_lineage(connection, artifact_id)
            parents = tuple(
                row["parent_artifact_id"]
                for row in connection.execute(
                    """
                    SELECT parent_artifact_id
                    FROM artifact_lineage
                    WHERE child_artifact_id = ?
                    ORDER BY parent_artifact_id
                    """,
                    (artifact_id,),
                )
            )
            children = tuple(
                row["child_artifact_id"]
                for row in connection.execute(
                    """
                    SELECT child_artifact_id
                    FROM artifact_lineage
                    WHERE parent_artifact_id = ?
                    ORDER BY child_artifact_id
                    """,
                    (artifact_id,),
                )
            )
            return ArtifactLineage(
                artifact_id=artifact_id,
                parent_artifact_ids=parents,
                child_artifact_ids=children,
            )

    @staticmethod
    def _assert_acyclic_lineage(
        connection: sqlite3.Connection, artifact_id: str
    ) -> None:
        """Reject a persisted ancestor cycle without unbounded recursion.

        Normal registration cannot create a cycle because a child is new and
        every parent must already exist. This guard protects readers from a
        legacy migration or out-of-band database writer. ``hex`` encodes IDs
        before path membership checks, so arbitrary punctuation in an ID
        cannot collide with the comma delimiter.
        """

        cycle = connection.execute(
            """
            WITH RECURSIVE ancestry(node_id, visited_path, has_cycle) AS (
                SELECT
                    ?,
                    ',' || hex(CAST(? AS BLOB)) || ',',
                    0
                UNION ALL
                SELECT
                    lineage.parent_artifact_id,
                    ancestry.visited_path
                        || hex(CAST(lineage.parent_artifact_id AS BLOB)) || ',',
                    CASE
                        WHEN instr(
                            ancestry.visited_path,
                            ',' || hex(CAST(lineage.parent_artifact_id AS BLOB)) || ','
                        ) > 0 THEN 1
                        ELSE 0
                    END
                FROM ancestry
                JOIN artifact_lineage AS lineage
                    ON lineage.child_artifact_id = ancestry.node_id
                WHERE ancestry.has_cycle = 0
            )
            SELECT 1 FROM ancestry WHERE has_cycle = 1 LIMIT 1
            """,
            (artifact_id, artifact_id),
        ).fetchone()
        if cycle is not None:
            raise ArtifactRegistryError(
                "LINEAGE_CYCLE", f"artifact {artifact_id} has cyclic persisted lineage"
            )

    def _transition_from(
        self,
        artifact_id: str,
        *,
        expected: ArtifactStatus,
        target: ArtifactStatus,
        reason: str,
        invalid_message: str,
    ) -> ArtifactPublicView:
        with self._transaction(write=True) as connection:
            record = self._require_record(connection, artifact_id)
            if record.status is not expected:
                raise ArtifactRegistryError("INVALID_STATE", invalid_message)
            updated = connection.execute(
                """
                UPDATE artifacts
                SET status = ?, failure_reason = ?
                WHERE artifact_id = ? AND status = ?
                """,
                (target.value, reason, artifact_id, expected.value),
            )
            if updated.rowcount != 1:
                raise ArtifactRegistryError("INVALID_STATE", "artifact status changed concurrently")
            return self._require_record(connection, artifact_id).to_public_view()

    def _initialize_schema(self) -> None:
        artifact_types = ", ".join(f"'{value.value}'" for value in ArtifactType)
        statuses = ", ".join(f"'{value.value}'" for value in ArtifactStatus)
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL CHECK (length(trim(case_id)) > 0),
                artifact_type TEXT NOT NULL CHECK (artifact_type IN ({artifact_types})),
                status TEXT NOT NULL CHECK (status IN ({statuses})),
                internal_uri TEXT NOT NULL CHECK (length(trim(internal_uri)) > 0),
                created_by TEXT NOT NULL CHECK (length(trim(created_by)) > 0),
                idempotency_key TEXT NOT NULL CHECK (length(trim(idempotency_key)) > 0),
                producer_name TEXT NOT NULL CHECK (length(trim(producer_name)) > 0),
                producer_version TEXT NOT NULL CHECK (length(trim(producer_version)) > 0),
                geometry_json TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                registration_fingerprint TEXT NOT NULL CHECK (length(registration_fingerprint) = 64),
                checksum_sha256 TEXT,
                size_bytes INTEGER CHECK (size_bytes IS NULL OR size_bytes >= 0),
                failure_reason TEXT,
                CHECK (
                    status != 'AVAILABLE'
                    OR (
                        checksum_sha256 IS NOT NULL
                        AND length(checksum_sha256) = 64
                        AND size_bytes IS NOT NULL
                    )
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS artifact_lineage (
                child_artifact_id TEXT NOT NULL,
                parent_artifact_id TEXT NOT NULL,
                PRIMARY KEY (child_artifact_id, parent_artifact_id),
                CHECK (child_artifact_id != parent_artifact_id),
                FOREIGN KEY (child_artifact_id) REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
                FOREIGN KEY (parent_artifact_id) REFERENCES artifacts(artifact_id) ON DELETE RESTRICT
            )
            """,
            """
            DROP INDEX IF EXISTS uq_available_artifact_idempotency_key
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_artifact_case_idempotency_key
            ON artifacts(case_id, idempotency_key)
            WHERE status IN ('PENDING', 'AVAILABLE')
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_artifact_case_idempotency_status
            ON artifacts(case_id, idempotency_key, status)
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_artifact_lineage_parent
            ON artifact_lineage(parent_artifact_id)
            """,
        )
        with self._transaction(write=True) as connection:
            for statement in statements:
                connection.execute(statement)

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield self._connection
                self._secure_database_files()
                self._connection.commit()
            except ArtifactRegistryError:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
            except sqlite3.OperationalError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise ArtifactRegistryError(
                        "REGISTRY_BUSY", "artifact registry is busy", retryable=True
                    ) from exc
                raise ArtifactRegistryError(
                    "STORAGE_ERROR", "artifact registry operation failed"
                ) from exc
            except sqlite3.DatabaseError as exc:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise ArtifactRegistryError(
                    "STORAGE_ERROR", "artifact registry operation failed"
                ) from exc
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactRegistryError("REGISTRY_CLOSED", "artifact registry is closed")

    @staticmethod
    def _prepare_database_path(database_path: str | os.PathLike[str]) -> str:
        raw_path = os.fspath(database_path)
        if raw_path == ":memory:":
            return raw_path
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if path.is_symlink():
            raise ArtifactRegistryError(
                "INVALID_DATABASE_PATH", "artifact registry database must not be a symlink"
            )
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactRegistryError(
                    "INVALID_DATABASE_PATH", "artifact registry database is not a regular file"
                )
            os.fchmod(descriptor, 0o600)
        except ArtifactRegistryError:
            raise
        except OSError as exc:
            raise ArtifactRegistryError(
                "INVALID_DATABASE_PATH", "artifact registry database path is unsafe"
            ) from exc
        finally:
            if "descriptor" in locals():
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
                    raise ArtifactRegistryError(
                        "INVALID_DATABASE_PATH",
                        "artifact registry database sidecar is not a safe regular file",
                    )
                path.chmod(0o600)
            except FileNotFoundError:
                continue
            except ArtifactRegistryError:
                raise
            except OSError as exc:
                raise ArtifactRegistryError(
                    "STORAGE_PERMISSION_ERROR",
                    "artifact registry database permissions could not be secured",
                ) from exc

    @staticmethod
    def _select_artifact_row(
        connection: sqlite3.Connection, artifact_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()

    def _require_record(
        self, connection: sqlite3.Connection, artifact_id: str
    ) -> _StoredArtifact:
        row = self._select_artifact_row(connection, artifact_id)
        if row is None:
            raise ArtifactRegistryError("NOT_FOUND", f"artifact {artifact_id} was not found")
        return self._row_to_record(connection, row)

    def _find_available_record(
        self,
        connection: sqlite3.Connection,
        key: str,
        *,
        case_id: str | None = None,
    ) -> _StoredArtifact | None:
        if case_id is not None:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE case_id = ? AND idempotency_key = ? AND status = ?
                """,
                (case_id, key, ArtifactStatus.AVAILABLE.value),
            ).fetchone()
            return self._row_to_record(connection, row) if row is not None else None

        rows = connection.execute(
            """
            SELECT * FROM artifacts
            WHERE idempotency_key = ? AND status IN (?, ?)
            ORDER BY case_id
            LIMIT 2
            """,
            (
                key,
                ArtifactStatus.PENDING.value,
                ArtifactStatus.AVAILABLE.value,
            ),
        ).fetchall()
        if len(rows) > 1:
            raise ArtifactRegistryError(
                "AMBIGUOUS_IDEMPOTENCY_KEY",
                "case_id is required when an idempotency key exists in multiple cases",
            )
        if not rows or ArtifactStatus(rows[0]["status"]) is not ArtifactStatus.AVAILABLE:
            return None
        return self._row_to_record(connection, rows[0])

    def _find_active_record(
        self, connection: sqlite3.Connection, case_id: str, key: str
    ) -> _StoredArtifact | None:
        row = connection.execute(
            """
            SELECT * FROM artifacts
            WHERE case_id = ?
              AND idempotency_key = ?
              AND status IN (?, ?)
            """,
            (
                case_id,
                key,
                ArtifactStatus.PENDING.value,
                ArtifactStatus.AVAILABLE.value,
            ),
        ).fetchone()
        return self._row_to_record(connection, row) if row is not None else None

    def _row_to_record(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> _StoredArtifact:
        parent_ids = tuple(
            parent["parent_artifact_id"]
            for parent in connection.execute(
                """
                SELECT parent_artifact_id
                FROM artifact_lineage
                WHERE child_artifact_id = ?
                ORDER BY parent_artifact_id
                """,
                (row["artifact_id"],),
            )
        )
        try:
            raw_metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ArtifactRegistryError("STORAGE_ERROR", "stored artifact metadata is invalid") from exc
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ArtifactRegistryError("STORAGE_ERROR", "stored artifact metadata is invalid")
        return _StoredArtifact(
            artifact_id=row["artifact_id"],
            case_id=row["case_id"],
            artifact_type=ArtifactType(row["artifact_type"]),
            status=ArtifactStatus(row["status"]),
            internal_uri=row["internal_uri"],
            created_by=row["created_by"],
            idempotency_key=row["idempotency_key"],
            producer_name=row["producer_name"],
            producer_version=row["producer_version"],
            geometry=self._decode_geometry(row["geometry_json"]),
            metadata=raw_metadata,
            checksum_sha256=row["checksum_sha256"],
            size_bytes=row["size_bytes"],
            failure_reason=row["failure_reason"],
            parent_artifact_ids=parent_ids,
            registration_fingerprint=row["registration_fingerprint"],
        )

    @staticmethod
    def _compute_registration_fingerprint(
        *,
        case_id: str,
        artifact_type: ArtifactType,
        producer_name: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...],
        geometry: VolumeGeometry | None,
        metadata: Mapping[str, str],
    ) -> str:
        payload = canonical_json(
            {
                "identity_schema_version": "1",
                "case_id": case_id,
                "artifact_type": artifact_type,
                "producer_name": producer_name,
                "producer_version": producer_version,
                "parent_artifact_ids": sorted(parent_artifact_ids),
                "geometry_fingerprint": (
                    geometry.geometry_fingerprint if geometry is not None else None
                ),
                "metadata": dict(metadata),
            }
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _encode_geometry(geometry: VolumeGeometry | None) -> str | None:
        if geometry is None:
            return None
        return json.dumps(
            {
                "size_ijk": geometry.size_ijk,
                "spacing_mm": geometry.spacing_mm,
                "origin_mm": geometry.origin_mm,
                "direction_cosines": geometry.direction_cosines,
                "coordinate_system": geometry.coordinate_system.value,
                "geometry_fingerprint": geometry.geometry_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_geometry(payload: str | None) -> VolumeGeometry | None:
        if payload is None:
            return None
        try:
            raw = json.loads(payload)
            return VolumeGeometry(
                size_ijk=tuple(raw["size_ijk"]),
                spacing_mm=tuple(raw["spacing_mm"]),
                origin_mm=tuple(raw["origin_mm"]),
                direction_cosines=tuple(raw["direction_cosines"]),
                coordinate_system=CoordinateSystem(raw["coordinate_system"]),
                geometry_fingerprint=raw["geometry_fingerprint"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactRegistryError("STORAGE_ERROR", "stored artifact geometry is invalid") from exc

    @staticmethod
    def _to_ref(record: _StoredArtifact) -> ArtifactRef:
        if record.status is not ArtifactStatus.AVAILABLE or record.checksum_sha256 is None:
            raise ArtifactRegistryError("ARTIFACT_NOT_AVAILABLE", "artifact is not finalized")
        return ArtifactRef(
            artifact_id=record.artifact_id,
            case_id=record.case_id,
            artifact_type=record.artifact_type,
            uri=record.internal_uri,
            checksum_sha256=record.checksum_sha256,
            status=record.status,
            geometry=record.geometry,
            producer_name=record.producer_name,
            producer_version=record.producer_version,
            parent_artifact_ids=record.parent_artifact_ids,
            metadata=dict(record.metadata),
        )


__all__ = ["SQLiteArtifactRegistry"]
