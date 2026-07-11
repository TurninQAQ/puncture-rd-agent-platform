"""PostgreSQL Run/event repository with atomic version and event fencing."""

from __future__ import annotations

from contextlib import suppress
from hashlib import sha256
import json
import math
import re
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .errors import (
    ExecutionSuperseded,
    RunRepositoryConfigurationError,
    RunRepositoryError,
    RunRepositoryEventConflict,
    RunRepositoryIdempotencyConflict,
    RunRepositoryIntegrityError,
    RunRepositoryNotFound,
    RunRepositoryTransitionError,
    RunRepositoryUnavailable,
    RunRepositoryVersionConflict,
)
from .json_boundary import RuntimeJsonBoundaryError, copy_json_mapping
from .models import EventType, RunEvent, RunRequest, RunSnapshot, RunStatus
from .repository import (
    CreateRunResult,
    InMemoryRunRepository,
    RunEventDraft,
    VersionedRun,
    _RUNNING_STREAM_EVENTS,
    _canonical_json,
    _canonical_utc_millisecond,
    _event_fingerprint,
    _request_document,
    _request_fingerprint,
    _snapshot_fingerprint,
    _transition_fingerprint,
)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_TRANSIENT_SQLSTATES = {
    "40001",
    "40P01",
    "55P03",
    "57014",
    "57P01",
    "57P02",
    "57P03",
    "53300",
    "53400",
}
_CONFIGURATION_SQLSTATES = {
    "28P01",
    "3F000",
    "42501",
    "42703",
    "42P01",
}
_INTEGRITY_SQLSTATE_PREFIXES = ("22", "23")
_REQUEST_FIELDS = frozenset(
    {
        "case_id",
        "user_query",
        "task_type",
        "idempotency_key",
        "tenant_id",
        "principal_id",
        "artifact_ids",
        "metadata",
    }
)


def _quote_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("PostgreSQL schema must be a simple identifier")
    return f'"{value}"'


def _validate_connection_string(value: str) -> str:
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


def _validate_application_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("application_name must be non-empty")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("application_name must be valid UTF-8") from exc
    if len(encoded) > 63 or any(
        character in value for character in ("\r", "\n", "\x00")
    ):
        raise ValueError("application_name must fit PostgreSQL's 63-byte limit")
    return value


def _validate_timeout_seconds(value: float) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError("connect_timeout_seconds must be positive and finite")
    return max(1, math.ceil(float(value)))


def _validate_timeout_ms(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_json_loads(value: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        value,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _canonical_json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise RunRepositoryIntegrityError()
    try:
        document = _strict_json_loads(value)
        if not isinstance(document, dict):
            raise ValueError("canonical JSON value must be an object")
        if _canonical_json(document) != value:
            raise ValueError("JSON object is not canonical")
        return copy_json_mapping(document)
    except (RuntimeJsonBoundaryError, TypeError, ValueError) as exc:
        raise RunRepositoryIntegrityError() from exc


def _request_from_canonical(
    canonical: str,
    *,
    fingerprint: str,
    tenant_id: str,
    idempotency_key: str,
) -> RunRequest:
    if not isinstance(canonical, str) or not isinstance(fingerprint, str):
        raise RunRepositoryIntegrityError()
    if sha256(canonical.encode("utf-8")).hexdigest() != fingerprint:
        raise RunRepositoryIntegrityError()
    try:
        document = _strict_json_loads(canonical)
        if not isinstance(document, dict) or set(document) != _REQUEST_FIELDS:
            raise ValueError("request document shape is invalid")
        if _canonical_json(document) != canonical:
            raise ValueError("request document is not canonical")
        if document["tenant_id"] != tenant_id:
            raise ValueError("request tenant does not match row tenant")
        if document["idempotency_key"] != idempotency_key:
            raise ValueError("request idempotency key does not match row")
        artifact_ids = document["artifact_ids"]
        metadata = document["metadata"]
        if not isinstance(artifact_ids, list) or not isinstance(metadata, dict):
            raise ValueError("request collections are invalid")
        return RunRequest(
            case_id=document["case_id"],
            user_query=document["user_query"],
            task_type=document["task_type"],
            idempotency_key=document["idempotency_key"],
            tenant_id=document["tenant_id"],
            principal_id=document["principal_id"],
            artifact_ids=tuple(artifact_ids),
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, RuntimeJsonBoundaryError) as exc:
        raise RunRepositoryIntegrityError() from exc


class PostgresRunRepository:
    """One-transaction-per-operation PostgreSQL implementation.

    Migrations are explicit: call :meth:`migrate` during deployment/startup,
    never from a request path.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        schema: str = "puncture_runtime",
        connect_timeout_seconds: float = 5.0,
        statement_timeout_ms: int = 5_000,
        lock_timeout_ms: int = 1_000,
        application_name: str = "puncture-api-run-repository",
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._connection_string = _validate_connection_string(connection_string)
        self._schema_name = schema
        self._schema = _quote_identifier(schema)
        self._connect_timeout_seconds = _validate_timeout_seconds(
            connect_timeout_seconds
        )
        self._statement_timeout_ms = _validate_timeout_ms(
            "statement_timeout_ms",
            statement_timeout_ms,
        )
        self._lock_timeout_ms = _validate_timeout_ms(
            "lock_timeout_ms",
            lock_timeout_ms,
        )
        self._application_name = _validate_application_name(application_name)
        if connection_factory is not None and not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory
        self._runs = f'{self._schema}."runs"'
        self._events = f'{self._schema}."run_events"'
        self._mutations = f'{self._schema}."run_mutations"'
        self._migrations = f'{self._schema}."schema_migrations"'
        self._run_select = f"""
            SELECT
                run_id,
                tenant_id,
                idempotency_key,
                request_fingerprint,
                request_canonical,
                snapshot_fingerprint,
                status,
                version,
                trace_id,
                to_char(created_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                to_char(updated_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                final_report_canonical,
                checkpoint_canonical,
                approval_id,
                error_canonical,
                last_event_sequence
            FROM {self._runs}
        """
        self._event_select = """
            SELECT
                run_id,
                sequence,
                event_type,
                node_name,
                to_char(occurred_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                payload_canonical,
                trace_id,
                event_key,
                execution_version,
                event_fingerprint
        """

    @property
    def schema(self) -> str:
        return self._schema_name

    def migrate(self) -> None:
        statements = self._migration_statements()
        checksum = sha256("\n".join(statements).encode("utf-8")).hexdigest()
        connection: Any | None = None
        try:
            connection = self._connect()
            lock_key = int.from_bytes(
                sha256(
                    f"puncture-run-repository-migration:{self._schema_name}".encode(
                        "utf-8"
                    )
                ).digest()[:8],
                byteorder="big",
                signed=True,
            )
            self._execute_discard(
                connection,
                "SELECT pg_advisory_xact_lock(%s::bigint)",
                (lock_key,),
            )
            self._execute_discard(
                connection,
                f"CREATE SCHEMA IF NOT EXISTS {self._schema}",
            )
            self._execute_discard(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS {self._migrations} (
                    version integer PRIMARY KEY,
                    name text NOT NULL,
                    checksum_sha256 text NOT NULL CHECK (
                        checksum_sha256 ~ '^[0-9a-f]{64}$'
                    ),
                    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
                )
                """,
            )
            row = self._fetchone(
                connection,
                f"SELECT name, checksum_sha256 FROM {self._migrations} "
                "WHERE version = %s",
                (1,),
            )
            if row is not None:
                if row != ("initial-run-event-store", checksum):
                    raise RunRepositoryConfigurationError()
            else:
                for statement in statements:
                    self._execute_discard(connection, statement)
                self._execute_discard(
                    connection,
                    f"""
                    INSERT INTO {self._migrations}
                        (version, name, checksum_sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (1, "initial-run-event-store", checksum),
                )
            connection.commit()
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc, setup=True) from exc
        finally:
            self._close(connection)

    setup = migrate

    def create_or_get_started(
        self,
        snapshot: RunSnapshot,
        initial_events: tuple[RunEventDraft, ...],
    ) -> CreateRunResult:
        normalized = InMemoryRunRepository._copy_snapshot(snapshot)
        events = tuple(
            InMemoryRunRepository._copy_event(event) for event in initial_events
        )
        InMemoryRunRepository._validate_snapshot_state(normalized)
        if normalized.status is not RunStatus.RUNNING:
            raise RunRepositoryTransitionError(
                "new runs must enter the repository in RUNNING state"
            )
        if tuple(event.event_type for event in events) != (
            EventType.RUN_CREATED,
            EventType.RUN_STARTED,
        ):
            raise RunRepositoryTransitionError(
                "new runs require RUN_CREATED then RUN_STARTED"
            )
        if any(event.event_key is not None for event in events):
            raise RunRepositoryTransitionError(
                "initial lifecycle events must not define event_key"
            )
        snapshot_fingerprint = _snapshot_fingerprint(normalized)
        request_document = _request_document(normalized.request)
        request_canonical = _canonical_json(request_document)
        request_fingerprint = _request_fingerprint(normalized.request)
        final_report_canonical = _canonical_json(normalized.final_report)
        checkpoint_canonical = _canonical_json(normalized.checkpoint)
        error_canonical = (
            _canonical_json(normalized.error)
            if normalized.error is not None
            else None
        )

        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                f"""
                INSERT INTO {self._runs} (
                    run_id,
                    tenant_id,
                    idempotency_key,
                    request_fingerprint,
                    request_canonical,
                    request_json,
                    snapshot_fingerprint,
                    status,
                    version,
                    trace_id,
                    created_at,
                    updated_at,
                    final_report_canonical,
                    final_report,
                    checkpoint_canonical,
                    checkpoint,
                    approval_id,
                    error_canonical,
                    error_json,
                    last_event_sequence
                ) VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s,
                    %s, 1, %s, %s::timestamptz, %s::timestamptz,
                    %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s::jsonb, 2
                )
                ON CONFLICT DO NOTHING
                RETURNING run_id
                """,
                (
                    normalized.run_id,
                    normalized.request.tenant_id,
                    normalized.request.idempotency_key,
                    request_fingerprint,
                    request_canonical,
                    request_canonical,
                    snapshot_fingerprint,
                    normalized.status.value,
                    normalized.trace_id,
                    normalized.created_at,
                    normalized.updated_at,
                    final_report_canonical,
                    final_report_canonical,
                    checkpoint_canonical,
                    checkpoint_canonical,
                    normalized.approval_id,
                    error_canonical,
                    error_canonical,
                ),
            )
            if row is None:
                existing_row = self._fetchone(
                    connection,
                    self._run_select
                    + " WHERE tenant_id = %s AND idempotency_key = %s FOR SHARE",
                    (
                        normalized.request.tenant_id,
                        normalized.request.idempotency_key,
                    ),
                )
                if existing_row is None:
                    raise RunRepositoryIdempotencyConflict()
                existing, _, _ = self._decode_run(existing_row)
                if (
                    existing_row[3] != request_fingerprint
                    or existing_row[4] != request_canonical
                ):
                    raise RunRepositoryIdempotencyConflict()
                connection.commit()
                return CreateRunResult(existing, created=False)

            for sequence, event in enumerate(events, start=1):
                self._insert_event(
                    connection,
                    run_id=normalized.run_id,
                    tenant_id=normalized.request.tenant_id,
                    sequence=sequence,
                    event=event,
                    event_key=None,
                    execution_version=1,
                    trace_id=normalized.trace_id,
                )
            commit_attempted = True
            connection.commit()
            return CreateRunResult(VersionedRun(normalized, 1), created=True)
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted:
                return self._reconcile_create(
                    normalized,
                    request_fingerprint=request_fingerprint,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def get(self, run_id: str, *, tenant_id: str) -> VersionedRun:
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                self._run_select + " WHERE run_id = %s AND tenant_id = %s",
                (run_id, tenant_id),
            )
            if row is None:
                raise RunRepositoryNotFound()
            run, _, _ = self._decode_run(row)
            connection.commit()
            return run
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def get_events(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("after_sequence must be an integer")
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        connection: Any | None = None
        try:
            connection = self._connect()
            run_row = self._fetchone(
                connection,
                f"""
                SELECT last_event_sequence, trace_id
                FROM {self._runs}
                WHERE run_id = %s AND tenant_id = %s
                FOR SHARE
                """,
                (run_id, tenant_id),
            )
            if run_row is None:
                raise RunRepositoryNotFound()
            last_sequence, trace_id = run_row
            rows = self._fetchall(
                connection,
                self._event_select
                + f"""
                  FROM {self._events}
                  WHERE run_id = %s AND tenant_id = %s AND sequence > %s
                  ORDER BY sequence ASC
                """,
                (run_id, tenant_id, after_sequence),
            )
            events = tuple(self._decode_event(row)[0] for row in rows)
            expected_first = after_sequence + 1
            expected_count = max(0, int(last_sequence) - after_sequence)
            if len(events) != expected_count:
                raise RunRepositoryIntegrityError()
            if events and events[0].sequence != expected_first:
                raise RunRepositoryIntegrityError()
            if [event.sequence for event in events] != list(
                range(expected_first, expected_first + len(events))
            ):
                raise RunRepositoryIntegrityError()
            if any(event.trace_id != trace_id for event in events):
                raise RunRepositoryIntegrityError()
            connection.commit()
            return events
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def assert_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
    ) -> None:
        InMemoryRunRepository._validate_version(expected_version)
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                f"""
                SELECT version, status
                FROM {self._runs}
                WHERE run_id = %s AND tenant_id = %s
                """,
                (run_id, tenant_id),
            )
            if row is None:
                raise RunRepositoryNotFound()
            if row != (expected_version, RunStatus.RUNNING.value):
                raise ExecutionSuperseded("execution no longer owns this run")
            connection.commit()
        except (ExecutionSuperseded, RunRepositoryError):
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def append_if_running(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        event: RunEventDraft,
    ) -> RunEvent:
        InMemoryRunRepository._validate_version(expected_version)
        normalized = InMemoryRunRepository._copy_event(event)
        if normalized.event_type not in _RUNNING_STREAM_EVENTS:
            raise RunRepositoryTransitionError(
                "state events must be committed atomically with a state transition"
            )
        if normalized.event_key is None:
            raise RunRepositoryTransitionError(
                "running stream events require a stable event_key"
            )
        fingerprint = _event_fingerprint(normalized, expected_version)
        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            run_row = self._fetchone(
                connection,
                f"""
                SELECT version, status, trace_id, last_event_sequence
                FROM {self._runs}
                WHERE run_id = %s AND tenant_id = %s
                FOR UPDATE
                """,
                (run_id, tenant_id),
            )
            if run_row is None:
                raise RunRepositoryNotFound()
            existing_row = self._fetchone(
                connection,
                self._event_select
                + f"""
                  FROM {self._events}
                  WHERE run_id = %s AND tenant_id = %s AND event_key = %s
                """,
                (run_id, tenant_id, normalized.event_key),
            )
            if existing_row is not None:
                existing, event_key, execution_version, stored_fingerprint = (
                    self._decode_event(existing_row)
                )
                if existing.trace_id != run_row[2]:
                    raise RunRepositoryIntegrityError()
                if (
                    event_key != normalized.event_key
                    or execution_version != expected_version
                    or stored_fingerprint != fingerprint
                ):
                    raise RunRepositoryEventConflict()
                connection.commit()
                return existing
            version, status, trace_id, last_sequence = run_row
            if version != expected_version or status != RunStatus.RUNNING.value:
                raise ExecutionSuperseded("execution no longer owns this run")
            new_sequence = int(last_sequence) + 1
            self._execute_discard(
                connection,
                f"""
                UPDATE {self._runs}
                SET last_event_sequence = %s
                WHERE run_id = %s AND tenant_id = %s
                """,
                (new_sequence, run_id, tenant_id),
            )
            appended = self._insert_event(
                connection,
                run_id=run_id,
                tenant_id=tenant_id,
                sequence=new_sequence,
                event=normalized,
                event_key=normalized.event_key,
                execution_version=expected_version,
                trace_id=trace_id,
            )
            commit_attempted = True
            connection.commit()
            return appended
        except (ExecutionSuperseded, RunRepositoryError):
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted:
                return self._reconcile_append(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    event_key=normalized.event_key,
                    expected_version=expected_version,
                    fingerprint=fingerprint,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def compare_and_swap(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        InMemoryRunRepository._validate_version(expected_version)
        normalized = InMemoryRunRepository._copy_snapshot(snapshot)
        normalized_events = tuple(
            InMemoryRunRepository._copy_event(event) for event in events
        )
        InMemoryRunRepository._validate_snapshot_state(normalized)
        if any(event.event_key is not None for event in normalized_events):
            raise RunRepositoryTransitionError(
                "lifecycle transition events must not define event_key"
            )
        target_fingerprint = _snapshot_fingerprint(normalized)
        transition_fingerprint = _transition_fingerprint(
            normalized_events,
            expected_version,
        )
        final_report_canonical = _canonical_json(normalized.final_report)
        checkpoint_canonical = _canonical_json(normalized.checkpoint)
        error_canonical = (
            _canonical_json(normalized.error)
            if normalized.error is not None
            else None
        )
        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            current_row = self._fetchone(
                connection,
                self._run_select
                + " WHERE run_id = %s AND tenant_id = %s FOR UPDATE",
                (run_id, tenant_id),
            )
            if current_row is None:
                raise RunRepositoryNotFound()
            current, last_sequence, _ = self._decode_run(current_row)
            if current.version != expected_version:
                if self._cas_replay_matches(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    target_fingerprint=target_fingerprint,
                    transition_fingerprint=transition_fingerprint,
                    expected_version=expected_version,
                ):
                    connection.commit()
                    return VersionedRun(normalized, expected_version + 1)
                raise RunRepositoryVersionConflict()
            InMemoryRunRepository._validate_replacement(
                current.snapshot,
                normalized,
            )
            InMemoryRunRepository._validate_transition_events(
                normalized,
                normalized_events,
            )
            next_version = expected_version + 1
            next_sequence = last_sequence + len(normalized_events)
            self._execute_discard(
                connection,
                f"""
                UPDATE {self._runs}
                SET status = %s,
                    version = %s,
                    snapshot_fingerprint = %s,
                    updated_at = %s::timestamptz,
                    final_report_canonical = %s,
                    final_report = %s::jsonb,
                    checkpoint_canonical = %s,
                    checkpoint = %s::jsonb,
                    approval_id = %s,
                    error_canonical = %s,
                    error_json = %s::jsonb,
                    last_event_sequence = %s
                WHERE run_id = %s AND tenant_id = %s AND version = %s
                """,
                (
                    normalized.status.value,
                    next_version,
                    target_fingerprint,
                    normalized.updated_at,
                    final_report_canonical,
                    final_report_canonical,
                    checkpoint_canonical,
                    checkpoint_canonical,
                    normalized.approval_id,
                    error_canonical,
                    error_canonical,
                    next_sequence,
                    run_id,
                    tenant_id,
                    expected_version,
                ),
            )
            for offset, event in enumerate(normalized_events, start=1):
                self._insert_event(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    sequence=last_sequence + offset,
                    event=event,
                    event_key=None,
                    execution_version=expected_version,
                    trace_id=normalized.trace_id,
                )
            self._insert_mutation(
                connection,
                run_id=run_id,
                tenant_id=tenant_id,
                expected_version=expected_version,
                target_fingerprint=target_fingerprint,
                transition_fingerprint=transition_fingerprint,
            )
            commit_attempted = True
            connection.commit()
            return VersionedRun(normalized, next_version)
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted:
                return self._reconcile_cas(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    expected_version=expected_version,
                    target=normalized,
                    target_fingerprint=target_fingerprint,
                    transition_fingerprint=transition_fingerprint,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def _migration_statements(self) -> tuple[str, ...]:
        status_values = ", ".join(
            f"'{status.value}'" for status in RunStatus
        )
        event_values = ", ".join(
            f"'{event_type.value}'" for event_type in EventType
        )
        return (
            f"""
            CREATE TABLE {self._runs} (
                run_id text COLLATE "C" NOT NULL,
                tenant_id text COLLATE "C" NOT NULL,
                idempotency_key text COLLATE "C" NOT NULL,
                request_fingerprint text NOT NULL,
                request_canonical text NOT NULL,
                request_json jsonb NOT NULL,
                snapshot_fingerprint text NOT NULL,
                status text NOT NULL,
                version bigint NOT NULL,
                trace_id text COLLATE "C" NOT NULL,
                created_at timestamptz NOT NULL,
                updated_at timestamptz NOT NULL,
                final_report_canonical text NOT NULL DEFAULT '{{}}',
                final_report jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                checkpoint_canonical text NOT NULL DEFAULT '{{}}',
                checkpoint jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                approval_id text COLLATE "C",
                error_canonical text,
                error_json jsonb,
                last_event_sequence bigint NOT NULL,
                CONSTRAINT pk_runtime_runs PRIMARY KEY (run_id),
                CONSTRAINT uq_runtime_runs_tenant_run UNIQUE (tenant_id, run_id),
                CONSTRAINT uq_runtime_runs_tenant_idempotency
                    UNIQUE (tenant_id, idempotency_key),
                CONSTRAINT ck_runtime_runs_status
                    CHECK (status IN ({status_values})),
                CONSTRAINT ck_runtime_runs_version CHECK (version >= 1),
                CONSTRAINT ck_runtime_runs_sequence
                    CHECK (last_event_sequence >= 2),
                CONSTRAINT ck_runtime_runs_time_order
                    CHECK (updated_at >= created_at),
                CONSTRAINT ck_runtime_runs_time_precision CHECK (
                    created_at = date_trunc('milliseconds', created_at)
                    AND updated_at = date_trunc('milliseconds', updated_at)
                ),
                CONSTRAINT ck_runtime_runs_request_fingerprint CHECK (
                    request_fingerprint ~ '^[0-9a-f]{{64}}$'
                ),
                CONSTRAINT ck_runtime_runs_snapshot_fingerprint CHECK (
                    snapshot_fingerprint ~ '^[0-9a-f]{{64}}$'
                ),
                CONSTRAINT ck_runtime_runs_request_object
                    CHECK (jsonb_typeof(request_json) = 'object'),
                CONSTRAINT ck_runtime_runs_request_canonical
                    CHECK (request_json = request_canonical::jsonb),
                CONSTRAINT ck_runtime_runs_final_object
                    CHECK (jsonb_typeof(final_report) = 'object'),
                CONSTRAINT ck_runtime_runs_final_canonical
                    CHECK (final_report = final_report_canonical::jsonb),
                CONSTRAINT ck_runtime_runs_checkpoint_object
                    CHECK (jsonb_typeof(checkpoint) = 'object'),
                CONSTRAINT ck_runtime_runs_checkpoint_canonical
                    CHECK (checkpoint = checkpoint_canonical::jsonb),
                CONSTRAINT ck_runtime_runs_error_object CHECK (
                    error_json IS NULL OR jsonb_typeof(error_json) = 'object'
                ),
                CONSTRAINT ck_runtime_runs_error_canonical CHECK (
                    (error_canonical IS NULL AND error_json IS NULL)
                    OR (
                        error_canonical IS NOT NULL
                        AND error_json IS NOT NULL
                        AND error_json = error_canonical::jsonb
                    )
                ),
                CONSTRAINT ck_runtime_runs_request_tenant CHECK (
                    request_json ->> 'tenant_id' = tenant_id
                ),
                CONSTRAINT ck_runtime_runs_request_idempotency CHECK (
                    request_json ->> 'idempotency_key' = idempotency_key
                ),
                CONSTRAINT ck_runtime_runs_approval CHECK (
                    (
                        status = 'WAITING_APPROVAL'
                        AND approval_id IS NOT NULL
                        AND btrim(approval_id) <> ''
                    ) OR (
                        status <> 'WAITING_APPROVAL' AND approval_id IS NULL
                    )
                ),
                CONSTRAINT ck_runtime_runs_error CHECK (
                    (status = 'FAILED' AND error_json IS NOT NULL)
                    OR (status <> 'FAILED' AND error_json IS NULL)
                ),
                CONSTRAINT ck_runtime_runs_report CHECK (
                    status = 'SUCCEEDED' OR final_report = '{{}}'::jsonb
                )
            )
            """,
            f"""
            CREATE TABLE {self._events} (
                tenant_id text COLLATE "C" NOT NULL,
                run_id text COLLATE "C" NOT NULL,
                sequence bigint NOT NULL,
                event_key text COLLATE "C",
                execution_version bigint NOT NULL,
                event_fingerprint text NOT NULL,
                event_type text NOT NULL,
                node_name text,
                occurred_at timestamptz NOT NULL DEFAULT
                    date_trunc('milliseconds', clock_timestamp()),
                payload_canonical text NOT NULL,
                payload jsonb NOT NULL,
                trace_id text COLLATE "C" NOT NULL,
                CONSTRAINT pk_runtime_run_events
                    PRIMARY KEY (tenant_id, run_id, sequence),
                CONSTRAINT fk_runtime_run_events_run
                    FOREIGN KEY (tenant_id, run_id)
                    REFERENCES {self._runs} (tenant_id, run_id)
                    ON DELETE RESTRICT,
                CONSTRAINT ck_runtime_run_events_sequence CHECK (sequence >= 1),
                CONSTRAINT ck_runtime_run_events_execution_version
                    CHECK (execution_version >= 1),
                CONSTRAINT ck_runtime_run_events_time_precision CHECK (
                    occurred_at = date_trunc('milliseconds', occurred_at)
                ),
                CONSTRAINT ck_runtime_run_events_fingerprint CHECK (
                    event_fingerprint ~ '^[0-9a-f]{{64}}$'
                ),
                CONSTRAINT ck_runtime_run_events_key CHECK (
                    event_key IS NULL OR (
                        btrim(event_key) <> ''
                        AND event_key = btrim(event_key)
                        AND octet_length(event_key) <= 256
                    )
                ),
                CONSTRAINT ck_runtime_run_events_payload
                    CHECK (jsonb_typeof(payload) = 'object'),
                CONSTRAINT ck_runtime_run_events_payload_canonical
                    CHECK (payload = payload_canonical::jsonb),
                CONSTRAINT ck_runtime_run_events_type
                    CHECK (event_type IN ({event_values}))
            )
            """,
            f"""
            CREATE TABLE {self._mutations} (
                tenant_id text COLLATE "C" NOT NULL,
                run_id text COLLATE "C" NOT NULL,
                target_version bigint NOT NULL,
                expected_version bigint NOT NULL,
                snapshot_fingerprint text NOT NULL,
                transition_fingerprint text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
                CONSTRAINT pk_runtime_run_mutations
                    PRIMARY KEY (tenant_id, run_id, target_version),
                CONSTRAINT uq_runtime_run_mutations_expected
                    UNIQUE (tenant_id, run_id, expected_version),
                CONSTRAINT fk_runtime_run_mutations_run
                    FOREIGN KEY (tenant_id, run_id)
                    REFERENCES {self._runs} (tenant_id, run_id)
                    ON DELETE RESTRICT,
                CONSTRAINT ck_runtime_run_mutations_versions CHECK (
                    expected_version >= 1
                    AND target_version = expected_version + 1
                ),
                CONSTRAINT ck_runtime_run_mutations_snapshot_fingerprint CHECK (
                    snapshot_fingerprint ~ '^[0-9a-f]{{64}}$'
                ),
                CONSTRAINT ck_runtime_run_mutations_transition_fingerprint CHECK (
                    transition_fingerprint ~ '^[0-9a-f]{{64}}$'
                )
            )
            """,
            f"""
            CREATE UNIQUE INDEX
                "uq_runtime_run_event_key"
            ON {self._events} (tenant_id, run_id, event_key)
            WHERE event_key IS NOT NULL
            """,
            f"""
            CREATE INDEX
                "ix_runtime_runs_recovery"
            ON {self._runs} (status, updated_at, run_id)
            WHERE status IN ('RUNNING', 'WAITING_APPROVAL', 'FAILED')
            """,
        )

    def _connect(self) -> Any:
        factory = self._connection_factory
        if factory is None:
            try:
                import psycopg
            except (ImportError, ModuleNotFoundError) as exc:
                raise RunRepositoryConfigurationError() from exc
            factory = psycopg.connect
        connection: Any | None = None
        try:
            connection = factory(
                self._connection_string,
                autocommit=False,
                connect_timeout=self._connect_timeout_seconds,
                application_name=self._application_name,
            )
            if getattr(connection, "closed", False):
                raise RunRepositoryUnavailable()
            if getattr(connection, "autocommit", None) is not False:
                connection.autocommit = False
            self._execute_discard(
                connection,
                "SELECT set_config('statement_timeout', %s, true)",
                (str(self._statement_timeout_ms),),
            )
            self._execute_discard(
                connection,
                "SELECT set_config('lock_timeout', %s, true)",
                (str(self._lock_timeout_ms),),
            )
            return connection
        except RunRepositoryError:
            with suppress(Exception):
                connection.close()
            raise
        except Exception as exc:
            with suppress(Exception):
                connection.close()
            raise self._map_exception(exc) from exc

    @staticmethod
    def _fetchone(
        connection: Any,
        statement: str,
        parameters: tuple[Any, ...] = (),
    ) -> tuple[Any, ...] | None:
        cursor = connection.execute(statement, parameters)
        try:
            return cursor.fetchone()
        finally:
            with suppress(Exception):
                cursor.close()

    @staticmethod
    def _fetchall(
        connection: Any,
        statement: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]:
        cursor = connection.execute(statement, parameters)
        try:
            return list(cursor.fetchall())
        finally:
            with suppress(Exception):
                cursor.close()

    @staticmethod
    def _execute_discard(
        connection: Any,
        statement: str,
        parameters: tuple[Any, ...] = (),
    ) -> None:
        cursor = connection.execute(statement, parameters)
        with suppress(Exception):
            cursor.close()

    def _decode_run(
        self,
        row: tuple[Any, ...],
    ) -> tuple[VersionedRun, int, str]:
        if len(row) != 16:
            raise RunRepositoryIntegrityError()
        (
            run_id,
            tenant_id,
            idempotency_key,
            request_fingerprint,
            request_canonical,
            snapshot_fingerprint,
            status_value,
            version,
            trace_id,
            created_at,
            updated_at,
            final_report_canonical,
            checkpoint_canonical,
            approval_id,
            error_canonical,
            last_sequence,
        ) = row
        try:
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version < 1
                or isinstance(last_sequence, bool)
                or not isinstance(last_sequence, int)
                or last_sequence < 2
                or not isinstance(snapshot_fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", snapshot_fingerprint)
            ):
                raise ValueError("run counters are invalid")
            request = _request_from_canonical(
                request_canonical,
                fingerprint=request_fingerprint,
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
            )
            status = RunStatus(status_value)
            snapshot = RunSnapshot(
                run_id=run_id,
                request=request,
                status=status,
                trace_id=trace_id,
                created_at=_canonical_utc_millisecond(
                    created_at,
                    field_name="created_at",
                ),
                updated_at=_canonical_utc_millisecond(
                    updated_at,
                    field_name="updated_at",
                ),
                final_report=_canonical_json_object(final_report_canonical),
                checkpoint=_canonical_json_object(checkpoint_canonical),
                approval_id=approval_id,
                error=(
                    _canonical_json_object(error_canonical)
                    if error_canonical is not None
                    else None
                ),
            )
            normalized = InMemoryRunRepository._copy_snapshot(snapshot)
            InMemoryRunRepository._validate_snapshot_state(normalized)
            if _request_fingerprint(normalized.request) != request_fingerprint:
                raise ValueError("request fingerprint changed after decoding")
            if _snapshot_fingerprint(normalized) != snapshot_fingerprint:
                raise ValueError("snapshot fingerprint changed after decoding")
            return VersionedRun(normalized, version), last_sequence, snapshot_fingerprint
        except (TypeError, ValueError, RuntimeJsonBoundaryError) as exc:
            raise RunRepositoryIntegrityError() from exc

    def _decode_event(
        self,
        row: tuple[Any, ...],
    ) -> tuple[RunEvent, str | None, int, str]:
        if len(row) != 10:
            raise RunRepositoryIntegrityError()
        (
            run_id,
            sequence,
            event_type,
            node_name,
            timestamp,
            payload_canonical,
            trace_id,
            event_key,
            execution_version,
            fingerprint,
        ) = row
        try:
            if (
                isinstance(execution_version, bool)
                or not isinstance(execution_version, int)
                or execution_version < 1
                or not isinstance(fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
            ):
                raise ValueError("event identity is invalid")
            draft = RunEventDraft(
                event_type=EventType(event_type),
                node_name=node_name,
                payload=_canonical_json_object(payload_canonical),
                event_key=event_key,
            )
            if _event_fingerprint(draft, execution_version) != fingerprint:
                raise ValueError("event fingerprint changed after decoding")
            event = RunEvent(
                run_id=run_id,
                sequence=sequence,
                event_type=draft.event_type,
                node_name=draft.node_name,
                timestamp=_canonical_utc_millisecond(
                    timestamp,
                    field_name="event timestamp",
                ),
                payload=draft.payload,
                trace_id=trace_id,
            )
            return event, event_key, execution_version, fingerprint
        except (TypeError, ValueError, RuntimeJsonBoundaryError) as exc:
            raise RunRepositoryIntegrityError() from exc

    def _insert_event(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        sequence: int,
        event: RunEventDraft,
        event_key: str | None,
        execution_version: int,
        trace_id: str,
    ) -> RunEvent:
        fingerprint = _event_fingerprint(event, execution_version)
        payload_canonical = _canonical_json(event.payload)
        row = self._fetchone(
            connection,
            f"""
            INSERT INTO {self._events} (
                tenant_id,
                run_id,
                sequence,
                event_key,
                execution_version,
                event_fingerprint,
                event_type,
                node_name,
                payload_canonical,
                payload,
                trace_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            RETURNING
                run_id,
                sequence,
                event_type,
                node_name,
                to_char(occurred_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                payload_canonical,
                trace_id,
                event_key,
                execution_version,
                event_fingerprint
            """,
            (
                tenant_id,
                run_id,
                sequence,
                event_key,
                execution_version,
                fingerprint,
                event.event_type.value,
                event.node_name,
                payload_canonical,
                payload_canonical,
                trace_id,
            ),
        )
        if row is None:
            raise RunRepositoryIntegrityError()
        decoded, _, _, _ = self._decode_event(row)
        return RunEvent(
            run_id=run_id,
            sequence=decoded.sequence,
            event_type=decoded.event_type,
            node_name=decoded.node_name,
            timestamp=decoded.timestamp,
            payload=decoded.payload,
            trace_id=decoded.trace_id,
        )

    def _insert_mutation(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        expected_version: int,
        target_fingerprint: str,
        transition_fingerprint: str,
    ) -> None:
        self._execute_discard(
            connection,
            f"""
            INSERT INTO {self._mutations} (
                tenant_id,
                run_id,
                target_version,
                expected_version,
                snapshot_fingerprint,
                transition_fingerprint
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                tenant_id,
                run_id,
                expected_version + 1,
                expected_version,
                target_fingerprint,
                transition_fingerprint,
            ),
        )

    def _cas_replay_matches(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        target_fingerprint: str,
        transition_fingerprint: str,
        expected_version: int,
    ) -> bool:
        row = self._fetchone(
            connection,
            f"""
            SELECT snapshot_fingerprint, transition_fingerprint
            FROM {self._mutations}
            WHERE tenant_id = %s
              AND run_id = %s
              AND target_version = %s
              AND expected_version = %s
            """,
            (tenant_id, run_id, expected_version + 1, expected_version),
        )
        return row == (target_fingerprint, transition_fingerprint)

    def _reconcile_create(
        self,
        snapshot: RunSnapshot,
        *,
        request_fingerprint: str,
    ) -> CreateRunResult:
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                self._run_select
                + " WHERE tenant_id = %s AND idempotency_key = %s",
                (
                    snapshot.request.tenant_id,
                    snapshot.request.idempotency_key,
                ),
            )
            if row is None:
                raise RunRepositoryUnavailable()
            run, _, _ = self._decode_run(row)
            expected_canonical = _canonical_json(_request_document(snapshot.request))
            if row[3] != request_fingerprint or row[4] != expected_canonical:
                raise RunRepositoryIdempotencyConflict()
            events = self._fetchall(
                connection,
                self._event_select
                + f" FROM {self._events} "
                "WHERE tenant_id = %s AND run_id = %s ORDER BY sequence",
                (snapshot.request.tenant_id, run.snapshot.run_id),
            )
            event_types = [self._decode_event(event)[0].event_type for event in events]
            if event_types[:2] != [EventType.RUN_CREATED, EventType.RUN_STARTED]:
                raise RunRepositoryIntegrityError()
            connection.commit()
            return CreateRunResult(
                run,
                created=(run.snapshot.run_id == snapshot.run_id),
            )
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def _reconcile_append(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_key: str,
        expected_version: int,
        fingerprint: str,
    ) -> RunEvent:
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                self._event_select
                + f" FROM {self._events} "
                "WHERE tenant_id = %s AND run_id = %s AND event_key = %s",
                (tenant_id, run_id, event_key),
            )
            if row is None:
                raise RunRepositoryUnavailable()
            event, stored_key, stored_version, stored_fingerprint = (
                self._decode_event(row)
            )
            if (
                stored_key != event_key
                or stored_version != expected_version
                or stored_fingerprint != fingerprint
            ):
                raise RunRepositoryEventConflict()
            connection.commit()
            return RunEvent(
                run_id=run_id,
                sequence=event.sequence,
                event_type=event.event_type,
                node_name=event.node_name,
                timestamp=event.timestamp,
                payload=event.payload,
                trace_id=event.trace_id,
            )
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def _reconcile_cas(
        self,
        *,
        run_id: str,
        tenant_id: str,
        expected_version: int,
        target: RunSnapshot,
        target_fingerprint: str,
        transition_fingerprint: str,
    ) -> VersionedRun:
        connection: Any | None = None
        try:
            connection = self._connect()
            if not self._cas_replay_matches(
                connection,
                run_id=run_id,
                tenant_id=tenant_id,
                target_fingerprint=target_fingerprint,
                transition_fingerprint=transition_fingerprint,
                expected_version=expected_version,
            ):
                raise RunRepositoryUnavailable()
            connection.commit()
            return VersionedRun(target, expected_version + 1)
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    @staticmethod
    def _map_exception(exc: BaseException, *, setup: bool = False) -> RunRepositoryError:
        sqlstate = getattr(exc, "sqlstate", None)
        if isinstance(sqlstate, str):
            if sqlstate.startswith("08") or sqlstate in _TRANSIENT_SQLSTATES:
                return RunRepositoryUnavailable()
            if sqlstate in _CONFIGURATION_SQLSTATES:
                return RunRepositoryConfigurationError()
            if sqlstate.startswith(_INTEGRITY_SQLSTATE_PREFIXES):
                return RunRepositoryIntegrityError()
        class_name = type(exc).__name__
        module_name = type(exc).__module__
        if module_name.startswith("psycopg") and class_name in {
            "InterfaceError",
            "OperationalError",
        }:
            return RunRepositoryUnavailable()
        if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
            return RunRepositoryUnavailable()
        if setup:
            return RunRepositoryConfigurationError()
        return RunRepositoryIntegrityError()

    @staticmethod
    def _rollback(connection: Any | None) -> None:
        if connection is None:
            return
        with suppress(Exception):
            connection.rollback()

    @staticmethod
    def _close(connection: Any | None) -> None:
        if connection is None:
            return
        with suppress(Exception):
            connection.close()


def migrate_postgres_run_repository(
    connection_string: str,
    **kwargs: Any,
) -> None:
    PostgresRunRepository(connection_string, **kwargs).migrate()


__all__ = [
    "PostgresRunRepository",
    "migrate_postgres_run_repository",
]
