"""PostgreSQL Run/event repository with atomic version and event fencing."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
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
from .models import (
    ApprovalDecision,
    EventType,
    RunEvent,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from .repository import (
    CreateRunResult,
    InMemoryRunRepository,
    RunEventDraft,
    RunEventPage,
    RunExecutionClaim,
    RunExecutionIntent,
    RunExecutionIntentKind,
    VersionedRun,
    _RUNNING_STREAM_EVENTS,
    _canonical_json,
    _canonical_utc_millisecond,
    _event_fingerprint,
    _execution_intent_document,
    _execution_intent_fingerprint,
    _request_document,
    _request_fingerprint,
    _snapshot_fingerprint,
    _transition_fingerprint,
    _validate_event_page_request,
    _validate_execution_identifier,
    _validate_execution_lease_seconds,
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
_EXECUTION_INTENT_FIELDS = frozenset({"kind", "approval"})
_APPROVAL_FIELDS = frozenset(
    {"approval_id", "approved", "principal_id", "comment"}
)


@dataclass(frozen=True, slots=True)
class _DecodedExecutionJob:
    tenant_id: str
    run_id: str
    execution_version: int
    intent: RunExecutionIntent
    generation: int
    owner_token: str | None
    worker_id: str | None
    claimed_at: str | None
    heartbeat_at: str | None
    lease_expires_at: str | None
    released_at: str | None
    completed_at: str | None


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


def _execution_intent_from_canonical(
    canonical: str,
    *,
    fingerprint: str,
) -> RunExecutionIntent:
    if not isinstance(canonical, str) or not isinstance(fingerprint, str):
        raise RunRepositoryIntegrityError()
    if sha256(canonical.encode("utf-8")).hexdigest() != fingerprint:
        raise RunRepositoryIntegrityError()
    try:
        document = _strict_json_loads(canonical)
        if (
            not isinstance(document, dict)
            or set(document) != _EXECUTION_INTENT_FIELDS
            or _canonical_json(document) != canonical
        ):
            raise ValueError("execution intent document is invalid")
        kind = RunExecutionIntentKind(document["kind"])
        approval_document = document["approval"]
        approval: ApprovalDecision | None = None
        if approval_document is not None:
            if (
                not isinstance(approval_document, dict)
                or set(approval_document) != _APPROVAL_FIELDS
                or not isinstance(approval_document["approval_id"], str)
                or type(approval_document["approved"]) is not bool
                or not isinstance(approval_document["principal_id"], str)
                or not isinstance(approval_document["comment"], str)
            ):
                raise ValueError("approval execution intent is invalid")
            approval = ApprovalDecision(
                approval_id=approval_document["approval_id"],
                approved=approval_document["approved"],
                principal_id=approval_document["principal_id"],
                comment=approval_document["comment"],
            )
        intent = RunExecutionIntent(kind=kind, approval=approval)
        if _execution_intent_fingerprint(intent) != fingerprint:
            raise ValueError("execution intent fingerprint changed after decoding")
        return intent
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeJsonBoundaryError,
    ) as exc:
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
        self._execution_jobs = f'{self._schema}."run_execution_jobs"'
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
        self._execution_job_select = """
            SELECT
                tenant_id,
                run_id,
                execution_version,
                intent_fingerprint,
                intent_canonical,
                generation,
                owner_token,
                worker_id,
                to_char(claimed_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                to_char(heartbeat_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                to_char(lease_expires_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                to_char(released_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),
                to_char(completed_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"')
        """

    @property
    def schema(self) -> str:
        return self._schema_name

    def check_health(self) -> None:
        """Verify connectivity and every deployed migration checksum."""

        connection: Any | None = None
        try:
            connection = self._connect()
            for version, name, statements in self._migration_versions():
                checksum = sha256(
                    "\n".join(statements).encode("utf-8")
                ).hexdigest()
                row = self._fetchone(
                    connection,
                    f"SELECT name, checksum_sha256 FROM {self._migrations} "
                    "WHERE version = %s",
                    (version,),
                )
                if row != (name, checksum):
                    raise RunRepositoryConfigurationError()
            self._execute_discard(
                connection,
                f"SELECT 1 FROM {self._runs} LIMIT 0",
            )
            self._execute_discard(
                connection,
                f"SELECT 1 FROM {self._events} LIMIT 0",
            )
            self._execute_discard(
                connection,
                f"SELECT 1 FROM {self._mutations} LIMIT 0",
            )
            self._execute_discard(
                connection,
                f"SELECT 1 FROM {self._execution_jobs} LIMIT 0",
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

    def migrate(self) -> None:
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
                        checksum_sha256 ~ '^[0-9a-f]{{64}}$'
                    ),
                    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
                )
                """,
            )
            for version, name, statements in self._migration_versions():
                checksum = sha256(
                    "\n".join(statements).encode("utf-8")
                ).hexdigest()
                row = self._fetchone(
                    connection,
                    f"SELECT name, checksum_sha256 FROM {self._migrations} "
                    "WHERE version = %s",
                    (version,),
                )
                if row is not None:
                    if row != (name, checksum):
                        raise RunRepositoryConfigurationError()
                    continue
                for statement in statements:
                    self._execute_discard(connection, statement)
                self._execute_discard(
                    connection,
                    f"""
                    INSERT INTO {self._migrations}
                        (version, name, checksum_sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (version, name, checksum),
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
                if not self._execution_job_intent_matches(
                    connection,
                    run_id=existing.snapshot.run_id,
                    tenant_id=existing.snapshot.request.tenant_id,
                    execution_version=1,
                    intent=RunExecutionIntent(RunExecutionIntentKind.CREATE),
                ):
                    raise RunRepositoryIntegrityError()
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
            self._insert_execution_job(
                connection,
                run_id=normalized.run_id,
                tenant_id=normalized.request.tenant_id,
                execution_version=1,
                intent=RunExecutionIntent(RunExecutionIntentKind.CREATE),
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

    def get_event_page(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
        limit: int = 128,
    ) -> RunEventPage:
        _validate_event_page_request(after_sequence, limit)
        connection: Any | None = None
        try:
            connection = self._connect()
            run_row = self._fetchone(
                connection,
                f"""
                SELECT last_event_sequence, trace_id, status
                FROM {self._runs}
                WHERE run_id = %s AND tenant_id = %s
                """,
                (run_id, tenant_id),
            )
            if run_row is None:
                raise RunRepositoryNotFound()
            last_sequence, trace_id, status_value = run_row
            if (
                isinstance(last_sequence, bool)
                or not isinstance(last_sequence, int)
                or last_sequence < 2
                or not isinstance(trace_id, str)
            ):
                raise RunRepositoryIntegrityError()
            try:
                status = RunStatus(status_value)
            except (TypeError, ValueError) as exc:
                raise RunRepositoryIntegrityError() from exc
            page_end = min(last_sequence, after_sequence + limit)
            if page_end <= after_sequence:
                rows: list[tuple[Any, ...]] = []
            else:
                rows = self._fetchall(
                    connection,
                    self._event_select
                    + f"""
                      FROM {self._events}
                      WHERE run_id = %s
                        AND tenant_id = %s
                        AND sequence > %s
                        AND sequence <= %s
                      ORDER BY sequence ASC
                      LIMIT %s
                    """,
                    (
                        run_id,
                        tenant_id,
                        after_sequence,
                        page_end,
                        limit,
                    ),
                )
            events = tuple(self._decode_event(row)[0] for row in rows)
            expected_count = max(0, page_end - after_sequence)
            if len(events) != expected_count:
                raise RunRepositoryIntegrityError()
            if events and events[0].sequence != after_sequence + 1:
                raise RunRepositoryIntegrityError()
            if any(
                event.sequence != after_sequence + index
                for index, event in enumerate(events, start=1)
            ):
                raise RunRepositoryIntegrityError()
            if any(event.trace_id != trace_id for event in events):
                raise RunRepositoryIntegrityError()
            page = RunEventPage(
                events=events,
                high_water_sequence=last_sequence,
                status=status,
            )
            connection.commit()
            return page
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
        return self._compare_and_swap(
            run_id,
            tenant_id=tenant_id,
            expected_version=expected_version,
            snapshot=snapshot,
            events=events,
            intent=None,
            execution_claim=None,
        )

    def compare_and_swap_and_enqueue(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        intent: RunExecutionIntent,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        if not isinstance(intent, RunExecutionIntent):
            raise TypeError("execution intent is invalid")
        return self._compare_and_swap(
            run_id,
            tenant_id=tenant_id,
            expected_version=expected_version,
            snapshot=snapshot,
            events=events,
            intent=intent,
            execution_claim=None,
        )

    def compare_and_swap_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...] = (),
    ) -> VersionedRun:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        return self._compare_and_swap(
            claim.run.snapshot.run_id,
            tenant_id=claim.run.snapshot.request.tenant_id,
            expected_version=claim.run.version,
            snapshot=snapshot,
            events=events,
            intent=None,
            execution_claim=claim,
        )

    def _compare_and_swap(
        self,
        run_id: str,
        *,
        tenant_id: str,
        expected_version: int,
        snapshot: RunSnapshot,
        events: tuple[RunEventDraft, ...],
        intent: RunExecutionIntent | None,
        execution_claim: RunExecutionClaim | None,
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
                replay_matches = self._cas_replay_matches(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    target_fingerprint=target_fingerprint,
                    transition_fingerprint=transition_fingerprint,
                    expected_version=expected_version,
                )
                if replay_matches and intent is not None:
                    replay_matches = self._execution_job_intent_matches(
                        connection,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        execution_version=expected_version + 1,
                        intent=intent,
                    )
                if replay_matches and execution_claim is not None:
                    replay_matches = self._execution_claim_identity_matches(
                        connection,
                        execution_claim,
                    )
                if replay_matches:
                    connection.commit()
                    return VersionedRun(normalized, expected_version + 1)
                raise RunRepositoryVersionConflict()
            if execution_claim is not None:
                self._require_active_execution_claim(
                    connection,
                    execution_claim,
                )
            InMemoryRunRepository._validate_replacement(
                current.snapshot,
                normalized,
            )
            InMemoryRunRepository._validate_transition_events(
                normalized,
                normalized_events,
            )
            if intent is not None and normalized.status is not RunStatus.RUNNING:
                raise RunRepositoryTransitionError(
                    "execution intent requires a RUNNING target"
                )
            if intent is not None:
                expected_intent = {
                    RunStatus.WAITING_APPROVAL: RunExecutionIntentKind.APPROVAL,
                    RunStatus.FAILED: RunExecutionIntentKind.RESUME,
                }.get(current.snapshot.status)
                if intent.kind is not expected_intent:
                    raise RunRepositoryTransitionError(
                        "execution intent does not match the lifecycle transition"
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
            if current.snapshot.status is RunStatus.RUNNING:
                self._complete_execution_job(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    execution_version=expected_version,
                )
            if intent is not None:
                self._insert_execution_job(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    execution_version=next_version,
                    intent=intent,
                )
            commit_attempted = True
            connection.commit()
            return VersionedRun(normalized, next_version)
        except (ExecutionSuperseded, RunRepositoryError):
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
                    intent=intent,
                    execution_claim=execution_claim,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def claim_next_execution(
        self,
        *,
        worker_id: str,
        owner_token: str,
        lease_seconds: float,
    ) -> RunExecutionClaim | None:
        _validate_execution_identifier("worker_id", worker_id)
        _validate_execution_identifier("owner_token", owner_token)
        lease = _validate_execution_lease_seconds(lease_seconds)
        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            owner_lock_key = int.from_bytes(
                sha256(
                    f"puncture-execution-owner:{owner_token}".encode("utf-8")
                ).digest()[:8],
                byteorder="big",
                signed=True,
            )
            self._execute_discard(
                connection,
                "SELECT pg_advisory_xact_lock(%s::bigint)",
                (owner_lock_key,),
            )
            existing_identity = self._fetchone(
                connection,
                f"""
                SELECT
                    j.tenant_id,
                    j.run_id,
                    j.execution_version,
                    j.worker_id,
                    r.version,
                    r.status
                FROM {self._execution_jobs} AS j
                JOIN {self._runs} AS r
                  ON r.tenant_id = j.tenant_id AND r.run_id = j.run_id
                WHERE j.owner_token = %s
                FOR UPDATE OF j
                """,
                (owner_token,),
            )
            if existing_identity is not None:
                (
                    tenant_id,
                    run_id,
                    execution_version,
                    stored_worker_id,
                    current_version,
                    current_status,
                ) = existing_identity
                if (
                    stored_worker_id != worker_id
                    or current_version != execution_version
                    or current_status != RunStatus.RUNNING.value
                ):
                    raise RunRepositoryTransitionError(
                        "execution owner token is already in use"
                    )
                claim = self._load_execution_claim(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    execution_version=execution_version,
                )
                if not self._execution_claim_is_active(connection, claim):
                    raise RunRepositoryTransitionError(
                        "execution owner token is already in use"
                    )
                connection.commit()
                return claim

            candidate = self._fetchone(
                connection,
                f"""
                SELECT j.tenant_id, j.run_id, j.execution_version
                FROM {self._execution_jobs} AS j
                JOIN {self._runs} AS r
                  ON r.tenant_id = j.tenant_id AND r.run_id = j.run_id
                WHERE r.status = %s
                  AND r.version = j.execution_version
                  AND j.completed_at IS NULL
                  AND (
                      j.owner_token IS NULL
                      OR j.released_at IS NOT NULL
                      OR j.lease_expires_at <=
                          date_trunc('milliseconds', clock_timestamp())
                  )
                ORDER BY r.updated_at ASC, r.run_id ASC
                FOR UPDATE OF j SKIP LOCKED
                LIMIT 1
                """,
                (RunStatus.RUNNING.value,),
            )
            if candidate is None:
                connection.commit()
                return None
            tenant_id, run_id, execution_version = candidate
            updated = self._fetchone(
                connection,
                f"""
                WITH db_clock AS (
                    SELECT date_trunc(
                        'milliseconds', clock_timestamp()
                    ) AS now
                )
                UPDATE {self._execution_jobs} AS j
                SET generation = j.generation + 1,
                    owner_token = %s,
                    worker_id = %s,
                    claimed_at = db_clock.now,
                    heartbeat_at = db_clock.now,
                    lease_expires_at = date_trunc(
                        'milliseconds',
                        db_clock.now + (%s * interval '1 second')
                    ),
                    released_at = NULL
                FROM db_clock
                WHERE j.tenant_id = %s
                  AND j.run_id = %s
                  AND j.execution_version = %s
                RETURNING j.generation
                """,
                (
                    owner_token,
                    worker_id,
                    lease,
                    tenant_id,
                    run_id,
                    execution_version,
                ),
            )
            if updated is None:
                raise RunRepositoryIntegrityError()
            claim = self._load_execution_claim(
                connection,
                run_id=run_id,
                tenant_id=tenant_id,
                execution_version=execution_version,
            )
            commit_attempted = True
            connection.commit()
            return claim
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted:
                return self._reconcile_execution_claim(
                    worker_id=worker_id,
                    owner_token=owner_token,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def heartbeat_execution_claim(
        self,
        claim: RunExecutionClaim,
        *,
        lease_seconds: float,
    ) -> RunExecutionClaim:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        lease = _validate_execution_lease_seconds(lease_seconds)
        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            updated = self._fetchone(
                connection,
                f"""
                WITH db_clock AS (
                    SELECT date_trunc(
                        'milliseconds', clock_timestamp()
                    ) AS now
                )
                UPDATE {self._execution_jobs} AS j
                SET heartbeat_at = db_clock.now,
                    lease_expires_at = date_trunc(
                        'milliseconds',
                        db_clock.now + (%s * interval '1 second')
                    )
                FROM db_clock
                WHERE j.tenant_id = %s
                  AND j.run_id = %s
                  AND j.execution_version = %s
                  AND j.generation = %s
                  AND j.owner_token = %s
                  AND j.worker_id = %s
                  AND j.intent_fingerprint = %s
                  AND j.released_at IS NULL
                  AND j.completed_at IS NULL
                  AND j.lease_expires_at > db_clock.now
                  AND EXISTS (
                      SELECT 1
                      FROM {self._runs} AS r
                      WHERE r.tenant_id = j.tenant_id
                        AND r.run_id = j.run_id
                        AND r.version = j.execution_version
                        AND r.status = %s
                  )
                RETURNING j.generation
                """,
                (
                    lease,
                    claim.run.snapshot.request.tenant_id,
                    claim.run.snapshot.run_id,
                    claim.run.version,
                    claim.generation,
                    claim.owner_token,
                    claim.worker_id,
                    _execution_intent_fingerprint(claim.intent),
                    RunStatus.RUNNING.value,
                ),
            )
            if updated is None:
                self._raise_missing_or_superseded_claim(connection, claim)
            refreshed = self._load_execution_claim(
                connection,
                run_id=claim.run.snapshot.run_id,
                tenant_id=claim.run.snapshot.request.tenant_id,
                execution_version=claim.run.version,
            )
            commit_attempted = True
            connection.commit()
            return refreshed
        except (ExecutionSuperseded, RunRepositoryError):
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted:
                return self._reconcile_execution_claim(
                    worker_id=claim.worker_id,
                    owner_token=claim.owner_token,
                )
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def release_execution_claim(self, claim: RunExecutionClaim) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        connection: Any | None = None
        commit_attempted = False
        try:
            connection = self._connect()
            run_exists = self._fetchone(
                connection,
                f"""
                SELECT 1
                FROM {self._runs}
                WHERE tenant_id = %s AND run_id = %s
                """,
                (
                    claim.run.snapshot.request.tenant_id,
                    claim.run.snapshot.run_id,
                ),
            )
            if run_exists is None:
                raise RunRepositoryNotFound()
            self._execute_discard(
                connection,
                f"""
                WITH db_clock AS (
                    SELECT date_trunc(
                        'milliseconds', clock_timestamp()
                    ) AS now
                )
                UPDATE {self._execution_jobs} AS j
                SET released_at = COALESCE(j.released_at, db_clock.now),
                    lease_expires_at = LEAST(
                        j.lease_expires_at, db_clock.now
                    )
                FROM db_clock
                WHERE j.tenant_id = %s
                  AND j.run_id = %s
                  AND j.execution_version = %s
                  AND j.generation = %s
                  AND j.owner_token = %s
                  AND j.worker_id = %s
                  AND j.intent_fingerprint = %s
                """,
                (
                    claim.run.snapshot.request.tenant_id,
                    claim.run.snapshot.run_id,
                    claim.run.version,
                    claim.generation,
                    claim.owner_token,
                    claim.worker_id,
                    _execution_intent_fingerprint(claim.intent),
                ),
            )
            commit_attempted = True
            connection.commit()
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            if commit_attempted and self._execution_claim_was_released(claim):
                return
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def abandon_execution_claim(self, claim: RunExecutionClaim) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        # Deliberately preserve the lease. Shutdown grace expiry must stop
        # heartbeats and let PostgreSQL's TTL fence the old execution before a
        # replacement worker can claim it.
        return None

    def assert_execution_claim(self, claim: RunExecutionClaim) -> None:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        connection: Any | None = None
        try:
            connection = self._connect()
            self._require_active_execution_claim(connection, claim)
            connection.commit()
        except (ExecutionSuperseded, RunRepositoryError):
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def append_if_claimed(
        self,
        claim: RunExecutionClaim,
        *,
        event: RunEventDraft,
    ) -> RunEvent:
        if not isinstance(claim, RunExecutionClaim):
            raise TypeError("execution claim is invalid")
        normalized = InMemoryRunRepository._copy_event(event)
        expected_version = claim.run.version
        if normalized.event_type not in _RUNNING_STREAM_EVENTS:
            raise RunRepositoryTransitionError(
                "state events must be committed atomically with a state transition"
            )
        if normalized.event_key is None:
            raise RunRepositoryTransitionError(
                "running stream events require a stable event_key"
            )
        fingerprint = _event_fingerprint(normalized, expected_version)
        run_id = claim.run.snapshot.run_id
        tenant_id = claim.run.snapshot.request.tenant_id
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
                if not self._execution_claim_identity_matches(connection, claim):
                    raise ExecutionSuperseded("execution claim was superseded")
                connection.commit()
                return existing
            self._require_active_execution_claim(connection, claim)
            version, status, trace_id, last_sequence = run_row
            if version != expected_version or status != RunStatus.RUNNING.value:
                raise ExecutionSuperseded("execution claim is no longer active")
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
                return self._reconcile_claimed_append(
                    claim=claim,
                    event_key=normalized.event_key,
                    fingerprint=fingerprint,
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

    def _migration_versions(
        self,
    ) -> tuple[tuple[int, str, tuple[str, ...]], ...]:
        return (
            (1, "initial-run-event-store", self._migration_statements()),
            (2, "durable-run-execution-jobs", self._execution_job_statements()),
        )

    def _execution_job_statements(self) -> tuple[str, ...]:
        intent_values = ", ".join(
            f"'{intent_kind.value}'" for intent_kind in RunExecutionIntentKind
        )
        create_intent = RunExecutionIntent(RunExecutionIntentKind.CREATE)
        resume_intent = RunExecutionIntent(RunExecutionIntentKind.RESUME)
        create_canonical = _canonical_json(
            _execution_intent_document(create_intent)
        )
        resume_canonical = _canonical_json(
            _execution_intent_document(resume_intent)
        )
        create_fingerprint = _execution_intent_fingerprint(create_intent)
        resume_fingerprint = _execution_intent_fingerprint(resume_intent)
        return (
            f"""
            CREATE TABLE {self._execution_jobs} (
                tenant_id text COLLATE "C" NOT NULL,
                run_id text COLLATE "C" NOT NULL,
                execution_version bigint NOT NULL,
                intent_kind text NOT NULL,
                intent_fingerprint text NOT NULL,
                intent_canonical text NOT NULL,
                intent_json jsonb NOT NULL,
                generation bigint NOT NULL DEFAULT 0,
                owner_token text COLLATE "C",
                worker_id text COLLATE "C",
                claimed_at timestamptz,
                heartbeat_at timestamptz,
                lease_expires_at timestamptz,
                released_at timestamptz,
                completed_at timestamptz,
                created_at timestamptz NOT NULL DEFAULT
                    date_trunc('milliseconds', clock_timestamp()),
                CONSTRAINT pk_runtime_run_execution_jobs PRIMARY KEY (
                    tenant_id, run_id, execution_version
                ),
                CONSTRAINT uq_runtime_run_execution_jobs_owner
                    UNIQUE (owner_token),
                CONSTRAINT fk_runtime_run_execution_jobs_run
                    FOREIGN KEY (tenant_id, run_id)
                    REFERENCES {self._runs} (tenant_id, run_id)
                    ON DELETE RESTRICT,
                CONSTRAINT ck_runtime_run_execution_jobs_version
                    CHECK (execution_version >= 1),
                CONSTRAINT ck_runtime_run_execution_jobs_kind
                    CHECK (intent_kind IN ({intent_values})),
                CONSTRAINT ck_runtime_run_execution_jobs_intent_fingerprint
                    CHECK (intent_fingerprint ~ '^[0-9a-f]{{64}}$'),
                CONSTRAINT ck_runtime_run_execution_jobs_intent_object
                    CHECK (jsonb_typeof(intent_json) = 'object'),
                CONSTRAINT ck_runtime_run_execution_jobs_intent_canonical
                    CHECK (intent_json = intent_canonical::jsonb),
                CONSTRAINT ck_runtime_run_execution_jobs_intent_kind
                    CHECK (intent_json ->> 'kind' = intent_kind),
                CONSTRAINT ck_runtime_run_execution_jobs_generation
                    CHECK (generation >= 0),
                CONSTRAINT ck_runtime_run_execution_jobs_claim_shape CHECK (
                    (
                        generation = 0
                        AND owner_token IS NULL
                        AND worker_id IS NULL
                        AND claimed_at IS NULL
                        AND heartbeat_at IS NULL
                        AND lease_expires_at IS NULL
                        AND (
                            (
                                released_at IS NULL
                                AND completed_at IS NULL
                            ) OR (
                                released_at IS NOT NULL
                                AND completed_at IS NOT NULL
                            )
                        )
                    ) OR (
                        generation >= 1
                        AND owner_token IS NOT NULL
                        AND worker_id IS NOT NULL
                        AND claimed_at IS NOT NULL
                        AND heartbeat_at IS NOT NULL
                        AND lease_expires_at IS NOT NULL
                    )
                ),
                CONSTRAINT ck_runtime_run_execution_jobs_owner_token CHECK (
                    owner_token IS NULL OR (
                        btrim(owner_token) <> ''
                        AND owner_token = btrim(owner_token)
                        AND octet_length(owner_token) <= 128
                    )
                ),
                CONSTRAINT ck_runtime_run_execution_jobs_worker_id CHECK (
                    worker_id IS NULL OR (
                        btrim(worker_id) <> ''
                        AND worker_id = btrim(worker_id)
                        AND octet_length(worker_id) <= 128
                    )
                ),
                CONSTRAINT ck_runtime_run_execution_jobs_time_precision CHECK (
                    created_at = date_trunc('milliseconds', created_at)
                    AND (
                        claimed_at IS NULL OR claimed_at =
                            date_trunc('milliseconds', claimed_at)
                    )
                    AND (
                        heartbeat_at IS NULL OR heartbeat_at =
                            date_trunc('milliseconds', heartbeat_at)
                    )
                    AND (
                        lease_expires_at IS NULL OR lease_expires_at =
                            date_trunc('milliseconds', lease_expires_at)
                    )
                    AND (
                        released_at IS NULL OR released_at =
                            date_trunc('milliseconds', released_at)
                    )
                    AND (
                        completed_at IS NULL OR completed_at =
                            date_trunc('milliseconds', completed_at)
                    )
                ),
                CONSTRAINT ck_runtime_run_execution_jobs_time_order CHECK (
                    claimed_at IS NULL OR (
                        heartbeat_at >= claimed_at
                        AND lease_expires_at >= heartbeat_at
                        AND (released_at IS NULL OR released_at >= claimed_at)
                        AND (completed_at IS NULL OR completed_at >= claimed_at)
                    )
                ),
                CONSTRAINT ck_runtime_run_execution_jobs_completion CHECK (
                    completed_at IS NULL OR released_at IS NOT NULL
                )
            )
            """,
            f"""
            WITH db_clock AS (
                SELECT date_trunc('milliseconds', clock_timestamp()) AS now
            )
            INSERT INTO {self._execution_jobs} (
                tenant_id,
                run_id,
                execution_version,
                intent_kind,
                intent_fingerprint,
                intent_canonical,
                intent_json,
                released_at,
                completed_at
            )
            SELECT
                tenant_id,
                run_id,
                1,
                'CREATE',
                '{create_fingerprint}',
                '{create_canonical}',
                '{create_canonical}'::jsonb,
                CASE
                    WHEN status = 'RUNNING' AND version = 1 THEN NULL
                    ELSE db_clock.now
                END,
                CASE
                    WHEN status = 'RUNNING' AND version = 1 THEN NULL
                    ELSE db_clock.now
                END
            FROM {self._runs}
            CROSS JOIN db_clock
            ON CONFLICT DO NOTHING
            """,
            f"""
            INSERT INTO {self._execution_jobs} (
                tenant_id,
                run_id,
                execution_version,
                intent_kind,
                intent_fingerprint,
                intent_canonical,
                intent_json
            )
            SELECT
                tenant_id,
                run_id,
                version,
                'RESUME',
                '{resume_fingerprint}',
                '{resume_canonical}',
                '{resume_canonical}'::jsonb
            FROM {self._runs}
            WHERE status = 'RUNNING' AND version > 1
            ON CONFLICT DO NOTHING
            """,
            f"""
            CREATE INDEX "ix_runtime_run_execution_jobs_claim"
            ON {self._execution_jobs} (
                completed_at,
                released_at,
                lease_expires_at,
                tenant_id,
                run_id,
                execution_version
            )
            WHERE completed_at IS NULL
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

    def _insert_execution_job(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        execution_version: int,
        intent: RunExecutionIntent,
    ) -> None:
        canonical = _canonical_json(_execution_intent_document(intent))
        self._execute_discard(
            connection,
            f"""
            INSERT INTO {self._execution_jobs} (
                tenant_id,
                run_id,
                execution_version,
                intent_kind,
                intent_fingerprint,
                intent_canonical,
                intent_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                tenant_id,
                run_id,
                execution_version,
                intent.kind.value,
                _execution_intent_fingerprint(intent),
                canonical,
                canonical,
            ),
        )

    def _complete_execution_job(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        execution_version: int,
    ) -> None:
        row = self._fetchone(
            connection,
            f"""
            WITH db_clock AS (
                SELECT date_trunc('milliseconds', clock_timestamp()) AS now
            )
            UPDATE {self._execution_jobs} AS j
            SET released_at = COALESCE(j.released_at, db_clock.now),
                completed_at = COALESCE(j.completed_at, db_clock.now)
            FROM db_clock
            WHERE j.tenant_id = %s
              AND j.run_id = %s
              AND j.execution_version = %s
            RETURNING j.execution_version
            """,
            (tenant_id, run_id, execution_version),
        )
        if row != (execution_version,):
            raise RunRepositoryIntegrityError()

    def _execution_job_intent_matches(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        execution_version: int,
        intent: RunExecutionIntent,
    ) -> bool:
        row = self._fetchone(
            connection,
            f"""
            SELECT intent_fingerprint, intent_canonical
            FROM {self._execution_jobs}
            WHERE tenant_id = %s
              AND run_id = %s
              AND execution_version = %s
            """,
            (tenant_id, run_id, execution_version),
        )
        expected_fingerprint = _execution_intent_fingerprint(intent)
        expected_canonical = _canonical_json(_execution_intent_document(intent))
        return row == (expected_fingerprint, expected_canonical)

    def _decode_execution_job(
        self,
        row: tuple[Any, ...],
    ) -> _DecodedExecutionJob:
        if len(row) != 13:
            raise RunRepositoryIntegrityError()
        (
            tenant_id,
            run_id,
            execution_version,
            intent_fingerprint,
            intent_canonical,
            generation,
            owner_token,
            worker_id,
            claimed_at,
            heartbeat_at,
            lease_expires_at,
            released_at,
            completed_at,
        ) = row
        try:
            if (
                not isinstance(tenant_id, str)
                or not isinstance(run_id, str)
                or isinstance(execution_version, bool)
                or not isinstance(execution_version, int)
                or execution_version < 1
                or isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise ValueError("execution job identity is invalid")
            intent = _execution_intent_from_canonical(
                intent_canonical,
                fingerprint=intent_fingerprint,
            )
            claim_values = (
                owner_token,
                worker_id,
                claimed_at,
                heartbeat_at,
                lease_expires_at,
            )
            if generation == 0:
                if any(value is not None for value in claim_values):
                    raise ValueError("unclaimed execution job has claim fields")
            else:
                if any(value is None for value in claim_values):
                    raise ValueError("claimed execution job is incomplete")
                _validate_execution_identifier("owner_token", owner_token)
                _validate_execution_identifier("worker_id", worker_id)
            for field_name, value in (
                ("claimed_at", claimed_at),
                ("heartbeat_at", heartbeat_at),
                ("lease_expires_at", lease_expires_at),
                ("released_at", released_at),
                ("completed_at", completed_at),
            ):
                if value is not None:
                    _canonical_utc_millisecond(value, field_name=field_name)
            return _DecodedExecutionJob(
                tenant_id=tenant_id,
                run_id=run_id,
                execution_version=execution_version,
                intent=intent,
                generation=generation,
                owner_token=owner_token,
                worker_id=worker_id,
                claimed_at=claimed_at,
                heartbeat_at=heartbeat_at,
                lease_expires_at=lease_expires_at,
                released_at=released_at,
                completed_at=completed_at,
            )
        except (TypeError, ValueError, RuntimeJsonBoundaryError) as exc:
            raise RunRepositoryIntegrityError() from exc

    def _load_execution_claim(
        self,
        connection: Any,
        *,
        run_id: str,
        tenant_id: str,
        execution_version: int,
    ) -> RunExecutionClaim:
        run_row = self._fetchone(
            connection,
            self._run_select + " WHERE run_id = %s AND tenant_id = %s",
            (run_id, tenant_id),
        )
        if run_row is None:
            raise RunRepositoryNotFound()
        run, _, _ = self._decode_run(run_row)
        job_row = self._fetchone(
            connection,
            self._execution_job_select
            + f"""
              FROM {self._execution_jobs}
              WHERE tenant_id = %s
                AND run_id = %s
                AND execution_version = %s
            """,
            (tenant_id, run_id, execution_version),
        )
        if job_row is None:
            raise RunRepositoryIntegrityError()
        job = self._decode_execution_job(job_row)
        if (
            run.version != execution_version
            or run.snapshot.status is not RunStatus.RUNNING
            or job.generation < 1
            or job.owner_token is None
            or job.worker_id is None
            or job.claimed_at is None
            or job.heartbeat_at is None
            or job.lease_expires_at is None
        ):
            raise ExecutionSuperseded("execution claim is no longer active")
        return RunExecutionClaim(
            run=run,
            intent=job.intent,
            generation=job.generation,
            owner_token=job.owner_token,
            worker_id=job.worker_id,
            claimed_at=job.claimed_at,
            heartbeat_at=job.heartbeat_at,
            lease_expires_at=job.lease_expires_at,
        )

    def _execution_claim_identity_matches(
        self,
        connection: Any,
        claim: RunExecutionClaim,
    ) -> bool:
        row = self._fetchone(
            connection,
            self._execution_job_select
            + f"""
              FROM {self._execution_jobs}
              WHERE tenant_id = %s
                AND run_id = %s
                AND execution_version = %s
            """,
            (
                claim.run.snapshot.request.tenant_id,
                claim.run.snapshot.run_id,
                claim.run.version,
            ),
        )
        if row is None:
            return False
        job = self._decode_execution_job(row)
        return bool(
            job.generation == claim.generation
            and job.owner_token == claim.owner_token
            and job.worker_id == claim.worker_id
            and _execution_intent_fingerprint(job.intent)
            == _execution_intent_fingerprint(claim.intent)
        )

    def _execution_claim_is_active(
        self,
        connection: Any,
        claim: RunExecutionClaim,
    ) -> bool:
        row = self._fetchone(
            connection,
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM {self._execution_jobs} AS j
                JOIN {self._runs} AS r
                  ON r.tenant_id = j.tenant_id AND r.run_id = j.run_id
                WHERE j.tenant_id = %s
                  AND j.run_id = %s
                  AND j.execution_version = %s
                  AND j.generation = %s
                  AND j.owner_token = %s
                  AND j.worker_id = %s
                  AND j.intent_fingerprint = %s
                  AND j.released_at IS NULL
                  AND j.completed_at IS NULL
                  AND j.lease_expires_at >
                      date_trunc('milliseconds', clock_timestamp())
                  AND r.version = j.execution_version
                  AND r.status = %s
            )
            """,
            (
                claim.run.snapshot.request.tenant_id,
                claim.run.snapshot.run_id,
                claim.run.version,
                claim.generation,
                claim.owner_token,
                claim.worker_id,
                _execution_intent_fingerprint(claim.intent),
                RunStatus.RUNNING.value,
            ),
        )
        return row == (True,)

    def _require_active_execution_claim(
        self,
        connection: Any,
        claim: RunExecutionClaim,
    ) -> None:
        row = self._fetchone(
            connection,
            f"""
            SELECT j.generation
            FROM {self._execution_jobs} AS j
            JOIN {self._runs} AS r
              ON r.tenant_id = j.tenant_id AND r.run_id = j.run_id
            WHERE j.tenant_id = %s
              AND j.run_id = %s
              AND j.execution_version = %s
              AND j.generation = %s
              AND j.owner_token = %s
              AND j.worker_id = %s
              AND j.intent_fingerprint = %s
              AND j.released_at IS NULL
              AND j.completed_at IS NULL
              AND j.lease_expires_at >
                  date_trunc('milliseconds', clock_timestamp())
              AND r.version = j.execution_version
              AND r.status = %s
            FOR UPDATE OF j
            """,
            (
                claim.run.snapshot.request.tenant_id,
                claim.run.snapshot.run_id,
                claim.run.version,
                claim.generation,
                claim.owner_token,
                claim.worker_id,
                _execution_intent_fingerprint(claim.intent),
                RunStatus.RUNNING.value,
            ),
        )
        if row is None:
            self._raise_missing_or_superseded_claim(connection, claim)

    def _raise_missing_or_superseded_claim(
        self,
        connection: Any,
        claim: RunExecutionClaim,
    ) -> None:
        row = self._fetchone(
            connection,
            f"""
            SELECT 1
            FROM {self._runs}
            WHERE tenant_id = %s AND run_id = %s
            """,
            (
                claim.run.snapshot.request.tenant_id,
                claim.run.snapshot.run_id,
            ),
        )
        if row is None:
            raise RunRepositoryNotFound()
        raise ExecutionSuperseded("execution claim is no longer active")

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
            if not self._execution_job_intent_matches(
                connection,
                run_id=run.snapshot.run_id,
                tenant_id=run.snapshot.request.tenant_id,
                execution_version=1,
                intent=RunExecutionIntent(RunExecutionIntentKind.CREATE),
            ):
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

    def _reconcile_execution_claim(
        self,
        *,
        worker_id: str,
        owner_token: str,
    ) -> RunExecutionClaim:
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                f"""
                SELECT tenant_id, run_id, execution_version, worker_id
                FROM {self._execution_jobs}
                WHERE owner_token = %s
                """,
                (owner_token,),
            )
            if row is None or row[3] != worker_id:
                raise RunRepositoryUnavailable()
            claim = self._load_execution_claim(
                connection,
                run_id=row[1],
                tenant_id=row[0],
                execution_version=row[2],
            )
            if not self._execution_claim_is_active(connection, claim):
                raise RunRepositoryUnavailable()
            connection.commit()
            return claim
        except ExecutionSuperseded as exc:
            self._rollback(connection)
            raise RunRepositoryUnavailable() from exc
        except RunRepositoryError:
            self._rollback(connection)
            raise
        except Exception as exc:
            self._rollback(connection)
            raise self._map_exception(exc) from exc
        finally:
            self._close(connection)

    def _execution_claim_was_released(self, claim: RunExecutionClaim) -> bool:
        connection: Any | None = None
        try:
            connection = self._connect()
            row = self._fetchone(
                connection,
                f"""
                SELECT released_at IS NOT NULL
                FROM {self._execution_jobs}
                WHERE tenant_id = %s
                  AND run_id = %s
                  AND execution_version = %s
                  AND generation = %s
                  AND owner_token = %s
                  AND worker_id = %s
                  AND intent_fingerprint = %s
                """,
                (
                    claim.run.snapshot.request.tenant_id,
                    claim.run.snapshot.run_id,
                    claim.run.version,
                    claim.generation,
                    claim.owner_token,
                    claim.worker_id,
                    _execution_intent_fingerprint(claim.intent),
                ),
            )
            connection.commit()
            return row == (True,)
        except Exception:
            self._rollback(connection)
            return False
        finally:
            self._close(connection)

    def _reconcile_claimed_append(
        self,
        *,
        claim: RunExecutionClaim,
        event_key: str,
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
                (
                    claim.run.snapshot.request.tenant_id,
                    claim.run.snapshot.run_id,
                    event_key,
                ),
            )
            if row is None:
                raise RunRepositoryUnavailable()
            event, stored_key, stored_version, stored_fingerprint = (
                self._decode_event(row)
            )
            if (
                stored_key != event_key
                or stored_version != claim.run.version
                or stored_fingerprint != fingerprint
            ):
                raise RunRepositoryEventConflict()
            if not self._execution_claim_identity_matches(connection, claim):
                raise ExecutionSuperseded("execution claim was superseded")
            connection.commit()
            return event
        except (ExecutionSuperseded, RunRepositoryError):
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
        intent: RunExecutionIntent | None,
        execution_claim: RunExecutionClaim | None,
    ) -> VersionedRun:
        connection: Any | None = None
        try:
            connection = self._connect()
            replay_matches = self._cas_replay_matches(
                connection,
                run_id=run_id,
                tenant_id=tenant_id,
                target_fingerprint=target_fingerprint,
                transition_fingerprint=transition_fingerprint,
                expected_version=expected_version,
            )
            if replay_matches and intent is not None:
                replay_matches = self._execution_job_intent_matches(
                    connection,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    execution_version=expected_version + 1,
                    intent=intent,
                )
            if replay_matches and execution_claim is not None:
                replay_matches = self._execution_claim_identity_matches(
                    connection,
                    execution_claim,
                )
            if not replay_matches:
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
