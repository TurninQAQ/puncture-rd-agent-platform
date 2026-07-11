"""Process-level FastAPI SIGTERM and durable worker-reclaim probe.

CI invokes ``orchestrate``.  Process A creates a Run through the real
PostgreSQL FastAPI composition, commits one idempotent injected-port side
effect, and then deliberately ignores cooperative shutdown.  The orchestrator
sends SIGTERM, allowing TestClient's lifespan shutdown to stop heartbeats after
the configured grace period.  Process B starts the same composition and
reclaims the expired durable job without duplicating the side effect.

The executor below is a connectivity fixture for an injected company-algorithm
port.  It does not implement any company algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from threading import Event
import time
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from puncture_agent.api.fastapi_app import (  # noqa: E402
    ApiPermission,
    AuthorizedCase,
)
from puncture_agent.api.http_contracts import AuthenticatedPrincipal  # noqa: E402
from puncture_agent.api.postgres_app import (  # noqa: E402
    PostgresApiSettings,
    create_postgres_app,
)
from puncture_agent.runtime.models import (  # noqa: E402
    EventType,
    ExecutionOutcome,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from puncture_agent.runtime.service import (  # noqa: E402
    Emit,
    RunExecutionContext,
)
from puncture_agent.runtime.worker import WorkerState  # noqa: E402


DSN_ENV = "PUNCTURE_TEST_POSTGRES_DSN"
EVIDENCE_DIR_ENV = "PUNCTURE_API_SIGTERM_EVIDENCE_DIR"
SCHEMA_ENV = "PUNCTURE_API_SIGTERM_SCHEMA"
RUN_ID_ENV = "PUNCTURE_API_SIGTERM_RUN_ID"
MODULE = "tests.api.postgres_api_sigterm_probe"

TENANT_ID = "tenant-recovery"
PRINCIPAL_ID = "principal-recovery"
CASE_ID = "Case-Sigterm-Recovery"
TOKEN = "sigterm-probe-token"
AUTH_HEADERS = {"Authorization": f"Bearer {TOKEN}"}
LEDGER_TABLE = "side_effect_ledger"
PORT_NAME = "company-algorithm-port"

POLL_SECONDS = 0.02
HEARTBEAT_SECONDS = 0.10
LEASE_SECONDS = 0.60
SHUTDOWN_GRACE_SECONDS = 0.15
CHILD_TIMEOUT_SECONDS = 30.0
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class ProbeFailure(RuntimeError):
    """A failed API lifecycle/reclaim invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProbeFailure(f"required environment variable {name} is missing")
    return value


def evidence_directory() -> Path:
    directory = Path(require_env(EVIDENCE_DIR_ENV)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def schema_name() -> str:
    schema = require_env(SCHEMA_ENV)
    if not _IDENTIFIER.fullmatch(schema):
        raise ProbeFailure("probe schema is not a safe PostgreSQL identifier")
    return schema


def qualified_table(schema: str, table: str) -> str:
    if not _IDENTIFIER.fullmatch(schema) or not _IDENTIFIER.fullmatch(table):
        raise ProbeFailure("unsafe PostgreSQL identifier")
    return f'"{schema}"."{table}"'


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeFailure(f"failed to read evidence file {path.name}") from exc
    if not isinstance(value, dict):
        raise ProbeFailure(f"evidence file {path.name} must contain an object")
    return value


def wait_for_json(
    path: Path,
    *,
    timeout_seconds: float,
    process: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return read_json(path)
        if process is not None and process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            detail = stderr.strip() or stdout.strip() or str(process.returncode)
            raise ProbeFailure(f"child exited before {path.name}: {detail}")
        time.sleep(POLL_SECONDS)
    raise ProbeFailure(f"timed out waiting for {path.name}")


def psycopg_module() -> Any:
    try:
        import psycopg
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProbeFailure("psycopg is required for the SIGTERM probe") from exc
    return psycopg


class ProbeAuthenticator:
    def authenticate(self, bearer_token: str) -> AuthenticatedPrincipal:
        if bearer_token != TOKEN:
            raise ProbeFailure("probe received an unexpected bearer token")
        return AuthenticatedPrincipal(TENANT_ID, PRINCIPAL_ID)


class ProbeAuthorizer:
    @staticmethod
    def _case(case_id: str) -> AuthorizedCase:
        if case_id != CASE_ID:
            raise ProbeFailure("probe received an unexpected case")
        return AuthorizedCase(
            tenant_id=TENANT_ID,
            project_id="project-recovery",
            case_id=CASE_ID,
        )

    def require_case(
        self,
        principal: AuthenticatedPrincipal,
        *,
        case_id: str,
        permission: ApiPermission,
    ) -> AuthorizedCase:
        del permission
        require(principal.tenant_id == TENANT_ID, "unexpected principal tenant")
        return self._case(case_id)

    def require_run(
        self,
        principal: AuthenticatedPrincipal,
        *,
        snapshot: RunSnapshot,
        permission: ApiPermission,
    ) -> AuthorizedCase:
        del permission
        require(principal.tenant_id == TENANT_ID, "unexpected principal tenant")
        return self._case(snapshot.request.case_id)


class InjectedPortExecutor:
    """Recovery-safe connectivity fixture with a PostgreSQL replay ledger."""

    recovery_safe = True

    def __init__(
        self,
        dsn: str,
        schema: str,
        directory: Path,
        *,
        block_after_effect: bool,
    ) -> None:
        self._dsn = dsn
        self._ledger = qualified_table(schema, LEDGER_TABLE)
        self._directory = directory
        self._block_after_effect = block_after_effect

    def execute_claimed(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        context: RunExecutionContext,
        checkpoint: Mapping[str, Any] | None = None,
        approval: Any = None,
    ) -> ExecutionOutcome:
        del checkpoint, approval
        context.assert_active()
        call_id = f"{context.run_id}:{PORT_NAME}:v1"
        result = {
            "accepted": True,
            "case_id": request.case_id,
            "port": PORT_NAME,
        }
        emit(
            EventType.TOOL_CALLED,
            PORT_NAME,
            {"call_id": call_id, "port": PORT_NAME},
            event_key="company-port-call-v1",
        )

        psycopg = psycopg_module()
        with psycopg.connect(self._dsn, autocommit=True) as connection:
            inserted = connection.execute(
                f"""
                INSERT INTO {self._ledger} (
                    call_id, run_id, result_json, first_generation
                )
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (call_id) DO NOTHING
                RETURNING call_id
                """,
                (
                    call_id,
                    context.run_id,
                    canonical_json_bytes(result).decode("utf-8"),
                    context.generation,
                ),
            ).fetchone()
            stored = connection.execute(
                f"""
                SELECT run_id, result_json, first_generation
                FROM {self._ledger}
                WHERE call_id = %s
                """,
                (call_id,),
            ).fetchone()
        require(stored is not None, "side-effect ledger row was not durable")
        stored_run_id, stored_result, first_generation = stored
        require(stored_run_id == context.run_id, "stable call_id changed Run identity")
        require(stored_result == result, "stable call_id changed its durable result")

        marker = {
            "call_id": call_id,
            "first_generation": int(first_generation),
            "generation": context.generation,
            "inserted": inserted is not None,
            "phase": "process-a" if self._block_after_effect else "process-b",
            "pid": os.getpid(),
            "recovering": context.recovering,
            "result_sha256": sha256_value(result),
            "run_id": context.run_id,
            "schema_version": "1",
            "trace_id": context.trace_id,
        }
        marker_name = (
            "a_effect_committed.json"
            if self._block_after_effect
            else "b_effect_replayed.json"
        )
        write_json(self._directory / marker_name, marker)

        if self._block_after_effect:
            # Deliberately model a non-cooperative external/company port.  The
            # API lifespan must finish after grace without releasing the lease;
            # process exit removes this daemon execution thread.
            Event().wait(120.0)
            raise ProbeFailure("process A executor unexpectedly resumed")

        emit(
            EventType.TOOL_RESULT,
            PORT_NAME,
            {"call_id": call_id, "result": result},
            event_key="company-port-result-v1",
        )
        context.assert_active()
        return ExecutionOutcome(
            status=RunStatus.SUCCEEDED,
            final_report={
                "call_id": call_id,
                "connectivity": "verified",
                "port": PORT_NAME,
            },
        )


def settings(dsn: str, schema: str, worker_id: str) -> PostgresApiSettings:
    return PostgresApiSettings(
        connection_string=dsn,
        schema=schema,
        migrate_on_startup=True,
        worker_id=worker_id,
        worker_poll_interval_seconds=POLL_SECONDS,
        worker_heartbeat_interval_seconds=HEARTBEAT_SECONDS,
        worker_lease_seconds=LEASE_SECONDS,
        worker_shutdown_grace_seconds=SHUTDOWN_GRACE_SECONDS,
    )


def create_body() -> dict[str, Any]:
    return {
        "case_id": CASE_ID,
        "user_query": "verify the injected company algorithm port",
        "task_type": "DATA_MODEL_VALIDATION",
        "idempotency_key": "sigterm-recovery-create-v1",
        "artifact_ids": [],
        "metadata": {},
    }


def create_ledger(dsn: str, schema: str) -> None:
    psycopg = psycopg_module()
    table = qualified_table(schema, LEDGER_TABLE)
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                call_id text PRIMARY KEY,
                run_id text NOT NULL,
                result_json jsonb NOT NULL,
                first_generation bigint NOT NULL CHECK (first_generation >= 1),
                committed_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
            """
        )


def process_a() -> int:
    from fastapi.testclient import TestClient

    dsn = require_env(DSN_ENV)
    schema = schema_name()
    directory = evidence_directory()
    termination_requested = Event()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_shutdown(signum: int, frame: Any) -> None:
        del frame
        if signum == signal.SIGTERM:
            termination_requested.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    executor = InjectedPortExecutor(
        dsn,
        schema,
        directory,
        block_after_effect=True,
    )
    app = create_postgres_app(
        settings(dsn, schema, "api-sigterm-process-a"),
        executor=executor,
        authenticator=ProbeAuthenticator(),
        authorizer=ProbeAuthorizer(),
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            create_ledger(dsn, schema)
            response = client.post(
                "/api/v1/runs",
                headers=AUTH_HEADERS,
                json=create_body(),
            )
            require(response.status_code == 200, f"Run create failed: {response.text}")
            created = response.json()
            require(created.get("status") == RunStatus.RUNNING.value, "Run was not deferred")
            effect = wait_for_json(
                directory / "a_effect_committed.json",
                timeout_seconds=CHILD_TIMEOUT_SECONDS,
            )
            require(effect.get("run_id") == created.get("run_id"), "executor used a different Run")
            write_json(
                directory / "a_ready.json",
                {
                    "call_id": effect["call_id"],
                    "generation": effect["generation"],
                    "http_status": response.status_code,
                    "phase": "process-a-ready",
                    "pid": os.getpid(),
                    "run_id": created["run_id"],
                    "schema_version": "1",
                    "status": created["status"],
                    "trace_id": created["trace_id"],
                },
            )
            require(
                termination_requested.wait(CHILD_TIMEOUT_SECONDS),
                "process A did not receive SIGTERM",
            )
        worker = app.state.run_worker
        worker_status = worker.status
        metrics = worker.metrics.snapshot()
        write_json(
            directory / "a_shutdown.json",
            {
                "active_executions": worker_status.active_executions,
                "phase": "process-a-shutdown",
                "pid": os.getpid(),
                "schema_version": "1",
                "shutdown_timeouts_total": metrics.shutdown_timeouts_total,
                "sigterm_observed": termination_requested.is_set(),
                "worker_state": worker_status.state.value,
            },
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


def process_b() -> int:
    from fastapi.testclient import TestClient

    dsn = require_env(DSN_ENV)
    schema = schema_name()
    run_id = require_env(RUN_ID_ENV)
    directory = evidence_directory()
    executor = InjectedPortExecutor(
        dsn,
        schema,
        directory,
        block_after_effect=False,
    )
    app = create_postgres_app(
        settings(dsn, schema, "api-sigterm-process-b"),
        executor=executor,
        authenticator=ProbeAuthenticator(),
        authorizer=ProbeAuthorizer(),
    )
    started = time.monotonic()
    with TestClient(app, raise_server_exceptions=False) as client:
        deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
        terminal: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            response = client.get(
                f"/api/v1/runs/{run_id}",
                headers=AUTH_HEADERS,
            )
            require(response.status_code == 200, f"Run read failed: {response.text}")
            snapshot = response.json()
            if snapshot.get("status") in {
                RunStatus.SUCCEEDED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                terminal = snapshot
                break
            time.sleep(POLL_SECONDS)
        require(terminal is not None, "process B did not reach a terminal Run")
        events_response = client.get(
            f"/api/v1/runs/{run_id}/events?after_sequence=0&limit=512",
            headers=AUTH_HEADERS,
        )
        require(events_response.status_code == 200, "event replay request failed")
        events = events_response.json()
        require(isinstance(events, list), "event replay did not return a list")
        effect = wait_for_json(
            directory / "b_effect_replayed.json",
            timeout_seconds=CHILD_TIMEOUT_SECONDS,
        )
    worker_status = app.state.run_worker.status
    write_json(
        directory / "b_result.json",
        {
            "call_id": effect["call_id"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "events": events,
            "generation": effect["generation"],
            "phase": "process-b-result",
            "pid": os.getpid(),
            "recovering": effect["recovering"],
            "run_id": terminal["run_id"],
            "schema_version": "1",
            "status": terminal["status"],
            "trace_id": terminal["trace_id"],
            "worker_active_executions": worker_status.active_executions,
            "worker_state": worker_status.state.value,
        },
    )
    return 0


def child_environment(dsn: str, schema: str, directory: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment[DSN_ENV] = dsn
    environment[EVIDENCE_DIR_ENV] = str(directory)
    environment[SCHEMA_ENV] = schema
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def child_command(mode: str) -> list[str]:
    return [sys.executable, "-m", MODULE, mode]


def stop_child(process: subprocess.Popen[str]) -> tuple[str, str]:
    process.terminate()
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        raise ProbeFailure(
            "process A did not complete graceful lifespan shutdown: "
            + (stderr.strip() or stdout.strip() or "no child output")
        )


def run_process_b(
    dsn: str,
    schema: str,
    directory: Path,
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    environment = child_environment(dsn, schema, directory)
    environment[RUN_ID_ENV] = run_id
    return subprocess.run(
        child_command("process-b"),
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS + 10,
        check=False,
    )


def final_database_state(dsn: str, schema: str, run_id: str) -> dict[str, Any]:
    psycopg = psycopg_module()
    jobs = qualified_table(schema, "run_execution_jobs")
    runs = qualified_table(schema, "runs")
    ledger = qualified_table(schema, LEDGER_TABLE)
    with psycopg.connect(dsn, autocommit=True) as connection:
        job = connection.execute(
            f"""
            SELECT generation, completed_at IS NOT NULL, released_at IS NOT NULL
            FROM {jobs}
            WHERE tenant_id = %s AND run_id = %s
            """,
            (TENANT_ID, run_id),
        ).fetchone()
        run = connection.execute(
            f"""
            SELECT status, version
            FROM {runs}
            WHERE tenant_id = %s AND run_id = %s
            """,
            (TENANT_ID, run_id),
        ).fetchone()
        effect_count = connection.execute(
            f"SELECT count(*) FROM {ledger} WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        server_version = connection.execute(
            "SELECT current_setting('server_version_num')::integer"
        ).fetchone()
    require(job is not None, "durable execution job is missing")
    require(run is not None, "durable Run is missing")
    require(effect_count is not None, "side-effect count query failed")
    require(server_version is not None, "PostgreSQL version query failed")
    return {
        "effect_count": int(effect_count[0]),
        "generation": int(job[0]),
        "job_completed": bool(job[1]),
        "job_released": bool(job[2]),
        "run_status": str(run[0]),
        "run_version": int(run[1]),
        "server_version_num": int(server_version[0]),
    }


def drop_schema(dsn: str, schema: str) -> None:
    psycopg = psycopg_module()
    if not _IDENTIFIER.fullmatch(schema):
        raise ProbeFailure("refusing to drop an unsafe schema name")
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def orchestrate() -> int:
    require(os.name == "posix", "SIGTERM recovery probe requires POSIX signals")
    dsn = require_env(DSN_ENV)
    directory = evidence_directory()
    schema = f"api_sigterm_{uuid4().hex[:24]}"
    for name in (
        "a_effect_committed.json",
        "a_ready.json",
        "a_shutdown.json",
        "b_effect_replayed.json",
        "b_result.json",
        "orchestrator.json",
    ):
        (directory / name).unlink(missing_ok=True)

    environment = child_environment(dsn, schema, directory)
    process_a_child = subprocess.Popen(
        child_command("process-a"),
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    schema_cleaned = False
    try:
        a_ready = wait_for_json(
            directory / "a_ready.json",
            timeout_seconds=CHILD_TIMEOUT_SECONDS,
            process=process_a_child,
        )
        stdout_a, stderr_a = stop_child(process_a_child)
        require(
            process_a_child.returncode == 0,
            "process A failed after SIGTERM: "
            + (stderr_a.strip() or stdout_a.strip() or str(process_a_child.returncode)),
        )
        a_effect = read_json(directory / "a_effect_committed.json")
        a_shutdown = read_json(directory / "a_shutdown.json")
        require(a_shutdown.get("sigterm_observed") is True, "SIGTERM was not observed")
        require(
            a_shutdown.get("worker_state") == WorkerState.STOPPED.value,
            "process A worker did not stop through API lifespan",
        )
        require(
            int(a_shutdown.get("shutdown_timeouts_total", 0)) >= 1,
            "process A did not exercise the over-grace shutdown path",
        )

        run_id = str(a_ready.get("run_id", ""))
        require(run_id and run_id == a_effect.get("run_id"), "process A Run identity changed")
        process_b_child = run_process_b(dsn, schema, directory, run_id)
        require(
            process_b_child.returncode == 0,
            "process B failed: "
            + (
                process_b_child.stderr.strip()
                or process_b_child.stdout.strip()
                or str(process_b_child.returncode)
            ),
        )
        b_effect = read_json(directory / "b_effect_replayed.json")
        b_result = read_json(directory / "b_result.json")
        database = final_database_state(dsn, schema, run_id)

        require(a_effect.get("inserted") is True, "process A did not commit the side effect")
        require(b_effect.get("inserted") is False, "process B duplicated the side effect")
        require(a_effect.get("call_id") == b_effect.get("call_id"), "stable call_id changed")
        require(a_effect.get("trace_id") == b_result.get("trace_id"), "trace_id changed")
        require(b_result.get("run_id") == run_id, "process B recovered a different Run")
        require(b_result.get("status") == RunStatus.SUCCEEDED.value, "Run did not succeed")
        require(b_result.get("recovering") is True, "process B was not marked recovering")
        generation_a = int(a_effect.get("generation", 0))
        generation_b = int(b_effect.get("generation", 0))
        require(generation_a >= 1, "process A claim generation is invalid")
        require(generation_b >= 2 and generation_b > generation_a, "job was not reclaimed")
        require(database["generation"] == generation_b, "database generation changed")
        require(database["effect_count"] == 1, "side effect was duplicated or omitted")
        require(
            database["run_status"] == RunStatus.SUCCEEDED.value,
            "database Run is not terminal",
        )
        require(database["job_completed"] is True, "execution job was not completed")
        require(database["job_released"] is True, "execution job was not released")
        require(
            b_result.get("worker_state") == WorkerState.STOPPED.value,
            "process B worker did not stop through API lifespan",
        )
        require(
            int(b_result.get("worker_active_executions", -1)) == 0,
            "process B retained an active execution after shutdown",
        )

        events = b_result.get("events")
        require(isinstance(events, list) and events, "recovered Run has no events")
        sequences = [int(event.get("sequence", 0)) for event in events]
        require(sequences == list(range(1, len(events) + 1)), "event sequence is not contiguous")
        terminal_types = {
            EventType.RUN_COMPLETED.value,
            EventType.RUN_FAILED.value,
            EventType.RUN_CANCELLED.value,
        }
        terminal_events = [
            event for event in events if event.get("event_type") in terminal_types
        ]
        require(len(terminal_events) == 1, "Run does not have exactly one terminal event")
        require(
            terminal_events[0].get("event_type") == EventType.RUN_COMPLETED.value,
            "recovered Run has the wrong terminal event",
        )
        require(
            sum(event.get("event_type") == EventType.TOOL_CALLED.value for event in events) == 1,
            "replayed TOOL_CALLED event was duplicated",
        )
        require(
            sum(event.get("event_type") == EventType.TOOL_RESULT.value for event in events) == 1,
            "TOOL_RESULT event count is incorrect",
        )

        evidence = {
            "assertions": {
                "api_lifespan_stopped_both_workers": True,
                "event_sequence_contiguous": True,
                "injected_port_only_no_company_algorithm": True,
                "same_run_reclaimed_after_sigterm": True,
                "side_effect_exactly_once_via_stable_call_id": True,
                "single_terminal_event": True,
            },
            "event_count": len(events),
            "generation": {"process_a": generation_a, "process_b": generation_b},
            "phase": "orchestrate",
            "processes": {"process_a": a_effect["pid"], "process_b": b_effect["pid"]},
            "run_id": run_id,
            "schema_version": "1",
            "side_effect_count": database["effect_count"],
            "status": database["run_status"],
            "storage": {
                "job_completed": database["job_completed"],
                "job_released": database["job_released"],
                "postgres_server_version_num": database["server_version_num"],
                "run_version": database["run_version"],
            },
            "terminal_event_type": terminal_events[0]["event_type"],
        }
        encoded_evidence = canonical_json_bytes(evidence).decode("utf-8")
        require(dsn not in encoded_evidence, "evidence contains the PostgreSQL DSN")
        drop_schema(dsn, schema)
        schema_cleaned = True
        evidence["schema_cleaned"] = True
        write_json(directory / "orchestrator.json", evidence)
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if process_a_child.poll() is None:
            process_a_child.kill()
            process_a_child.communicate(timeout=5)
        if not schema_cleaned:
            try:
                drop_schema(dsn, schema)
            except Exception:
                pass


def main(arguments: list[str] | None = None) -> int:
    if arguments is None:
        arguments = sys.argv[1:]
    modes = {
        "orchestrate": orchestrate,
        "process-a": process_a,
        "process-b": process_b,
    }
    if len(arguments) != 1 or arguments[0] not in modes:
        print(
            "usage: postgres_api_sigterm_probe.py orchestrate|process-a|process-b",
            file=sys.stderr,
        )
        return 2
    dsn = os.environ.get(DSN_ENV, "")
    try:
        return modes[arguments[0]]()
    except Exception as exc:
        message = str(exc)
        if dsn:
            message = message.replace(dsn, "<redacted-postgres-dsn>")
        print(
            f"{arguments[0]} failed: {type(exc).__name__}: {message}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
