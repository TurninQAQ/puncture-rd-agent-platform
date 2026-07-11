"""Two-process PostgreSQL service-restart recovery probe for GitHub Actions.

This module is intentionally not named ``test_*.py``.  CI invokes ``prepare``
in one Python process, restarts the PostgreSQL service container, then invokes
``recover`` in a fresh process.  The JSON evidence contains no connection
string or storage credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    LangGraphRuntime,
    PostgresAdvisoryThreadExecutionLeaseManager,
    build_mock_handlers,
    langgraph_available,
    open_postgres_checkpointer,
)
from puncture_agent.agent.nodes import DeterministicMockToolExecutor  # noqa: E402


DSN_ENV = "PUNCTURE_TEST_POSTGRES_DSN"
EVIDENCE_DIR_ENV = "PUNCTURE_POSTGRES_RESTART_EVIDENCE_DIR"
BEFORE_FILE = "before.json"
AFTER_FILE = "after.json"
TOOL_CALLS_FILE = "tool_calls.jsonl"
EXPECTED_INTERRUPTED_COUNTS = {"generate_candidate_paths": 1}
EXPECTED_FINAL_COUNTS = {
    "evaluate_intraoperative_risk": 1,
    "evaluate_path_safety": 1,
    "generate_candidate_paths": 1,
    "verify_skin_penetration": 1,
}


class ProbeFailure(RuntimeError):
    """A failed restart/recovery invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProbeFailure(f"required environment variable {name} is missing")
    return value


def _evidence_directory() -> Path:
    directory = Path(_require_env(EVIDENCE_DIR_ENV)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with temporary.open("x", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeFailure(f"failed to read valid evidence file {path.name}") from exc
    if not isinstance(value, dict):
        raise ProbeFailure(f"evidence file {path.name} must contain an object")
    return value


def _postgres_identity(dsn: str) -> dict[str, Any]:
    try:
        import psycopg
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProbeFailure("psycopg is required for the PostgreSQL restart probe") from exc

    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT
                pg_postmaster_start_time(),
                system_identifier::text,
                current_setting('server_version'),
                current_setting('server_version_num')
            FROM pg_control_system()
            """
        ).fetchone()
    _require(row is not None and len(row) == 4, "PostgreSQL identity query returned no row")
    started_at, system_identifier, server_version, server_version_num = row
    return {
        "postmaster_started_at": started_at.isoformat(),
        "server_version": str(server_version),
        "server_version_num": int(server_version_num),
        "system_identifier": str(system_identifier),
    }


class JournaledCountingExecutor:
    """Run deterministic tools and durably append one non-secret call record."""

    def __init__(self, journal_path: Path, *, process_nonce: str) -> None:
        self._delegate = DeterministicMockToolExecutor()
        self._journal_path = journal_path
        self._process_nonce = process_nonce

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> Any:
        result = self._delegate.execute(tool_name, request)
        record = {
            "pid": os.getpid(),
            "process_nonce": self._process_nonce,
            "request_sha256": _sha256(dict(request)),
            "tool_name": tool_name,
        }
        descriptor = os.open(
            self._journal_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(_canonical_json_bytes(record).decode("utf-8"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return result


def _tool_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ProbeFailure(
                f"tool journal line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("tool_name"), str):
            raise ProbeFailure(f"tool journal line {line_number} has an invalid shape")
        entries.append(value)
    return entries


def _tool_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        tool_name = str(entry["tool_name"])
        counts[tool_name] = counts.get(tool_name, 0) + 1
    return dict(sorted(counts.items()))


def _handlers(executor: JournaledCountingExecutor) -> dict[str, Any]:
    from langgraph.types import interrupt

    handlers = dict(build_mock_handlers(executor))
    original_router = handlers["candidate_router"]

    def approval_gate(state: AgentState, context: Any) -> Any:
        decision = interrupt(
            {
                "kind": "candidate_review",
                "prompt": "approve generated candidates after PostgreSQL restart",
            }
        )
        state.metadata["candidate_review"] = decision
        return original_router(state, context)

    handlers["candidate_router"] = approval_gate
    return handlers


def _runtime(dsn: str, saver: Any, executor: JournaledCountingExecutor) -> LangGraphRuntime:
    return LangGraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        _handlers(executor),
        checkpointer=saver,
        execution_lease_manager=PostgresAdvisoryThreadExecutionLeaseManager(dsn),
    )


def prepare() -> dict[str, Any]:
    dsn = _require_env(DSN_ENV)
    directory = _evidence_directory()
    before_path = directory / BEFORE_FILE
    after_path = directory / AFTER_FILE
    journal_path = directory / TOOL_CALLS_FILE
    for path in (before_path, after_path, journal_path):
        path.unlink(missing_ok=True)

    _require(langgraph_available(), "LangGraph production dependencies are unavailable")
    process_nonce = uuid4().hex
    thread_id = f"postgres-service-restart-{uuid4().hex}"
    executor = JournaledCountingExecutor(journal_path, process_nonce=process_nonce)
    postgres = _postgres_identity(dsn)
    _require(
        postgres["server_version_num"] // 10_000 == 16,
        "the restart evidence job requires PostgreSQL 16",
    )

    with open_postgres_checkpointer(dsn, setup=True) as saver:
        runtime = _runtime(dsn, saver, executor)
        interrupted = runtime.run(
            AgentState(
                user_query="对 Case-994 做路径规划和安全评估",
                session_id=thread_id,
            )
        )
        persisted = runtime.checkpoint_state(thread_id=thread_id)

    _require(interrupted.status == AgentStatus.AWAITING_INPUT, "graph did not interrupt")
    _require(persisted.status == AgentStatus.AWAITING_INPUT, "checkpoint is not awaiting input")
    pending_interrupts = persisted.metadata.get("pending_interrupts")
    pending_value = (
        pending_interrupts[0].get("value")
        if isinstance(pending_interrupts, list)
        and bool(pending_interrupts)
        and isinstance(pending_interrupts[0], dict)
        else None
    )
    _require(
        isinstance(pending_value, dict)
        and pending_value.get("kind") == "candidate_review",
        "candidate-review interrupt was not persisted",
    )
    checkpoint_state = persisted.to_dict()
    entries = _tool_entries(journal_path)
    counts = _tool_counts(entries)
    _require(counts == EXPECTED_INTERRUPTED_COUNTS, "unexpected pre-restart tool calls")
    _require(
        all(entry.get("process_nonce") == process_nonce for entry in entries),
        "pre-restart tool journal contains another process",
    )

    evidence = {
        "checkpoint_sha256": _sha256(checkpoint_state),
        "checkpoint_state": checkpoint_state,
        "phase": "prepare",
        "postgres": postgres,
        "process": {"nonce": process_nonce, "pid": os.getpid()},
        "schema_version": "1",
        "thread_id": thread_id,
        "tool_counts": counts,
    }
    _write_json(before_path, evidence)
    return evidence


def recover() -> dict[str, Any]:
    dsn = _require_env(DSN_ENV)
    directory = _evidence_directory()
    before = _read_json(directory / BEFORE_FILE)
    journal_path = directory / TOOL_CALLS_FILE
    _require(langgraph_available(), "LangGraph production dependencies are unavailable")
    _require(before.get("schema_version") == "1", "unsupported before evidence schema")
    _require(before.get("phase") == "prepare", "before evidence has the wrong phase")
    thread_id = before.get("thread_id")
    _require(isinstance(thread_id, str) and thread_id, "before evidence has no thread ID")
    before_process = before.get("process")
    before_postgres = before.get("postgres")
    before_state = before.get("checkpoint_state")
    _require(isinstance(before_process, dict), "before evidence has no process identity")
    _require(isinstance(before_postgres, dict), "before evidence has no PostgreSQL identity")
    _require(isinstance(before_state, dict), "before evidence has no checkpoint state")
    _require(
        before.get("checkpoint_sha256") == _sha256(before_state),
        "before checkpoint hash does not match its state",
    )
    _require(
        _tool_counts(_tool_entries(journal_path)) == EXPECTED_INTERRUPTED_COUNTS,
        "tool journal changed before recovery",
    )

    process_nonce = uuid4().hex
    current_pid = os.getpid()
    _require(before_process.get("pid") != current_pid, "recover reused the prepare PID")
    _require(
        before_process.get("nonce") != process_nonce,
        "recover reused the prepare process nonce",
    )
    postgres = _postgres_identity(dsn)
    _require(
        postgres["server_version_num"] // 10_000 == 16,
        "the restart evidence job requires PostgreSQL 16",
    )
    _require(
        before_postgres.get("system_identifier") == postgres["system_identifier"],
        "PostgreSQL system identifier changed across restart",
    )
    _require(
        before_postgres.get("server_version_num") == postgres["server_version_num"],
        "PostgreSQL server version changed across restart",
    )
    _require(
        before_postgres.get("postmaster_started_at")
        != postgres["postmaster_started_at"],
        "PostgreSQL postmaster start time did not change",
    )

    executor = JournaledCountingExecutor(journal_path, process_nonce=process_nonce)
    with open_postgres_checkpointer(dsn, setup=False) as saver:
        runtime = _runtime(dsn, saver, executor)
        restored = runtime.checkpoint_state(thread_id=thread_id)
        restored_state = restored.to_dict()
        restored_sha256 = _sha256(restored_state)
        _require(
            restored_sha256 == before["checkpoint_sha256"],
            "checkpoint changed across PostgreSQL service restart",
        )
        _require(
            _canonical_json_bytes(restored_state) == _canonical_json_bytes(before_state),
            "restored checkpoint does not equal the prepare checkpoint",
        )
        resumed = runtime.resume(
            thread_id=thread_id,
            resume_value={"approved": True},
        )

    _require(resumed.status == AgentStatus.SUCCEEDED, "recovered graph did not succeed")
    _require(
        resumed.metadata.get("candidate_review") == {"approved": True},
        "resume value was not applied",
    )
    entries = _tool_entries(journal_path)
    counts = _tool_counts(entries)
    _require(counts == EXPECTED_FINAL_COUNTS, "completed tool calls were replayed or omitted")
    candidate_entries = [
        entry for entry in entries if entry["tool_name"] == "generate_candidate_paths"
    ]
    _require(len(candidate_entries) == 1, "candidate generation did not remain exactly once")
    _require(
        candidate_entries[0].get("process_nonce") == before_process.get("nonce"),
        "candidate generation was replayed by the recovery process",
    )
    _require(
        all(
            entry.get("process_nonce") == process_nonce
            for entry in entries
            if entry["tool_name"] != "generate_candidate_paths"
        ),
        "post-interrupt tools did not execute in the recovery process",
    )

    final_state = resumed.to_dict()
    evidence = {
        "assertions": {
            "checkpoint_equal_after_restart": True,
            "completed_tool_not_replayed": True,
            "fresh_python_process": True,
            "postgres_service_restarted": True,
            "same_postgres_cluster": True,
        },
        "before_checkpoint_sha256": before["checkpoint_sha256"],
        "final_checkpoint_sha256": _sha256(final_state),
        "final_state": final_state,
        "phase": "recover",
        "postgres": postgres,
        "process": {"nonce": process_nonce, "pid": current_pid},
        "restored_checkpoint_sha256": restored_sha256,
        "schema_version": "1",
        "thread_id": thread_id,
        "tool_counts": counts,
    }
    _write_json(directory / AFTER_FILE, evidence)
    return evidence


def main(arguments: list[str] | None = None) -> int:
    phases = {"prepare": prepare, "recover": recover}
    if arguments is None:
        arguments = sys.argv[1:]
    if len(arguments) != 1 or arguments[0] not in phases:
        print("usage: postgres_service_restart_probe.py prepare|recover", file=sys.stderr)
        return 2
    phase = arguments[0]
    dsn = os.environ.get(DSN_ENV, "")
    try:
        evidence = phases[phase]()
    except Exception as exc:
        message = str(exc).replace(dsn, "<redacted-postgres-dsn>") if dsn else str(exc)
        print(f"{phase} failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1
    summary = {
        "checkpoint_sha256": evidence.get(
            "checkpoint_sha256", evidence.get("restored_checkpoint_sha256")
        ),
        "phase": phase,
        "postgres": evidence["postgres"],
        "process": evidence["process"],
        "thread_id": evidence["thread_id"],
        "tool_counts": evidence["tool_counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
