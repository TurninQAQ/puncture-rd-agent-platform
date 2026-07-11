"""Kill a graph worker after an MCP response but before its node checkpoint."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from contracts.artifacts import ArtifactRef  # noqa: E402
from contracts.enums import (  # noqa: E402
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
)
from contracts.geometry import VolumeGeometry  # noqa: E402
from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    LangGraphRuntime,
    PostgresAdvisoryThreadExecutionLeaseManager,
    build_mock_handlers,
    langgraph_available,
    open_postgres_checkpointer,
)
from puncture_agent.agent.tool_bridge import McpToolExecutor  # noqa: E402
from puncture_agent.mcp import (  # noqa: E402
    InMemoryArtifactResolver,
    McpPrincipal,
    McpToolRuntime,
    SQLiteToolReplayLedger,
)
from puncture_agent.mocks.tool_mocks import MOCK_HANDLERS  # noqa: E402
from puncture_agent.tooling import TOOL_DEFINITIONS  # noqa: E402
from puncture_agent.tooling.registry import ToolRegistry  # noqa: E402


DSN_ENV = "PUNCTURE_TEST_POSTGRES_DSN"
WORK_DIR_ENV = "PUNCTURE_PROCESS_KILL_WORK_DIR"
CASE_ID = "Case-995"
TARGET_TOOL = "generate_candidate_paths"
PLANNING_TOOLS = (
    "generate_candidate_paths",
    "evaluate_path_safety",
    "evaluate_intraoperative_risk",
    "verify_skin_penetration",
)
MODULE = "tests.graph.postgres_tool_process_kill_probe"


class ProbeFailure(RuntimeError):
    """A failed process-kill/replay invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProbeFailure(f"required environment variable {name} is missing")
    return value


def work_directory() -> Path:
    directory = Path(require_env(WORK_DIR_ENV)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
    fsync_directory(path.parent)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeFailure(f"failed to read valid evidence file {path.name}") from exc
    if not isinstance(value, dict):
        raise ProbeFailure(f"evidence file {path.name} must contain an object")
    return value


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(value).decode("utf-8"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeFailure(f"invalid JSONL at {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ProbeFailure(f"invalid JSONL object at {path.name}:{line_number}")
        result.append(value)
    return result


def geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0, 1.0, 1.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        coordinate_system=CoordinateSystem.LPS,
    )


def artifact(suffix: str, artifact_type: ArtifactType) -> ArtifactRef:
    artifact_id = f"artifact-{CASE_ID}-{suffix}"
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id=CASE_ID,
        artifact_type=artifact_type,
        uri=f"mock://private/{artifact_id}",
        checksum_sha256="a" * 64,
        status=ArtifactStatus.AVAILABLE,
        geometry=geometry(),
        producer_name="process-kill-probe",
        producer_version="1",
    )


def planning_artifacts() -> tuple[ArtifactRef, ...]:
    return (
        artifact("ct", ArtifactType.CT_VOLUME),
        artifact("skin-surface", ArtifactType.SKIN_SURFACE_MASK),
        artifact("target", ArtifactType.TARGET_MASK),
        artifact("skin", ArtifactType.SEGMENTATION_MASK),
        artifact("heart", ArtifactType.DANGER_MASK),
        artifact("bone", ArtifactType.DANGER_MASK),
        artifact("bronchus", ArtifactType.DANGER_MASK),
        artifact("vessel", ArtifactType.DANGER_MASK),
        artifact("lung", ArtifactType.SEGMENTATION_MASK),
    )


def counted_registry(side_effects_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for name, definition in TOOL_DEFINITIONS.items():
        handler = MOCK_HANDLERS[name]
        if name in PLANNING_TOOLS:

            def counted(request: Any, *, _name: str = name, _handler: Any = handler) -> Any:
                append_jsonl(
                    side_effects_path,
                    {
                        "case_id": request.context.case_id,
                        "pid": os.getpid(),
                        "request_id": request.context.request_id,
                        "tool_name": _name,
                    },
                )
                return _handler(request)

            handler = counted
        registry.register(definition, handler)
    return registry


class ObservingCaller:
    def __init__(self, delegate: McpToolRuntime, observations_path: Path) -> None:
        self.delegate = delegate
        self.tool_names = delegate.tool_names
        self.observations_path = observations_path

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> Any:
        result = self.delegate.call_tool(name, arguments, principal=principal)
        structured = result.structured_content
        append_jsonl(
            self.observations_path,
            {
                "idempotent_replay": bool(
                    result.meta.get("com.turninqaq/idempotentReplay", False)
                ),
                "pid": os.getpid(),
                "request_id": structured.get("request_id"),
                "result_sha256": value_sha256(structured.get("result")),
                "tool_name": name,
                "trace_id": structured.get("trace_id"),
            },
        )
        return result


class CrashAfterResponseExecutor:
    """Kill only after the target bridge response has returned to the node."""

    def __init__(
        self,
        delegate: McpToolExecutor,
        *,
        crash_after_target: bool,
        crash_marker_path: Path,
    ) -> None:
        self.delegate = delegate
        self.crash_after_target = crash_after_target
        self.crash_marker_path = crash_marker_path

    @contextmanager
    def bind_state(self, state: AgentState) -> Iterator["CrashAfterResponseExecutor"]:
        with self.delegate.bind_state(state):
            yield self

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self.delegate.execute(tool_name, request)
        if self.crash_after_target and tool_name == TARGET_TOOL:
            write_json(
                self.crash_marker_path,
                {
                    "case_id": request.get("case_id"),
                    "phase": "after_mcp_response_before_node_return",
                    "pid": os.getpid(),
                    "response_result_sha256": value_sha256(response.get("result")),
                    "schema_version": "1",
                    "tool_name": tool_name,
                },
            )
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL unexpectedly returned")
        return response


def build_runtime(
    dsn: str,
    saver: Any,
    directory: Path,
    *,
    crash_after_target: bool,
) -> tuple[LangGraphRuntime, SQLiteToolReplayLedger]:
    ledger = SQLiteToolReplayLedger(directory / "tool-replay.sqlite3")
    mcp_runtime = McpToolRuntime(
        counted_registry(directory / "side_effects.jsonl"),
        InMemoryArtifactResolver(planning_artifacts()),
        server_name="planning-safety",
        replay_ledger=ledger,
        replay_authorizer=lambda name, value, principal: True,
        replay_response_validator=lambda name, value, response, principal: True,
    )
    caller = ObservingCaller(mcp_runtime, directory / "mcp_observations.jsonl")
    bridge = McpToolExecutor(
        caller,
        principal=McpPrincipal("process-kill-probe", (CASE_ID,)),
    )
    executor = CrashAfterResponseExecutor(
        bridge,
        crash_after_target=crash_after_target,
        crash_marker_path=directory / "crash_marker.json",
    )
    runtime = LangGraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(executor),
        checkpointer=saver,
        execution_lease_manager=PostgresAdvisoryThreadExecutionLeaseManager(dsn),
    )
    return runtime, ledger


def prepare_worker() -> int:
    require(os.name == "posix", "process-kill probe requires POSIX signals")
    require(langgraph_available(), "LangGraph production dependencies are unavailable")
    dsn = require_env(DSN_ENV)
    directory = work_directory()
    thread_id = f"postgres-tool-process-kill-{uuid4().hex}"
    write_json(
        directory / "before.json",
        {
            "case_id": CASE_ID,
            "phase": "prepare",
            "pid": os.getpid(),
            "schema_version": "1",
            "thread_id": thread_id,
        },
    )
    with open_postgres_checkpointer(dsn, setup=True) as saver:
        runtime, ledger = build_runtime(
            dsn,
            saver,
            directory,
            crash_after_target=True,
        )
        try:
            runtime.run(
                AgentState(
                    user_query=f"对 {CASE_ID} 做路径规划和皮肤穿透安全评估",
                    session_id=thread_id,
                )
            )
        finally:
            ledger.close()
    raise ProbeFailure("prepare worker reached the end without SIGKILL")


def recover_worker() -> int:
    require(langgraph_available(), "LangGraph production dependencies are unavailable")
    dsn = require_env(DSN_ENV)
    directory = work_directory()
    before = read_json(directory / "before.json")
    thread_id = before.get("thread_id")
    require(isinstance(thread_id, str) and thread_id, "before evidence has no thread ID")
    require(before.get("pid") != os.getpid(), "recovery reused the killed process PID")

    with open_postgres_checkpointer(dsn, setup=False) as saver:
        runtime, ledger = build_runtime(
            dsn,
            saver,
            directory,
            crash_after_target=False,
        )
        try:
            pre_resume = runtime.checkpoint_state(thread_id=thread_id)
            require(
                "planning_safety_subgraph.generate_candidate_paths"
                not in pre_resume.visited_nodes,
                "target node was checkpointed despite the process kill",
            )
            require(
                not any(
                    call.get("tool_name") == TARGET_TOOL
                    for call in pre_resume.tool_calls
                    if isinstance(call, Mapping)
                ),
                "target tool call was checkpointed despite the process kill",
            )
            resumed = runtime.resume(thread_id=thread_id)
            terminal = runtime.checkpoint_state(thread_id=thread_id)
        finally:
            ledger.close()

    require(resumed.status == AgentStatus.SUCCEEDED, "recovered graph did not succeed")
    require(resumed.session_id == thread_id, "recovered graph changed session identity")
    require(resumed.case_id == CASE_ID, "recovered graph changed case identity")
    require(
        canonical_json_bytes(resumed.to_dict()) == canonical_json_bytes(terminal.to_dict()),
        "terminal checkpoint differs from the recovered state",
    )
    final_tools = [
        str(call.get("tool_name"))
        for call in resumed.tool_calls
        if isinstance(call, Mapping)
    ]
    require(final_tools == list(PLANNING_TOOLS), "recovered tool sequence is incorrect")
    write_json(
        directory / "after.json",
        {
            "case_id": resumed.case_id,
            "final_state_sha256": value_sha256(resumed.to_dict()),
            "phase": "recover",
            "pid": os.getpid(),
            "pre_resume_state_sha256": value_sha256(pre_resume.to_dict()),
            "schema_version": "1",
            "status": resumed.status,
            "thread_id": thread_id,
            "tool_names": final_tools,
        },
    )
    return 0


def run_child(mode: str, dsn: str, directory: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment[DSN_ENV] = dsn
    environment[WORK_DIR_ENV] = str(directory)
    environment["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, "-m", MODULE, mode],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def orchestrate() -> int:
    require(os.name == "posix", "process-kill probe requires POSIX signals")
    dsn = require_env(DSN_ENV)
    directory = work_directory()
    for name in (
        "before.json",
        "crash_marker.json",
        "after.json",
        "orchestrator.json",
        "side_effects.jsonl",
        "mcp_observations.jsonl",
        "tool-replay.sqlite3",
        "tool-replay.sqlite3-wal",
        "tool-replay.sqlite3-shm",
    ):
        (directory / name).unlink(missing_ok=True)

    prepare = run_child("prepare-worker", dsn, directory)
    require(
        prepare.returncode == -signal.SIGKILL,
        "prepare worker was not terminated by SIGKILL: "
        + (prepare.stderr.strip() or prepare.stdout.strip() or str(prepare.returncode)),
    )
    before = read_json(directory / "before.json")
    crash = read_json(directory / "crash_marker.json")
    require(crash.get("pid") == before.get("pid"), "crash marker has the wrong PID")
    require(
        crash.get("phase") == "after_mcp_response_before_node_return",
        "crash marker has the wrong boundary",
    )
    initial_effects = read_jsonl(directory / "side_effects.jsonl")
    require(
        [entry.get("tool_name") for entry in initial_effects] == [TARGET_TOOL],
        "prepare worker executed an unexpected underlying side effect",
    )
    initial_observations = read_jsonl(directory / "mcp_observations.jsonl")
    require(len(initial_observations) == 1, "prepare worker has the wrong MCP call count")
    require(
        initial_observations[0].get("idempotent_replay") is False,
        "the first target call was unexpectedly a replay",
    )
    require(
        initial_observations[0].get("result_sha256")
        == crash.get("response_result_sha256"),
        "crash marker does not match the committed MCP response",
    )

    recover = run_child("recover-worker", dsn, directory)
    require(
        recover.returncode == 0,
        "recover worker failed: "
        + (recover.stderr.strip() or recover.stdout.strip() or str(recover.returncode)),
    )
    after = read_json(directory / "after.json")
    require(after.get("pid") != before.get("pid"), "recovery did not use a fresh process")
    effects = read_jsonl(directory / "side_effects.jsonl")
    require(
        [entry.get("tool_name") for entry in effects] == list(PLANNING_TOOLS),
        "underlying tool side effects were duplicated or omitted",
    )
    observations = read_jsonl(directory / "mcp_observations.jsonl")
    observed_tools = [entry.get("tool_name") for entry in observations]
    require(
        observed_tools == [TARGET_TOOL, *PLANNING_TOOLS],
        "MCP observation sequence is incorrect",
    )
    first_target, replayed_target = observations[0], observations[1]
    require(
        replayed_target.get("idempotent_replay") is True,
        "recovered target call did not use the durable replay ledger",
    )
    require(
        first_target.get("request_id") == replayed_target.get("request_id")
        and first_target.get("trace_id") == replayed_target.get("trace_id")
        and first_target.get("result_sha256") == replayed_target.get("result_sha256"),
        "replayed target identity or result changed across processes",
    )
    require(
        all(entry.get("idempotent_replay") is False for entry in observations[2:]),
        "a downstream first execution was incorrectly marked as replay",
    )

    evidence = {
        "assertions": {
            "checkpoint_excluded_crashed_node": True,
            "fresh_recovery_process": True,
            "mcp_response_committed_before_sigkill": True,
            "target_replayed_without_side_effect": True,
            "terminal_checkpoint_verified": True,
        },
        "case_id": CASE_ID,
        "killed_pid": before["pid"],
        "phase": "orchestrate",
        "recovery_pid": after["pid"],
        "schema_version": "1",
        "side_effect_counts": {
            tool_name: sum(
                entry.get("tool_name") == tool_name for entry in effects
            )
            for tool_name in PLANNING_TOOLS
        },
        "thread_id": before["thread_id"],
    }
    write_json(directory / "orchestrator.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


def main(arguments: list[str] | None = None) -> int:
    if arguments is None:
        arguments = sys.argv[1:]
    modes = {
        "orchestrate": orchestrate,
        "prepare-worker": prepare_worker,
        "recover-worker": recover_worker,
    }
    if len(arguments) != 1 or arguments[0] not in modes:
        print(
            "usage: postgres_tool_process_kill_probe.py "
            "orchestrate|prepare-worker|recover-worker",
            file=sys.stderr,
        )
        return 2
    dsn = os.environ.get(DSN_ENV, "")
    try:
        return modes[arguments[0]]()
    except Exception as exc:
        message = str(exc).replace(dsn, "<redacted-postgres-dsn>") if dsn else str(exc)
        print(
            f"{arguments[0]} failed: {type(exc).__name__}: {message}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
