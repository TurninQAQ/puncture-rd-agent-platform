"""Reproducible PostgreSQL checkpoint save/resume engineering benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
from math import ceil
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

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
ENFORCE_ENV = "PUNCTURE_ENFORCE_PERFORMANCE_GATES"
SAVE_P95_LIMIT_MS = 50.0
RESUME_P95_LIMIT_MS = 150.0
EXPECTED_TOOLS = (
    "evaluate_intraoperative_risk",
    "evaluate_path_safety",
    "generate_candidate_paths",
    "verify_skin_penetration",
)


class BenchmarkFailure(RuntimeError):
    """A correctness or benchmark-contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkFailure(message)


def nearest_rank(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile used by the checked-in gates."""

    if not values:
        raise ValueError("percentile values must not be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("percentile fraction must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[ceil(len(ordered) * fraction) - 1]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def summarize_rounds(
    raw_rounds: Sequence[Sequence[float]],
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Build reproducible aggregate and per-round statistics."""

    normalized = [[float(value) for value in values] for values in raw_rounds]
    flattened = [value for values in normalized for value in values]
    if not flattened:
        if not allow_empty:
            raise ValueError("measurement rounds must not be empty")
        return {
            "count": 0,
            "max": None,
            "p50": None,
            "p95": None,
            "raw_rounds": normalized,
            "round_p95": [None for _ in normalized],
        }
    require(
        all(values for values in normalized),
        "every measured round must contain at least one sample",
    )
    return {
        "count": len(flattened),
        "max": round(max(flattened), 6),
        "p50": round(float(statistics.median(flattened)), 6),
        "p95": round(nearest_rank(flattened, 0.95), 6),
        "raw_rounds": [
            [round(value, 6) for value in values] for values in normalized
        ],
        "round_p95": [round(nearest_rank(values, 0.95), 6) for values in normalized],
    }


@dataclass(frozen=True)
class OperationSample:
    operation: str
    phase: str
    duration_ms: float


class TimingRecorder:
    """Capture successful synchronous saver operations by benchmark phase."""

    def __init__(self) -> None:
        self._phase = "unclassified"
        self.samples: list[OperationSample] = []

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        previous = self._phase
        self._phase = phase
        try:
            yield
        finally:
            self._phase = previous

    def record(self, operation: str, duration_ms: float) -> None:
        self.samples.append(
            OperationSample(
                operation=operation,
                phase=self._phase,
                duration_ms=duration_ms,
            )
        )

    def clear(self) -> None:
        self.samples.clear()

    def durations(self, operation: str, phases: set[str]) -> list[float]:
        return [
            sample.duration_ms
            for sample in self.samples
            if sample.operation == operation and sample.phase in phases
        ]


def timing_saver(delegate: Any, recorder: TimingRecorder) -> Any:
    """Wrap a real saver while preserving the BaseCheckpointSaver contract."""

    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
    except (ImportError, ModuleNotFoundError) as exc:
        raise BenchmarkFailure("LangGraph checkpoint dependencies are unavailable") from exc

    class TimingSaver(BaseCheckpointSaver):
        def __init__(self) -> None:
            super().__init__(serde=delegate.serde)
            self.delegate = delegate

        @property
        def config_specs(self) -> list[Any]:
            return list(self.delegate.config_specs)

        def get_tuple(self, config: Mapping[str, Any]) -> Any:
            return self.delegate.get_tuple(config)

        def list(
            self,
            config: Mapping[str, Any] | None,
            *,
            filter: dict[str, Any] | None = None,
            before: Mapping[str, Any] | None = None,
            limit: int | None = None,
        ) -> Iterator[Any]:
            return self.delegate.list(
                config,
                filter=filter,
                before=before,
                limit=limit,
            )

        def put(
            self,
            config: Mapping[str, Any],
            checkpoint: Mapping[str, Any],
            metadata: Mapping[str, Any],
            new_versions: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            started = time.perf_counter_ns()
            result = self.delegate.put(config, checkpoint, metadata, new_versions)
            recorder.record(
                "put",
                (time.perf_counter_ns() - started) / 1_000_000.0,
            )
            return result

        def put_writes(
            self,
            config: Mapping[str, Any],
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            started = time.perf_counter_ns()
            self.delegate.put_writes(config, writes, task_id, task_path)
            recorder.record(
                "put_writes",
                (time.perf_counter_ns() - started) / 1_000_000.0,
            )

        def get_next_version(self, current: Any, channel: None) -> Any:
            return self.delegate.get_next_version(current, channel)

        def delete_thread(self, thread_id: str) -> None:
            self.delegate.delete_thread(thread_id)

        def with_allowlist(
            self,
            extra_allowlist: Collection[tuple[str, ...]],
        ) -> Any:
            return timing_saver(
                self.delegate.with_allowlist(extra_allowlist),
                recorder,
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

    return TimingSaver()


class CountingExecutor:
    """Count deterministic fake tool calls without adding external latency."""

    def __init__(self) -> None:
        self.delegate = DeterministicMockToolExecutor()
        self.counts: dict[str, int] = {}

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> Any:
        self.counts[tool_name] = self.counts.get(tool_name, 0) + 1
        return self.delegate.execute(tool_name, request)


def benchmark_handlers(executor: CountingExecutor) -> dict[str, Any]:
    """Interrupt immediately before the fake report handler."""

    from langgraph.types import interrupt

    handlers = dict(build_mock_handlers(executor))
    original_report = handlers["report_generator"]

    def report_approval_gate(state: AgentState, context: Any) -> Any:
        decision = interrupt(
            {
                "kind": "benchmark_report_approval",
                "prompt": "resume the terminal report checkpoint benchmark",
            }
        )
        state.metadata["benchmark_report_approval"] = decision
        return original_report(state, context)

    handlers["report_generator"] = report_approval_gate
    return handlers


def build_benchmark_runtime(
    dsn: str,
    saver: Any,
    executor: CountingExecutor,
) -> LangGraphRuntime:
    return LangGraphRuntime(
        ROOT / "graph" / "main_graph.json",
        benchmark_handlers(executor),
        checkpointer=saver,
        execution_lease_manager=PostgresAdvisoryThreadExecutionLeaseManager(dsn),
    )


@dataclass
class BatchMeasurements:
    checkpoint_state_bytes: list[float]
    resume_ms: list[float]
    run_to_interrupt_ms: list[float]


def expected_tool_counts(sample_count: int) -> dict[str, int]:
    return {tool_name: sample_count for tool_name in EXPECTED_TOOLS}


def run_batch(
    *,
    dsn: str,
    recorder: TimingRecorder,
    label: str,
    sample_count: int,
    setup: bool,
) -> BatchMeasurements:
    """Create interrupted checkpoints, reconnect, then resume every session."""

    executor = CountingExecutor()
    thread_prefix = f"postgres-checkpoint-benchmark-{label}-{uuid4().hex}"
    thread_ids: list[str] = []
    expected_cases: dict[str, str] = {}
    checkpoint_state_bytes: list[float] = []
    run_to_interrupt_ms: list[float] = []

    with open_postgres_checkpointer(dsn, setup=setup) as prepare_delegate:
        prepare_saver = timing_saver(prepare_delegate, recorder)
        prepare_runtime = build_benchmark_runtime(dsn, prepare_saver, executor)
        for sample_index in range(sample_count):
            thread_id = f"{thread_prefix}-{sample_index:04d}"
            case_number = 200_000 + sample_index
            expected_case = f"Case-{case_number}"
            started = time.perf_counter_ns()
            with recorder.phase(f"{label}:prepare"):
                interrupted = prepare_runtime.run(
                    AgentState(
                        user_query=(
                            f"对 Case-{case_number} 做路径规划和安全评估"
                        ),
                        session_id=thread_id,
                    )
                )
            run_to_interrupt_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
            require(
                interrupted.status == AgentStatus.AWAITING_INPUT,
                "benchmark graph did not stop at the report interrupt",
            )
            pending = interrupted.metadata.get("pending_interrupts")
            pending_value = (
                pending[0].get("value")
                if isinstance(pending, list)
                and pending
                and isinstance(pending[0], dict)
                else None
            )
            require(
                isinstance(pending_value, dict)
                and pending_value.get("kind") == "benchmark_report_approval",
                "benchmark report interrupt was not durable",
            )
            checkpoint_state_bytes.append(
                float(
                    len(
                        json.dumps(
                            interrupted.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                )
            )
            thread_ids.append(thread_id)
            expected_cases[thread_id] = expected_case

    expected_counts = expected_tool_counts(sample_count)
    require(
        executor.counts == expected_counts,
        "prepare phase did not execute the expected fake tools exactly once",
    )
    counts_before_resume = dict(executor.counts)
    resume_ms: list[float] = []
    resumed_states: dict[str, dict[str, Any]] = {}

    with open_postgres_checkpointer(dsn, setup=False) as resume_delegate:
        resume_saver = timing_saver(resume_delegate, recorder)
        resume_runtime = build_benchmark_runtime(dsn, resume_saver, executor)
        for thread_id in thread_ids:
            started = time.perf_counter_ns()
            with recorder.phase(f"{label}:resume"):
                resumed = resume_runtime.resume(
                    thread_id=thread_id,
                    resume_value={"approved": True},
                )
            resume_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            require(
                resumed.status == AgentStatus.SUCCEEDED,
                "benchmark resume did not reach a successful terminal state",
            )
            require(
                resumed.session_id == thread_id,
                "benchmark resume returned the wrong session identity",
            )
            require(
                resumed.case_id == expected_cases[thread_id],
                "benchmark resume returned the wrong case identity",
            )
            require(
                resumed.metadata.get("benchmark_report_approval")
                == {"approved": True},
                "benchmark resume value was not applied",
            )
            require(
                executor.counts == counts_before_resume,
                "checkpoint resume replayed a completed fake tool",
            )
            resumed_states[thread_id] = resumed.to_dict()
        for thread_id in thread_ids:
            terminal = resume_runtime.checkpoint_state(thread_id=thread_id).to_dict()
            require(
                canonical_json_bytes(terminal)
                == canonical_json_bytes(resumed_states[thread_id]),
                "terminal checkpoint differs from the resumed state",
            )
        for thread_id in thread_ids:
            resume_delegate.delete_thread(thread_id)

    return BatchMeasurements(
        checkpoint_state_bytes=checkpoint_state_bytes,
        resume_ms=resume_ms,
        run_to_interrupt_ms=run_to_interrupt_ms,
    )


def safe_label(value: str, option: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise BenchmarkFailure(f"{option} must not be empty")
    if any(token in normalized for token in ("://", "@", "\r", "\n", "\x00")):
        raise BenchmarkFailure(f"{option} must be a non-secret descriptive label")
    return normalized


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def postgres_environment(dsn: str) -> dict[str, Any]:
    try:
        import psycopg
    except (ImportError, ModuleNotFoundError) as exc:
        raise BenchmarkFailure("psycopg is required for the PostgreSQL benchmark") from exc
    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT
                current_setting('server_version'),
                current_setting('server_version_num'),
                current_setting('synchronous_commit')
            """
        ).fetchone()
    require(row is not None and len(row) == 3, "PostgreSQL environment query failed")
    return {
        "server_version": str(row[0]),
        "server_version_num": int(row[1]),
        "synchronous_commit": str(row[2]),
    }


def write_json(path: Path, result: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_github_summary(result: Mapping[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return
    measurements = result["measurements"]
    thresholds = result["thresholds"]
    gate = result["gate"]
    lines = [
        "## PostgreSQL checkpoint benchmark",
        "",
        "| Metric | P50 | P95 | Limit |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Checkpoint save | "
            f"{measurements['checkpoint_save_ms']['p50']:.3f} ms | "
            f"{measurements['checkpoint_save_ms']['p95']:.3f} ms | "
            f"{thresholds['save_p95_ms']:.0f} ms |"
        ),
        (
            "| Resume end-to-end | "
            f"{measurements['resume_end_to_end_ms']['p50']:.3f} ms | "
            f"{measurements['resume_end_to_end_ms']['p95']:.3f} ms | "
            f"{thresholds['resume_p95_ms']:.0f} ms |"
        ),
        "",
        f"Gate mode: `{gate['mode']}`; threshold observation: `{thresholds['passed']}`.",
        "GitHub-hosted runner values are an engineering baseline, not a production SLA.",
        "",
    ]
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except OSError as exc:
        print(f"warning: failed to append GITHUB_STEP_SUMMARY: {exc}", file=sys.stderr)


def build_result(args: argparse.Namespace, dsn: str) -> dict[str, Any]:
    require(langgraph_available(), "LangGraph production dependencies are unavailable")
    recorder = TimingRecorder()
    setup_required = True

    if args.warmups:
        run_batch(
            dsn=dsn,
            recorder=recorder,
            label="warmup",
            sample_count=args.warmups,
            setup=setup_required,
        )
        setup_required = False
        recorder.clear()

    save_rounds: list[list[float]] = []
    put_writes_rounds: list[list[float]] = []
    resume_rounds: list[list[float]] = []
    run_rounds: list[list[float]] = []
    checkpoint_size_rounds: list[list[float]] = []
    for round_index in range(args.rounds):
        label = f"round-{round_index}"
        batch = run_batch(
            dsn=dsn,
            recorder=recorder,
            label=label,
            sample_count=args.samples_per_round,
            setup=setup_required,
        )
        setup_required = False
        phases = {f"{label}:prepare", f"{label}:resume"}
        save_rounds.append(recorder.durations("put", phases))
        put_writes_rounds.append(recorder.durations("put_writes", phases))
        resume_rounds.append(batch.resume_ms)
        run_rounds.append(batch.run_to_interrupt_ms)
        checkpoint_size_rounds.append(batch.checkpoint_state_bytes)

    checkpoint_save = summarize_rounds(save_rounds)
    resume_end_to_end = summarize_rounds(resume_rounds)
    put_writes = summarize_rounds(put_writes_rounds, allow_empty=True)
    run_to_interrupt = summarize_rounds(run_rounds)
    checkpoint_state_bytes = summarize_rounds(checkpoint_size_rounds)
    measured_sessions = args.rounds * args.samples_per_round
    require(
        checkpoint_save["count"] > measured_sessions,
        "the timing proxy did not capture the expected checkpoint put calls",
    )

    required_rounds = ceil(args.rounds / 2)
    save_aggregate_pass = checkpoint_save["p95"] <= SAVE_P95_LIMIT_MS
    resume_aggregate_pass = resume_end_to_end["p95"] <= RESUME_P95_LIMIT_MS
    save_rounds_passed = sum(
        value <= SAVE_P95_LIMIT_MS for value in checkpoint_save["round_p95"]
    )
    resume_rounds_passed = sum(
        value <= RESUME_P95_LIMIT_MS
        for value in resume_end_to_end["round_p95"]
    )
    stability_pass = (
        save_rounds_passed >= required_rounds
        and resume_rounds_passed >= required_rounds
    )
    threshold_pass = save_aggregate_pass and resume_aggregate_pass and stability_pass
    enforce = bool(args.enforce_thresholds or os.environ.get(ENFORCE_ENV) == "1")

    commit = os.environ.get("GITHUB_SHA", "").strip() or git_value("rev-parse", "HEAD")
    dirty_output = git_value("status", "--porcelain")
    source_command = [sys.executable, *sys.argv]
    result: dict[str, Any] = {
        "benchmark_id": "langgraph-postgres-checkpoint",
        "correctness": {
            "errors": [],
            "interrupted": measured_sessions,
            "resumed": measured_sessions,
            "terminal_checkpoints_verified": measured_sessions,
            "tool_counts_unchanged": True,
        },
        "environment": {
            "cpu_model": cpu_model(),
            "label": args.environment_label,
            "langgraph": package_version("langgraph"),
            "langgraph_checkpoint_postgres": package_version(
                "langgraph-checkpoint-postgres"
            ),
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "postgres": postgres_environment(dsn),
            "psycopg": package_version("psycopg"),
            "python": platform.python_version(),
            "storage_label": args.storage_label,
        },
        "gate": {
            "mode": "enforce" if enforce else "record",
            "passed": threshold_pass if enforce else True,
        },
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "measurements": {
            "checkpoint_save_ms": checkpoint_save,
            "diagnostics": {
                "checkpoint_state_bytes": checkpoint_state_bytes,
                "put_writes_ms": put_writes,
                "run_to_interrupt_ms": run_to_interrupt,
            },
            "resume_end_to_end_ms": resume_end_to_end,
        },
        "notes": [
            "Fake model, RAG and tool dependencies isolate runtime/checkpoint overhead.",
            "GitHub-hosted runner results are an engineering baseline, not a production SLA.",
            "Saver connection setup, migration and graph compilation are outside timed resume samples; production advisory-lease acquisition remains inside public resume().",
            "checkpoint_save_ms measures successful synchronous PostgresSaver.put() calls; put_writes is reported separately and is not folded into that metric.",
            "delete_thread removes benchmark rows logically but does not VACUUM; later rounds may include autovacuum or dead-tuple jitter.",
        ],
        "schema_version": "1",
        "source": {
            "command": source_command,
            "commit": commit or "unknown",
            "dirty": bool(dirty_output) if dirty_output is not None else None,
        },
        "thresholds": {
            "aggregate_pass": save_aggregate_pass and resume_aggregate_pass,
            "passed": threshold_pass,
            "required_rounds_passed": required_rounds,
            "resume_p95_ms": RESUME_P95_LIMIT_MS,
            "resume_rounds_passed": resume_rounds_passed,
            "save_p95_ms": SAVE_P95_LIMIT_MS,
            "save_rounds_passed": save_rounds_passed,
            "stability_pass": stability_pass,
        },
        "workload": {
            "checkpoint_save_measurement": "successful synchronous PostgresSaver.put() call",
            "concurrency": 1,
            "durability": "sync",
            "fake_dependencies": True,
            "graph": "graph/main_graph.json",
            "interrupt_before": "report_generator",
            "p50_method": "median",
            "p95_method": "nearest-rank",
            "put_writes_in_checkpoint_save": False,
            "resume_connection_model": "new saver and runtime after prepare phase",
            "resume_measurement": "public LangGraphRuntime.resume() through terminal state",
            "rounds": args.rounds,
            "samples_per_round": args.samples_per_round,
            "setup_timed": False,
            "warmups": args.warmups,
        },
    }
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples-per-round", type=int, default=50)
    parser.add_argument("--environment-label", default="local-postgresql")
    parser.add_argument("--storage-label", default="explicit-local-postgresql")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-thresholds", action="store_true")
    args = parser.parse_args(arguments)
    if args.warmups < 0:
        parser.error("--warmups must not be negative")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.samples_per_round < 1:
        parser.error("--samples-per-round must be positive")
    try:
        args.environment_label = safe_label(
            args.environment_label,
            "--environment-label",
        )
        args.storage_label = safe_label(args.storage_label, "--storage-label")
    except BenchmarkFailure as exc:
        parser.error(str(exc))
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn:
        print(f"benchmark failed: required environment variable {DSN_ENV} is missing", file=sys.stderr)
        return 1
    try:
        result = build_result(args, dsn)
        if args.output is not None:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        append_github_summary(result)
    except Exception as exc:
        message = str(exc).replace(dsn, "<redacted-postgres-dsn>")
        print(f"benchmark failed: {type(exc).__name__}: {message}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            annotation = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error title=PostgreSQL checkpoint benchmark::{annotation}")
        return 1

    save = result["measurements"]["checkpoint_save_ms"]
    resume = result["measurements"]["resume_end_to_end_ms"]
    print(
        "POSTGRES_CHECKPOINT_BENCHMARK "
        f"save_p50_ms={save['p50']:.3f} save_p95_ms={save['p95']:.3f} "
        f"resume_p50_ms={resume['p50']:.3f} resume_p95_ms={resume['p95']:.3f} "
        f"threshold_pass={result['thresholds']['passed']} "
        f"gate_mode={result['gate']['mode']}"
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            "::notice title=PostgreSQL checkpoint benchmark::"
            f"save P50/P95 {save['p50']:.3f}/{save['p95']:.3f} ms; "
            f"resume P50/P95 {resume['p50']:.3f}/{resume['p95']:.3f} ms; "
            f"threshold_pass={result['thresholds']['passed']}; "
            f"gate_mode={result['gate']['mode']}"
        )
    if not result["thresholds"]["passed"]:
        message = (
            f"observed save/resume P95 {save['p95']:.3f}/{resume['p95']:.3f} ms "
            f"against {SAVE_P95_LIMIT_MS:.0f}/{RESUME_P95_LIMIT_MS:.0f} ms gates"
        )
        if result["gate"]["mode"] == "enforce":
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::error title=Performance gate::{message}")
            return 1
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=Engineering baseline exceeded::{message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
