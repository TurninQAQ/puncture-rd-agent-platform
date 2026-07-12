"""Compare graph-only latency with and without in-memory tracing.

Records P50/P95 for warm runs. The initial Task 08 engineering gate is that
batched/in-memory tracing should add <= 5% P95 latency on the graph-only mock
path in a controlled environment. Default mode always records; set
``PUNCTURE_ENFORCE_PERFORMANCE_GATES=1`` or ``--enforce`` to fail on the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from math import ceil
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from puncture_agent.agent import AgentState, GraphRuntime, build_mock_handlers  # noqa: E402
from puncture_agent.observability.tracing import (  # noqa: E402
    CompositeTraceExporter,
    InMemoryOtlpTraceExporter,
    InMemoryTraceExporter,
    TraceRecorder,
)

ENFORCE_ENV = "PUNCTURE_ENFORCE_PERFORMANCE_GATES"
OVERHEAD_P95_LIMIT_RATIO = 0.05


def nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(v) for v in values)
    return ordered[ceil(len(ordered) * fraction) - 1]


def run_once(runtime: GraphRuntime, query: str) -> float:
    started = time.perf_counter()
    runtime.run(AgentState(user_query=query))
    return (time.perf_counter() - started) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--environment-label",
        default="local",
        help="Non-secret environment label for the evidence record",
    )
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmups < 0:
        parser.error("invalid sample/warmup counts")

    graph = ROOT / "graph" / "main_graph.json"
    handlers = build_mock_handlers()
    query = "对 Case-901 做路径规划和安全评估"

    plain = GraphRuntime(graph, handlers)
    for _ in range(args.warmups):
        run_once(plain, query)
    plain_samples = [run_once(plain, query) for _ in range(args.samples)]

    exporter = InMemoryTraceExporter()
    otlp = InMemoryOtlpTraceExporter()
    tracer = TraceRecorder(CompositeTraceExporter([exporter, otlp]))
    traced = GraphRuntime(graph, handlers, tracer=tracer)
    for _ in range(args.warmups):
        run_once(traced, query)
    exporter.clear()
    otlp.clear()
    traced_samples = [run_once(traced, query) for _ in range(args.samples)]

    plain_p50 = statistics.median(plain_samples)
    plain_p95 = nearest_rank(plain_samples, 0.95)
    traced_p50 = statistics.median(traced_samples)
    traced_p95 = nearest_rank(traced_samples, 0.95)
    overhead_ratio = (traced_p95 - plain_p95) / plain_p95 if plain_p95 else 0.0

    payload = {
        "benchmark": "tracing-overhead-v1",
        "environment_label": args.environment_label,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmups": args.warmups,
        "samples": args.samples,
        "exporter": "InMemoryTraceExporter+InMemoryOtlpTraceExporter",
        "batch_settings": "synchronous in-process export (no remote OTLP)",
        "plain_ms": {
            "p50": plain_p50,
            "p95": plain_p95,
            "mean": statistics.fmean(plain_samples),
        },
        "traced_ms": {
            "p50": traced_p50,
            "p95": traced_p95,
            "mean": statistics.fmean(traced_samples),
        },
        "overhead_p95_ratio": overhead_ratio,
        "gate_overhead_p95_ratio": OVERHEAD_P95_LIMIT_RATIO,
        "spans_exported_last_clear_window": len(exporter.spans()),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    enforce = args.enforce or os.environ.get(ENFORCE_ENV) == "1"
    if enforce and overhead_ratio > OVERHEAD_P95_LIMIT_RATIO:
        print(
            f"ENFORCE failed: overhead_p95_ratio={overhead_ratio:.4f} "
            f"> {OVERHEAD_P95_LIMIT_RATIO}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
