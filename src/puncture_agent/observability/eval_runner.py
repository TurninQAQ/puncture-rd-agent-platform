"""Optional traced evaluation runner and offline CLI helpers.

Keeps instrumentation out of graph/runtime modules: a case-level parent span is
opened on a shared :class:`TraceRecorder` before ``GraphRuntime.run``, so nested
``agent.graph`` / ``agent.node`` spans inherit one ``trace_id`` via contextvars.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from puncture_agent.agent.runtime import GraphRuntime
from puncture_agent.observability.dataset import load_eval_dataset
from puncture_agent.observability.eval_harness import (
    AgentEvalHarness,
    EvalCase,
    EvalCaseResult,
    EvalReport,
    compare_reports,
    evaluate_case,
    load_report,
)
from puncture_agent.observability.tracing import (
    InMemoryTraceExporter,
    TraceRecorder,
)


@dataclass(frozen=True)
class TracedEvalCaseResult:
    result: EvalCaseResult
    trace_id: str
    span_count: int


def run_case_with_tracing(
    case: EvalCase,
    runtime_factory: Callable[[], GraphRuntime],
    *,
    tracer: TraceRecorder | None = None,
    exporter: InMemoryTraceExporter | None = None,
) -> tuple[EvalCaseResult, TraceRecorder, InMemoryTraceExporter]:
    """Execute one case under a shared tracer and stamp ``trace_id`` on the result."""

    memory = exporter or InMemoryTraceExporter()
    recorder = tracer or TraceRecorder(memory)
    runtime = runtime_factory()
    # Attach the same recorder when the runtime accepts a tracer kw/attr.
    if getattr(runtime, "tracer", None) is None:
        try:
            runtime.tracer = recorder  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        runtime.tracer = recorder  # type: ignore[attr-defined]

    started = time.perf_counter()
    with recorder.start_span(
        "eval.case",
        attributes={
            "case_id": case.case_id,
            "component": "eval",
        },
    ) as root:
        state = runtime.run(case.build_state())
        # Persist correlation for diagnostics even on the mock runtime.
        state.metadata["trace_id"] = root.trace_id
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = evaluate_case(case, state)
    observed = dict(result.observed)
    observed["latency_ms"] = elapsed_ms
    observed["trace_id"] = root.trace_id
    observed["require_trace_on_failure"] = True
    result = EvalCaseResult(
        case_id=result.case_id,
        passed=result.passed,
        routing_correct=result.routing_correct,
        terminal_status_correct=result.terminal_status_correct,
        required_node_coverage=result.required_node_coverage,
        forbidden_node_violation_count=result.forbidden_node_violation_count,
        tool_recall=result.tool_recall,
        citation_requirement_met=result.citation_requirement_met,
        schema_valid=result.schema_valid,
        failures=result.failures,
        observed=observed,
        tool_precision=result.tool_precision,
        tool_parameter_validity=result.tool_parameter_validity,
        forbidden_tool_violation_count=result.forbidden_tool_violation_count,
        recovery_correct=result.recovery_correct,
        current_version_hit=result.current_version_hit,
        acl_violation_count=result.acl_violation_count,
        trace_id=root.trace_id,
        diagnostics=result.diagnostics,
    )
    return result, recorder, memory


class TracedAgentEvalHarness:
    """Like :class:`AgentEvalHarness` but always records a per-case trace."""

    def __init__(self, runtime_factory: Callable[[], GraphRuntime]) -> None:
        self.runtime_factory = runtime_factory
        self.last_exporters: list[InMemoryTraceExporter] = []

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        dataset_version: str | None = None,
    ) -> EvalReport:
        case_list = list(cases)
        if not case_list:
            raise ValueError("At least one EvalCase is required")
        results: list[EvalCaseResult] = []
        self.last_exporters = []
        for case in case_list:
            exporter = InMemoryTraceExporter()
            result, _, memory = run_case_with_tracing(
                case,
                self.runtime_factory,
                exporter=exporter,
            )
            results.append(result)
            self.last_exporters.append(memory)
        # Reuse aggregate/report construction from the untraced harness path.
        base = AgentEvalHarness(self.runtime_factory)
        # Build report without re-running cases.
        from puncture_agent.observability.eval_harness import (
            EvalReport as _EvalReport,
            _aggregate_metrics,
            evaluate_release_gates,
        )

        report = _EvalReport(
            case_count=len(results),
            passed_case_count=sum(item.passed for item in results),
            metrics=_aggregate_metrics(results),
            cases=tuple(results),
            dataset_version=dataset_version,
        )
        blocked, reasons = evaluate_release_gates(report)
        if blocked or reasons:
            report = _EvalReport(
                case_count=report.case_count,
                passed_case_count=report.passed_case_count,
                metrics=report.metrics,
                cases=report.cases,
                schema_version=report.schema_version,
                metrics_schema_version=report.metrics_schema_version,
                dataset_version=report.dataset_version,
                release_blocked=blocked,
                release_block_reasons=reasons,
            )
        del base
        return report


def _default_runtime_factory() -> GraphRuntime:
    from puncture_agent.agent import build_mock_handlers

    root = Path(__file__).resolve().parents[3]
    return GraphRuntime(root / "graph" / "main_graph.json", build_mock_handlers())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline agent evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a versioned dataset")
    run_parser.add_argument("--dataset", required=True, help="JSON/JSONL dataset path")
    run_parser.add_argument("--output", help="Write EvalReport JSON to this path")
    run_parser.add_argument(
        "--traced",
        action="store_true",
        help="Attach TraceRecorder and stamp trace_id on each case",
    )
    run_parser.add_argument(
        "--fail-on-release-block",
        action="store_true",
        help="Exit non-zero when release gates fail",
    )

    compare_parser = sub.add_parser("compare", help="Compare baseline and candidate reports")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", help="Write RegressionReport JSON")
    compare_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when release is blocked",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        schema, version, cases = load_eval_dataset(args.dataset)
        if args.traced:
            report = TracedAgentEvalHarness(_default_runtime_factory).run(
                cases, dataset_version=version
            )
        else:
            report = AgentEvalHarness(_default_runtime_factory).run(
                cases, dataset_version=version
            )
        payload = report.to_json()
        if args.output:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        print(
            f"# schema={schema} dataset={version} "
            f"passed={report.passed_case_count}/{report.case_count} "
            f"blocked={report.release_blocked}",
            file=sys.stderr,
        )
        if args.fail_on_release_block and report.release_blocked:
            return 2
        return 0 if report.passed_case_count == report.case_count else 1

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    regression = compare_reports(baseline, candidate)
    payload = regression.to_json()
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.fail_on_regression and regression.release_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
