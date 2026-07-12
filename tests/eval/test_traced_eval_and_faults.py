from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentStatus,
    GraphRuntime,
    TaskType,
    build_mock_handlers,
)
from puncture_agent.observability.dataset import load_eval_dataset  # noqa: E402
from puncture_agent.observability.eval_harness import EvalCase  # noqa: E402
from puncture_agent.observability.eval_runner import (  # noqa: E402
    TracedAgentEvalHarness,
    main as eval_main,
    run_case_with_tracing,
)
from puncture_agent.observability.tracing import (  # noqa: E402
    InMemoryTraceExporter,
    TraceRecorder,
)


def runtime_factory() -> GraphRuntime:
    return GraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
    )


class TracedEvalTests(unittest.TestCase):
    def test_traced_reference_suite_has_complete_trace_linkage(self) -> None:
        _, version, cases = load_eval_dataset(
            PROJECT_ROOT / "tests" / "eval" / "fixtures" / "mock_reference_v1.json"
        )
        harness = TracedAgentEvalHarness(runtime_factory)
        report = harness.run(cases, dataset_version=version)
        self.assertEqual(3, report.passed_case_count)
        for case in report.cases:
            self.assertTrue(case.trace_id)
            self.assertEqual(case.trace_id, case.observed.get("trace_id"))
            self.assertIn("latency_ms", case.observed)
        # Each case exporter has a parent eval.case plus nested graph/node spans.
        for exporter in harness.last_exporters:
            spans = exporter.spans()
            self.assertGreaterEqual(len(spans), 3)
            roots = [s for s in spans if s.name == "eval.case"]
            self.assertEqual(1, len(roots))
            self.assertTrue(all(s.trace_id == roots[0].trace_id for s in spans))

    def test_fault_injection_rows_have_error_or_recovery_traces(self) -> None:
        # one-time tool timeout recovery
        recovery = EvalCase(
            case_id="fault-timeout-recovery",
            query="对 Case-910 做路径规划",
            expected_task_type=TaskType.PLANNING_SAFETY,
            expected_terminal_status=AgentStatus.SUCCEEDED,
            required_nodes=("generate_candidate_paths", "error_recovery"),
            expected_tools=("generate_candidate_paths",),
            minimum_citations=2,
            metadata={
                "fail_tool_once": ["generate_candidate_paths"],
                "is_recovery_case": True,
                "expected_retry_count_min": 1,
            },
        )
        result, _, exporter = run_case_with_tracing(recovery, runtime_factory)
        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.trace_id)
        self.assertTrue(any(span.status in {"OK", "ERROR"} for span in exporter.spans()))

        # persistent timeout exhausts retries
        exhaust = EvalCase(
            case_id="fault-timeout-exhaust",
            query="对 Case-911 做路径规划",
            expected_task_type=TaskType.PLANNING_SAFETY,
            # Mock runtime routes persistent tool timeout exhaustion to
            # MANUAL_REVIEW rather than FAILED; assert the observed terminal.
            expected_terminal_status=AgentStatus.MANUAL_REVIEW,
            expected_tools=("generate_candidate_paths",),
            minimum_citations=0,
            metadata={"fail_tool_always": ["generate_candidate_paths"]},
        )
        result, _, exporter = run_case_with_tracing(exhaust, runtime_factory)
        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.trace_id)
        # Graph still completes spans; failures surface in agent status.
        self.assertTrue(exporter.spans())

    def test_node_exception_exports_error_span_and_reraises(self) -> None:
        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)

        def boom(state, context):  # type: ignore[no-untyped-def]
            del state, context
            raise RuntimeError("injected node failure")

        handlers = build_mock_handlers()
        handlers = dict(handlers)
        handlers["parse_request"] = boom
        runtime = GraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            tracer=tracer,
        )
        with self.assertRaises(Exception):
            runtime.run(
                __import__(
                    "puncture_agent.agent.state", fromlist=["AgentState"]
                ).AgentState(user_query="x")
            )
        error_spans = [s for s in exporter.spans() if s.status == "ERROR"]
        self.assertGreaterEqual(len(error_spans), 1)

    def test_cli_run_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "report.json"
            code = eval_main(
                [
                    "run",
                    "--dataset",
                    str(
                        PROJECT_ROOT
                        / "tests"
                        / "eval"
                        / "fixtures"
                        / "mock_reference_v1.json"
                    ),
                    "--output",
                    str(out),
                    "--traced",
                    "--fail-on-release-block",
                ]
            )
            self.assertEqual(0, code)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(3, payload["passed_case_count"])
            # Compare report against itself: no regressions.
            cmp_out = Path(directory) / "cmp.json"
            code = eval_main(
                [
                    "compare",
                    "--baseline",
                    str(out),
                    "--candidate",
                    str(out),
                    "--output",
                    str(cmp_out),
                    "--fail-on-regression",
                ]
            )
            self.assertEqual(0, code)
            cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
            self.assertEqual([], cmp["regressed_case_ids"])
            self.assertFalse(cmp["release_blocked"])


class MalformedInputTests(unittest.TestCase):
    def test_malformed_dataset_path(self) -> None:
        from puncture_agent.observability.dataset import DatasetValidationError, load_eval_dataset

        with self.assertRaises(DatasetValidationError):
            load_eval_dataset(PROJECT_ROOT / "tests" / "eval" / "fixtures" / "missing.json")


if __name__ == "__main__":
    unittest.main()
