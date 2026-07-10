from __future__ import annotations

import sys
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
from puncture_agent.observability.eval_harness import (  # noqa: E402
    AgentEvalHarness,
    EvalCase,
)


def runtime_factory() -> GraphRuntime:
    return GraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
    )


class EvalHarnessTests(unittest.TestCase):
    def test_reference_cases_pass_all_contract_metrics(self) -> None:
        cases = [
            EvalCase(
                case_id="planning-success",
                query="对 Case-701 做路径规划和安全评估",
                expected_task_type=TaskType.PLANNING_SAFETY,
                expected_terminal_status=AgentStatus.SUCCEEDED,
                required_nodes=(
                    "generate_candidate_paths",
                    "evaluate_path_safety",
                    "verify_skin_penetration",
                    "result_verifier",
                ),
                forbidden_nodes=("convert_mcs_to_nifti",),
                expected_tools=(
                    "generate_candidate_paths",
                    "evaluate_path_safety",
                    "evaluate_intraoperative_risk",
                    "verify_skin_penetration",
                ),
                minimum_citations=2,
            ),
            EvalCase(
                case_id="data-success",
                query="检查 Case-702 的 MCS 标签与分割结果",
                expected_task_type=TaskType.DATA_MODEL_VALIDATION,
                expected_terminal_status=AgentStatus.SUCCEEDED,
                required_nodes=(
                    "convert_mcs_to_nifti",
                    "validate_label_schema",
                    "run_segmentation",
                ),
                forbidden_nodes=("generate_candidate_paths",),
                expected_tools=(
                    "inspect_case_metadata",
                    "convert_mcs_to_nifti",
                    "validate_label_schema",
                    "run_segmentation",
                    "validate_segmentation_result",
                    "extract_skin_surface",
                ),
                minimum_citations=2,
            ),
        ]
        report = AgentEvalHarness(runtime_factory).run(cases)
        self.assertEqual(2, report.passed_case_count)
        self.assertEqual(1.0, report.metrics["task_success_rate"])
        self.assertEqual(0.0, report.metrics["forbidden_node_violation_rate"])
        self.assertEqual(1.0, report.metrics["report_schema_validity_rate"])

    def test_failed_expectation_contains_actionable_diagnostics(self) -> None:
        case = EvalCase(
            case_id="intentional-failure",
            query="对 Case-703 做路径规划",
            expected_task_type=TaskType.DATA_MODEL_VALIDATION,
            expected_terminal_status=AgentStatus.FAILED,
            required_nodes=("convert_mcs_to_nifti",),
            expected_tools=("run_segmentation",),
            minimum_citations=10,
        )
        report = AgentEvalHarness(runtime_factory).run([case])
        self.assertEqual(0, report.passed_case_count)
        failures = report.cases[0].failures
        self.assertGreaterEqual(len(failures), 4)
        self.assertTrue(any("task_type" in item for item in failures))
        self.assertTrue(any("required nodes" in item for item in failures))

    def test_empty_eval_suite_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgentEvalHarness(runtime_factory).run([])


if __name__ == "__main__":
    unittest.main()
