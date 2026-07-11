from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentStatus,
    LangGraphRuntime,
    TaskType,
    build_mock_handlers,
    langgraph_available,
)
from puncture_agent.observability.eval_harness import (  # noqa: E402
    AgentEvalHarness,
    EvalCase,
)
from tests.graph.test_langgraph_runtime import FakeLangGraphApi  # noqa: E402


def langgraph_runtime_factory() -> LangGraphRuntime:
    options = {} if langgraph_available() else {"langgraph_api": FakeLangGraphApi()}
    return LangGraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
        **options,
    )


class LangGraphEvalTests(unittest.TestCase):
    def test_reference_cases_pass_contract_metrics_on_production_runtime(self) -> None:
        cases = (
            EvalCase(
                case_id="langgraph-planning-success",
                query="对 Case-721 做路径规划和安全评估",
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
                case_id="langgraph-data-success",
                query="检查 Case-722 的 MCS 标签与分割结果",
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
        )

        report = AgentEvalHarness(langgraph_runtime_factory).run(cases)

        self.assertEqual(2, report.case_count)
        self.assertEqual(2, report.passed_case_count)
        self.assertEqual(1.0, report.metrics["task_success_rate"])
        self.assertEqual(1.0, report.metrics["routing_accuracy"])
        self.assertEqual(1.0, report.metrics["terminal_status_accuracy"])
        self.assertEqual(1.0, report.metrics["required_node_coverage"])
        self.assertEqual(1.0, report.metrics["tool_recall"])
        self.assertEqual(1.0, report.metrics["citation_pass_rate"])
        self.assertEqual(1.0, report.metrics["report_schema_validity_rate"])
        self.assertEqual(0.0, report.metrics["forbidden_node_violation_rate"])
        self.assertTrue(all(case.passed for case in report.cases))


if __name__ == "__main__":
    unittest.main()
