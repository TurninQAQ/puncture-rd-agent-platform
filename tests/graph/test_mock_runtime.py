from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    GraphRuntime,
    TaskType,
    VerificationStatus,
    build_mock_handlers,
)


def build_runtime() -> GraphRuntime:
    return GraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
    )


class MockRuntimeTests(unittest.TestCase):
    def test_planning_and_safety_flow_reaches_verified_report(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="请对 Case-102 做路径规划和皮肤穿透安全评估"
            )
        )
        self.assertEqual(TaskType.PLANNING_SAFETY, state.task_type)
        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(VerificationStatus.PASS, state.verification_status)
        self.assertEqual(3, len(state.candidate_paths))
        self.assertEqual(["path-001", "path-002"], state.safety_result["accepted_candidate_ids"])
        self.assertTrue(state.skin_penetration_result["penetrated"])
        self.assertIn(
            "planning_safety_subgraph.generate_candidate_paths", state.visited_nodes
        )
        self.assertEqual("Case-102", state.final_report["case_id"])

    def test_mcs_data_flow_runs_conversion_and_segmentation(self) -> None:
        state = build_runtime().run(
            AgentState(user_query="检查 Case-203 的 MCS 标签和分割模型结果")
        )
        self.assertEqual(TaskType.DATA_MODEL_VALIDATION, state.task_type)
        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertTrue(state.metadata["requires_conversion"])
        self.assertIn("labels_nifti", state.artifacts)
        self.assertIn("segmentation_masks", state.artifacts)
        called_tools = [item["tool_name"] for item in state.tool_calls]
        self.assertIn("convert_mcs_to_nifti", called_tools)
        self.assertIn("run_segmentation", called_tools)

    def test_missing_case_id_stops_before_any_algorithm_tool(self) -> None:
        state = build_runtime().run(
            AgentState(user_query="执行路径规划和安全评估")
        )
        self.assertEqual(AgentStatus.AWAITING_INPUT, state.status)
        self.assertIn("case_id", state.final_report["missing_fields"])
        self.assertFalse(state.tool_calls)
        self.assertNotIn("planning_safety_subgraph", state.visited_nodes)

    def test_no_feasible_path_is_a_valid_terminal_outcome(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="为 Case-304 进行针道规划",
                metadata={"force_no_feasible_path": True},
            )
        )
        self.assertEqual(AgentStatus.COMPLETED_WITH_NO_RESULT, state.status)
        self.assertEqual(
            VerificationStatus.NO_FEASIBLE_PATH, state.verification_status
        )
        self.assertEqual([], state.candidate_paths)
        self.assertNotIn(
            "planning_safety_subgraph.evaluate_path_safety", state.visited_nodes
        )

    def test_one_time_tool_timeout_is_retried_from_subgraph_boundary(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="对 Case-405 做路径规划和风险判断",
                metadata={"fail_tool_once": ["generate_candidate_paths"]},
            )
        )
        calls = [
            call
            for call in state.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(2, len(calls))
        self.assertEqual(["FAILED", "SUCCESS"], [call["status"] for call in calls])
        self.assertEqual(1, state.retry_count)
        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(VerificationStatus.PASS, state.verification_status)

    def test_persistent_timeout_exhausts_retry_budget(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="对 Case-406 做路径规划和风险判断",
                metadata={"fail_tool_always": ["generate_candidate_paths"]},
            )
        )
        calls = [
            call
            for call in state.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(2, len(calls))
        self.assertTrue(all(call["status"] == "FAILED" for call in calls))
        self.assertEqual(1, state.retry_count)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, state.status)
        self.assertEqual(VerificationStatus.MANUAL_REVIEW, state.verification_status)

    def test_non_retryable_tool_error_is_not_retried(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="对 Case-407 做路径规划和风险判断",
                metadata={"fail_tool_non_retryable": ["generate_candidate_paths"]},
            )
        )
        calls = [
            call
            for call in state.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(1, len(calls))
        self.assertEqual(0, state.retry_count)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, state.status)

    def test_missing_planning_artifact_stops_before_planner(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="对 Case-408 做路径规划",
                metadata={"missing_required_artifacts": ["target"]},
            )
        )
        self.assertEqual(AgentStatus.AWAITING_INPUT, state.status)
        self.assertIn("target", state.final_report["missing_fields"])
        self.assertFalse(state.tool_calls)

    def test_geometry_failure_requires_manual_review_without_segmentation(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="检查 Case-409 的标签和分割",
                metadata={"force_geometry_mismatch": True},
            )
        )
        called_tools = [call["tool_name"] for call in state.tool_calls]
        self.assertEqual(["inspect_case_metadata"], called_tools)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, state.status)
        self.assertEqual(VerificationStatus.MANUAL_REVIEW, state.verification_status)

    def test_checkpoint_round_trip_preserves_state(self) -> None:
        original = AgentState(
            user_query="检查 Case-506 标签",
            case_id="Case-506",
            metadata={"nested": {"value": 1}},
        )
        restored = AgentState.from_dict(original.to_dict())
        self.assertEqual(original.to_dict(), restored.to_dict())
        restored.metadata["nested"]["value"] = 2
        self.assertEqual(1, original.metadata["nested"]["value"])


if __name__ == "__main__":
    unittest.main()
