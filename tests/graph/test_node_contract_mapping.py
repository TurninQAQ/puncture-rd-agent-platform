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
    VerificationStatus,
    build_mock_handlers,
)


def success(result):
    return {"status": "SUCCESS", "result": result, "error": None}


class FrozenShapeExecutor:
    def __init__(self, *, risk_manual_review: bool = False) -> None:
        self.calls = []
        self.risk_manual_review = risk_manual_review

    def execute(self, tool_name, request):
        self.calls.append((tool_name, dict(request)))
        case_id = request["case_id"]
        candidate_paths = [
            {
                "candidate_id": "path-001",
                "entry_point_world_mm": {"x_mm": 1.0, "y_mm": 2.0, "z_mm": 3.0},
                "target_point_world_mm": {"x_mm": 4.0, "y_mm": 5.0, "z_mm": 6.0},
                "length_mm": 5.2,
                "insertion_angle_deg": 12.0,
                "angle_reference": "LOCAL_SURFACE_NORMAL",
                "rank_hint": 1,
                "path_artifact_id": None,
            },
            {
                "candidate_id": "path-002",
                "entry_point_world_mm": {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 30.0},
                "target_point_world_mm": {"x_mm": 40.0, "y_mm": 50.0, "z_mm": 60.0},
                "length_mm": 52.0,
                "insertion_angle_deg": 18.0,
                "angle_reference": "LOCAL_SURFACE_NORMAL",
                "rank_hint": 2,
                "path_artifact_id": None,
            },
        ]
        responses = {
            "inspect_case_metadata": {
                "case_id": case_id,
                "ct_geometry": {"geometry_fingerprint": "geometry-1"},
                "ready_for_next_stage": True,
                "all_geometries_compatible": True,
                "required_types_present": True,
            },
            "convert_mcs_to_nifti": {
                "output_artifact": {
                    "artifact_id": f"artifact-{case_id}-converted-labels"
                },
                "geometry_matches_reference": True,
            },
            "validate_label_schema": {
                "valid": True,
                "observed_label_values": [0, 1, 2, 3],
            },
            "run_segmentation": {
                "segmentation_artifact": {
                    "artifact_id": f"artifact-{case_id}-segmentation"
                },
                "produced_labels": [
                    {"label_name": "skin"},
                    {"label_name": "lung"},
                    {"label_name": "heart"},
                ],
                "inference_time_ms": 10.0,
                "peak_gpu_memory_mb": 128.0,
            },
            "validate_segmentation_result": {
                "valid": True,
                "geometry_matches_ct": True,
            },
            "extract_skin_surface": {
                "surface_artifact": {
                    "artifact_id": f"artifact-{case_id}-skin-surface-real"
                },
                "geometry_matches_source": True,
            },
            "generate_candidate_paths": {"candidates": candidate_paths},
            "evaluate_path_safety": {
                "assessments": [
                    {"candidate_id": "path-001", "disposition": "REJECTED"},
                    {"candidate_id": "path-002", "disposition": "ACCEPTED"},
                ],
                "accepted_candidate_ids": ["path-002"],
                "rejected_candidate_ids": ["path-001"],
                "safest_candidate_id": "path-002",
            },
            "evaluate_intraoperative_risk": {
                "overall_level": "SAFE",
                "flags": [],
                "needle_in_lung": True,
                "large_vessel_penetration": False,
                "requires_manual_review": self.risk_manual_review,
            },
            "verify_skin_penetration": {
                "status": "PENETRATED",
                "crossed_skin": True,
                "samples_evaluated": 12,
            },
        }
        return success(responses[tool_name])


def runtime(executor) -> GraphRuntime:
    return GraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(executor),
    )


class FrozenToolResultMappingTests(unittest.TestCase):
    def test_data_flow_accepts_frozen_artifact_result_fields(self) -> None:
        executor = FrozenShapeExecutor()
        state = runtime(executor).run(
            AgentState(user_query="检查 Case-901 的 MCS 标签和分割")
        )

        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(
            "artifact-Case-901-converted-labels", state.artifacts["labels_nifti"]
        )
        self.assertEqual(
            "artifact-Case-901-segmentation",
            state.artifacts["segmentation_artifact"],
        )
        self.assertEqual(
            "artifact-Case-901-skin-surface-real", state.artifacts["skin_surface"]
        )
        validation_request = next(
            request
            for name, request in executor.calls
            if name == "validate_segmentation_result"
        )
        self.assertEqual(
            {"segmentation": "artifact-Case-901-segmentation"},
            validation_request["mask_artifact_ids"],
        )

    def test_planning_uses_safest_accepted_candidate_and_filters_report(self) -> None:
        executor = FrozenShapeExecutor()
        state = runtime(executor).run(
            AgentState(user_query="对 Case-902 做路径规划和安全评估")
        )

        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        risk_request = next(
            request
            for name, request in executor.calls
            if name == "evaluate_intraoperative_risk"
        )
        self.assertEqual(
            {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 30.0},
            risk_request["planned_entry_point_world_mm"],
        )
        self.assertEqual("SAFE", state.risk_flags["overall_level"])
        self.assertTrue(state.skin_penetration_result["penetrated"])
        self.assertEqual(
            ["path-002"],
            [item["candidate_id"] for item in state.final_report["candidate_paths"]],
        )

    def test_frozen_manual_review_flag_is_fail_closed(self) -> None:
        state = runtime(FrozenShapeExecutor(risk_manual_review=True)).run(
            AgentState(user_query="对 Case-903 做路径规划和安全评估")
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, state.status)
        self.assertEqual(VerificationStatus.MANUAL_REVIEW, state.verification_status)


if __name__ == "__main__":
    unittest.main()
