"""Schema snapshots: any failure requires an explicit contract migration."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from contracts.artifacts import ArtifactPublicView  # noqa: E402
from contracts.common import ToolResponseEnvelope, to_json  # noqa: E402
from contracts.enums import ArtifactType, ToolExecutionStatus  # noqa: E402
from contracts.tool_inputs import TOOL_REQUEST_TYPES  # noqa: E402
from contracts.tool_outputs import TOOL_RESULT_TYPES  # noqa: E402
from puncture_agent.tooling import TOOL_DEFINITIONS, build_mock_registry  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from tools.helpers import inspect_request  # noqa: E402


EXPECTED_REQUEST_FIELDS = {
    "inspect_case_metadata": ("context", "ct_artifact", "related_artifacts", "required_artifact_types", "require_same_geometry", "verify_checksums"),
    "convert_mcs_to_nifti": ("context", "mcs_artifact", "reference_ct_artifact", "label_mapping", "output_coordinate_system", "output_dtype", "overwrite"),
    "validate_label_schema": ("context", "labelmap_artifact", "expected_labels", "allow_unknown_values", "require_all_required_labels"),
    "run_segmentation": ("context", "ct_artifact", "model_id", "model_version", "requested_labels", "precision", "device_id", "output_probability_maps"),
    "validate_segmentation_result": ("context", "ct_artifact", "segmentation_artifact", "expected_labels", "quality_thresholds", "require_geometry_match"),
    "extract_skin_surface": ("context", "skin_mask_artifact", "method", "thickness_mm", "connectivity", "keep_largest_component"),
    "generate_candidate_paths": ("context", "ct_artifact", "skin_surface_artifact", "target_artifact", "lesion_artifact", "target_point_world_mm", "max_needle_length_mm", "max_insertion_angle_deg", "angle_reference", "max_candidates", "entry_sampling_step_mm", "planner_version"),
    "evaluate_path_safety": ("context", "ct_artifact", "candidate_paths", "danger_masks", "needle_radius_mm", "path_sampling_step_mm", "reject_warning_intersection"),
    "evaluate_intraoperative_risk": ("context", "ct_artifact", "planned_entry_world_mm", "current_tip_world_mm", "insertion_depth_mm", "danger_masks", "lung_mask_artifact", "skin_mask_artifact", "risk_rule_version"),
    "verify_skin_penetration": ("context", "skin_mask_artifact", "planned_entry_world_mm", "current_tip_world_mm", "insertion_depth_mm", "sampling_step_voxel", "min_depth_for_slip_mm", "skin_label_value"),
}

EXPECTED_RESULT_FIELDS = {
    "inspect_case_metadata": ("case_id", "ct_geometry", "inspections", "required_types_present", "all_geometries_compatible", "ready_for_next_stage", "issues"),
    "convert_mcs_to_nifti": ("output_artifact", "applied_mappings", "geometry_matches_reference", "output_dtype", "total_nonzero_voxels"),
    "validate_label_schema": ("valid", "observed_label_values", "missing_required_label_names", "unknown_label_values", "issues"),
    "run_segmentation": ("segmentation_artifact", "model_id", "model_version", "precision", "produced_labels", "inference_time_ms", "peak_gpu_memory_mb"),
    "validate_segmentation_result": ("valid", "geometry_matches_ct", "label_results", "issues", "recommended_action"),
    "extract_skin_surface": ("surface_artifact", "source_voxel_count", "surface_voxel_count", "requested_thickness_mm", "effective_thickness_mm", "components_removed", "geometry_matches_source"),
    "generate_candidate_paths": ("candidates", "sampled_entry_point_count", "rejected_by_length_count", "rejected_by_angle_count", "planner_version", "elapsed_ms"),
    "evaluate_path_safety": ("assessments", "accepted_candidate_ids", "rejected_candidate_ids", "safest_candidate_id", "elapsed_ms"),
    "evaluate_intraoperative_risk": ("overall_level", "tip_world_mm", "insertion_depth_mm", "flags", "needle_in_lung", "large_vessel_penetration", "requires_manual_review", "rule_version"),
    "verify_skin_penetration": ("status", "crossed_skin", "crossing_point_world_mm", "first_skin_sample_index", "samples_evaluated", "path_length_mm", "insertion_depth_mm", "evidence"),
}


class ToolContractTests(unittest.TestCase):
    def test_exactly_ten_tools_share_request_and_result_catalogs(self) -> None:
        self.assertEqual(set(EXPECTED_REQUEST_FIELDS), set(TOOL_DEFINITIONS))
        self.assertEqual(set(TOOL_REQUEST_TYPES), set(TOOL_RESULT_TYPES))
        self.assertEqual(10, len(TOOL_DEFINITIONS))

    def test_request_field_snapshots(self) -> None:
        for name, expected in EXPECTED_REQUEST_FIELDS.items():
            with self.subTest(tool=name):
                self.assertEqual(expected, tuple(field.name for field in fields(TOOL_REQUEST_TYPES[name])))

    def test_result_field_snapshots(self) -> None:
        for name, expected in EXPECTED_RESULT_FIELDS.items():
            with self.subTest(tool=name):
                self.assertEqual(expected, tuple(field.name for field in fields(TOOL_RESULT_TYPES[name])))

    def test_mock_response_is_json_serializable_and_preserves_identity(self) -> None:
        response = build_mock_registry().execute("inspect_case_metadata", inspect_request())
        payload = json.loads(to_json(response))
        self.assertEqual("req-001", payload["request_id"])
        self.assertEqual("trace-001", payload["trace_id"])
        self.assertEqual("SUCCESS", payload["status"])
        self.assertEqual("inspect_case_metadata", payload["tool_name"])

    def test_public_artifact_projection_does_not_leak_uri_or_checksum(self) -> None:
        artifact = inspect_request().ct_artifact
        projection = artifact.to_public_view()
        self.assertIsInstance(projection, ArtifactPublicView)
        payload = json.loads(to_json(projection))
        self.assertNotIn("uri", payload)
        self.assertNotIn("checksum_sha256", payload)
        self.assertNotIn("metadata", payload)

    def test_response_invariants_reject_success_without_result(self) -> None:
        with self.assertRaises(ValueError):
            ToolResponseEnvelope(
                request_id="r",
                trace_id="t",
                tool_name="x",
                tool_version="1",
                status=ToolExecutionStatus.SUCCESS,
                result=None,
                artifacts=(),
                metrics=(),
                warnings=(),
                error=None,
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:00:00Z",
            )

    def test_registry_rejects_wrong_request_type(self) -> None:
        with self.assertRaises(TypeError):
            build_mock_registry().execute("inspect_case_metadata", object())

    def test_enum_wire_value_is_stable(self) -> None:
        self.assertEqual("CT_VOLUME", ArtifactType.CT_VOLUME.value)


if __name__ == "__main__":
    unittest.main()
