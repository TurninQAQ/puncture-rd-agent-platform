from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.enums import ArtifactType, ErrorCode  # noqa: E402
from puncture_agent.tooling import build_mock_registry  # noqa: E402
from tools.helpers import (  # noqa: E402
    artifact,
    conversion_request,
    geometry,
    inspect_request,
    label_validation_request,
)


class CaseDataToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_mock_registry()

    def test_inspect_happy_path_is_ready(self) -> None:
        response = self.registry.execute("inspect_case_metadata", inspect_request())
        self.assertTrue(response.ok)
        self.assertTrue(response.result.ready_for_next_stage)
        self.assertTrue(response.result.all_geometries_compatible)

    def test_inspect_reports_geometry_mismatch(self) -> None:
        bad_label = artifact(ArtifactType.NIFTI_LABELMAP, "bad", volume_geometry=geometry(origin_x=5.0))
        response = self.registry.execute(
            "inspect_case_metadata", inspect_request(related_artifacts=(bad_label,))
        )
        self.assertFalse(response.result.ready_for_next_stage)
        self.assertIn("GEOMETRY_MISMATCH", {issue.code for issue in response.result.issues})

    def test_conversion_preserves_reference_geometry_and_mapping(self) -> None:
        response = self.registry.execute("convert_mcs_to_nifti", conversion_request())
        self.assertEqual(ArtifactType.NIFTI_LABELMAP, response.result.output_artifact.artifact_type)
        self.assertTrue(response.result.geometry_matches_reference)
        self.assertEqual(2, len(response.result.applied_mappings))

    def test_conversion_injected_parser_failure_has_stable_error(self) -> None:
        request = conversion_request(
            mcs_artifact=artifact(ArtifactType.MCS_SEGMENTATION, "mcs", metadata={"mock_conversion_error": "true"})
        )
        response = self.registry.execute("convert_mcs_to_nifti", request)
        self.assertFalse(response.ok)
        self.assertEqual(ErrorCode.UNSUPPORTED_FORMAT, response.error.code)

    def test_schema_validation_detects_missing_and_unknown_labels(self) -> None:
        request = label_validation_request(
            labelmap_artifact=artifact(ArtifactType.NIFTI_LABELMAP, "labels", metadata={"label_values": "0,1,99"})
        )
        response = self.registry.execute("validate_label_schema", request)
        self.assertFalse(response.result.valid)
        self.assertEqual(("lung", "heart"), response.result.missing_required_label_names)
        self.assertEqual((99,), response.result.unknown_label_values)


if __name__ == "__main__":
    unittest.main()
