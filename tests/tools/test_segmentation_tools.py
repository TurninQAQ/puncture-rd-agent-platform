from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.enums import ArtifactType, ErrorCode  # noqa: E402
from puncture_agent.tooling import build_mock_registry  # noqa: E402
from tools.helpers import artifact, segmentation_request, segmentation_validation_request, skin_request  # noqa: E402


class SegmentationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_mock_registry()

    def test_inference_returns_versioned_artifact_and_metrics(self) -> None:
        response = self.registry.execute("run_segmentation", segmentation_request())
        self.assertEqual("v1", response.result.model_version)
        self.assertEqual(158.0, response.result.inference_time_ms)
        self.assertEqual(ArtifactType.SEGMENTATION_MASK, response.result.segmentation_artifact.artifact_type)
        self.assertEqual({"inference_time", "peak_gpu_memory"}, {metric.name for metric in response.metrics})

    def test_gpu_oom_is_retryable_structured_error(self) -> None:
        request = segmentation_request(
            ct_artifact=artifact(ArtifactType.CT_VOLUME, "ct", metadata={"mock_gpu_oom": "true"})
        )
        response = self.registry.execute("run_segmentation", request)
        self.assertEqual(ErrorCode.GPU_OUT_OF_MEMORY, response.error.code)
        self.assertTrue(response.error.retryable)

    def test_segmentation_validation_detects_required_missing_label(self) -> None:
        request = segmentation_validation_request(
            segmentation_artifact=artifact(ArtifactType.SEGMENTATION_MASK, "seg", metadata={"label_values": "0,1,2"})
        )
        response = self.registry.execute("validate_segmentation_result", request)
        self.assertFalse(response.result.valid)
        self.assertIn("REQUIRED_LABEL_MISSING", {issue.code for issue in response.result.issues})

    def test_skin_surface_preserves_geometry(self) -> None:
        response = self.registry.execute("extract_skin_surface", skin_request())
        self.assertTrue(response.result.geometry_matches_source)
        self.assertLess(response.result.surface_voxel_count, response.result.source_voxel_count)

    def test_empty_skin_mask_returns_error(self) -> None:
        request = skin_request(
            skin_mask_artifact=artifact(ArtifactType.SEGMENTATION_MASK, "skin", metadata={"mock_empty_mask": "true"})
        )
        response = self.registry.execute("extract_skin_surface", request)
        self.assertEqual(ErrorCode.EMPTY_SEGMENTATION, response.error.code)


if __name__ == "__main__":
    unittest.main()
