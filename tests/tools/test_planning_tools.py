from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.enums import ArtifactType, ErrorCode, PathDisposition, PenetrationStatus, RiskLevel, RiskStructure  # noqa: E402
from puncture_agent.tooling import build_mock_registry  # noqa: E402
from tools.helpers import (  # noqa: E402
    artifact,
    candidate_request,
    danger_specs,
    penetration_request,
    risk_request,
    safety_request,
)


class PlanningToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_mock_registry()

    def test_candidate_generation_obeys_count_length_and_angle(self) -> None:
        request = candidate_request(max_candidates=2, max_insertion_angle_deg=25.0)
        response = self.registry.execute("generate_candidate_paths", request)
        self.assertEqual(2, len(response.result.candidates))
        self.assertTrue(all(path.length_mm <= 120.0 for path in response.result.candidates))
        self.assertTrue(all(path.insertion_angle_deg <= 25.0 for path in response.result.candidates))

    def test_no_candidate_has_stable_error(self) -> None:
        response = self.registry.execute(
            "generate_candidate_paths", candidate_request(max_needle_length_mm=10.0)
        )
        self.assertEqual(ErrorCode.NO_CANDIDATE_PATH, response.error.code)

    def test_safety_rejects_injected_collision(self) -> None:
        specs = danger_specs(metadata={"mock_collision_candidate_ids": "path-002", "mock_clearance_mm": "10"})
        response = self.registry.execute("evaluate_path_safety", safety_request(danger_masks=specs))
        by_id = {item.candidate_id: item for item in response.result.assessments}
        self.assertEqual(PathDisposition.REJECTED, by_id["path-002"].disposition)
        self.assertIn("path-002", response.result.rejected_candidate_ids)

    def test_intraoperative_risk_enters_stop_zone(self) -> None:
        specs = danger_specs(metadata={"mock_tip_distance_mm": "1.0"})
        response = self.registry.execute(
            "evaluate_intraoperative_risk", risk_request(danger_masks=specs)
        )
        self.assertEqual(RiskLevel.STOP, response.result.overall_level)
        self.assertTrue(response.result.requires_manual_review)

    def test_skin_penetration_and_slip_are_distinguishable(self) -> None:
        penetrated = self.registry.execute("verify_skin_penetration", penetration_request())
        self.assertEqual(PenetrationStatus.PENETRATED, penetrated.result.status)
        slipped_request = penetration_request(
            skin_mask_artifact=artifact(ArtifactType.SKIN_SURFACE_MASK, "skin", metadata={"mock_crossed_skin": "false"}),
            insertion_depth_mm=20.0,
        )
        slipped = self.registry.execute("verify_skin_penetration", slipped_request)
        self.assertEqual(PenetrationStatus.SUSPECTED_SLIP, slipped.result.status)


if __name__ == "__main__":
    unittest.main()
