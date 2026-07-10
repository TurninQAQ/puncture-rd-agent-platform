from __future__ import annotations

import unittest
import os
import subprocess
import sys
from pathlib import Path

from contracts.enums import ArtifactType, CoordinateSystem
from contracts.geometry import VolumeGeometry
from puncture_agent.artifacts.identity import canonical_json, build_artifact_idempotency_key


class ArtifactIdentityTests(unittest.TestCase):
    def test_mapping_and_set_order_do_not_change_canonical_json(self) -> None:
        first = {"b": {3, 1, 2}, "a": {"y": 2, "x": 1}}
        second = {"a": {"x": 1, "y": 2}, "b": {2, 3, 1}}
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_enum_and_unicode_are_stable(self) -> None:
        encoded = canonical_json({"type": ArtifactType.CT_VOLUME, "name": "标签"})
        self.assertEqual('{"name":"标签","type":"CT_VOLUME"}', encoded)

    def test_non_finite_float_is_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                canonical_json({"value": value})

    def test_non_string_mapping_keys_cannot_collide(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json({1: "numeric"})
        with self.assertRaises(TypeError):
            canonical_json({1: "numeric", "1": "string"})

    def test_equivalent_invocations_have_same_key(self) -> None:
        first = build_artifact_idempotency_key(
            tool_name="run_segmentation",
            tool_version="1.2.0",
            input_artifact_ids=("ct-2", "ct-1"),
            parameters={"precision": "FP16", "device": 0},
            geometry_fingerprints=("geo-b", "geo-a"),
        )
        second = build_artifact_idempotency_key(
            tool_name="run_segmentation",
            tool_version="1.2.0",
            input_artifact_ids=("ct-1", "ct-2"),
            parameters={"device": 0, "precision": "FP16"},
            geometry_fingerprints=("geo-a", "geo-b"),
        )
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_version_parameter_and_geometry_changes_change_key(self) -> None:
        base = dict(
            tool_name="evaluate_path_safety",
            tool_version="1.0.0",
            input_artifact_ids=("path-1", "heart-1"),
            parameters={"radius_mm": 5.0},
            geometry_fingerprints=("geo-1",),
        )
        reference = build_artifact_idempotency_key(**base)
        variants = (
            {**base, "tool_version": "1.0.1"},
            {**base, "parameters": {"radius_mm": 4.0}},
            {**base, "geometry_fingerprints": ("geo-2",)},
        )
        for variant in variants:
            self.assertNotEqual(reference, build_artifact_idempotency_key(**variant))

    def test_geometry_fingerprint_changes_for_every_spatial_dimension(self) -> None:
        base = dict(
            size_ijk=(10, 11, 12),
            spacing_mm=(1.0, 1.1, 1.2),
            origin_mm=(2.0, 3.0, 4.0),
            direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            coordinate_system=CoordinateSystem.LPS,
        )
        reference = VolumeGeometry(**base).geometry_fingerprint
        variants = (
            {**base, "size_ijk": (10, 11, 13)},
            {**base, "spacing_mm": (1.0, 1.1, 1.3)},
            {**base, "origin_mm": (2.0, 3.0, 4.1)},
            {
                **base,
                "direction_cosines": (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            },
            {**base, "coordinate_system": CoordinateSystem.RAS},
        )
        for variant in variants:
            self.assertNotEqual(reference, VolumeGeometry(**variant).geometry_fingerprint)

    def test_geometry_fingerprint_is_stable_across_processes(self) -> None:
        script = """
from contracts.enums import CoordinateSystem
from contracts.geometry import VolumeGeometry
print(VolumeGeometry(
    size_ijk=(10, 11, 12),
    spacing_mm=(1.0, 1.1, 1.2),
    origin_mm=(2.0, 3.0, 4.0),
    direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    coordinate_system=CoordinateSystem.LPS,
).geometry_fingerprint)
"""
        project_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(project_root), str(project_root / "src"))
        )
        outputs = [
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=environment,
                text=True,
            ).strip()
            for _ in range(2)
        ]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(64, len(outputs[0]))


if __name__ == "__main__":
    unittest.main()
