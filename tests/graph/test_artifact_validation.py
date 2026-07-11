from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from contracts.artifacts import ArtifactPublicView  # noqa: E402
from contracts.common import to_primitive  # noqa: E402
from contracts.enums import (  # noqa: E402
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
)
from contracts.geometry import VolumeGeometry  # noqa: E402
from puncture_agent.agent.artifact_validation import (  # noqa: E402
    ArtifactValidationRecord,
    ArtifactValidationRejected,
    ArtifactValidationUnavailable,
    RegistryToolArtifactValidator,
)
from puncture_agent.artifacts import ArtifactRegistryError  # noqa: E402
from puncture_agent.mcp import McpPrincipal  # noqa: E402


CASE_ID = "case-001"
CALLER = "agent-runtime"


def geometry(*, spacing_delta: float = 0.0, origin_x: float = 0.0) -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0 + spacing_delta, 1.0, 1.5),
        origin_mm=(origin_x, 0.0, 0.0),
        direction_cosines=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        coordinate_system=CoordinateSystem.LPS,
    )


def record(
    artifact_id: str,
    artifact_type: ArtifactType,
    *,
    artifact_geometry: VolumeGeometry | None = None,
    case_id: str = CASE_ID,
    status: ArtifactStatus = ArtifactStatus.AVAILABLE,
    producer_name: str = "fixture",
    producer_version: str = "1",
    parents: tuple[str, ...] = (),
) -> ArtifactValidationRecord:
    selected_geometry = artifact_geometry if artifact_geometry is not None else geometry()
    return ArtifactValidationRecord(
        public_view=ArtifactPublicView(
            artifact_id=artifact_id,
            case_id=case_id,
            artifact_type=artifact_type,
            status=status,
            producer_name=producer_name,
            producer_version=producer_version,
            geometry_fingerprint=selected_geometry.geometry_fingerprint,
        ),
        geometry=selected_geometry,
        parent_artifact_ids=parents,
    )


class FakeValidationRegistry:
    def __init__(self, records: dict[str, ArtifactValidationRecord]) -> None:
        self.records = dict(records)
        self.unavailable_ids: set[str] = set()
        self.calls: list[str] = []

    def get_validation_record(self, artifact_id: str) -> ArtifactValidationRecord:
        self.calls.append(artifact_id)
        if artifact_id in self.unavailable_ids:
            raise ArtifactRegistryError(
                "STORAGE_ERROR",
                "database connection failed",
                retryable=True,
            )
        try:
            return self.records[artifact_id]
        except KeyError as exc:
            raise ArtifactRegistryError(
                "NOT_FOUND",
                "artifact was not found",
            ) from exc


def context() -> dict[str, Any]:
    return {
        "request_id": "req-001",
        "trace_id": "trace-001",
        "case_id": CASE_ID,
        "caller": CALLER,
        "idempotency_key": "idem-001",
        "requested_at": "2026-07-11T12:00:00Z",
        "deadline_epoch_ms": None,
    }


def response(
    tool_name: str,
    result_field: str,
    output: ArtifactValidationRecord,
) -> dict[str, Any]:
    public = to_primitive(output.public_view)
    return {
        "tool_name": tool_name,
        "tool_version": "1.0.0",
        "status": "SUCCESS",
        "result": {result_field: dict(public)},
        "artifacts": [dict(public)],
    }


class RegistryToolArtifactValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ct = record("ct-001", ArtifactType.CT_VOLUME)
        self.mcs = record("mcs-001", ArtifactType.MCS_SEGMENTATION)
        self.skin = record("skin-001", ArtifactType.SEGMENTATION_MASK)
        self.planning_skin = record(
            "planning-skin-001", ArtifactType.SKIN_SURFACE_MASK
        )
        self.target = record("target-001", ArtifactType.TARGET_MASK)
        self.lesion = record("lesion-001", ArtifactType.LESION_MASK)
        self.danger = record("danger-001", ArtifactType.DANGER_MASK)
        self.converted = record(
            "nifti-001",
            ArtifactType.NIFTI_LABELMAP,
            artifact_geometry=geometry(spacing_delta=0.00005),
            producer_name="convert_mcs_to_nifti",
            producer_version="1.0.0",
            parents=(self.mcs.public_view.artifact_id, self.ct.public_view.artifact_id),
        )
        self.segmentation = record(
            "seg-001",
            ArtifactType.SEGMENTATION_MASK,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parents=(self.ct.public_view.artifact_id,),
        )
        self.surface = record(
            "surface-001",
            ArtifactType.SKIN_SURFACE_MASK,
            producer_name="extract_skin_surface",
            producer_version="1.0.0",
            parents=(self.skin.public_view.artifact_id,),
        )
        planning_parent_ids = tuple(
            item.public_view.artifact_id
            for item in (self.ct, self.planning_skin, self.target, self.lesion)
        )
        self.path_one = record(
            "path-001",
            ArtifactType.PATH_MASK,
            producer_name="generate_candidate_paths",
            producer_version="1.0.0",
            parents=planning_parent_ids,
        )
        self.path_two = record(
            "path-002",
            ArtifactType.PATH_MASK,
            producer_name="generate_candidate_paths",
            producer_version="1.0.0",
            parents=planning_parent_ids,
        )
        self.registry = FakeValidationRegistry(
            {
                item.public_view.artifact_id: item
                for item in (
                    self.ct,
                    self.mcs,
                    self.skin,
                    self.planning_skin,
                    self.target,
                    self.lesion,
                    self.danger,
                    self.converted,
                    self.segmentation,
                    self.surface,
                    self.path_one,
                    self.path_two,
                )
            }
        )
        self.validator = RegistryToolArtifactValidator(self.registry)
        self.principal = McpPrincipal(CALLER, (CASE_ID,))

    def conversion_arguments(self) -> dict[str, Any]:
        return {
            "context": context(),
            "mcs_artifact": {"artifact_id": self.mcs.public_view.artifact_id},
            "reference_ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
        }

    def segmentation_arguments(self) -> dict[str, Any]:
        return {
            "context": context(),
            "ct_artifact": {
                "artifact_id": self.ct.public_view.artifact_id,
                "case_id": CASE_ID,
                "artifact_type": "CT_VOLUME",
            },
        }

    def surface_arguments(self) -> dict[str, Any]:
        return {
            "context": context(),
            "skin_mask_artifact": {"artifact_id": self.skin.public_view.artifact_id},
        }

    def planning_arguments(self) -> dict[str, Any]:
        return {
            "context": context(),
            "ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
            "skin_surface_artifact": {
                "artifact_id": self.planning_skin.public_view.artifact_id
            },
            "target_artifact": {"artifact_id": self.target.public_view.artifact_id},
            "lesion_artifact": {"artifact_id": self.lesion.public_view.artifact_id},
        }

    @staticmethod
    def candidate_response(
        path_ids: tuple[str | None, ...],
        envelope_records: tuple[ArtifactValidationRecord, ...],
    ) -> dict[str, Any]:
        return {
            "tool_name": "generate_candidate_paths",
            "tool_version": "1.0.0",
            "status": "SUCCESS",
            "result": {
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index + 1}",
                        "path_artifact_id": path_id,
                    }
                    for index, path_id in enumerate(path_ids)
                ]
            },
            "artifacts": [
                to_primitive(item.public_view) for item in envelope_records
            ],
        }

    def test_trusted_conversion_output_is_accepted_with_compatible_geometry(self) -> None:
        arguments = self.conversion_arguments()

        self.validator.validate_request(
            "convert_mcs_to_nifti", arguments, self.principal
        )
        self.validator.validate_response(
            "convert_mcs_to_nifti",
            arguments,
            response("convert_mcs_to_nifti", "output_artifact", self.converted),
            self.principal,
        )

    def test_trusted_segmentation_output_is_accepted(self) -> None:
        arguments = self.segmentation_arguments()

        self.validator.validate_request("run_segmentation", arguments, self.principal)
        self.validator.validate_response(
            "run_segmentation",
            arguments,
            response("run_segmentation", "segmentation_artifact", self.segmentation),
            self.principal,
        )

    def test_trusted_surface_output_is_accepted(self) -> None:
        arguments = self.surface_arguments()

        self.validator.validate_request(
            "extract_skin_surface", arguments, self.principal
        )
        self.validator.validate_response(
            "extract_skin_surface",
            arguments,
            response("extract_skin_surface", "surface_artifact", self.surface),
            self.principal,
        )

    def test_unregistered_remote_output_is_rejected(self) -> None:
        forged = record(
            "unregistered-001",
            ArtifactType.SEGMENTATION_MASK,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parents=(self.ct.public_view.artifact_id,),
        )

        with self.assertRaisesRegex(
            ArtifactValidationRejected,
            "not authorized and registered",
        ):
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                response("run_segmentation", "segmentation_artifact", forged),
                self.principal,
            )

    def test_authoritative_wrong_output_type_is_rejected(self) -> None:
        wrong = record(
            "wrong-type-001",
            ArtifactType.CT_VOLUME,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parents=(self.ct.public_view.artifact_id,),
        )
        self.registry.records[wrong.public_view.artifact_id] = wrong

        with self.assertRaisesRegex(ArtifactValidationRejected, "unexpected Artifact type"):
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                response("run_segmentation", "segmentation_artifact", wrong),
                self.principal,
            )

    def test_authoritative_non_available_output_is_rejected(self) -> None:
        pending = record(
            "pending-001",
            ArtifactType.SEGMENTATION_MASK,
            status=ArtifactStatus.PENDING,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parents=(self.ct.public_view.artifact_id,),
        )
        self.registry.records[pending.public_view.artifact_id] = pending

        with self.assertRaisesRegex(ArtifactValidationRejected, "not AVAILABLE"):
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                response("run_segmentation", "segmentation_artifact", pending),
                self.principal,
            )

    def test_authoritative_wrong_producer_or_version_is_rejected(self) -> None:
        scenarios = (
            ("attacker", "1.0.0", "unexpected authoritative producer"),
            ("run_segmentation", "999", "producer version"),
        )
        for producer, version, expected in scenarios:
            with self.subTest(producer=producer, version=version):
                wrong = record(
                    f"producer-{producer}-{version}",
                    ArtifactType.SEGMENTATION_MASK,
                    producer_name=producer,
                    producer_version=version,
                    parents=(self.ct.public_view.artifact_id,),
                )
                self.registry.records[wrong.public_view.artifact_id] = wrong
                with self.assertRaisesRegex(ArtifactValidationRejected, expected):
                    self.validator.validate_response(
                        "run_segmentation",
                        self.segmentation_arguments(),
                        response("run_segmentation", "segmentation_artifact", wrong),
                        self.principal,
                    )

    def test_remote_fingerprint_must_exactly_match_registry_public_view(self) -> None:
        forged_response = response(
            "run_segmentation", "segmentation_artifact", self.segmentation
        )
        forged_response["result"]["segmentation_artifact"][
            "geometry_fingerprint"
        ] = "forged-fingerprint"
        forged_response["artifacts"][0]["geometry_fingerprint"] = "forged-fingerprint"

        with self.assertRaisesRegex(ArtifactValidationRejected, "authoritative Registry view"):
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                forged_response,
                self.principal,
            )

    def test_output_requires_exact_direct_parent_lineage(self) -> None:
        wrong = record(
            "wrong-lineage-001",
            ArtifactType.SEGMENTATION_MASK,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parents=(self.mcs.public_view.artifact_id,),
        )
        self.registry.records[wrong.public_view.artifact_id] = wrong

        with self.assertRaisesRegex(ArtifactValidationRejected, "direct lineage"):
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                response("run_segmentation", "segmentation_artifact", wrong),
                self.principal,
            )

    def test_principal_acl_is_checked_before_registry_access(self) -> None:
        denied = McpPrincipal(CALLER, ("case-other",))

        with self.assertRaisesRegex(ArtifactValidationRejected, "not permitted"):
            self.validator.validate_request(
                "run_segmentation",
                self.segmentation_arguments(),
                denied,
            )
        self.assertEqual([], self.registry.calls)

    def test_registry_unavailability_is_retryable_and_fail_closed(self) -> None:
        self.registry.unavailable_ids.add(self.segmentation.public_view.artifact_id)

        with self.assertRaises(ArtifactValidationUnavailable) as caught:
            self.validator.validate_response(
                "run_segmentation",
                self.segmentation_arguments(),
                response("run_segmentation", "segmentation_artifact", self.segmentation),
                self.principal,
            )
        self.assertTrue(caught.exception.retryable)
        self.assertEqual("DEPENDENCY_FAILED", caught.exception.code)

    def test_request_rejects_wrong_case_status_type_and_geometry(self) -> None:
        scenarios = (
            (
                replace(
                    self.mcs,
                    public_view=replace(self.mcs.public_view, case_id="case-other"),
                ),
                "another case",
            ),
            (
                replace(
                    self.mcs,
                    public_view=replace(
                        self.mcs.public_view,
                        status=ArtifactStatus.INVALID,
                    ),
                ),
                "not AVAILABLE",
            ),
            (
                replace(
                    self.mcs,
                    public_view=replace(
                        self.mcs.public_view,
                        artifact_type=ArtifactType.CT_VOLUME,
                    ),
                ),
                "unexpected Artifact type",
            ),
            (
                record(
                    self.mcs.public_view.artifact_id,
                    ArtifactType.MCS_SEGMENTATION,
                    artifact_geometry=geometry(origin_x=25.0),
                ),
                "geometry is incompatible",
            ),
        )
        for replacement, expected in scenarios:
            with self.subTest(expected=expected):
                self.registry.records[self.mcs.public_view.artifact_id] = replacement
                with self.assertRaisesRegex(ArtifactValidationRejected, expected):
                    self.validator.validate_request(
                        "convert_mcs_to_nifti",
                        self.conversion_arguments(),
                        self.principal,
                    )
                self.registry.records[self.mcs.public_view.artifact_id] = self.mcs

    def test_segmentation_and_surface_require_exact_output_geometry(self) -> None:
        compatible_but_not_exact = geometry(spacing_delta=0.00005)
        scenarios = (
            (
                "run_segmentation",
                self.segmentation_arguments(),
                "segmentation_artifact",
                record(
                    "seg-near-001",
                    ArtifactType.SEGMENTATION_MASK,
                    artifact_geometry=compatible_but_not_exact,
                    producer_name="run_segmentation",
                    producer_version="1.0.0",
                    parents=(self.ct.public_view.artifact_id,),
                ),
            ),
            (
                "extract_skin_surface",
                self.surface_arguments(),
                "surface_artifact",
                record(
                    "surface-near-001",
                    ArtifactType.SKIN_SURFACE_MASK,
                    artifact_geometry=compatible_but_not_exact,
                    producer_name="extract_skin_surface",
                    producer_version="1.0.0",
                    parents=(self.skin.public_view.artifact_id,),
                ),
            ),
        )
        for tool_name, arguments, result_field, output in scenarios:
            with self.subTest(tool_name=tool_name):
                self.registry.records[output.public_view.artifact_id] = output
                with self.assertRaisesRegex(ArtifactValidationRejected, "not exact"):
                    self.validator.validate_response(
                        tool_name,
                        arguments,
                        response(tool_name, result_field, output),
                        self.principal,
                    )

    def test_non_artifact_tool_cannot_smuggle_envelope_artifacts(self) -> None:
        arguments = {
            "context": context(),
            "ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
            "related_artifacts": [],
            "require_same_geometry": True,
        }
        normalized = {
            "tool_name": "inspect_case_metadata",
            "tool_version": "1.0.0",
            "status": "SUCCESS",
            "result": {"ready_for_next_stage": True},
            "artifacts": [to_primitive(self.segmentation.public_view)],
        }

        with self.assertRaisesRegex(ArtifactValidationRejected, "not allowed to publish"):
            self.validator.validate_response(
                "inspect_case_metadata",
                arguments,
                normalized,
                self.principal,
            )

    def test_request_artifact_type_policies_match_each_tool_role(self) -> None:
        self.validator.validate_request(
            "validate_label_schema",
            {
                "context": context(),
                "labelmap_artifact": {
                    "artifact_id": self.skin.public_view.artifact_id
                },
            },
            self.principal,
        )
        self.validator.validate_request(
            "validate_segmentation_result",
            {
                "context": context(),
                "ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
                "segmentation_artifact": {
                    "artifact_id": self.converted.public_view.artifact_id
                },
            },
            self.principal,
        )
        self.validator.validate_request(
            "evaluate_intraoperative_risk",
            {
                "context": context(),
                "ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
                "danger_masks": [
                    {"artifact": {"artifact_id": self.danger.public_view.artifact_id}}
                ],
                "lung_mask_artifact": {
                    "artifact_id": self.danger.public_view.artifact_id
                },
                "skin_mask_artifact": {
                    "artifact_id": self.planning_skin.public_view.artifact_id
                },
            },
            self.principal,
        )
        self.validator.validate_request(
            "verify_skin_penetration",
            {
                "context": context(),
                "skin_mask_artifact": {
                    "artifact_id": self.planning_skin.public_view.artifact_id
                },
            },
            self.principal,
        )

        with self.assertRaisesRegex(ArtifactValidationRejected, "unexpected Artifact type"):
            self.validator.validate_request(
                "evaluate_path_safety",
                {
                    "context": context(),
                    "ct_artifact": {"artifact_id": self.ct.public_view.artifact_id},
                    "candidate_paths": [],
                    "danger_masks": [
                        {"artifact": {"artifact_id": self.skin.public_view.artifact_id}}
                    ],
                },
                self.principal,
            )
        with self.assertRaisesRegex(ArtifactValidationRejected, "unexpected Artifact type"):
            self.validator.validate_request(
                "verify_skin_penetration",
                {
                    "context": context(),
                    "skin_mask_artifact": {
                        "artifact_id": self.converted.public_view.artifact_id
                    },
                },
                self.principal,
            )

    def test_candidate_generation_accepts_zero_or_multiple_trusted_path_outputs(self) -> None:
        arguments = self.planning_arguments()
        self.validator.validate_request(
            "generate_candidate_paths", arguments, self.principal
        )

        self.validator.validate_response(
            "generate_candidate_paths",
            arguments,
            self.candidate_response((None,), ()),
            self.principal,
        )
        self.validator.validate_response(
            "generate_candidate_paths",
            arguments,
            self.candidate_response(
                (
                    self.path_one.public_view.artifact_id,
                    self.path_two.public_view.artifact_id,
                ),
                (self.path_two, self.path_one),
            ),
            self.principal,
        )

    def test_candidate_path_output_ids_and_envelope_must_match_exactly(self) -> None:
        arguments = self.planning_arguments()
        scenarios = (
            (
                self.candidate_response(
                    (
                        self.path_one.public_view.artifact_id,
                        self.path_one.public_view.artifact_id,
                    ),
                    (self.path_one,),
                ),
                "must be unique",
            ),
            (
                self.candidate_response(
                    (
                        self.path_one.public_view.artifact_id,
                        self.path_two.public_view.artifact_id,
                    ),
                    (self.path_one,),
                ),
                "exactly match",
            ),
            (
                self.candidate_response((None,), (self.path_one,)),
                "exactly match",
            ),
        )
        for normalized, expected in scenarios:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ArtifactValidationRejected, expected):
                    self.validator.validate_response(
                        "generate_candidate_paths",
                        arguments,
                        normalized,
                        self.principal,
                    )

    def test_candidate_path_registry_claims_fail_closed(self) -> None:
        arguments = self.planning_arguments()
        expected_parents = tuple(
            item.public_view.artifact_id
            for item in (self.ct, self.planning_skin, self.target, self.lesion)
        )
        compatible_but_not_exact = geometry(spacing_delta=0.00005)
        scenarios = (
            (
                record(
                    "path-pending",
                    ArtifactType.PATH_MASK,
                    status=ArtifactStatus.PENDING,
                    producer_name="generate_candidate_paths",
                    producer_version="1.0.0",
                    parents=expected_parents,
                ),
                "not AVAILABLE",
            ),
            (
                record(
                    "path-wrong-case",
                    ArtifactType.PATH_MASK,
                    case_id="case-other",
                    producer_name="generate_candidate_paths",
                    producer_version="1.0.0",
                    parents=expected_parents,
                ),
                "another case",
            ),
            (
                record(
                    "path-wrong-type",
                    ArtifactType.CANDIDATE_PATH_SET,
                    producer_name="generate_candidate_paths",
                    producer_version="1.0.0",
                    parents=expected_parents,
                ),
                "unexpected Artifact type",
            ),
            (
                record(
                    "path-wrong-producer",
                    ArtifactType.PATH_MASK,
                    producer_name="attacker",
                    producer_version="1.0.0",
                    parents=expected_parents,
                ),
                "unexpected authoritative producer",
            ),
            (
                record(
                    "path-wrong-lineage",
                    ArtifactType.PATH_MASK,
                    producer_name="generate_candidate_paths",
                    producer_version="1.0.0",
                    parents=(self.ct.public_view.artifact_id,),
                ),
                "direct lineage",
            ),
            (
                record(
                    "path-wrong-geometry",
                    ArtifactType.PATH_MASK,
                    artifact_geometry=compatible_but_not_exact,
                    producer_name="generate_candidate_paths",
                    producer_version="1.0.0",
                    parents=expected_parents,
                ),
                "not exact",
            ),
        )
        for output, expected in scenarios:
            with self.subTest(expected=expected):
                self.registry.records[output.public_view.artifact_id] = output
                with self.assertRaisesRegex(ArtifactValidationRejected, expected):
                    self.validator.validate_response(
                        "generate_candidate_paths",
                        arguments,
                        self.candidate_response(
                            (output.public_view.artifact_id,),
                            (output,),
                        ),
                        self.principal,
                    )

        forged = self.candidate_response(
            (self.path_one.public_view.artifact_id,),
            (self.path_one,),
        )
        forged["artifacts"][0]["geometry_fingerprint"] = "forged"
        with self.assertRaisesRegex(ArtifactValidationRejected, "Registry view"):
            self.validator.validate_response(
                "generate_candidate_paths",
                arguments,
                forged,
                self.principal,
            )

        with self.assertRaisesRegex(ArtifactValidationRejected, "reuse an input"):
            self.validator.validate_response(
                "generate_candidate_paths",
                arguments,
                self.candidate_response(
                    (self.ct.public_view.artifact_id,),
                    (self.ct,),
                ),
                self.principal,
            )


if __name__ == "__main__":
    unittest.main()
