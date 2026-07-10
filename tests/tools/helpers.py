"""Shared dependency-free fixtures for tool tests."""

from __future__ import annotations

from dataclasses import replace

from contracts.artifacts import ArtifactRef
from contracts.common import ToolCallContext
from contracts.domain import CandidatePath, DangerMaskSpec, LabelDefinition, LabelQualityThreshold, SafetyMargin
from contracts.enums import (
    AngleReference,
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
    RiskStructure,
)
from contracts.geometry import VolumeGeometry, WorldPoint
from contracts.tool_inputs import (
    ConvertMcsToNiftiRequest,
    EvaluateIntraoperativeRiskRequest,
    EvaluatePathSafetyRequest,
    ExtractSkinSurfaceRequest,
    GenerateCandidatePathsRequest,
    InspectCaseMetadataRequest,
    LabelMappingEntry,
    RunSegmentationRequest,
    ValidateLabelSchemaRequest,
    ValidateSegmentationResultRequest,
    VerifySkinPenetrationRequest,
)


def geometry(*, origin_x: float = 0.0) -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0, 1.0, 1.5),
        origin_mm=(origin_x, 0.0, 0.0),
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=CoordinateSystem.LPS,
    )


def context() -> ToolCallContext:
    return ToolCallContext(
        request_id="req-001",
        trace_id="trace-001",
        case_id="case-001",
        caller="unit-test",
        idempotency_key="idem-001",
        requested_at="2026-07-10T00:00:00Z",
    )


def artifact(
    artifact_type: ArtifactType,
    suffix: str,
    *,
    metadata: dict[str, str] | None = None,
    volume_geometry: VolumeGeometry | None = None,
    status: ArtifactStatus = ArtifactStatus.AVAILABLE,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"case-001-{suffix}",
        case_id="case-001",
        artifact_type=artifact_type,
        uri=f"mock://case-001/{suffix}",
        checksum_sha256="a" * 64,
        status=status,
        geometry=volume_geometry or geometry(),
        producer_name="unit-test",
        producer_version="1",
        metadata=metadata or {},
    )


def labels() -> tuple[LabelDefinition, ...]:
    return (
        LabelDefinition("background", 0, True),
        LabelDefinition("skin", 1, True),
        LabelDefinition("lung", 2, True),
        LabelDefinition("heart", 3, True),
    )


def danger_specs(*, metadata: dict[str, str] | None = None) -> tuple[DangerMaskSpec, ...]:
    return (
        DangerMaskSpec(
            structure=RiskStructure.HEART,
            artifact=artifact(ArtifactType.DANGER_MASK, "heart", metadata=metadata),
            safety_margin=SafetyMargin(warning_mm=8.0, stop_mm=3.0),
        ),
        DangerMaskSpec(
            structure=RiskStructure.BONE,
            artifact=artifact(ArtifactType.DANGER_MASK, "bone"),
            safety_margin=SafetyMargin(warning_mm=5.0, stop_mm=2.0),
        ),
    )


def candidate_paths() -> tuple[CandidatePath, ...]:
    return (
        CandidatePath("path-001", WorldPoint(20, 40, 10), WorldPoint(55, 65, 60), 65.6, 18, "LOCAL_SURFACE_NORMAL", 1),
        CandidatePath("path-002", WorldPoint(23, 37, 11), WorldPoint(55, 65, 60), 64.8, 24, "LOCAL_SURFACE_NORMAL", 2),
    )


def inspect_request(**changes) -> InspectCaseMetadataRequest:
    request = InspectCaseMetadataRequest(
        context(),
        artifact(ArtifactType.CT_VOLUME, "ct"),
        (artifact(ArtifactType.NIFTI_LABELMAP, "labels"),),
        (ArtifactType.NIFTI_LABELMAP,),
    )
    return replace(request, **changes)


def conversion_request(**changes) -> ConvertMcsToNiftiRequest:
    request = ConvertMcsToNiftiRequest(
        context=context(),
        mcs_artifact=artifact(ArtifactType.MCS_SEGMENTATION, "mcs"),
        reference_ct_artifact=artifact(ArtifactType.CT_VOLUME, "ct"),
        label_mapping=(LabelMappingEntry("Skin", 1, "skin", 1), LabelMappingEntry("Lung", 2, "lung", 2)),
        output_coordinate_system=CoordinateSystem.LPS,
    )
    return replace(request, **changes)


def label_validation_request(**changes) -> ValidateLabelSchemaRequest:
    request = ValidateLabelSchemaRequest(
        context(),
        artifact(ArtifactType.NIFTI_LABELMAP, "labels", metadata={"label_values": "0,1,2,3"}),
        labels(),
    )
    return replace(request, **changes)


def segmentation_request(**changes) -> RunSegmentationRequest:
    request = RunSegmentationRequest(
        context(),
        artifact(ArtifactType.CT_VOLUME, "ct"),
        "nnunet-puncture",
        "v1",
        ("skin", "lung", "heart"),
    )
    return replace(request, **changes)


def segmentation_validation_request(**changes) -> ValidateSegmentationResultRequest:
    request = ValidateSegmentationResultRequest(
        context(),
        artifact(ArtifactType.CT_VOLUME, "ct"),
        artifact(ArtifactType.SEGMENTATION_MASK, "seg", metadata={"label_values": "0,1,2,3"}),
        labels(),
        (LabelQualityThreshold("skin", 100, 10), LabelQualityThreshold("heart", 100, 5)),
    )
    return replace(request, **changes)


def skin_request(**changes) -> ExtractSkinSurfaceRequest:
    request = ExtractSkinSurfaceRequest(context(), artifact(ArtifactType.SEGMENTATION_MASK, "skin-mask"))
    return replace(request, **changes)


def candidate_request(**changes) -> GenerateCandidatePathsRequest:
    request = GenerateCandidatePathsRequest(
        context=context(),
        ct_artifact=artifact(ArtifactType.CT_VOLUME, "ct"),
        skin_surface_artifact=artifact(ArtifactType.SKIN_SURFACE_MASK, "skin-surface"),
        target_artifact=artifact(ArtifactType.TARGET_MASK, "target"),
        lesion_artifact=None,
        target_point_world_mm=WorldPoint(55, 65, 60),
        max_needle_length_mm=120.0,
        max_insertion_angle_deg=45.0,
        angle_reference=AngleReference.LOCAL_SURFACE_NORMAL,
        max_candidates=3,
        entry_sampling_step_mm=2.0,
        planner_version="planner-v1",
    )
    return replace(request, **changes)


def safety_request(**changes) -> EvaluatePathSafetyRequest:
    request = EvaluatePathSafetyRequest(
        context(), artifact(ArtifactType.CT_VOLUME, "ct"), candidate_paths(), danger_specs(), 1.0
    )
    return replace(request, **changes)


def risk_request(**changes) -> EvaluateIntraoperativeRiskRequest:
    request = EvaluateIntraoperativeRiskRequest(
        context=context(),
        ct_artifact=artifact(ArtifactType.CT_VOLUME, "ct"),
        planned_entry_world_mm=WorldPoint(20, 40, 10),
        current_tip_world_mm=WorldPoint(30, 45, 25),
        insertion_depth_mm=20.0,
        danger_masks=danger_specs(),
        lung_mask_artifact=artifact(ArtifactType.SEGMENTATION_MASK, "lung", metadata={"mock_tip_inside": "true"}),
        skin_mask_artifact=artifact(ArtifactType.SEGMENTATION_MASK, "skin"),
        risk_rule_version="risk-v1",
    )
    return replace(request, **changes)


def penetration_request(**changes) -> VerifySkinPenetrationRequest:
    request = VerifySkinPenetrationRequest(
        context(),
        artifact(ArtifactType.SKIN_SURFACE_MASK, "skin", metadata={"mock_crossed_skin": "true"}),
        WorldPoint(20, 40, 10),
        WorldPoint(30, 45, 25),
        20.0,
    )
    return replace(request, **changes)
