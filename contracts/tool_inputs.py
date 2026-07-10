"""Fixed request contracts for the ten algorithm tools.

Large volumes are always passed by :class:`ArtifactRef`; no request embeds CT
voxels or masks.  Each request carries a :class:`ToolCallContext` so retries,
traces, and case isolation remain deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import ArtifactRef
from .common import ToolCallContext
from .domain import CandidatePath, DangerMaskSpec, LabelDefinition, LabelQualityThreshold
from .enums import AngleReference, ArtifactType, Connectivity, CoordinateSystem, InferencePrecision, SkinSurfaceMethod
from .geometry import WorldPoint


def _require_same_case(context: ToolCallContext, *artifacts: ArtifactRef | None) -> None:
    mismatches = [
        artifact.artifact_id
        for artifact in artifacts
        if artifact is not None and artifact.case_id != context.case_id
    ]
    if mismatches:
        raise ValueError(f"artifacts do not belong to context.case_id: {', '.join(mismatches)}")


def _require_unique_label_values(labels: tuple[LabelDefinition, ...]) -> None:
    values = [label.value for label in labels]
    names = [label.name for label in labels]
    if len(values) != len(set(values)) or len(names) != len(set(names)):
        raise ValueError("label names and values must be unique")


@dataclass(frozen=True, slots=True)
class LabelMappingEntry:
    source_name: str
    source_value: int
    target_name: str
    target_value: int

    def __post_init__(self) -> None:
        if not self.source_name or not self.target_name:
            raise ValueError("source_name and target_name are required")
        if self.source_value < 0 or self.target_value < 0:
            raise ValueError("label values must be non-negative")


@dataclass(frozen=True, slots=True)
class InspectCaseMetadataRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    related_artifacts: tuple[ArtifactRef, ...] = ()
    required_artifact_types: tuple[ArtifactType, ...] = ()
    require_same_geometry: bool = True
    verify_checksums: bool = True

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.ct_artifact, *self.related_artifacts)


@dataclass(frozen=True, slots=True)
class ConvertMcsToNiftiRequest:
    context: ToolCallContext
    mcs_artifact: ArtifactRef
    reference_ct_artifact: ArtifactRef
    label_mapping: tuple[LabelMappingEntry, ...]
    output_coordinate_system: CoordinateSystem
    output_dtype: str = "uint16"
    overwrite: bool = False

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.mcs_artifact, self.reference_ct_artifact)
        if self.mcs_artifact.artifact_type is not ArtifactType.MCS_SEGMENTATION:
            raise ValueError("mcs_artifact must have type MCS_SEGMENTATION")
        if self.reference_ct_artifact.artifact_type is not ArtifactType.CT_VOLUME:
            raise ValueError("reference_ct_artifact must have type CT_VOLUME")
        if not self.label_mapping:
            raise ValueError("label_mapping must not be empty")
        if self.output_dtype not in {"uint8", "uint16", "int16"}:
            raise ValueError("output_dtype must be uint8, uint16, or int16")
        source_values = [entry.source_value for entry in self.label_mapping]
        target_values = [entry.target_value for entry in self.label_mapping]
        if len(source_values) != len(set(source_values)) or len(target_values) != len(set(target_values)):
            raise ValueError("source and target label values must each be unique")


@dataclass(frozen=True, slots=True)
class ValidateLabelSchemaRequest:
    context: ToolCallContext
    labelmap_artifact: ArtifactRef
    expected_labels: tuple[LabelDefinition, ...]
    allow_unknown_values: bool = False
    require_all_required_labels: bool = True

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.labelmap_artifact)
        if not self.expected_labels:
            raise ValueError("expected_labels must not be empty")
        _require_unique_label_values(self.expected_labels)


@dataclass(frozen=True, slots=True)
class RunSegmentationRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    model_id: str
    model_version: str
    requested_labels: tuple[str, ...]
    precision: InferencePrecision = InferencePrecision.FP16
    device_id: int = 0
    output_probability_maps: bool = False

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.ct_artifact)
        if self.ct_artifact.artifact_type is not ArtifactType.CT_VOLUME:
            raise ValueError("ct_artifact must have type CT_VOLUME")
        if not self.model_id or not self.model_version or not self.requested_labels:
            raise ValueError("model_id, model_version, and requested_labels are required")
        if len(self.requested_labels) != len(set(self.requested_labels)):
            raise ValueError("requested_labels must be unique")
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidateSegmentationResultRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    segmentation_artifact: ArtifactRef
    expected_labels: tuple[LabelDefinition, ...]
    quality_thresholds: tuple[LabelQualityThreshold, ...]
    require_geometry_match: bool = True

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.ct_artifact, self.segmentation_artifact)
        if not self.expected_labels:
            raise ValueError("expected_labels must not be empty")
        _require_unique_label_values(self.expected_labels)
        threshold_names = [threshold.label_name for threshold in self.quality_thresholds]
        if len(threshold_names) != len(set(threshold_names)):
            raise ValueError("quality threshold label names must be unique")


@dataclass(frozen=True, slots=True)
class ExtractSkinSurfaceRequest:
    context: ToolCallContext
    skin_mask_artifact: ArtifactRef
    method: SkinSurfaceMethod = SkinSurfaceMethod.EROSION_DIFFERENCE
    thickness_mm: float = 2.0
    connectivity: Connectivity = Connectivity.TWENTY_SIX
    keep_largest_component: bool = True

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.skin_mask_artifact)
        if self.thickness_mm <= 0:
            raise ValueError("thickness_mm must be positive")


@dataclass(frozen=True, slots=True)
class GenerateCandidatePathsRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    skin_surface_artifact: ArtifactRef
    target_artifact: ArtifactRef
    lesion_artifact: ArtifactRef | None
    target_point_world_mm: WorldPoint | None
    max_needle_length_mm: float
    max_insertion_angle_deg: float
    angle_reference: AngleReference
    max_candidates: int
    entry_sampling_step_mm: float
    planner_version: str

    def __post_init__(self) -> None:
        _require_same_case(
            self.context,
            self.ct_artifact,
            self.skin_surface_artifact,
            self.target_artifact,
            self.lesion_artifact,
        )
        if self.max_needle_length_mm <= 0:
            raise ValueError("max_needle_length_mm must be positive")
        if not 0 < self.max_insertion_angle_deg <= 90:
            raise ValueError("max_insertion_angle_deg must be in (0, 90]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.entry_sampling_step_mm <= 0:
            raise ValueError("entry_sampling_step_mm must be positive")
        if not self.planner_version:
            raise ValueError("planner_version is required")


@dataclass(frozen=True, slots=True)
class EvaluatePathSafetyRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    candidate_paths: tuple[CandidatePath, ...]
    danger_masks: tuple[DangerMaskSpec, ...]
    needle_radius_mm: float
    path_sampling_step_mm: float = 0.5
    reject_warning_intersection: bool = False

    def __post_init__(self) -> None:
        _require_same_case(
            self.context,
            self.ct_artifact,
            *(spec.artifact for spec in self.danger_masks),
        )
        if not self.candidate_paths:
            raise ValueError("candidate_paths must not be empty")
        if not self.danger_masks:
            raise ValueError("danger_masks must not be empty")
        if self.needle_radius_mm <= 0 or self.path_sampling_step_mm <= 0:
            raise ValueError("needle radius and sampling step must be positive")
        candidate_ids = [path.candidate_id for path in self.candidate_paths]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_ids must be unique")


@dataclass(frozen=True, slots=True)
class EvaluateIntraoperativeRiskRequest:
    context: ToolCallContext
    ct_artifact: ArtifactRef
    planned_entry_world_mm: WorldPoint
    current_tip_world_mm: WorldPoint
    insertion_depth_mm: float
    danger_masks: tuple[DangerMaskSpec, ...]
    lung_mask_artifact: ArtifactRef | None
    skin_mask_artifact: ArtifactRef | None
    risk_rule_version: str

    def __post_init__(self) -> None:
        _require_same_case(
            self.context,
            self.ct_artifact,
            self.lung_mask_artifact,
            self.skin_mask_artifact,
            *(spec.artifact for spec in self.danger_masks),
        )
        if self.insertion_depth_mm < 0:
            raise ValueError("insertion_depth_mm must be non-negative")
        if not self.danger_masks:
            raise ValueError("danger_masks must not be empty")
        if not self.risk_rule_version:
            raise ValueError("risk_rule_version is required")


@dataclass(frozen=True, slots=True)
class VerifySkinPenetrationRequest:
    context: ToolCallContext
    skin_mask_artifact: ArtifactRef
    planned_entry_world_mm: WorldPoint
    current_tip_world_mm: WorldPoint
    insertion_depth_mm: float
    sampling_step_voxel: float = 0.5
    min_depth_for_slip_mm: float = 5.0
    skin_label_value: int = 1

    def __post_init__(self) -> None:
        _require_same_case(self.context, self.skin_mask_artifact)
        if self.insertion_depth_mm < 0:
            raise ValueError("insertion_depth_mm must be non-negative")
        if not 0 < self.sampling_step_voxel <= 1:
            raise ValueError("sampling_step_voxel must be in (0, 1]")
        if self.min_depth_for_slip_mm <= 0:
            raise ValueError("min_depth_for_slip_mm must be positive")
        if self.skin_label_value < 0:
            raise ValueError("skin_label_value must be non-negative")


TOOL_REQUEST_TYPES = {
    "inspect_case_metadata": InspectCaseMetadataRequest,
    "convert_mcs_to_nifti": ConvertMcsToNiftiRequest,
    "validate_label_schema": ValidateLabelSchemaRequest,
    "run_segmentation": RunSegmentationRequest,
    "validate_segmentation_result": ValidateSegmentationResultRequest,
    "extract_skin_surface": ExtractSkinSurfaceRequest,
    "generate_candidate_paths": GenerateCandidatePathsRequest,
    "evaluate_path_safety": EvaluatePathSafetyRequest,
    "evaluate_intraoperative_risk": EvaluateIntraoperativeRiskRequest,
    "verify_skin_penetration": VerifySkinPenetrationRequest,
}
