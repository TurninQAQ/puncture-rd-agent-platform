"""Fixed result contracts for the ten algorithm tools."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import ArtifactRef
from .domain import CandidatePath, PathSafetyAssessment, RiskFlag, ValidationIssue
from .enums import InferencePrecision, PenetrationStatus, RecommendedAction, RiskLevel
from .geometry import VolumeGeometry, WorldPoint


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    artifact_id: str
    artifact_type: str
    available: bool
    checksum_valid: bool
    geometry_matches_ct: bool | None


@dataclass(frozen=True, slots=True)
class CaseMetadataResult:
    case_id: str
    ct_geometry: VolumeGeometry
    inspections: tuple[ArtifactInspection, ...]
    required_types_present: bool
    all_geometries_compatible: bool
    ready_for_next_stage: bool
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class AppliedLabelMapping:
    source_name: str
    source_value: int
    target_name: str
    target_value: int
    voxel_count: int


@dataclass(frozen=True, slots=True)
class McsToNiftiResult:
    output_artifact: ArtifactRef
    applied_mappings: tuple[AppliedLabelMapping, ...]
    geometry_matches_reference: bool
    output_dtype: str
    total_nonzero_voxels: int


@dataclass(frozen=True, slots=True)
class LabelSchemaValidationResult:
    valid: bool
    observed_label_values: tuple[int, ...]
    missing_required_label_names: tuple[str, ...]
    unknown_label_values: tuple[int, ...]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class LabelStatistics:
    label_name: str
    label_value: int
    voxel_count: int
    volume_ml: float
    component_count: int
    touches_volume_border: bool


@dataclass(frozen=True, slots=True)
class SegmentationInferenceResult:
    segmentation_artifact: ArtifactRef
    model_id: str
    model_version: str
    precision: InferencePrecision
    produced_labels: tuple[LabelStatistics, ...]
    inference_time_ms: float
    peak_gpu_memory_mb: float


@dataclass(frozen=True, slots=True)
class LabelValidationResult:
    statistics: LabelStatistics
    passed: bool
    issue_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SegmentationValidationResult:
    valid: bool
    geometry_matches_ct: bool
    label_results: tuple[LabelValidationResult, ...]
    issues: tuple[ValidationIssue, ...]
    recommended_action: RecommendedAction


@dataclass(frozen=True, slots=True)
class SkinSurfaceExtractionResult:
    surface_artifact: ArtifactRef
    source_voxel_count: int
    surface_voxel_count: int
    requested_thickness_mm: float
    effective_thickness_mm: tuple[float, float, float]
    components_removed: int
    geometry_matches_source: bool


@dataclass(frozen=True, slots=True)
class CandidatePathGenerationResult:
    candidates: tuple[CandidatePath, ...]
    sampled_entry_point_count: int
    rejected_by_length_count: int
    rejected_by_angle_count: int
    planner_version: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class PathSafetyEvaluationResult:
    assessments: tuple[PathSafetyAssessment, ...]
    accepted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    safest_candidate_id: str | None
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class IntraoperativeRiskResult:
    overall_level: RiskLevel
    tip_world_mm: WorldPoint
    insertion_depth_mm: float
    flags: tuple[RiskFlag, ...]
    needle_in_lung: bool | None
    large_vessel_penetration: bool | None
    requires_manual_review: bool
    rule_version: str


@dataclass(frozen=True, slots=True)
class SkinPenetrationResult:
    status: PenetrationStatus
    crossed_skin: bool
    crossing_point_world_mm: WorldPoint | None
    first_skin_sample_index: int | None
    samples_evaluated: int
    path_length_mm: float
    insertion_depth_mm: float
    evidence: str


TOOL_RESULT_TYPES = {
    "inspect_case_metadata": CaseMetadataResult,
    "convert_mcs_to_nifti": McsToNiftiResult,
    "validate_label_schema": LabelSchemaValidationResult,
    "run_segmentation": SegmentationInferenceResult,
    "validate_segmentation_result": SegmentationValidationResult,
    "extract_skin_surface": SkinSurfaceExtractionResult,
    "generate_candidate_paths": CandidatePathGenerationResult,
    "evaluate_path_safety": PathSafetyEvaluationResult,
    "evaluate_intraoperative_risk": IntraoperativeRiskResult,
    "verify_skin_penetration": SkinPenetrationResult,
}
