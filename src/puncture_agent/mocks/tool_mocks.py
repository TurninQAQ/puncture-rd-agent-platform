"""Deterministic mocks for the ten tool contracts.

Metadata keys beginning with ``mock_`` are explicit failure/result injection
points used by unit and Agent recovery tests.  Production adapters must ignore
these keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from math import dist
from typing import Any, Callable

from contracts.artifacts import ArtifactRef
from contracts.common import MetricValue, ToolResponseEnvelope
from contracts.domain import (
    CandidatePath,
    PathSafetyAssessment,
    PathStructureClearance,
    RiskFlag,
    ValidationIssue,
)
from contracts.enums import (
    ArtifactStatus,
    ArtifactType,
    ErrorCode,
    PathDisposition,
    PenetrationStatus,
    RecommendedAction,
    RiskLevel,
    RiskStructure,
    ToolExecutionStatus,
    ValidationSeverity,
)
from contracts.errors import ErrorDetail
from contracts.geometry import WorldPoint
from contracts.tool_inputs import (
    ConvertMcsToNiftiRequest,
    EvaluateIntraoperativeRiskRequest,
    EvaluatePathSafetyRequest,
    ExtractSkinSurfaceRequest,
    GenerateCandidatePathsRequest,
    InspectCaseMetadataRequest,
    RunSegmentationRequest,
    ValidateLabelSchemaRequest,
    ValidateSegmentationResultRequest,
    VerifySkinPenetrationRequest,
)
from contracts.tool_outputs import (
    AppliedLabelMapping,
    ArtifactInspection,
    CandidatePathGenerationResult,
    CaseMetadataResult,
    IntraoperativeRiskResult,
    LabelSchemaValidationResult,
    LabelStatistics,
    LabelValidationResult,
    McsToNiftiResult,
    PathSafetyEvaluationResult,
    SegmentationInferenceResult,
    SegmentationValidationResult,
    SkinPenetrationResult,
    SkinSurfaceExtractionResult,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _success(request: Any, name: str, result: Any, *, artifacts: tuple[ArtifactRef, ...] = (), metrics=(), warnings=()):
    now = _now()
    return ToolResponseEnvelope(
        request_id=request.context.request_id,
        trace_id=request.context.trace_id,
        tool_name=name,
        tool_version="1.0.0",
        status=ToolExecutionStatus.SUCCESS,
        result=result,
        artifacts=artifacts,
        metrics=tuple(metrics),
        warnings=tuple(warnings),
        error=None,
        started_at=now,
        finished_at=now,
    )


def _failed(request: Any, name: str, code: ErrorCode, message: str, *, retryable: bool = False):
    now = _now()
    return ToolResponseEnvelope(
        request_id=request.context.request_id,
        trace_id=request.context.trace_id,
        tool_name=name,
        tool_version="1.0.0",
        status=ToolExecutionStatus.FAILED,
        result=None,
        artifacts=(),
        metrics=(),
        warnings=(),
        error=ErrorDetail(code=code, message=message, retryable=retryable),
        started_at=now,
        finished_at=now,
    )


def _metadata_ints(artifact: ArtifactRef, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = artifact.metadata.get(key)
    if raw is None or not raw.strip():
        return default
    return tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))


def _mock_artifact(
    source: ArtifactRef,
    *,
    suffix: str,
    artifact_type: ArtifactType,
    producer_name: str,
    metadata: dict[str, str] | None = None,
) -> ArtifactRef:
    artifact_id = f"{source.case_id}-{suffix}"
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id=source.case_id,
        artifact_type=artifact_type,
        uri=f"mock://artifacts/{artifact_id}",
        checksum_sha256=sha256(artifact_id.encode("utf-8")).hexdigest(),
        status=ArtifactStatus.AVAILABLE,
        geometry=source.geometry,
        producer_name=producer_name,
        producer_version="mock-1",
        parent_artifact_ids=(source.artifact_id,),
        metadata=metadata or {},
    )


def mock_inspect_case_metadata(request: InspectCaseMetadataRequest):
    name = "inspect_case_metadata"
    if request.ct_artifact.geometry is None:
        return _failed(request, name, ErrorCode.GEOMETRY_MISMATCH, "CT geometry is missing")
    artifacts = (request.ct_artifact,) + request.related_artifacts
    inspections = []
    issues = []
    for artifact in artifacts:
        available = artifact.status is ArtifactStatus.AVAILABLE
        geometry_match = (
            None
            if artifact.geometry is None
            else request.ct_artifact.geometry.is_compatible_with(artifact.geometry)
        )
        checksum_valid = artifact.metadata.get("mock_checksum_invalid", "false") != "true"
        inspections.append(
            ArtifactInspection(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type.value,
                available=available,
                checksum_valid=checksum_valid,
                geometry_matches_ct=geometry_match,
            )
        )
        if not available:
            issues.append(ValidationIssue("ARTIFACT_NOT_AVAILABLE", ValidationSeverity.ERROR, "artifact unavailable", artifact.artifact_id))
        if request.verify_checksums and not checksum_valid:
            issues.append(ValidationIssue("CHECKSUM_MISMATCH", ValidationSeverity.ERROR, "checksum mismatch", artifact.artifact_id))
        if request.require_same_geometry and geometry_match is False:
            issues.append(ValidationIssue("GEOMETRY_MISMATCH", ValidationSeverity.ERROR, "geometry differs from CT", artifact.artifact_id))
    present_types = {artifact.artifact_type for artifact in artifacts}
    missing_types = [value.value for value in request.required_artifact_types if value not in present_types]
    if missing_types:
        issues.append(ValidationIssue("MISSING_ARTIFACT_TYPE", ValidationSeverity.ERROR, f"missing types: {', '.join(missing_types)}"))
    ready = not any(issue.severity is ValidationSeverity.ERROR for issue in issues)
    result = CaseMetadataResult(
        case_id=request.context.case_id,
        ct_geometry=request.ct_artifact.geometry,
        inspections=tuple(inspections),
        required_types_present=not missing_types,
        all_geometries_compatible=all(item.geometry_matches_ct is not False for item in inspections),
        ready_for_next_stage=ready,
        issues=tuple(issues),
    )
    return _success(request, name, result)


def mock_convert_mcs_to_nifti(request: ConvertMcsToNiftiRequest):
    name = "convert_mcs_to_nifti"
    if request.reference_ct_artifact.geometry is None:
        return _failed(request, name, ErrorCode.GEOMETRY_MISMATCH, "reference CT geometry is missing")
    if request.mcs_artifact.metadata.get("mock_conversion_error") == "true":
        return _failed(request, name, ErrorCode.UNSUPPORTED_FORMAT, "mock MCS parser rejected the file")
    mapping_results = tuple(
        AppliedLabelMapping(
            source_name=entry.source_name,
            source_value=entry.source_value,
            target_name=entry.target_name,
            target_value=entry.target_value,
            voxel_count=1000 * (index + 1),
        )
        for index, entry in enumerate(request.label_mapping)
    )
    output = _mock_artifact(
        request.reference_ct_artifact,
        suffix="labels-nifti",
        artifact_type=ArtifactType.NIFTI_LABELMAP,
        producer_name=name,
        metadata={"label_values": ",".join(str(entry.target_value) for entry in request.label_mapping)},
    )
    result = McsToNiftiResult(
        output_artifact=output,
        applied_mappings=mapping_results,
        geometry_matches_reference=True,
        output_dtype=request.output_dtype,
        total_nonzero_voxels=sum(item.voxel_count for item in mapping_results if item.target_value != 0),
    )
    return _success(request, name, result, artifacts=(output,), metrics=(MetricValue("conversion_time", 12.0, "ms"),))


def mock_validate_label_schema(request: ValidateLabelSchemaRequest):
    name = "validate_label_schema"
    observed = _metadata_ints(request.labelmap_artifact, "label_values", (0, 1, 2, 3))
    expected_by_value = {label.value: label for label in request.expected_labels}
    missing = tuple(
        label.name
        for label in request.expected_labels
        if label.required and label.value not in observed
    )
    unknown = tuple(value for value in observed if value not in expected_by_value)
    issues = []
    if request.require_all_required_labels:
        issues.extend(
            ValidationIssue("REQUIRED_LABEL_MISSING", ValidationSeverity.ERROR, f"required label missing: {label_name}")
            for label_name in missing
        )
    if not request.allow_unknown_values:
        issues.extend(
            ValidationIssue("UNKNOWN_LABEL_VALUE", ValidationSeverity.ERROR, f"unknown label value: {value}")
            for value in unknown
        )
    result = LabelSchemaValidationResult(
        valid=not issues,
        observed_label_values=observed,
        missing_required_label_names=missing,
        unknown_label_values=unknown,
        issues=tuple(issues),
    )
    return _success(request, name, result)


def mock_run_segmentation(request: RunSegmentationRequest):
    name = "run_segmentation"
    if request.ct_artifact.metadata.get("mock_gpu_oom") == "true":
        return _failed(request, name, ErrorCode.GPU_OUT_OF_MEMORY, "mock GPU out of memory", retryable=True)
    if request.model_version == "missing":
        return _failed(request, name, ErrorCode.MODEL_NOT_FOUND, "requested model version was not found")
    labels = tuple(
        LabelStatistics(
            label_name=label_name,
            label_value=index + 1,
            voxel_count=50_000 * (index + 1),
            volume_ml=50.0 * (index + 1),
            component_count=1,
            touches_volume_border=label_name.lower() in {"skin", "body"},
        )
        for index, label_name in enumerate(request.requested_labels)
    )
    output = _mock_artifact(
        request.ct_artifact,
        suffix=f"seg-{request.model_version}",
        artifact_type=ArtifactType.SEGMENTATION_MASK,
        producer_name=name,
        metadata={"label_values": ",".join(str(item.label_value) for item in labels)},
    )
    result = SegmentationInferenceResult(
        segmentation_artifact=output,
        model_id=request.model_id,
        model_version=request.model_version,
        precision=request.precision,
        produced_labels=labels,
        inference_time_ms=158.0,
        peak_gpu_memory_mb=2795.0,
    )
    return _success(
        request,
        name,
        result,
        artifacts=(output,),
        metrics=(MetricValue("inference_time", 158.0, "ms"), MetricValue("peak_gpu_memory", 2795.0, "MB")),
    )


def mock_validate_segmentation_result(request: ValidateSegmentationResultRequest):
    name = "validate_segmentation_result"
    geometry_matches = bool(
        request.ct_artifact.geometry
        and request.segmentation_artifact.geometry
        and request.ct_artifact.geometry.is_compatible_with(request.segmentation_artifact.geometry)
    )
    observed = _metadata_ints(request.segmentation_artifact, "label_values", tuple(label.value for label in request.expected_labels))
    issues = []
    if request.require_geometry_match and not geometry_matches:
        issues.append(ValidationIssue("GEOMETRY_MISMATCH", ValidationSeverity.ERROR, "segmentation geometry differs from CT"))
    threshold_by_name = {item.label_name: item for item in request.quality_thresholds}
    label_results = []
    for label in request.expected_labels:
        voxel_count = 50_000 if label.value in observed else 0
        stats = LabelStatistics(label.name, label.value, voxel_count, voxel_count / 1000.0, 1 if voxel_count else 0, label.name.lower() == "skin")
        threshold = threshold_by_name.get(label.name)
        passed = label.value in observed
        codes = []
        if label.required and not passed:
            codes.append("REQUIRED_LABEL_MISSING")
            issues.append(ValidationIssue("REQUIRED_LABEL_MISSING", ValidationSeverity.ERROR, f"missing {label.name}"))
        if threshold and voxel_count < threshold.min_voxel_count:
            passed = False
            codes.append("VOXEL_COUNT_TOO_LOW")
            issues.append(ValidationIssue("VOXEL_COUNT_TOO_LOW", ValidationSeverity.ERROR, f"{label.name} below threshold"))
        label_results.append(LabelValidationResult(stats, passed, tuple(codes)))
    valid = not any(issue.severity is ValidationSeverity.ERROR for issue in issues)
    result = SegmentationValidationResult(
        valid=valid,
        geometry_matches_ct=geometry_matches,
        label_results=tuple(label_results),
        issues=tuple(issues),
        recommended_action=RecommendedAction.CONTINUE if valid else RecommendedAction.MANUAL_REVIEW,
    )
    return _success(request, name, result)


def mock_extract_skin_surface(request: ExtractSkinSurfaceRequest):
    name = "extract_skin_surface"
    if request.skin_mask_artifact.metadata.get("mock_empty_mask") == "true":
        return _failed(request, name, ErrorCode.EMPTY_SEGMENTATION, "skin mask is empty")
    output = _mock_artifact(
        request.skin_mask_artifact,
        suffix="skin-surface",
        artifact_type=ArtifactType.SKIN_SURFACE_MASK,
        producer_name=name,
        metadata={"label_values": "0,1", "mock_crossed_skin": "true"},
    )
    spacing = request.skin_mask_artifact.geometry.spacing_mm if request.skin_mask_artifact.geometry else (1.0, 1.0, 1.0)
    result = SkinSurfaceExtractionResult(
        surface_artifact=output,
        source_voxel_count=480_000,
        surface_voxel_count=82_000,
        requested_thickness_mm=request.thickness_mm,
        effective_thickness_mm=tuple(max(axis, request.thickness_mm) for axis in spacing),
        components_removed=3 if request.keep_largest_component else 0,
        geometry_matches_source=True,
    )
    return _success(request, name, result, artifacts=(output,), metrics=(MetricValue("morphology_time", 23.0, "ms"),))


def mock_generate_candidate_paths(request: GenerateCandidatePathsRequest):
    name = "generate_candidate_paths"
    target = request.target_point_world_mm or WorldPoint(55.0, 65.0, 60.0)
    entry_points = (
        WorldPoint(20.0, 40.0, 10.0),
        WorldPoint(23.0, 37.0, 11.0),
        WorldPoint(18.0, 44.0, 12.0),
        WorldPoint(26.0, 42.0, 9.0),
        WorldPoint(16.0, 36.0, 13.0),
    )
    angles = (18.0, 24.0, 32.0, 38.0, 47.0)
    candidates = []
    rejected_length = 0
    rejected_angle = 0
    for index, (entry, angle) in enumerate(zip(entry_points, angles), start=1):
        length = dist(entry.as_tuple(), target.as_tuple())
        if length > request.max_needle_length_mm:
            rejected_length += 1
            continue
        if angle > request.max_insertion_angle_deg:
            rejected_angle += 1
            continue
        candidates.append(
            CandidatePath(
                candidate_id=f"path-{index:03d}",
                entry_point_world_mm=entry,
                target_point_world_mm=target,
                length_mm=round(length, 3),
                insertion_angle_deg=angle,
                angle_reference=request.angle_reference.value,
                rank_hint=len(candidates) + 1,
            )
        )
        if len(candidates) >= request.max_candidates:
            break
    if not candidates:
        return _failed(request, name, ErrorCode.NO_CANDIDATE_PATH, "all mock candidates violate length or angle constraints")
    result = CandidatePathGenerationResult(
        candidates=tuple(candidates),
        sampled_entry_point_count=500,
        rejected_by_length_count=rejected_length,
        rejected_by_angle_count=rejected_angle,
        planner_version=request.planner_version,
        elapsed_ms=34.0,
    )
    return _success(request, name, result, metrics=(MetricValue("planning_time", 34.0, "ms"),))


def mock_evaluate_path_safety(request: EvaluatePathSafetyRequest):
    name = "evaluate_path_safety"
    missing = [spec.structure.value for spec in request.danger_masks if spec.required and spec.artifact.status is not ArtifactStatus.AVAILABLE]
    if missing:
        return _failed(request, name, ErrorCode.REQUIRED_DANGER_MASK_MISSING, f"unavailable danger masks: {', '.join(missing)}")
    assessments = []
    for index, candidate in enumerate(request.candidate_paths):
        clearances = []
        reasons = []
        disposition = PathDisposition.ACCEPTED
        for spec in request.danger_masks:
            collision_ids = {
                value.strip()
                for value in spec.artifact.metadata.get("mock_collision_candidate_ids", "").split(",")
                if value.strip()
            }
            clearance = float(spec.artifact.metadata.get("mock_clearance_mm", str(8.2 - index * 2.2)))
            stop = candidate.candidate_id in collision_ids or clearance <= spec.safety_margin.stop_mm
            warning = stop or clearance <= spec.safety_margin.warning_mm
            clearances.append(PathStructureClearance(spec.structure, clearance, stop, warning))
            if stop:
                disposition = PathDisposition.REJECTED
                reasons.append(f"{spec.structure.value}_STOP_ENVELOPE_INTERSECTION")
            elif warning and disposition is not PathDisposition.REJECTED:
                disposition = PathDisposition.REJECTED if request.reject_warning_intersection else PathDisposition.ACCEPTED_WITH_WARNING
                if disposition is PathDisposition.REJECTED:
                    reasons.append(f"{spec.structure.value}_WARNING_ENVELOPE_INTERSECTION")
        minimum = min(item.minimum_clearance_mm for item in clearances)
        assessments.append(PathSafetyAssessment(candidate.candidate_id, disposition, minimum, tuple(clearances), tuple(reasons)))
    accepted = tuple(item.candidate_id for item in assessments if item.disposition is not PathDisposition.REJECTED)
    rejected = tuple(item.candidate_id for item in assessments if item.disposition is PathDisposition.REJECTED)
    safest = max(
        (item for item in assessments if item.candidate_id in accepted),
        key=lambda item: item.minimum_clearance_mm,
        default=None,
    )
    result = PathSafetyEvaluationResult(tuple(assessments), accepted, rejected, safest.candidate_id if safest else None, 18.0)
    return _success(request, name, result, metrics=(MetricValue("safety_evaluation_time", 18.0, "ms"),))


def mock_evaluate_intraoperative_risk(request: EvaluateIntraoperativeRiskRequest):
    name = "evaluate_intraoperative_risk"
    flags = []
    overall = RiskLevel.SAFE
    large_vessel = False
    for spec in request.danger_masks:
        distance_mm = float(spec.artifact.metadata.get("mock_tip_distance_mm", "12.0"))
        if distance_mm <= spec.safety_margin.stop_mm:
            level = RiskLevel.STOP
            overall = RiskLevel.STOP
        elif distance_mm <= spec.safety_margin.warning_mm:
            level = RiskLevel.WARNING
            if overall is RiskLevel.SAFE:
                overall = RiskLevel.WARNING
        else:
            level = RiskLevel.SAFE
        if spec.structure is RiskStructure.LARGE_VESSEL:
            large_vessel = spec.artifact.metadata.get("mock_vessel_core_contains_tip", "false") == "true"
            if large_vessel:
                level = RiskLevel.STOP
                overall = RiskLevel.STOP
        flags.append(
            RiskFlag(
                structure=spec.structure,
                level=level,
                reason_code=f"{spec.structure.value}_{level.value}",
                message=f"tip distance to {spec.structure.value.lower()} is {distance_mm:.2f} mm",
                distance_mm=distance_mm,
                evidence_artifact_ids=(spec.artifact.artifact_id,),
            )
        )
    needle_in_lung = None if request.lung_mask_artifact is None else request.lung_mask_artifact.metadata.get("mock_tip_inside", "false") == "true"
    result = IntraoperativeRiskResult(
        overall_level=overall,
        tip_world_mm=request.current_tip_world_mm,
        insertion_depth_mm=request.insertion_depth_mm,
        flags=tuple(flags),
        needle_in_lung=needle_in_lung,
        large_vessel_penetration=large_vessel,
        requires_manual_review=overall in (RiskLevel.WARNING, RiskLevel.STOP),
        rule_version=request.risk_rule_version,
    )
    return _success(request, name, result)


def mock_verify_skin_penetration(request: VerifySkinPenetrationRequest):
    name = "verify_skin_penetration"
    if request.skin_mask_artifact.geometry is None:
        return _failed(request, name, ErrorCode.GEOMETRY_MISMATCH, "skin mask geometry is missing")
    crossed = request.skin_mask_artifact.metadata.get("mock_crossed_skin", "true") == "true"
    path_length = dist(request.planned_entry_world_mm.as_tuple(), request.current_tip_world_mm.as_tuple())
    samples = max(2, int(path_length / (min(request.skin_mask_artifact.geometry.spacing_mm) * request.sampling_step_voxel)) + 1)
    if crossed:
        status = PenetrationStatus.PENETRATED
        crossing = request.planned_entry_world_mm
        first_index = 1
        evidence = "sampled ray intersected at least one skin voxel"
    elif request.insertion_depth_mm >= request.min_depth_for_slip_mm:
        status = PenetrationStatus.SUSPECTED_SLIP
        crossing = None
        first_index = None
        evidence = "no skin voxel intersection despite insertion depth exceeding threshold"
    else:
        status = PenetrationStatus.NOT_PENETRATED
        crossing = None
        first_index = None
        evidence = "no skin voxel intersection and insertion depth remains below slip threshold"
    result = SkinPenetrationResult(status, crossed, crossing, first_index, samples, path_length, request.insertion_depth_mm, evidence)
    return _success(request, name, result, metrics=(MetricValue("ray_samples", float(samples), "count"),))


MOCK_HANDLERS: dict[str, Callable[[Any], ToolResponseEnvelope[Any]]] = {
    "inspect_case_metadata": mock_inspect_case_metadata,
    "convert_mcs_to_nifti": mock_convert_mcs_to_nifti,
    "validate_label_schema": mock_validate_label_schema,
    "run_segmentation": mock_run_segmentation,
    "validate_segmentation_result": mock_validate_segmentation_result,
    "extract_skin_surface": mock_extract_skin_surface,
    "generate_candidate_paths": mock_generate_candidate_paths,
    "evaluate_path_safety": mock_evaluate_path_safety,
    "evaluate_intraoperative_risk": mock_evaluate_intraoperative_risk,
    "verify_skin_penetration": mock_verify_skin_penetration,
}
