"""Fail-closed planning/safety adapters with an injectable company-kernel port.

The deterministic backend in this module is a local development fixture.  It
returns scripted geometric summaries and never reads voxel data or claims to be
the production planning, distance-field, risk, or ray-tracing implementation.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Protocol

from contracts.artifacts import ArtifactRef
from contracts.common import MetricValue, ToolResponseEnvelope, to_primitive
from contracts.domain import (
    CandidatePath,
    DangerMaskSpec,
    PathSafetyAssessment,
    PathStructureClearance,
    RiskFlag,
)
from contracts.enums import (
    AngleReference,
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
    ErrorCode,
    PathDisposition,
    PenetrationStatus,
    RiskLevel,
    RiskStructure,
    ToolExecutionStatus,
)
from contracts.errors import ErrorDetail
from contracts.geometry import VolumeGeometry, WorldPoint
from contracts.tool_inputs import (
    EvaluateIntraoperativeRiskRequest,
    EvaluatePathSafetyRequest,
    GenerateCandidatePathsRequest,
    VerifySkinPenetrationRequest,
)
from contracts.tool_outputs import (
    CandidatePathGenerationResult,
    IntraoperativeRiskResult,
    PathSafetyEvaluationResult,
    SkinPenetrationResult,
)

from .catalog import TOOL_DEFINITIONS


PLANNING_TOOL_NAMES = (
    "generate_candidate_paths",
    "evaluate_path_safety",
    "evaluate_intraoperative_risk",
    "verify_skin_penetration",
)
_TOLERANCE_MM = 1e-3
_TOLERANCE_DEG = 1e-3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class PlanningKernelManifest:
    """Immutable compatibility and authorization manifest for one backend release."""

    backend_name: str
    backend_version: str
    supported_planner_versions: tuple[str, ...]
    supported_risk_rule_versions: tuple[str, ...]
    allowed_callers: tuple[str, ...]
    supported_coordinate_systems: tuple[CoordinateSystem, ...] = (
        CoordinateSystem.LPS,
        CoordinateSystem.RAS,
    )
    needle_device_axis_calibrated: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("backend_name", self.backend_name),
            ("backend_version", self.backend_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        for name, values in (
            ("supported_planner_versions", self.supported_planner_versions),
            ("supported_risk_rule_versions", self.supported_risk_rule_versions),
            ("allowed_callers", self.allowed_callers),
            ("supported_coordinate_systems", self.supported_coordinate_systems),
        ):
            normalized = tuple(values)
            if not normalized or len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must contain unique values")
            if name != "supported_coordinate_systems" and any(
                not isinstance(value, str) or not value.strip() for value in normalized
            ):
                raise ValueError(f"{name} contains an invalid string")
            if name == "supported_coordinate_systems" and any(
                not isinstance(value, CoordinateSystem) for value in normalized
            ):
                raise ValueError("supported_coordinate_systems contains an invalid value")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.needle_device_axis_calibrated, bool):
            raise ValueError("needle_device_axis_calibrated must be boolean")


@dataclass(frozen=True, slots=True)
class ArtifactHandle:
    artifact_id: str
    artifact_type: ArtifactType
    checksum_sha256: str
    geometry_fingerprint: str


@dataclass(frozen=True, slots=True)
class DangerHandle:
    structure: RiskStructure
    artifact: ArtifactHandle
    warning_mm: float
    stop_mm: float
    required: bool
    vessel_core_erosion_mm: float


@dataclass(frozen=True, slots=True)
class CandidateGenerationCommand:
    case_id: str
    trace_id: str
    geometry: VolumeGeometry
    ct: ArtifactHandle
    skin: ArtifactHandle
    target: ArtifactHandle
    lesion: ArtifactHandle | None
    target_point_world_mm: WorldPoint | None
    max_needle_length_mm: float
    max_insertion_angle_deg: float
    angle_reference: str
    max_candidates: int
    entry_sampling_step_mm: float
    planner_version: str


@dataclass(frozen=True, slots=True)
class PathSafetyCommand:
    case_id: str
    trace_id: str
    geometry: VolumeGeometry
    ct: ArtifactHandle
    candidates: tuple[CandidatePath, ...]
    danger_masks: tuple[DangerHandle, ...]
    needle_radius_mm: float
    path_sampling_step_mm: float


@dataclass(frozen=True, slots=True)
class TipRiskCommand:
    case_id: str
    trace_id: str
    geometry: VolumeGeometry
    ct: ArtifactHandle
    planned_entry_world_mm: WorldPoint
    current_tip_world_mm: WorldPoint
    insertion_depth_mm: float
    danger_masks: tuple[DangerHandle, ...]
    lung: ArtifactHandle | None
    skin: ArtifactHandle | None
    risk_rule_version: str


@dataclass(frozen=True, slots=True)
class SkinTraversalCommand:
    case_id: str
    trace_id: str
    geometry: VolumeGeometry
    skin: ArtifactHandle
    planned_entry_world_mm: WorldPoint
    current_tip_world_mm: WorldPoint
    insertion_depth_mm: float
    sampling_step_voxel: float
    skin_label_value: int


@dataclass(frozen=True, slots=True)
class NativeCandidate:
    candidate_id: str
    entry_point_world_mm: WorldPoint
    target_point_world_mm: WorldPoint
    length_mm: float
    insertion_angle_deg: float
    angle_reference: str
    rank_hint: int
    path_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class NativeCandidateBatch:
    candidates: tuple[NativeCandidate, ...]
    sampled_entry_point_count: int
    rejected_by_length_count: int
    rejected_by_angle_count: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class NativeClearance:
    structure: RiskStructure
    minimum_clearance_mm: float
    intersects_stop_region: bool
    intersects_warning_region: bool


@dataclass(frozen=True, slots=True)
class NativePathAssessment:
    candidate_id: str
    clearances: tuple[NativeClearance, ...]


@dataclass(frozen=True, slots=True)
class NativeSafetyBatch:
    assessments: tuple[NativePathAssessment, ...]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class NativeRiskFlag:
    structure: RiskStructure
    level: RiskLevel
    reason_code: str
    message: str
    distance_mm: float | None
    evidence_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeRiskState:
    flags: tuple[NativeRiskFlag, ...]
    needle_in_lung: bool | None
    large_vessel_penetration: bool | None
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class NativeSkinTraversal:
    crossed_skin: bool
    crossing_point_world_mm: WorldPoint | None
    first_skin_sample_index: int | None
    samples_evaluated: int
    path_length_mm: float
    skin_label_present: bool
    evidence: str
    elapsed_ms: float


class PlanningKernelPort(Protocol):
    @property
    def manifest(self) -> PlanningKernelManifest: ...

    def generate(self, command: CandidateGenerationCommand) -> NativeCandidateBatch: ...

    def path_clearance(self, command: PathSafetyCommand) -> NativeSafetyBatch: ...

    def tip_risk(self, command: TipRiskCommand) -> NativeRiskState: ...

    def traverse_skin(self, command: SkinTraversalCommand) -> NativeSkinTraversal: ...


class PlanningBackendTimeout(TimeoutError):
    pass


class PlanningBackendUnavailable(RuntimeError):
    pass


class PlanningBackendInvalidArgument(ValueError):
    pass


class PlanningBackendGeometryMismatch(RuntimeError):
    pass


class PlanningBackendMissingArtifact(RuntimeError):
    pass


class PlanningBackendNoCandidate(RuntimeError):
    pass


class PlanningBackendSafetyFailure(RuntimeError):
    pass


class PlanningBackendRiskFailure(RuntimeError):
    pass


class PlanningBackendPenetrationUndetermined(RuntimeError):
    pass


class _AdapterError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        field_path: str | None = None,
        dependency: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.field_path = field_path
        self.dependency = dependency
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class PlanningTraceRecord:
    request_id: str
    trace_id: str
    case_id: str
    tool_name: str
    tool_version: str
    status: ToolExecutionStatus
    backend_name: str
    backend_version: str
    idempotency_key_sha256: str
    idempotent_replay: bool
    result_summary: Mapping[str, Any] = field(default_factory=dict)
    error_code: ErrorCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_summary", MappingProxyType(dict(self.result_summary)))


class PlanningTraceSink(Protocol):
    def record(self, trace: PlanningTraceRecord) -> None: ...


class NullPlanningTraceSink:
    def record(self, trace: PlanningTraceRecord) -> None:
        return None


class InMemoryPlanningTraceSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[PlanningTraceRecord] = []

    def record(self, trace: PlanningTraceRecord) -> None:
        with self._lock:
            self._records.append(trace)

    @property
    def records(self) -> tuple[PlanningTraceRecord, ...]:
        with self._lock:
            return tuple(self._records)


class DeterministicPlanningBackend:
    """Scriptable local fake; it performs no voxel/mask algorithm."""

    def __init__(
        self,
        *,
        manifest: PlanningKernelManifest | None = None,
        candidate_batch: NativeCandidateBatch | None = None,
        safety_batch: NativeSafetyBatch | None = None,
        risk_state: NativeRiskState | None = None,
        skin_traversal: NativeSkinTraversal | None = None,
        safety_clearances: Mapping[tuple[str, RiskStructure], float] | None = None,
        risk_distances: Mapping[RiskStructure, float] | None = None,
        needle_in_lung: bool | None = True,
        large_vessel_penetration: bool | None = None,
        crossed_skin: bool = True,
        skin_label_present: bool = True,
        failures: Mapping[str, BaseException] | None = None,
    ) -> None:
        self._manifest = manifest or PlanningKernelManifest(
            backend_name="deterministic-planning-fixture",
            backend_version="fixture-v1",
            supported_planner_versions=("planner-v1",),
            supported_risk_rule_versions=("risk-v1",),
            allowed_callers=("unit-test", "local-demo"),
        )
        self.candidate_batch = candidate_batch
        self.safety_batch = safety_batch
        self.risk_state = risk_state
        self.skin_traversal = skin_traversal
        self.safety_clearances = dict(safety_clearances or {})
        self.risk_distances = dict(risk_distances or {})
        self.needle_in_lung = needle_in_lung
        self.large_vessel_penetration = large_vessel_penetration
        self.crossed_skin = crossed_skin
        self.skin_label_present = skin_label_present
        self.failures = dict(failures or {})
        self.call_counts = {name: 0 for name in ("generate", "path_clearance", "tip_risk", "traverse_skin")}

    @property
    def manifest(self) -> PlanningKernelManifest:
        return self._manifest

    def _begin(self, operation: str) -> None:
        self.call_counts[operation] += 1
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure

    def generate(self, command: CandidateGenerationCommand) -> NativeCandidateBatch:
        self._begin("generate")
        if self.candidate_batch is not None:
            return self.candidate_batch
        target = command.target_point_world_mm or WorldPoint(55.0, 65.0, 60.0)
        fixtures = (
            ("path-001", WorldPoint(20.0, 40.0, 10.0), 18.0),
            ("path-002", WorldPoint(23.0, 37.0, 11.0), 24.0),
            ("path-003", WorldPoint(18.0, 44.0, 12.0), 32.0),
            ("path-004", WorldPoint(26.0, 42.0, 9.0), 38.0),
            ("path-005", WorldPoint(16.0, 36.0, 13.0), 47.0),
        )
        accepted: list[NativeCandidate] = []
        rejected_length = 0
        rejected_angle = 0
        for candidate_id, entry, angle in fixtures:
            length = math.dist(entry.as_tuple(), target.as_tuple())
            if length > command.max_needle_length_mm:
                rejected_length += 1
                continue
            if angle > command.max_insertion_angle_deg:
                rejected_angle += 1
                continue
            if len(accepted) < command.max_candidates:
                accepted.append(
                    NativeCandidate(
                        candidate_id=candidate_id,
                        entry_point_world_mm=entry,
                        target_point_world_mm=target,
                        length_mm=length,
                        insertion_angle_deg=angle,
                        angle_reference=command.angle_reference,
                        rank_hint=len(accepted) + 1,
                    )
                )
        return NativeCandidateBatch(
            candidates=tuple(accepted),
            sampled_entry_point_count=len(fixtures),
            rejected_by_length_count=rejected_length,
            rejected_by_angle_count=rejected_angle,
            elapsed_ms=34.0,
        )

    def path_clearance(self, command: PathSafetyCommand) -> NativeSafetyBatch:
        self._begin("path_clearance")
        if self.safety_batch is not None:
            return self.safety_batch
        assessments: list[NativePathAssessment] = []
        for candidate in command.candidates:
            clearances: list[NativeClearance] = []
            for danger in command.danger_masks:
                clearance = self.safety_clearances.get((candidate.candidate_id, danger.structure), 12.0)
                stop = clearance <= danger.stop_mm
                warning = stop or clearance <= danger.warning_mm
                clearances.append(NativeClearance(danger.structure, clearance, stop, warning))
            assessments.append(NativePathAssessment(candidate.candidate_id, tuple(clearances)))
        return NativeSafetyBatch(tuple(assessments), elapsed_ms=18.0)

    def tip_risk(self, command: TipRiskCommand) -> NativeRiskState:
        self._begin("tip_risk")
        if self.risk_state is not None:
            return self.risk_state
        flags: list[NativeRiskFlag] = []
        has_large_vessel = False
        large_vessel = self.large_vessel_penetration
        for danger in command.danger_masks:
            distance = self.risk_distances.get(danger.structure, 12.0)
            if distance <= danger.stop_mm:
                level = RiskLevel.STOP
            elif distance <= danger.warning_mm:
                level = RiskLevel.WARNING
            else:
                level = RiskLevel.SAFE
            if danger.structure is RiskStructure.LARGE_VESSEL:
                has_large_vessel = True
                if large_vessel is None:
                    large_vessel = False
                if large_vessel:
                    level = RiskLevel.STOP
            flags.append(
                NativeRiskFlag(
                    structure=danger.structure,
                    level=level,
                    reason_code=f"{danger.structure.value}_{level.value}",
                    message=f"deterministic fixture distance for {danger.structure.value.lower()}",
                    distance_mm=distance,
                    evidence_artifact_ids=(danger.artifact.artifact_id,),
                )
            )
        return NativeRiskState(
            flags=tuple(flags),
            needle_in_lung=self.needle_in_lung if command.lung is not None else None,
            large_vessel_penetration=large_vessel if has_large_vessel else None,
            elapsed_ms=4.0,
        )

    def traverse_skin(self, command: SkinTraversalCommand) -> NativeSkinTraversal:
        self._begin("traverse_skin")
        if self.skin_traversal is not None:
            return self.skin_traversal
        length = math.dist(
            command.planned_entry_world_mm.as_tuple(),
            command.current_tip_world_mm.as_tuple(),
        )
        samples = max(2, int(math.ceil(length / max(command.sampling_step_voxel, 1e-6))) + 1)
        return NativeSkinTraversal(
            crossed_skin=self.crossed_skin,
            crossing_point_world_mm=command.planned_entry_world_mm if self.crossed_skin else None,
            first_skin_sample_index=1 if self.crossed_skin else None,
            samples_evaluated=samples,
            path_length_mm=length,
            skin_label_present=self.skin_label_present,
            evidence=(
                "fixture traversal crossed configured skin label"
                if self.crossed_skin
                else "fixture traversal found no configured skin label"
            ),
            elapsed_ms=2.0,
        )


@dataclass(frozen=True, slots=True)
class _Execution:
    result: Any
    metrics: tuple[MetricValue, ...]
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)


class PlanningToolAdapters:
    """Typed, idempotent registry handlers around a planning kernel backend."""

    def __init__(
        self,
        backend: PlanningKernelPort,
        *,
        trace_sink: PlanningTraceSink | None = None,
        clock: Callable[[], str] = _utc_now,
        epoch_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(backend.manifest, PlanningKernelManifest):
            raise ValueError("planning backend manifest is invalid")
        self.backend = backend
        self.trace_sink = trace_sink or NullPlanningTraceSink()
        self.clock = clock
        self.epoch_ms = epoch_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str, str, str], tuple[str, ToolResponseEnvelope[Any]]] = {}

    def handlers(self) -> Mapping[str, Callable[[Any], ToolResponseEnvelope[Any]]]:
        return MappingProxyType(
            {
                "generate_candidate_paths": self.generate_candidate_paths,
                "evaluate_path_safety": self.evaluate_path_safety,
                "evaluate_intraoperative_risk": self.evaluate_intraoperative_risk,
                "verify_skin_penetration": self.verify_skin_penetration,
            }
        )

    def generate_candidate_paths(
        self, request: GenerateCandidatePathsRequest
    ) -> ToolResponseEnvelope[CandidatePathGenerationResult]:
        return self._execute("generate_candidate_paths", request, self._run_generate)

    def evaluate_path_safety(
        self, request: EvaluatePathSafetyRequest
    ) -> ToolResponseEnvelope[PathSafetyEvaluationResult]:
        return self._execute("evaluate_path_safety", request, self._run_safety)

    def evaluate_intraoperative_risk(
        self, request: EvaluateIntraoperativeRiskRequest
    ) -> ToolResponseEnvelope[IntraoperativeRiskResult]:
        return self._execute("evaluate_intraoperative_risk", request, self._run_risk)

    def verify_skin_penetration(
        self, request: VerifySkinPenetrationRequest
    ) -> ToolResponseEnvelope[SkinPenetrationResult]:
        return self._execute("verify_skin_penetration", request, self._run_penetration)

    def _execute(
        self,
        tool_name: str,
        request: Any,
        operation: Callable[[Any], _Execution],
    ) -> ToolResponseEnvelope[Any]:
        started_at = self.clock()
        try:
            fingerprint = _request_fingerprint(request)
        except (TypeError, ValueError) as exc:
            response = self._failed(
                tool_name,
                request,
                started_at,
                _AdapterError(ErrorCode.INVALID_ARGUMENT, str(exc)),
                replay=False,
            )
            self._trace(tool_name, request, response, replay=False, summary={})
            return response
        key = (
            tool_name,
            request.context.case_id,
            request.context.caller,
            request.context.idempotency_key,
        )
        with self._lock:
            try:
                self._preflight(tool_name, request)
            except Exception as exc:
                error = self._normalize_error(tool_name, exc)
                response = self._failed(tool_name, request, started_at, error, replay=False)
                self._trace(tool_name, request, response, replay=False, summary={})
                return response
            cached = self._cache.get(key)
            if cached is not None:
                cached_fingerprint, cached_response = cached
                if cached_fingerprint != fingerprint:
                    response = self._failed(
                        tool_name,
                        request,
                        started_at,
                        _AdapterError(
                            ErrorCode.CONTRACT_VIOLATION,
                            "idempotency key was reused with a different request payload",
                        ),
                        replay=False,
                    )
                    self._trace(tool_name, request, response, replay=False, summary={})
                    return response
                replay_response = ToolResponseEnvelope(
                    request_id=request.context.request_id,
                    trace_id=request.context.trace_id,
                    tool_name=cached_response.tool_name,
                    tool_version=cached_response.tool_version,
                    status=cached_response.status,
                    result=cached_response.result,
                    artifacts=cached_response.artifacts,
                    metrics=cached_response.metrics,
                    warnings=cached_response.warnings,
                    error=cached_response.error,
                    started_at=started_at,
                    finished_at=self.clock(),
                )
                self._trace(tool_name, request, replay_response, replay=True, summary={"cache_hit": True})
                return replay_response
            try:
                execution = operation(request)
                response = ToolResponseEnvelope(
                    request_id=request.context.request_id,
                    trace_id=request.context.trace_id,
                    tool_name=tool_name,
                    tool_version=TOOL_DEFINITIONS[tool_name].version,
                    status=ToolExecutionStatus.SUCCESS,
                    result=execution.result,
                    artifacts=(),
                    metrics=execution.metrics,
                    warnings=execution.warnings,
                    error=None,
                    started_at=started_at,
                    finished_at=self.clock(),
                )
            except Exception as exc:  # adapter boundary: normalize every backend failure
                error = self._normalize_error(tool_name, exc)
                response = self._failed(tool_name, request, started_at, error, replay=False)
            if response.status is ToolExecutionStatus.SUCCESS:
                self._cache[key] = (fingerprint, response)
                summary = execution.summary
            else:
                summary = {}
            self._trace(tool_name, request, response, replay=False, summary=summary)
            return response

    def _preflight(self, tool_name: str, request: Any) -> None:
        manifest = self.backend.manifest
        if request.context.caller not in manifest.allowed_callers:
            raise _AdapterError(ErrorCode.PERMISSION_DENIED, "caller is not permitted to execute planning tools")
        deadline = request.context.deadline_epoch_ms
        if deadline is not None and self.epoch_ms() >= deadline:
            raise _AdapterError(ErrorCode.TIMEOUT, "tool deadline expired before execution", retryable=True)
        if tool_name == "generate_candidate_paths" and request.planner_version not in manifest.supported_planner_versions:
            raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "planner_version is not supported")
        if (
            tool_name == "generate_candidate_paths"
            and request.angle_reference is AngleReference.NEEDLE_DEVICE_AXIS
            and not manifest.needle_device_axis_calibrated
        ):
            raise _AdapterError(
                ErrorCode.INVALID_ARGUMENT,
                "NEEDLE_DEVICE_AXIS requires an approved calibrated device axis",
            )
        if tool_name == "evaluate_intraoperative_risk" and request.risk_rule_version not in manifest.supported_risk_rule_versions:
            raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "risk_rule_version is not supported")

    def _run_generate(self, request: GenerateCandidatePathsRequest) -> _Execution:
        _require_artifact(request.ct_artifact, (ArtifactType.CT_VOLUME,), "ct_artifact")
        geometry = _required_geometry(request.ct_artifact, "ct_artifact")
        _require_coordinate_support(geometry, self.backend.manifest)
        for field_path, label_artifact in (
            ("skin_surface_artifact", request.skin_surface_artifact),
            ("target_artifact", request.target_artifact),
        ):
            if label_artifact.status is not ArtifactStatus.AVAILABLE:
                raise _AdapterError(
                    ErrorCode.REQUIRED_LABEL_MISSING,
                    "required planning label is unavailable",
                    field_path=field_path,
                )
        _require_artifact(
            request.skin_surface_artifact,
            (ArtifactType.SKIN_SURFACE_MASK,),
            "skin_surface_artifact",
        )
        _require_artifact(request.target_artifact, (ArtifactType.TARGET_MASK,), "target_artifact")
        _require_compatible(geometry, request.skin_surface_artifact, "skin_surface_artifact")
        _require_compatible(geometry, request.target_artifact, "target_artifact")
        if request.lesion_artifact is not None:
            _require_artifact(request.lesion_artifact, (ArtifactType.LESION_MASK,), "lesion_artifact")
            _require_compatible(geometry, request.lesion_artifact, "lesion_artifact")
        _require_positive_finite(request.max_needle_length_mm, "max_needle_length_mm")
        _require_positive_finite(request.entry_sampling_step_mm, "entry_sampling_step_mm")
        _require_finite(request.max_insertion_angle_deg, "max_insertion_angle_deg")
        if request.target_point_world_mm is not None and not _point_in_volume(
            geometry, request.target_point_world_mm
        ):
            raise _AdapterError(ErrorCode.TARGET_OUT_OF_BOUNDS, "target point is outside CT geometry")
        command = CandidateGenerationCommand(
            case_id=request.context.case_id,
            trace_id=request.context.trace_id,
            geometry=geometry,
            ct=_handle(request.ct_artifact),
            skin=_handle(request.skin_surface_artifact),
            target=_handle(request.target_artifact),
            lesion=_handle(request.lesion_artifact) if request.lesion_artifact else None,
            target_point_world_mm=request.target_point_world_mm,
            max_needle_length_mm=request.max_needle_length_mm,
            max_insertion_angle_deg=request.max_insertion_angle_deg,
            angle_reference=request.angle_reference.value,
            max_candidates=request.max_candidates,
            entry_sampling_step_mm=request.entry_sampling_step_mm,
            planner_version=request.planner_version,
        )
        native = self.backend.generate(command)
        candidates = self._validate_candidates(request, geometry, native)
        result = CandidatePathGenerationResult(
            candidates=candidates,
            sampled_entry_point_count=native.sampled_entry_point_count,
            rejected_by_length_count=native.rejected_by_length_count,
            rejected_by_angle_count=native.rejected_by_angle_count,
            planner_version=request.planner_version,
            elapsed_ms=native.elapsed_ms,
        )
        return _Execution(
            result=result,
            metrics=(MetricValue("planning_time", native.elapsed_ms, "ms"),),
            summary={"candidate_count": len(candidates)},
        )

    def _validate_candidates(
        self,
        request: GenerateCandidatePathsRequest,
        geometry: VolumeGeometry,
        native: NativeCandidateBatch,
    ) -> tuple[CandidatePath, ...]:
        if not isinstance(native, NativeCandidateBatch):
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "planning backend returned an invalid batch")
        _validate_elapsed(native.elapsed_ms)
        for name, value in (
            ("sampled_entry_point_count", native.sampled_entry_point_count),
            ("rejected_by_length_count", native.rejected_by_length_count),
            ("rejected_by_angle_count", native.rejected_by_angle_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, f"{name} is invalid")
        if not native.candidates:
            raise _AdapterError(ErrorCode.NO_CANDIDATE_PATH, "backend produced no candidate path")
        if len(native.candidates) > request.max_candidates:
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "backend exceeded max_candidates")
        if any(not isinstance(item, NativeCandidate) for item in native.candidates):
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate payload is malformed")
        ordered = sorted(native.candidates, key=lambda item: (item.rank_hint, item.candidate_id))
        if [item.rank_hint for item in ordered] != list(range(1, len(ordered) + 1)):
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate ranks are not contiguous")
        if len({item.candidate_id for item in ordered}) != len(ordered):
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate IDs are not unique")
        output: list[CandidatePath] = []
        for item in ordered:
            if not isinstance(item.candidate_id, str) or not item.candidate_id.strip():
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate payload is malformed")
            if item.path_artifact_id is not None:
                raise _AdapterError(
                    ErrorCode.CONTRACT_VIOLATION,
                    "path_artifact_id cannot be returned without an injected artifact commit port",
                )
            if not _point_in_volume(geometry, item.entry_point_world_mm) or not _point_in_volume(
                geometry, item.target_point_world_mm
            ):
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate endpoint is outside CT geometry")
            if request.target_point_world_mm is not None and math.dist(
                item.target_point_world_mm.as_tuple(), request.target_point_world_mm.as_tuple()
            ) > _TOLERANCE_MM:
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "backend changed authoritative target point")
            actual_length = math.dist(
                item.entry_point_world_mm.as_tuple(), item.target_point_world_mm.as_tuple()
            )
            _require_finite(item.length_mm, "candidate.length_mm", contract=True)
            _require_finite(item.insertion_angle_deg, "candidate.insertion_angle_deg", contract=True)
            if abs(item.length_mm - actual_length) > max(_TOLERANCE_MM, actual_length * 1e-6):
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate physical length is inconsistent")
            if item.length_mm > request.max_needle_length_mm + _TOLERANCE_MM:
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate exceeds needle length constraint")
            if item.insertion_angle_deg > request.max_insertion_angle_deg + _TOLERANCE_DEG:
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate exceeds insertion angle constraint")
            if item.angle_reference != request.angle_reference.value:
                raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "candidate angle reference is inconsistent")
            output.append(
                CandidatePath(
                    candidate_id=item.candidate_id,
                    entry_point_world_mm=item.entry_point_world_mm,
                    target_point_world_mm=item.target_point_world_mm,
                    length_mm=item.length_mm,
                    insertion_angle_deg=item.insertion_angle_deg,
                    angle_reference=item.angle_reference,
                    rank_hint=item.rank_hint,
                )
            )
        if native.sampled_entry_point_count < len(output):
            raise _AdapterError(ErrorCode.CONTRACT_VIOLATION, "sampled count is smaller than candidate count")
        return tuple(output)

    def _run_safety(self, request: EvaluatePathSafetyRequest) -> _Execution:
        geometry, danger_specs, danger_warnings = self._prepare_dangers(
            request.ct_artifact,
            request.danger_masks,
            empty_code=ErrorCode.SAFETY_CHECK_FAILED,
        )
        _require_positive_finite(request.needle_radius_mm, "needle_radius_mm")
        _require_positive_finite(request.path_sampling_step_mm, "path_sampling_step_mm")
        for candidate in request.candidate_paths:
            self._validate_existing_candidate(candidate, geometry)
        command = PathSafetyCommand(
            case_id=request.context.case_id,
            trace_id=request.context.trace_id,
            geometry=geometry,
            ct=_handle(request.ct_artifact),
            candidates=request.candidate_paths,
            danger_masks=tuple(_danger_handle(spec) for spec in danger_specs),
            needle_radius_mm=request.needle_radius_mm,
            path_sampling_step_mm=request.path_sampling_step_mm,
        )
        native = self.backend.path_clearance(command)
        if not isinstance(native, NativeSafetyBatch):
            raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "safety backend returned an invalid batch")
        _validate_elapsed(native.elapsed_ms, ErrorCode.SAFETY_CHECK_FAILED)
        by_candidate: dict[str, NativePathAssessment] = {}
        for assessment in native.assessments:
            if not isinstance(assessment, NativePathAssessment):
                raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "safety assessment is malformed")
            if assessment.candidate_id in by_candidate:
                raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "duplicate safety assessment")
            by_candidate[assessment.candidate_id] = assessment
        expected_ids = {item.candidate_id for item in request.candidate_paths}
        if set(by_candidate) != expected_ids:
            raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "safety assessments do not cover every candidate")
        specs = {spec.structure: spec for spec in danger_specs}
        assessments: list[PathSafetyAssessment] = []
        warning_present = False
        for candidate in request.candidate_paths:
            native_assessment = by_candidate[candidate.candidate_id]
            clearances_by_structure: dict[RiskStructure, NativeClearance] = {}
            for clearance in native_assessment.clearances:
                if not isinstance(clearance, NativeClearance):
                    raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "structure clearance is malformed")
                if clearance.structure in clearances_by_structure:
                    raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "duplicate structure clearance")
                clearances_by_structure[clearance.structure] = clearance
            if set(clearances_by_structure) != set(specs):
                raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "structure clearances are incomplete")
            clearances: list[PathStructureClearance] = []
            reasons: list[str] = []
            has_stop = False
            has_warning = False
            for structure in specs:
                native_clearance = clearances_by_structure[structure]
                spec = specs[structure]
                _require_finite(
                    native_clearance.minimum_clearance_mm,
                    "minimum_clearance_mm",
                    contract=True,
                    code=ErrorCode.SAFETY_CHECK_FAILED,
                )
                if not isinstance(native_clearance.intersects_stop_region, bool) or not isinstance(
                    native_clearance.intersects_warning_region, bool
                ):
                    raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "clearance flags must be boolean")
                expected_stop = native_clearance.minimum_clearance_mm <= spec.safety_margin.stop_mm
                expected_warning = (
                    expected_stop
                    or native_clearance.minimum_clearance_mm <= spec.safety_margin.warning_mm
                )
                if expected_stop and not native_clearance.intersects_stop_region:
                    raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "backend downgraded a stop boundary")
                if (expected_warning or native_clearance.intersects_stop_region) and not native_clearance.intersects_warning_region:
                    raise _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "backend downgraded a warning boundary")
                has_stop = has_stop or native_clearance.intersects_stop_region
                has_warning = has_warning or native_clearance.intersects_warning_region
                if native_clearance.intersects_stop_region:
                    reasons.append(f"{structure.value}_STOP_ENVELOPE_INTERSECTION")
                clearances.append(
                    PathStructureClearance(
                        structure=structure,
                        minimum_clearance_mm=native_clearance.minimum_clearance_mm,
                        intersects_stop_region=native_clearance.intersects_stop_region,
                        intersects_warning_region=native_clearance.intersects_warning_region,
                    )
                )
            minimum = min(item.minimum_clearance_mm for item in clearances)
            if has_stop:
                disposition = PathDisposition.REJECTED
            elif has_warning and request.reject_warning_intersection:
                disposition = PathDisposition.REJECTED
                reasons.append("WARNING_INTERSECTION_REJECTED_BY_POLICY")
            elif has_warning:
                disposition = PathDisposition.ACCEPTED_WITH_WARNING
                warning_present = True
            else:
                disposition = PathDisposition.ACCEPTED
            assessments.append(
                PathSafetyAssessment(
                    candidate_id=candidate.candidate_id,
                    disposition=disposition,
                    minimum_clearance_mm=minimum,
                    clearances=tuple(clearances),
                    rejection_reasons=tuple(reasons),
                )
            )
        accepted = tuple(
            item.candidate_id for item in assessments if item.disposition is not PathDisposition.REJECTED
        )
        rejected = tuple(
            item.candidate_id for item in assessments if item.disposition is PathDisposition.REJECTED
        )
        candidate_by_id = {item.candidate_id: item for item in request.candidate_paths}
        safest = None
        if accepted:
            safest = sorted(
                (item for item in assessments if item.candidate_id in accepted),
                key=lambda item: (
                    -item.minimum_clearance_mm,
                    candidate_by_id[item.candidate_id].length_mm,
                    candidate_by_id[item.candidate_id].insertion_angle_deg,
                    item.candidate_id,
                ),
            )[0].candidate_id
        result = PathSafetyEvaluationResult(
            assessments=tuple(assessments),
            accepted_candidate_ids=accepted,
            rejected_candidate_ids=rejected,
            safest_candidate_id=safest,
            elapsed_ms=native.elapsed_ms,
        )
        warnings = tuple(
            [*danger_warnings, *(("PATH_WARNING_INTERSECTION",) if warning_present else ())]
        )
        return _Execution(
            result,
            (MetricValue("safety_evaluation_time", native.elapsed_ms, "ms"),),
            warnings,
            {"accepted_count": len(accepted), "rejected_count": len(rejected)},
        )

    def _run_risk(self, request: EvaluateIntraoperativeRiskRequest) -> _Execution:
        geometry, danger_specs, danger_warnings = self._prepare_dangers(
            request.ct_artifact,
            request.danger_masks,
            empty_code=ErrorCode.RISK_EVALUATION_FAILED,
        )
        _require_nonnegative_finite(request.insertion_depth_mm, "insertion_depth_mm")
        if not _point_in_volume(geometry, request.current_tip_world_mm):
            raise _AdapterError(ErrorCode.TARGET_OUT_OF_BOUNDS, "current needle tip is outside CT geometry")
        if not _point_in_volume(geometry, request.planned_entry_world_mm):
            raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "planned entry is outside CT geometry")
        warnings: list[str] = list(danger_warnings)
        lung = self._optional_artifact(
            request.lung_mask_artifact,
            geometry,
            (ArtifactType.SEGMENTATION_MASK, ArtifactType.DANGER_MASK),
            "lung_mask_artifact",
            warnings,
        )
        skin = self._optional_artifact(
            request.skin_mask_artifact,
            geometry,
            (ArtifactType.SEGMENTATION_MASK, ArtifactType.SKIN_SURFACE_MASK),
            "skin_mask_artifact",
            warnings,
        )
        command = TipRiskCommand(
            case_id=request.context.case_id,
            trace_id=request.context.trace_id,
            geometry=geometry,
            ct=_handle(request.ct_artifact),
            planned_entry_world_mm=request.planned_entry_world_mm,
            current_tip_world_mm=request.current_tip_world_mm,
            insertion_depth_mm=request.insertion_depth_mm,
            danger_masks=tuple(_danger_handle(spec) for spec in danger_specs),
            lung=lung,
            skin=skin,
            risk_rule_version=request.risk_rule_version,
        )
        native = self.backend.tip_risk(command)
        if not isinstance(native, NativeRiskState):
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk backend returned an invalid state")
        _validate_elapsed(native.elapsed_ms, ErrorCode.RISK_EVALUATION_FAILED)
        specs = {spec.structure: spec for spec in danger_specs}
        by_structure: dict[RiskStructure, NativeRiskFlag] = {}
        for flag in native.flags:
            if not isinstance(flag, NativeRiskFlag) or not isinstance(flag.level, RiskLevel):
                raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk flag is malformed")
            if flag.structure in by_structure:
                raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "duplicate risk structure flag")
            by_structure[flag.structure] = flag
        if set(by_structure) != set(specs):
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk flags do not cover every danger structure")
        flags: list[RiskFlag] = []
        for structure in specs:
            native_flag = by_structure[structure]
            spec = specs[structure]
            if not native_flag.reason_code.strip() or not native_flag.message.strip():
                raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk flag explanation is missing")
            if native_flag.distance_mm is not None:
                _require_finite(
                    native_flag.distance_mm,
                    "risk distance",
                    contract=True,
                    code=ErrorCode.RISK_EVALUATION_FAILED,
                )
                if native_flag.distance_mm <= spec.safety_margin.stop_mm and native_flag.level is not RiskLevel.STOP:
                    raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk backend downgraded a stop boundary")
                if (
                    native_flag.distance_mm <= spec.safety_margin.warning_mm
                    and native_flag.level is RiskLevel.SAFE
                ):
                    raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk backend downgraded a warning boundary")
            if spec.artifact.artifact_id not in native_flag.evidence_artifact_ids:
                raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "risk flag lacks required evidence artifact")
            flags.append(
                RiskFlag(
                    structure=structure,
                    level=native_flag.level,
                    reason_code=native_flag.reason_code,
                    message=native_flag.message,
                    distance_mm=native_flag.distance_mm,
                    evidence_artifact_ids=tuple(native_flag.evidence_artifact_ids),
                )
            )
        has_vessel = RiskStructure.LARGE_VESSEL in specs
        if native.needle_in_lung is not None and not isinstance(native.needle_in_lung, bool):
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "needle_in_lung must be boolean or null")
        if native.large_vessel_penetration is not None and not isinstance(
            native.large_vessel_penetration, bool
        ):
            raise _AdapterError(
                ErrorCode.RISK_EVALUATION_FAILED,
                "large_vessel_penetration must be boolean or null",
            )
        if not has_vessel and native.large_vessel_penetration is not None:
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "large-vessel state exists without evidence")
        if has_vessel and native.large_vessel_penetration is None:
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "large-vessel state is missing")
        if native.large_vessel_penetration and by_structure[RiskStructure.LARGE_VESSEL].level is not RiskLevel.STOP:
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "large-vessel penetration was not promoted to STOP")
        if lung is None and native.needle_in_lung is not None:
            raise _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "lung state exists without lung evidence")
        precedence = {RiskLevel.UNKNOWN: 0, RiskLevel.SAFE: 1, RiskLevel.WARNING: 2, RiskLevel.STOP: 3}
        overall = max((flag.level for flag in flags), key=lambda level: precedence[level])
        manual_review = overall in {RiskLevel.WARNING, RiskLevel.STOP, RiskLevel.UNKNOWN} or bool(warnings)
        result = IntraoperativeRiskResult(
            overall_level=overall,
            tip_world_mm=request.current_tip_world_mm,
            insertion_depth_mm=request.insertion_depth_mm,
            flags=tuple(flags),
            needle_in_lung=native.needle_in_lung,
            large_vessel_penetration=native.large_vessel_penetration,
            requires_manual_review=manual_review,
            rule_version=request.risk_rule_version,
        )
        return _Execution(
            result,
            (MetricValue("risk_evaluation_time", native.elapsed_ms, "ms"),),
            tuple(warnings),
            {"overall_level": overall.value, "flag_count": len(flags)},
        )

    def _run_penetration(self, request: VerifySkinPenetrationRequest) -> _Execution:
        _require_artifact(
            request.skin_mask_artifact,
            (ArtifactType.SKIN_SURFACE_MASK, ArtifactType.SEGMENTATION_MASK),
            "skin_mask_artifact",
        )
        geometry = _required_geometry(request.skin_mask_artifact, "skin_mask_artifact")
        _require_coordinate_support(geometry, self.backend.manifest)
        _require_nonnegative_finite(request.insertion_depth_mm, "insertion_depth_mm")
        _require_positive_finite(request.sampling_step_voxel, "sampling_step_voxel")
        if not _point_in_volume(geometry, request.planned_entry_world_mm) or not _point_in_volume(
            geometry, request.current_tip_world_mm
        ):
            raise _AdapterError(
                ErrorCode.SKIN_PENETRATION_UNDETERMINED,
                "skin traversal segment is outside mask geometry",
            )
        physical_length = math.dist(
            request.planned_entry_world_mm.as_tuple(), request.current_tip_world_mm.as_tuple()
        )
        if physical_length <= _TOLERANCE_MM:
            raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "skin traversal segment has zero length")
        native = self.backend.traverse_skin(
            SkinTraversalCommand(
                case_id=request.context.case_id,
                trace_id=request.context.trace_id,
                geometry=geometry,
                skin=_handle(request.skin_mask_artifact),
                planned_entry_world_mm=request.planned_entry_world_mm,
                current_tip_world_mm=request.current_tip_world_mm,
                insertion_depth_mm=request.insertion_depth_mm,
                sampling_step_voxel=request.sampling_step_voxel,
                skin_label_value=request.skin_label_value,
            )
        )
        if not isinstance(native, NativeSkinTraversal):
            raise _AdapterError(
                ErrorCode.SKIN_PENETRATION_UNDETERMINED,
                "skin traversal backend returned an invalid result",
            )
        _validate_elapsed(native.elapsed_ms, ErrorCode.SKIN_PENETRATION_UNDETERMINED)
        if not isinstance(native.skin_label_present, bool):
            raise _AdapterError(
                ErrorCode.SKIN_PENETRATION_UNDETERMINED,
                "skin_label_present is not boolean",
            )
        if not native.skin_label_present:
            raise _AdapterError(ErrorCode.REQUIRED_LABEL_MISSING, "configured skin label is absent")
        if (
            isinstance(native.samples_evaluated, bool)
            or not isinstance(native.samples_evaluated, int)
            or native.samples_evaluated < 1
        ):
            raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin sample count is invalid")
        _require_finite(
            native.path_length_mm,
            "skin path length",
            contract=True,
            code=ErrorCode.SKIN_PENETRATION_UNDETERMINED,
        )
        if abs(native.path_length_mm - physical_length) > max(_TOLERANCE_MM, physical_length * 1e-6):
            raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin path length is inconsistent")
        if not isinstance(native.crossed_skin, bool):
            raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "crossed_skin is not boolean")
        if native.crossed_skin:
            if native.crossing_point_world_mm is None or native.first_skin_sample_index is None:
                raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin crossing evidence is incomplete")
            if (
                isinstance(native.first_skin_sample_index, bool)
                or not isinstance(native.first_skin_sample_index, int)
                or not 0 <= native.first_skin_sample_index < native.samples_evaluated
            ):
                raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin crossing sample index is invalid")
            if not _point_on_segment(
                request.planned_entry_world_mm,
                request.current_tip_world_mm,
                native.crossing_point_world_mm,
            ):
                raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin crossing point is off segment")
            status = PenetrationStatus.PENETRATED
        else:
            if native.crossing_point_world_mm is not None or native.first_skin_sample_index is not None:
                raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "non-crossing result contains crossing evidence")
            status = (
                PenetrationStatus.SUSPECTED_SLIP
                if request.insertion_depth_mm >= request.min_depth_for_slip_mm
                else PenetrationStatus.NOT_PENETRATED
            )
        if not isinstance(native.evidence, str) or not native.evidence.strip() or len(native.evidence) > 1024:
            raise _AdapterError(ErrorCode.SKIN_PENETRATION_UNDETERMINED, "skin evidence summary is invalid")
        result = SkinPenetrationResult(
            status=status,
            crossed_skin=native.crossed_skin,
            crossing_point_world_mm=native.crossing_point_world_mm,
            first_skin_sample_index=native.first_skin_sample_index,
            samples_evaluated=native.samples_evaluated,
            path_length_mm=native.path_length_mm,
            insertion_depth_mm=request.insertion_depth_mm,
            evidence=native.evidence,
        )
        warnings = ("SUSPECTED_SKIN_SLIP",) if status is PenetrationStatus.SUSPECTED_SLIP else ()
        return _Execution(
            result,
            (MetricValue("ray_samples", float(native.samples_evaluated), "count"),),
            warnings,
            {"penetration_status": status.value, "samples_evaluated": native.samples_evaluated},
        )

    def _prepare_dangers(
        self,
        ct_artifact: ArtifactRef,
        danger_masks: Sequence[DangerMaskSpec],
        *,
        empty_code: ErrorCode,
    ) -> tuple[VolumeGeometry, tuple[DangerMaskSpec, ...], tuple[str, ...]]:
        _require_artifact(ct_artifact, (ArtifactType.CT_VOLUME,), "ct_artifact")
        geometry = _required_geometry(ct_artifact, "ct_artifact")
        _require_coordinate_support(geometry, self.backend.manifest)
        structures: set[RiskStructure] = set()
        available: list[DangerMaskSpec] = []
        warnings: list[str] = []
        for index, spec in enumerate(danger_masks):
            if spec.structure in structures:
                raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "danger structures must be unique")
            structures.add(spec.structure)
            if spec.artifact.status is not ArtifactStatus.AVAILABLE:
                if spec.required:
                    raise _AdapterError(
                        ErrorCode.REQUIRED_DANGER_MASK_MISSING,
                        "required danger mask is unavailable",
                        field_path=f"danger_masks[{index}].artifact",
                    )
                warnings.append(f"OPTIONAL_{spec.structure.value}_DANGER_MASK_UNAVAILABLE")
                continue
            _require_artifact(spec.artifact, (ArtifactType.DANGER_MASK,), f"danger_masks[{index}].artifact")
            _require_compatible(geometry, spec.artifact, f"danger_masks[{index}].artifact")
            available.append(spec)
        if not available:
            raise _AdapterError(empty_code, "no danger mask is available for evaluation")
        return geometry, tuple(available), tuple(warnings)

    @staticmethod
    def _validate_existing_candidate(candidate: CandidatePath, geometry: VolumeGeometry) -> None:
        if not _point_in_volume(geometry, candidate.entry_point_world_mm) or not _point_in_volume(
            geometry, candidate.target_point_world_mm
        ):
            raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "candidate endpoint is outside CT geometry")
        _require_positive_finite(candidate.length_mm, "candidate.length_mm")

    @staticmethod
    def _optional_artifact(
        artifact: ArtifactRef | None,
        geometry: VolumeGeometry,
        expected_types: tuple[ArtifactType, ...],
        field_path: str,
        warnings: list[str],
    ) -> ArtifactHandle | None:
        if artifact is None:
            return None
        if artifact.status is not ArtifactStatus.AVAILABLE:
            warnings.append(f"{field_path.upper()}_UNAVAILABLE")
            return None
        _require_artifact(artifact, expected_types, field_path)
        _require_compatible(geometry, artifact, field_path)
        return _handle(artifact)

    def _normalize_error(self, tool_name: str, exc: Exception) -> _AdapterError:
        if isinstance(exc, _AdapterError):
            return exc
        dependency = self.backend.manifest.backend_name
        if isinstance(exc, PlanningBackendTimeout):
            return _AdapterError(ErrorCode.TIMEOUT, "planning backend timed out", retryable=True, dependency=dependency)
        if isinstance(exc, PlanningBackendUnavailable):
            return _AdapterError(
                ErrorCode.DEPENDENCY_FAILED,
                "planning backend is unavailable",
                retryable=True,
                dependency=dependency,
            )
        if isinstance(exc, PlanningBackendInvalidArgument):
            return _AdapterError(
                ErrorCode.INVALID_ARGUMENT,
                "planning backend rejected the request",
                dependency=dependency,
            )
        if isinstance(exc, PlanningBackendGeometryMismatch):
            return _AdapterError(ErrorCode.GEOMETRY_MISMATCH, "backend geometry mismatch", dependency=dependency)
        if isinstance(exc, PlanningBackendMissingArtifact):
            if tool_name in {"evaluate_path_safety", "evaluate_intraoperative_risk"}:
                code = ErrorCode.REQUIRED_DANGER_MASK_MISSING
            elif tool_name == "generate_candidate_paths":
                code = ErrorCode.REQUIRED_LABEL_MISSING
            else:
                code = ErrorCode.MISSING_ARTIFACT
            return _AdapterError(code, "backend could not resolve a required artifact", dependency=dependency)
        if isinstance(exc, PlanningBackendNoCandidate):
            return _AdapterError(ErrorCode.NO_CANDIDATE_PATH, "backend produced no candidate path", dependency=dependency)
        if isinstance(exc, PlanningBackendSafetyFailure):
            return _AdapterError(ErrorCode.SAFETY_CHECK_FAILED, "backend safety computation failed", dependency=dependency)
        if isinstance(exc, PlanningBackendRiskFailure):
            return _AdapterError(ErrorCode.RISK_EVALUATION_FAILED, "backend risk computation failed", dependency=dependency)
        if isinstance(exc, PlanningBackendPenetrationUndetermined):
            return _AdapterError(
                ErrorCode.SKIN_PENETRATION_UNDETERMINED,
                "backend could not determine skin penetration",
                dependency=dependency,
            )
        return _AdapterError(
            ErrorCode.DEPENDENCY_FAILED,
            f"planning backend failed with {type(exc).__name__}",
            retryable=False,
            dependency=dependency,
        )

    def _failed(
        self,
        tool_name: str,
        request: Any,
        started_at: str,
        error: _AdapterError,
        *,
        replay: bool,
    ) -> ToolResponseEnvelope[Any]:
        response = ToolResponseEnvelope(
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            tool_version=TOOL_DEFINITIONS[tool_name].version,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(),
            warnings=(),
            error=ErrorDetail(
                code=error.code,
                message=str(error),
                retryable=error.retryable,
                field_path=error.field_path,
                dependency=error.dependency,
                details=error.details,
            ),
            started_at=started_at,
            finished_at=self.clock(),
        )
        if replay:
            self._trace(tool_name, request, response, replay=True, summary={})
        return response

    def _trace(
        self,
        tool_name: str,
        request: Any,
        response: ToolResponseEnvelope[Any],
        *,
        replay: bool,
        summary: Mapping[str, Any],
    ) -> None:
        record = PlanningTraceRecord(
                request_id=request.context.request_id,
                trace_id=request.context.trace_id,
                case_id=request.context.case_id,
                tool_name=tool_name,
                tool_version=TOOL_DEFINITIONS[tool_name].version,
                status=response.status,
                backend_name=self.backend.manifest.backend_name,
                backend_version=self.backend.manifest.backend_version,
                idempotency_key_sha256=sha256(
                    request.context.idempotency_key.encode("utf-8")
                ).hexdigest(),
                idempotent_replay=replay,
                result_summary=summary,
                error_code=response.error.code if response.error else None,
            )
        try:
            self.trace_sink.record(record)
        except Exception:
            # Telemetry must never change a deterministic planning/safety result.
            return


def build_planning_handlers(
    backend: PlanningKernelPort | None = None,
    *,
    trace_sink: PlanningTraceSink | None = None,
    clock: Callable[[], str] = _utc_now,
    epoch_ms: Callable[[], int] | None = None,
) -> tuple[PlanningToolAdapters, Mapping[str, Callable[[Any], ToolResponseEnvelope[Any]]]]:
    adapters = PlanningToolAdapters(
        backend if backend is not None else DeterministicPlanningBackend(),
        trace_sink=trace_sink,
        clock=clock,
        epoch_ms=epoch_ms,
    )
    return adapters, adapters.handlers()


def build_local_planning_adapter(
    *,
    trace_sink: PlanningTraceSink | None = None,
    clock: Callable[[], str] = _utc_now,
    epoch_ms: Callable[[], int] | None = None,
) -> PlanningToolAdapters:
    """Build the deterministic local fixture without company algorithm dependencies."""

    return PlanningToolAdapters(
        DeterministicPlanningBackend(),
        trace_sink=trace_sink,
        clock=clock,
        epoch_ms=epoch_ms,
    )


_DEFAULT_ADAPTER = build_local_planning_adapter()


def generate_candidate_paths(
    request: GenerateCandidatePathsRequest,
) -> ToolResponseEnvelope[CandidatePathGenerationResult]:
    """Module-level local handler suitable for explicit registry binding."""

    return _DEFAULT_ADAPTER.generate_candidate_paths(request)


def evaluate_path_safety(
    request: EvaluatePathSafetyRequest,
) -> ToolResponseEnvelope[PathSafetyEvaluationResult]:
    """Module-level local handler suitable for explicit registry binding."""

    return _DEFAULT_ADAPTER.evaluate_path_safety(request)


def evaluate_intraoperative_risk(
    request: EvaluateIntraoperativeRiskRequest,
) -> ToolResponseEnvelope[IntraoperativeRiskResult]:
    """Module-level local handler suitable for explicit registry binding."""

    return _DEFAULT_ADAPTER.evaluate_intraoperative_risk(request)


def verify_skin_penetration(
    request: VerifySkinPenetrationRequest,
) -> ToolResponseEnvelope[SkinPenetrationResult]:
    """Module-level local handler suitable for explicit registry binding."""

    return _DEFAULT_ADAPTER.verify_skin_penetration(request)


def _request_fingerprint(request: Any) -> str:
    primitive = to_primitive(request)
    if isinstance(primitive, dict) and isinstance(primitive.get("context"), dict):
        primitive["context"] = {
            "case_id": request.context.case_id,
            "caller": request.context.caller,
        }
    payload = json.dumps(
        primitive,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _handle(artifact: ArtifactRef | None) -> ArtifactHandle:
    if artifact is None or artifact.geometry is None:
        raise _AdapterError(ErrorCode.GEOMETRY_MISMATCH, "artifact geometry is missing")
    return ArtifactHandle(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        checksum_sha256=artifact.checksum_sha256,
        geometry_fingerprint=artifact.geometry.geometry_fingerprint,
    )


def _danger_handle(spec: DangerMaskSpec) -> DangerHandle:
    return DangerHandle(
        structure=spec.structure,
        artifact=_handle(spec.artifact),
        warning_mm=spec.safety_margin.warning_mm,
        stop_mm=spec.safety_margin.stop_mm,
        required=spec.required,
        vessel_core_erosion_mm=spec.vessel_core_erosion_mm,
    )


def _require_artifact(
    artifact: ArtifactRef,
    expected_types: tuple[ArtifactType, ...],
    field_path: str,
) -> None:
    if artifact.status is not ArtifactStatus.AVAILABLE:
        raise _AdapterError(
            ErrorCode.ARTIFACT_NOT_AVAILABLE,
            "required artifact is not available",
            field_path=field_path,
        )
    if artifact.artifact_type not in expected_types:
        raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "artifact type is invalid", field_path=field_path)


def _required_geometry(artifact: ArtifactRef, field_path: str) -> VolumeGeometry:
    if artifact.geometry is None:
        raise _AdapterError(ErrorCode.GEOMETRY_MISMATCH, "artifact geometry is missing", field_path=field_path)
    return artifact.geometry


def _require_compatible(reference: VolumeGeometry, artifact: ArtifactRef, field_path: str) -> None:
    geometry = _required_geometry(artifact, field_path)
    if not reference.is_compatible_with(geometry):
        raise _AdapterError(ErrorCode.GEOMETRY_MISMATCH, "artifact geometry is incompatible", field_path=field_path)


def _require_coordinate_support(geometry: VolumeGeometry, manifest: PlanningKernelManifest) -> None:
    if geometry.coordinate_system not in manifest.supported_coordinate_systems:
        raise _AdapterError(ErrorCode.INVALID_ARGUMENT, "coordinate system is not supported")


def _require_finite(
    value: float,
    field_path: str,
    *,
    contract: bool = False,
    code: ErrorCode = ErrorCode.CONTRACT_VIOLATION,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise _AdapterError(
            code if contract else ErrorCode.INVALID_ARGUMENT,
            f"{field_path} must be finite",
            field_path=field_path,
        )


def _require_positive_finite(value: float, field_path: str) -> None:
    _require_finite(value, field_path)
    if value <= 0:
        raise _AdapterError(ErrorCode.INVALID_ARGUMENT, f"{field_path} must be positive", field_path=field_path)


def _require_nonnegative_finite(value: float, field_path: str) -> None:
    _require_finite(value, field_path)
    if value < 0:
        raise _AdapterError(ErrorCode.INVALID_ARGUMENT, f"{field_path} must be non-negative", field_path=field_path)


def _validate_elapsed(value: float, code: ErrorCode = ErrorCode.CONTRACT_VIOLATION) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise _AdapterError(code, "backend elapsed_ms is invalid")


def _world_to_voxel(geometry: VolumeGeometry, point: WorldPoint) -> tuple[float, float, float]:
    matrix = geometry.direction_cosines
    a, b, c, d, e, f, g, h, i = matrix
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise _AdapterError(ErrorCode.GEOMETRY_MISMATCH, "direction cosine matrix is singular")
    inverse = (
        (e * i - f * h) / determinant,
        (c * h - b * i) / determinant,
        (b * f - c * e) / determinant,
        (f * g - d * i) / determinant,
        (a * i - c * g) / determinant,
        (c * d - a * f) / determinant,
        (d * h - e * g) / determinant,
        (b * g - a * h) / determinant,
        (a * e - b * d) / determinant,
    )
    delta = (
        point.x_mm - geometry.origin_mm[0],
        point.y_mm - geometry.origin_mm[1],
        point.z_mm - geometry.origin_mm[2],
    )
    physical_index = (
        inverse[0] * delta[0] + inverse[1] * delta[1] + inverse[2] * delta[2],
        inverse[3] * delta[0] + inverse[4] * delta[1] + inverse[5] * delta[2],
        inverse[6] * delta[0] + inverse[7] * delta[1] + inverse[8] * delta[2],
    )
    return tuple(value / spacing for value, spacing in zip(physical_index, geometry.spacing_mm))  # type: ignore[return-value]


def _point_in_volume(geometry: VolumeGeometry, point: WorldPoint) -> bool:
    index = _world_to_voxel(geometry, point)
    return all(-1e-6 <= value <= size - 1 + 1e-6 for value, size in zip(index, geometry.size_ijk))


def _point_on_segment(start: WorldPoint, end: WorldPoint, point: WorldPoint) -> bool:
    start_values = start.as_tuple()
    end_values = end.as_tuple()
    point_values = point.as_tuple()
    direction = tuple(right - left for left, right in zip(start_values, end_values))
    length_squared = sum(value * value for value in direction)
    if length_squared <= 0:
        return False
    parameter = sum(
        (value - left) * axis
        for value, left, axis in zip(point_values, start_values, direction)
    ) / length_squared
    if not -1e-6 <= parameter <= 1.0 + 1e-6:
        return False
    projection = tuple(left + parameter * axis for left, axis in zip(start_values, direction))
    return math.dist(projection, point_values) <= _TOLERANCE_MM


__all__ = [
    "ArtifactHandle",
    "CandidateGenerationCommand",
    "DangerHandle",
    "DeterministicPlanningBackend",
    "InMemoryPlanningTraceSink",
    "NativeCandidate",
    "NativeCandidateBatch",
    "NativeClearance",
    "NativePathAssessment",
    "NativeRiskFlag",
    "NativeRiskState",
    "NativeSafetyBatch",
    "NativeSkinTraversal",
    "PathSafetyCommand",
    "PlanningBackendGeometryMismatch",
    "PlanningBackendInvalidArgument",
    "PlanningBackendMissingArtifact",
    "PlanningBackendNoCandidate",
    "PlanningBackendPenetrationUndetermined",
    "PlanningBackendRiskFailure",
    "PlanningBackendSafetyFailure",
    "PlanningBackendTimeout",
    "PlanningBackendUnavailable",
    "PlanningKernelManifest",
    "PlanningKernelPort",
    "PlanningToolAdapters",
    "PlanningTraceRecord",
    "PlanningTraceSink",
    "SkinTraversalCommand",
    "TipRiskCommand",
    "build_planning_handlers",
    "build_local_planning_adapter",
    "evaluate_intraoperative_risk",
    "evaluate_path_safety",
    "generate_candidate_paths",
    "verify_skin_penetration",
]
