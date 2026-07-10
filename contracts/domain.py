"""Reusable domain objects shared by tool requests and results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .artifacts import ArtifactRef
from .enums import (
    PathDisposition,
    RiskLevel,
    RiskStructure,
    ValidationSeverity,
)
from .geometry import WorldPoint


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    name: str
    value: int
    required: bool
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("label name is required")
        if self.value < 0:
            raise ValueError("label value must be non-negative")


@dataclass(frozen=True, slots=True)
class LabelQualityThreshold:
    label_name: str
    min_voxel_count: int
    max_component_count: int
    min_volume_ml: float = 0.0
    max_volume_ml: float | None = None

    def __post_init__(self) -> None:
        if self.min_voxel_count < 0 or self.max_component_count < 1:
            raise ValueError("invalid label quality threshold")
        if self.min_volume_ml < 0:
            raise ValueError("min_volume_ml must be non-negative")
        if self.max_volume_ml is not None and self.max_volume_ml < self.min_volume_ml:
            raise ValueError("max_volume_ml must not be smaller than min_volume_ml")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    message: str
    artifact_id: str | None = None
    field_path: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("validation issue code and message are required")


@dataclass(frozen=True, slots=True)
class SafetyMargin:
    warning_mm: float
    stop_mm: float

    def __post_init__(self) -> None:
        if self.warning_mm < 0 or self.stop_mm < 0:
            raise ValueError("safety margins must be non-negative")
        if self.warning_mm < self.stop_mm:
            raise ValueError("warning_mm must be greater than or equal to stop_mm")


@dataclass(frozen=True, slots=True)
class DangerMaskSpec:
    structure: RiskStructure
    artifact: ArtifactRef
    safety_margin: SafetyMargin
    required: bool = True
    vessel_core_erosion_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.vessel_core_erosion_mm < 0:
            raise ValueError("vessel_core_erosion_mm must be non-negative")


@dataclass(frozen=True, slots=True)
class CandidatePath:
    candidate_id: str
    entry_point_world_mm: WorldPoint
    target_point_world_mm: WorldPoint
    length_mm: float
    insertion_angle_deg: float
    angle_reference: str
    rank_hint: int
    path_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not isfinite(self.length_mm) or self.length_mm <= 0:
            raise ValueError("length_mm must be a positive finite number")
        if not isfinite(self.insertion_angle_deg) or not 0 <= self.insertion_angle_deg <= 180:
            raise ValueError("insertion_angle_deg must be in [0, 180]")
        if self.rank_hint < 1:
            raise ValueError("rank_hint starts at 1")


@dataclass(frozen=True, slots=True)
class RiskFlag:
    structure: RiskStructure
    level: RiskLevel
    reason_code: str
    message: str
    distance_mm: float | None = None
    evidence_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason_code or not self.message:
            raise ValueError("risk flag reason_code and message are required")


@dataclass(frozen=True, slots=True)
class PathStructureClearance:
    structure: RiskStructure
    minimum_clearance_mm: float
    intersects_stop_region: bool
    intersects_warning_region: bool


@dataclass(frozen=True, slots=True)
class PathSafetyAssessment:
    candidate_id: str
    disposition: PathDisposition
    minimum_clearance_mm: float
    clearances: tuple[PathStructureClearance, ...]
    rejection_reasons: tuple[str, ...] = ()
