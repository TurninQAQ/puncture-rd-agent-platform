"""Contract-preserving adapters for the three segmentation tools.

The Agent layer deliberately does not implement TensorRT inference or medical
image morphology.  It coordinates three narrow ports and validates everything
that crosses those boundaries.  ``DeterministicSegmentationBackend`` is a
dependency-free manifest backend for local demos and adapter tests; production
code can inject company engine, image-algorithm, and artifact implementations.

Large voxel arrays never enter this module or an LLM-visible trace.  Inputs and
outputs remain :class:`contracts.artifacts.ArtifactRef` values.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil, isclose, isfinite
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, TypeVar

from contracts.artifacts import ArtifactRef
from contracts.common import MetricValue, ToolResponseEnvelope, to_primitive
from contracts.domain import LabelDefinition, ValidationIssue
from contracts.enums import (
    ArtifactStatus,
    ArtifactType,
    Connectivity,
    ErrorCode,
    InferencePrecision,
    RecommendedAction,
    SkinSurfaceMethod,
    ToolExecutionStatus,
    ValidationSeverity,
)
from contracts.errors import ErrorDetail
from contracts.geometry import VolumeGeometry
from contracts.tool_inputs import (
    ExtractSkinSurfaceRequest,
    RunSegmentationRequest,
    ValidateSegmentationResultRequest,
)
from contracts.tool_outputs import (
    LabelStatistics,
    LabelValidationResult,
    SegmentationInferenceResult,
    SegmentationValidationResult,
    SkinSurfaceExtractionResult,
)


TOOL_VERSION = "1.0.0"
_INTEGER_DTYPES = frozenset({"uint8", "uint16", "uint32", "int8", "int16", "int32"})
_SENSITIVE_DETAIL_FRAGMENTS = ("uri", "checksum", "path", "token", "secret", "credential")


@dataclass(frozen=True, slots=True)
class ModelLabel:
    """One immutable entry from the approved model registry label map."""

    name: str
    value: int
    required: bool = True
    deterministic_voxel_count: int = 50_000
    deterministic_component_count: int = 1
    touches_volume_border: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or self.value <= 0:
            raise ValueError("model label name is required and value must be positive")
        if self.deterministic_voxel_count < 0 or self.deterministic_component_count < 0:
            raise ValueError("deterministic label statistics must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Sanitized, versioned metadata for one deployable engine profile."""

    model_id: str
    model_version: str
    precision: InferencePrecision
    engine_hash: str
    labels: tuple[ModelLabel, ...]
    input_dtypes: tuple[str, ...] = ("int16", "float32")
    supported_device_ids: tuple[int, ...] = (0,)
    supports_label_selection: bool = False
    supports_probability_maps: bool = False
    preprocessing_profile: str = "ct-default-v1"
    runtime_name: str = "deterministic-manifest"

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.model_version.strip():
            raise ValueError("model_id and model_version are required")
        if len(self.engine_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.engine_hash.lower()
        ):
            raise ValueError("engine_hash must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "engine_hash", self.engine_hash.lower())
        if not self.labels:
            raise ValueError("model profile requires a label map")
        names = [item.name for item in self.labels]
        values = [item.value for item in self.labels]
        if len(names) != len(set(names)) or len(values) != len(set(values)):
            raise ValueError("model label names and values must be unique")
        normalized_dtypes = tuple(item.lower() for item in self.input_dtypes)
        if not normalized_dtypes:
            raise ValueError("at least one input dtype is required")
        object.__setattr__(self, "input_dtypes", normalized_dtypes)
        if not self.supported_device_ids or any(item < 0 for item in self.supported_device_ids):
            raise ValueError("supported_device_ids must contain non-negative device IDs")


@dataclass(frozen=True, slots=True)
class VolumeManifest:
    """Small local description of an artifact; never contains voxel arrays."""

    artifact_id: str
    geometry: VolumeGeometry | None
    dtype: str
    observed_label_values: tuple[int, ...] = ()
    label_statistics: tuple[LabelStatistics, ...] = ()
    checksum_valid: bool = True
    expected_checksum_sha256: str | None = None
    allowed_callers: frozenset[str] | None = None
    primary_label_value: int | None = None
    total_voxel_count: int | None = None
    deterministic_surface_voxel_count: int | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.dtype.strip():
            raise ValueError("manifest artifact_id and dtype are required")
        object.__setattr__(self, "dtype", self.dtype.lower())
        values = tuple(sorted(set(int(value) for value in self.observed_label_values)))
        if any(value < 0 for value in values):
            raise ValueError("manifest label values must be non-negative")
        object.__setattr__(self, "observed_label_values", values)
        stat_values = [item.label_value for item in self.label_statistics]
        if len(stat_values) != len(set(stat_values)):
            raise ValueError("manifest label statistics must have unique values")
        if self.expected_checksum_sha256 is not None:
            checksum = self.expected_checksum_sha256.lower()
            if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
                raise ValueError("expected checksum must be a SHA-256 hexadecimal digest")
            object.__setattr__(self, "expected_checksum_sha256", checksum)
        if self.primary_label_value is not None and self.primary_label_value < 0:
            raise ValueError("primary_label_value must be non-negative")
        if self.total_voxel_count is not None and self.total_voxel_count <= 0:
            raise ValueError("total_voxel_count must be positive")
        if self.deterministic_surface_voxel_count is not None and self.deterministic_surface_voxel_count < 0:
            raise ValueError("deterministic_surface_voxel_count must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Sanitized result returned by a TensorRT/C++ engine adapter."""

    geometry: VolumeGeometry
    output_dtype: str
    observed_label_values: tuple[int, ...]
    label_statistics: tuple[LabelStatistics, ...]
    inference_time_ms: float
    peak_gpu_memory_mb: float
    engine_hash: str
    runtime_name: str = "native"


@dataclass(frozen=True, slots=True)
class LabelAnalysis:
    """Independent image-operation result used by validation."""

    observed_label_values: tuple[int, ...]
    statistics: tuple[LabelStatistics, ...]
    total_voxel_count: int
    is_discrete: bool
    elapsed_ms: float
    runtime_name: str = "native"


@dataclass(frozen=True, slots=True)
class SkinSurfaceKernelResult:
    """Small result returned by the existing morphology implementation."""

    geometry: VolumeGeometry
    source_voxel_count: int
    surface_voxel_count: int
    effective_thickness_mm: tuple[float, float, float]
    components_removed: int
    binary_values: tuple[int, ...] = (0, 1)
    subset_of_source: bool = True
    external_surface_only: bool = True
    elapsed_ms: float = 0.0
    runtime_name: str = "native"


class SegmentationBackendError(RuntimeError):
    """Stable exception raised at native/storage boundaries."""

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
        self.public_message = message
        self.retryable = retryable
        self.field_path = field_path
        self.dependency = dependency
        self.details = dict(details or {})


class SegmentationEnginePort(Protocol):
    """Narrow port around model registry lookup and long-lived engine reuse."""

    def describe(
        self,
        model_id: str,
        version: str,
        precision: InferencePrecision,
    ) -> ModelProfile: ...

    def infer(self, ct: ArtifactRef, profile: ModelProfile, device_id: int) -> EngineResult: ...


class ImageAlgorithmPort(Protocol):
    """Narrow port around independent label and morphology operations."""

    def label_statistics(
        self,
        labelmap: ArtifactRef,
        schema: tuple[LabelDefinition, ...],
    ) -> LabelAnalysis: ...

    def extract_external_skin_surface(
        self,
        source: ArtifactRef,
        thickness_mm: float,
        method: SkinSurfaceMethod,
        connectivity: Connectivity,
        keep_largest_component: bool,
    ) -> SkinSurfaceKernelResult: ...


class SegmentationArtifactPort(Protocol):
    """Permission-aware artifact resolution and atomic commit boundary."""

    def resolve(self, artifact: ArtifactRef, *, caller: str, purpose: str) -> VolumeManifest: ...

    def commit(
        self,
        artifact: ArtifactRef,
        manifest: VolumeManifest,
        *,
        caller: str,
        commit_key: str,
        request_fingerprint: str,
    ) -> ArtifactRef: ...


class SegmentationBackendPort(
    SegmentationEnginePort,
    ImageAlgorithmPort,
    SegmentationArtifactPort,
    Protocol,
):
    """Convenience protocol for a backend implementing all three narrow ports."""


@dataclass(frozen=True, slots=True)
class SegmentationTraceEvent:
    """Safe trace projection containing IDs/metrics, never URIs or checksums."""

    timestamp: str
    request_id: str
    trace_id: str
    tool_name: str
    outcome: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", dict(self.attributes))


class SegmentationTraceSink(Protocol):
    def record(self, event: SegmentationTraceEvent) -> None: ...


class InMemorySegmentationTraceSink:
    """Thread-safe trace collector used by the local demo and tests."""

    def __init__(self) -> None:
        self._events: list[SegmentationTraceEvent] = []
        self._lock = RLock()

    def record(self, event: SegmentationTraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[SegmentationTraceEvent, ...]:
        with self._lock:
            return tuple(self._events)


class _NullTraceSink:
    def record(self, event: SegmentationTraceEvent) -> None:
        del event


def _default_profiles() -> tuple[ModelProfile, ...]:
    labels = (
        ModelLabel("skin", 1, deterministic_voxel_count=480_000, touches_volume_border=True),
        ModelLabel("lung", 2, deterministic_voxel_count=180_000),
        ModelLabel("heart", 3, deterministic_voxel_count=60_000),
    )
    profiles = []
    for precision in (InferencePrecision.FP16, InferencePrecision.FP32):
        digest = sha256(f"nnunet-puncture|v1|{precision.value}|local-demo".encode("utf-8")).hexdigest()
        profiles.append(
            ModelProfile(
                model_id="nnunet-puncture",
                model_version="v1",
                precision=precision,
                engine_hash=digest,
                labels=labels,
            )
        )
    return tuple(profiles)


class DeterministicSegmentationBackend(
    SegmentationEnginePort,
    ImageAlgorithmPort,
    SegmentationArtifactPort,
):
    """Manifest-only backend suitable for Python 3.10 local demos.

    It is intentionally not a medical algorithm.  Counters and explicit
    ``fail_next``/override hooks provide deterministic adapter verification.
    Metadata keys use ordinary names such as ``label_values``; legacy
    ``mock_*`` control flags are deliberately ignored.
    """

    def __init__(
        self,
        profiles: tuple[ModelProfile, ...] | None = None,
        *,
        allow_manifest_fallback: bool = True,
    ) -> None:
        selected_profiles = profiles if profiles is not None else _default_profiles()
        self._profiles = {
            (item.model_id, item.model_version, item.precision): item for item in selected_profiles
        }
        if len(self._profiles) != len(selected_profiles):
            raise ValueError("model profiles must have unique model/version/precision keys")
        self._allow_manifest_fallback = allow_manifest_fallback
        self._manifests: dict[str, VolumeManifest] = {}
        self._permissions: dict[str, frozenset[str] | None] = {}
        self._failures: dict[str, deque[SegmentationBackendError]] = defaultdict(deque)
        self._inference_overrides: dict[tuple[str, str, InferencePrecision], EngineResult] = {}
        self._analysis_overrides: dict[str, LabelAnalysis] = {}
        self._skin_overrides: dict[str, SkinSurfaceKernelResult] = {}
        self._loaded_engines: set[tuple[str, str, InferencePrecision, int]] = set()
        self._commits: dict[str, tuple[str, ArtifactRef]] = {}
        self._lock = RLock()
        self._engine_load_count = 0
        self._inference_count = 0
        self._commit_count = 0
        self._resolve_count = 0
        self._active_inferences = 0

    def register_manifest(self, manifest: VolumeManifest) -> None:
        with self._lock:
            self._manifests[manifest.artifact_id] = manifest
            if manifest.allowed_callers is not None:
                self._permissions[manifest.artifact_id] = manifest.allowed_callers

    def set_artifact_permissions(self, artifact_id: str, allowed_callers: set[str] | None) -> None:
        with self._lock:
            self._permissions[artifact_id] = (
                None if allowed_callers is None else frozenset(allowed_callers)
            )

    def register_profile(self, profile: ModelProfile) -> None:
        with self._lock:
            self._profiles[(profile.model_id, profile.model_version, profile.precision)] = profile

    def fail_next(self, operation: str, error: SegmentationBackendError) -> None:
        with self._lock:
            self._failures[operation].append(error)

    def set_inference_override(
        self,
        model_id: str,
        model_version: str,
        precision: InferencePrecision,
        result: EngineResult,
    ) -> None:
        with self._lock:
            self._inference_overrides[(model_id, model_version, precision)] = result

    def set_analysis_override(self, artifact_id: str, result: LabelAnalysis) -> None:
        with self._lock:
            self._analysis_overrides[artifact_id] = result

    def set_skin_override(self, artifact_id: str, result: SkinSurfaceKernelResult) -> None:
        with self._lock:
            self._skin_overrides[artifact_id] = result

    @property
    def engine_load_count(self) -> int:
        with self._lock:
            return self._engine_load_count

    @property
    def inference_count(self) -> int:
        with self._lock:
            return self._inference_count

    @property
    def commit_count(self) -> int:
        with self._lock:
            return self._commit_count

    @property
    def resolve_count(self) -> int:
        with self._lock:
            return self._resolve_count

    @property
    def active_inferences(self) -> int:
        with self._lock:
            return self._active_inferences

    def committed_manifest(self, artifact_id: str) -> VolumeManifest | None:
        with self._lock:
            return self._manifests.get(artifact_id)

    def _raise_scheduled(self, operation: str) -> None:
        with self._lock:
            if self._failures[operation]:
                raise self._failures[operation].popleft()

    def describe(
        self,
        model_id: str,
        version: str,
        precision: InferencePrecision,
    ) -> ModelProfile:
        self._raise_scheduled("describe")
        with self._lock:
            profile = self._profiles.get((model_id, version, precision))
            if profile is not None:
                return profile
            model_versions = {
                item.model_version for item in self._profiles.values() if item.model_id == model_id
            }
            if not model_versions:
                raise SegmentationBackendError(
                    ErrorCode.MODEL_NOT_FOUND,
                    "requested model is not registered",
                    dependency="model_registry",
                    details={"model_id": model_id},
                )
            if version not in model_versions:
                raise SegmentationBackendError(
                    ErrorCode.MODEL_VERSION_MISMATCH,
                    "requested model version is not registered",
                    dependency="model_registry",
                    details={"model_id": model_id, "model_version": version},
                )
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "requested precision has no approved engine profile",
                field_path="precision",
                details={"precision": precision.value},
            )

    def infer(self, ct: ArtifactRef, profile: ModelProfile, device_id: int) -> EngineResult:
        if device_id not in profile.supported_device_ids:
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "requested device is not approved for this engine profile",
                field_path="device_id",
                details={"device_id": str(device_id)},
            )
        engine_key = (profile.model_id, profile.model_version, profile.precision, device_id)
        with self._lock:
            self._active_inferences += 1
        try:
            self._raise_scheduled("infer")
            with self._lock:
                if engine_key not in self._loaded_engines:
                    self._loaded_engines.add(engine_key)
                    self._engine_load_count += 1
                self._inference_count += 1
                override = self._inference_overrides.get(
                    (profile.model_id, profile.model_version, profile.precision)
                )
            if override is not None:
                return override
            if ct.geometry is None:
                raise SegmentationBackendError(
                    ErrorCode.GEOMETRY_MISMATCH,
                    "CT geometry is required for inference",
                    field_path="ct_artifact.geometry",
                )
            voxel_volume_ml = _voxel_volume_ml(ct.geometry)
            stats = tuple(
                LabelStatistics(
                    label_name=item.name,
                    label_value=item.value,
                    voxel_count=item.deterministic_voxel_count,
                    volume_ml=item.deterministic_voxel_count * voxel_volume_ml,
                    component_count=item.deterministic_component_count,
                    touches_volume_border=item.touches_volume_border,
                )
                for item in profile.labels
            )
            return EngineResult(
                geometry=ct.geometry,
                output_dtype="uint8",
                observed_label_values=(0, *(item.value for item in profile.labels)),
                label_statistics=stats,
                inference_time_ms=12.0,
                peak_gpu_memory_mb=256.0,
                engine_hash=profile.engine_hash,
                runtime_name=profile.runtime_name,
            )
        finally:
            with self._lock:
                self._active_inferences -= 1

    def resolve(self, artifact: ArtifactRef, *, caller: str, purpose: str) -> VolumeManifest:
        del purpose
        with self._lock:
            explicit_rule = self._permissions.get(artifact.artifact_id)
            registered = self._manifests.get(artifact.artifact_id)
            manifest_rule = registered.allowed_callers if registered is not None else None
        allowed = explicit_rule if artifact.artifact_id in self._permissions else manifest_rule
        if allowed is None:
            metadata_callers = _csv_strings(artifact.metadata.get("allowed_callers", ""))
            allowed = frozenset(metadata_callers) if metadata_callers else None
        if allowed is not None and caller not in allowed:
            raise SegmentationBackendError(
                ErrorCode.PERMISSION_DENIED,
                "caller is not permitted to access the artifact",
                field_path="artifact",
                details={"artifact_id": artifact.artifact_id},
            )
        self._raise_scheduled("resolve")
        if artifact.status is ArtifactStatus.MISSING:
            raise SegmentationBackendError(
                ErrorCode.MISSING_ARTIFACT,
                "artifact does not exist",
                field_path="artifact.status",
                details={"artifact_id": artifact.artifact_id},
            )
        if artifact.status is not ArtifactStatus.AVAILABLE:
            raise SegmentationBackendError(
                ErrorCode.ARTIFACT_NOT_AVAILABLE,
                "artifact is not available",
                retryable=artifact.status is ArtifactStatus.PENDING,
                field_path="artifact.status",
                details={"artifact_id": artifact.artifact_id, "status": artifact.status.value},
            )
        with self._lock:
            manifest = self._manifests.get(artifact.artifact_id)
        if manifest is None:
            if not self._allow_manifest_fallback:
                raise SegmentationBackendError(
                    ErrorCode.MISSING_ARTIFACT,
                    "artifact manifest is not registered",
                    dependency="artifact_registry",
                    details={"artifact_id": artifact.artifact_id},
                )
            manifest = self._manifest_from_reference(artifact)
            with self._lock:
                self._manifests.setdefault(artifact.artifact_id, manifest)
        if not manifest.checksum_valid or (
            manifest.expected_checksum_sha256 is not None
            and manifest.expected_checksum_sha256 != artifact.checksum_sha256
        ):
            raise SegmentationBackendError(
                ErrorCode.CHECKSUM_MISMATCH,
                "artifact checksum validation failed",
                field_path="artifact.checksum_sha256",
                details={"artifact_id": artifact.artifact_id},
            )
        if artifact.geometry is not None and manifest.geometry is not None:
            if artifact.geometry.geometry_fingerprint != manifest.geometry.geometry_fingerprint:
                raise SegmentationBackendError(
                    ErrorCode.GEOMETRY_MISMATCH,
                    "artifact reference and stored image geometry differ",
                    field_path="artifact.geometry",
                    details={"artifact_id": artifact.artifact_id},
                )
        with self._lock:
            self._resolve_count += 1
        return manifest

    def label_statistics(
        self,
        labelmap: ArtifactRef,
        schema: tuple[LabelDefinition, ...],
    ) -> LabelAnalysis:
        self._raise_scheduled("label_statistics")
        with self._lock:
            override = self._analysis_overrides.get(labelmap.artifact_id)
            manifest = self._manifests.get(labelmap.artifact_id)
        if override is not None:
            return override
        if manifest is None:
            raise SegmentationBackendError(
                ErrorCode.MISSING_ARTIFACT,
                "labelmap manifest was not resolved",
                dependency="artifact_registry",
                details={"artifact_id": labelmap.artifact_id},
            )
        if manifest.geometry is None:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "labelmap geometry is missing",
                field_path="segmentation_artifact.geometry",
            )
        total_voxels = manifest.total_voxel_count or _geometry_voxel_count(manifest.geometry)
        schema_names = {item.value: item.name for item in schema}
        stored = {item.label_value: item for item in manifest.label_statistics}
        non_background = sum(
            item.voxel_count for value, item in stored.items() if value != 0
        )
        statistics = []
        for value in manifest.observed_label_values:
            item = stored.get(value)
            if item is None:
                voxel_count = max(0, total_voxels - non_background) if value == 0 else 0
                component_count = 1 if voxel_count else 0
                touches_border = value == 0 and voxel_count > 0
            else:
                voxel_count = item.voxel_count
                component_count = item.component_count
                touches_border = item.touches_volume_border
            statistics.append(
                LabelStatistics(
                    label_name=schema_names.get(value, item.label_name if item else f"label_{value}"),
                    label_value=value,
                    voxel_count=voxel_count,
                    volume_ml=voxel_count * _voxel_volume_ml(manifest.geometry),
                    component_count=component_count,
                    touches_volume_border=touches_border,
                )
            )
        return LabelAnalysis(
            observed_label_values=manifest.observed_label_values,
            statistics=tuple(statistics),
            total_voxel_count=total_voxels,
            is_discrete=manifest.dtype in _INTEGER_DTYPES,
            elapsed_ms=1.5,
            runtime_name="deterministic-manifest",
        )

    def extract_external_skin_surface(
        self,
        source: ArtifactRef,
        thickness_mm: float,
        method: SkinSurfaceMethod,
        connectivity: Connectivity,
        keep_largest_component: bool,
    ) -> SkinSurfaceKernelResult:
        del method, connectivity
        self._raise_scheduled("extract_skin_surface")
        with self._lock:
            override = self._skin_overrides.get(source.artifact_id)
            manifest = self._manifests.get(source.artifact_id)
        if override is not None:
            return override
        if manifest is None:
            raise SegmentationBackendError(
                ErrorCode.MISSING_ARTIFACT,
                "skin-mask manifest was not resolved",
                dependency="artifact_registry",
                details={"artifact_id": source.artifact_id},
            )
        if manifest.geometry is None:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "skin-mask geometry is missing",
                field_path="skin_mask_artifact.geometry",
            )
        if manifest.dtype not in _INTEGER_DTYPES:
            raise SegmentationBackendError(
                ErrorCode.LABEL_SCHEMA_ERROR,
                "skin mask must use a discrete integer label dtype",
                field_path="skin_mask_artifact",
            )
        skin_value = manifest.primary_label_value if manifest.primary_label_value is not None else 1
        stored = {item.label_value: item for item in manifest.label_statistics}
        skin = stored.get(skin_value)
        source_count = skin.voxel_count if skin is not None else 0
        if skin_value not in manifest.observed_label_values or source_count <= 0:
            raise SegmentationBackendError(
                ErrorCode.EMPTY_SEGMENTATION,
                "configured skin label is empty",
                field_path="skin_mask_artifact",
                details={"skin_label_value": str(skin_value)},
            )
        effective = tuple(
            round(ceil(thickness_mm / spacing) * spacing, 12)
            for spacing in manifest.geometry.spacing_mm
        )
        if manifest.deterministic_surface_voxel_count is not None:
            surface_count = manifest.deterministic_surface_voxel_count
        elif source_count == 1:
            surface_count = 1
        else:
            scale = max(0.05, min(0.8, sum(effective) / 30.0))
            surface_count = min(source_count - 1, max(1, round(source_count * scale)))
        components_removed = (
            max(0, skin.component_count - 1) if keep_largest_component else 0
        )
        return SkinSurfaceKernelResult(
            geometry=manifest.geometry,
            source_voxel_count=source_count,
            surface_voxel_count=surface_count,
            effective_thickness_mm=effective,
            components_removed=components_removed,
            elapsed_ms=2.5,
            runtime_name="deterministic-manifest",
        )

    def commit(
        self,
        artifact: ArtifactRef,
        manifest: VolumeManifest,
        *,
        caller: str,
        commit_key: str,
        request_fingerprint: str,
    ) -> ArtifactRef:
        del caller
        self._raise_scheduled("commit")
        if manifest.artifact_id != artifact.artifact_id:
            raise SegmentationBackendError(
                ErrorCode.CONTRACT_VIOLATION,
                "artifact and manifest identities differ",
                dependency="artifact_store",
            )
        if manifest.expected_checksum_sha256 != artifact.checksum_sha256:
            raise SegmentationBackendError(
                ErrorCode.CHECKSUM_MISMATCH,
                "output manifest checksum differs from artifact checksum",
                dependency="artifact_store",
            )
        with self._lock:
            previous = self._commits.get(commit_key)
            if previous is not None:
                previous_fingerprint, previous_artifact = previous
                if previous_fingerprint != request_fingerprint:
                    raise SegmentationBackendError(
                        ErrorCode.CONTRACT_VIOLATION,
                        "idempotency key was reused for different content",
                        field_path="context.idempotency_key",
                    )
                return previous_artifact
            existing = self._manifests.get(artifact.artifact_id)
            if existing is not None and existing.expected_checksum_sha256 not in (
                None,
                artifact.checksum_sha256,
            ):
                raise SegmentationBackendError(
                    ErrorCode.CONTRACT_VIOLATION,
                    "artifact ID already refers to different content",
                    dependency="artifact_store",
                )
            self._manifests[artifact.artifact_id] = manifest
            self._commits[commit_key] = (request_fingerprint, artifact)
            self._commit_count += 1
            return artifact

    def _manifest_from_reference(self, artifact: ArtifactRef) -> VolumeManifest:
        metadata = artifact.metadata
        checksum_valid = metadata.get("checksum_valid", "true").lower() != "false"
        dtype = metadata.get(
            "dtype",
            "int16" if artifact.artifact_type is ArtifactType.CT_VOLUME else "uint8",
        )
        if artifact.artifact_type is ArtifactType.CT_VOLUME:
            return VolumeManifest(
                artifact_id=artifact.artifact_id,
                geometry=artifact.geometry,
                dtype=dtype,
                checksum_valid=checksum_valid,
                expected_checksum_sha256=artifact.checksum_sha256,
            )
        values = _csv_ints(metadata.get("label_values", "0,1"), field_name="label_values")
        counts = _int_mapping(metadata.get("label_voxel_counts", ""), "label_voxel_counts")
        components = _int_mapping(
            metadata.get("label_component_counts", ""), "label_component_counts"
        )
        border_values = set(
            _csv_ints(metadata.get("border_label_values", "1"), field_name="border_label_values")
        )
        geometry = artifact.geometry
        total_voxels = _geometry_voxel_count(geometry) if geometry is not None else None
        voxel_volume_ml = _voxel_volume_ml(geometry) if geometry is not None else 0.0
        canonical_names = {0: "background", 1: "skin", 2: "lung", 3: "heart", 4: "liver"}
        stats = []
        non_background = 0
        for value in values:
            if value == 0:
                continue
            default_count = 480_000 if value == 1 else 50_000 * value
            count = counts.get(value, default_count)
            non_background += count
            stats.append(
                LabelStatistics(
                    label_name=canonical_names.get(value, f"label_{value}"),
                    label_value=value,
                    voxel_count=count,
                    volume_ml=count * voxel_volume_ml,
                    component_count=components.get(value, 1 if count else 0),
                    touches_volume_border=value in border_values,
                )
            )
        if 0 in values:
            background_count = max(0, (total_voxels or non_background) - non_background)
            stats.insert(
                0,
                LabelStatistics(
                    label_name="background",
                    label_value=0,
                    voxel_count=background_count,
                    volume_ml=background_count * voxel_volume_ml,
                    component_count=1 if background_count else 0,
                    touches_volume_border=background_count > 0,
                ),
            )
        primary = int(metadata.get("primary_label_value", "1"))
        surface_value = metadata.get("surface_voxel_count")
        return VolumeManifest(
            artifact_id=artifact.artifact_id,
            geometry=geometry,
            dtype=dtype,
            observed_label_values=values,
            label_statistics=tuple(stats),
            checksum_valid=checksum_valid,
            expected_checksum_sha256=artifact.checksum_sha256,
            primary_label_value=primary,
            total_voxel_count=total_voxels,
            deterministic_surface_voxel_count=(
                int(surface_value) if surface_value is not None else None
            ),
        )


_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class _CachedSuccess:
    request_fingerprint: str
    result: Any
    artifacts: tuple[ArtifactRef, ...]
    metrics: tuple[MetricValue, ...]
    warnings: tuple[str, ...]


class SegmentationToolAdapter:
    """Orchestrates and validates segmentation ports behind fixed contracts."""

    def __init__(
        self,
        backend: SegmentationBackendPort | None = None,
        *,
        engine: SegmentationEnginePort | None = None,
        algorithms: ImageAlgorithmPort | None = None,
        artifacts: SegmentationArtifactPort | None = None,
        trace_sink: SegmentationTraceSink | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if backend is not None:
            if engine is not None or algorithms is not None or artifacts is not None:
                raise ValueError("backend cannot be combined with individual port overrides")
            engine = backend
            algorithms = backend
            artifacts = backend
        fallback = DeterministicSegmentationBackend()
        self.engine = engine or fallback
        self.algorithms = algorithms or fallback
        self.artifacts = artifacts or fallback
        self.backend = (
            self.engine
            if self.engine is self.algorithms and self.engine is self.artifacts
            else None
        )
        self.local_backend = (
            self.backend
            if isinstance(self.backend, DeterministicSegmentationBackend)
            else None
        )
        self.trace_sink = trace_sink or _NullTraceSink()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._cache: dict[tuple[str, str, str, str], _CachedSuccess] = {}
        self._lock = RLock()

    def handlers(self) -> Mapping[str, Callable[[Any], ToolResponseEnvelope[Any]]]:
        """Return the three stable handlers without mutating the global registry."""

        return MappingProxyType(
            {
                "run_segmentation": self.run_segmentation,
                "validate_segmentation_result": self.validate_segmentation_result,
                "extract_skin_surface": self.extract_skin_surface,
            }
        )

    def run_segmentation(
        self,
        request: RunSegmentationRequest,
    ) -> ToolResponseEnvelope[SegmentationInferenceResult]:
        tool_name = "run_segmentation"
        with self._lock:
            started_at = self._utc_now()
            try:
                self._check_deadline(request.context.deadline_epoch_ms)
                ct_manifest = self._resolve(
                    request.ct_artifact,
                    caller=request.context.caller,
                    purpose=tool_name,
                    allowed_types=(ArtifactType.CT_VOLUME,),
                    field_path="ct_artifact",
                )
                fingerprint = _request_fingerprint(request)
                replay = self._replay(request, tool_name, fingerprint, started_at)
                if replay is not None:
                    return replay
                profile = self.engine.describe(
                    request.model_id,
                    request.model_version,
                    request.precision,
                )
                self._check_deadline(request.context.deadline_epoch_ms)
                self._validate_profile_for_request(profile, request, ct_manifest)
                engine_result = self.engine.infer(
                    request.ct_artifact,
                    profile,
                    request.device_id,
                )
                self._check_deadline(request.context.deadline_epoch_ms)
                produced = self._validate_engine_result(
                    engine_result,
                    profile,
                    request.ct_artifact.geometry,
                    request.requested_labels,
                )
                artifact = _build_output_artifact(
                    case_id=request.context.case_id,
                    suffix="seg",
                    artifact_type=ArtifactType.SEGMENTATION_MASK,
                    parent_ids=(request.ct_artifact.artifact_id,),
                    producer_name=tool_name,
                    geometry=engine_result.geometry,
                    content={
                        "request": fingerprint,
                        "model_id": profile.model_id,
                        "model_version": profile.model_version,
                        "precision": profile.precision.value,
                        "engine_hash": profile.engine_hash,
                        "labels": [item.label_value for item in engine_result.label_statistics],
                    },
                    metadata={
                        "model_id": profile.model_id,
                        "model_version": profile.model_version,
                        "precision": profile.precision.value,
                        "engine_hash": profile.engine_hash,
                        "preprocessing_profile": profile.preprocessing_profile,
                        "label_values": ",".join(
                            str(value) for value in engine_result.observed_label_values
                        ),
                        "runtime_name": engine_result.runtime_name,
                    },
                )
                output_manifest = VolumeManifest(
                    artifact_id=artifact.artifact_id,
                    geometry=engine_result.geometry,
                    dtype=engine_result.output_dtype,
                    observed_label_values=engine_result.observed_label_values,
                    label_statistics=engine_result.label_statistics,
                    expected_checksum_sha256=artifact.checksum_sha256,
                    allowed_callers=frozenset({request.context.caller}),
                    primary_label_value=1,
                    total_voxel_count=_geometry_voxel_count(engine_result.geometry),
                )
                committed = self.artifacts.commit(
                    artifact,
                    output_manifest,
                    caller=request.context.caller,
                    commit_key=self._commit_key(request, tool_name),
                    request_fingerprint=fingerprint,
                )
                result = SegmentationInferenceResult(
                    segmentation_artifact=committed,
                    model_id=profile.model_id,
                    model_version=profile.model_version,
                    precision=profile.precision,
                    produced_labels=produced,
                    inference_time_ms=engine_result.inference_time_ms,
                    peak_gpu_memory_mb=engine_result.peak_gpu_memory_mb,
                )
                metrics = (
                    MetricValue("inference_time", engine_result.inference_time_ms, "ms"),
                    MetricValue("peak_gpu_memory", engine_result.peak_gpu_memory_mb, "MB"),
                )
                warnings = _simulation_warnings(engine_result.runtime_name)
                response = self._success(
                    request,
                    tool_name,
                    result,
                    started_at,
                    artifacts=(committed,),
                    metrics=metrics,
                    warnings=warnings,
                )
                self._remember(request, tool_name, fingerprint, response)
                self._trace(
                    request,
                    tool_name,
                    "success",
                    {
                        "artifact_id": committed.artifact_id,
                        "model_id": profile.model_id,
                        "model_version": profile.model_version,
                        "precision": profile.precision.value,
                        "engine_hash": profile.engine_hash,
                        "device_id": request.device_id,
                        "inference_time_ms": engine_result.inference_time_ms,
                        "peak_gpu_memory_mb": engine_result.peak_gpu_memory_mb,
                    },
                )
                return response
            except SegmentationBackendError as exc:
                return self._failure(request, tool_name, started_at, exc)
            except Exception:
                return self._failure(
                    request,
                    tool_name,
                    started_at,
                    SegmentationBackendError(
                        ErrorCode.INTERNAL_ERROR,
                        "unexpected segmentation adapter failure",
                    ),
                )

    def validate_segmentation_result(
        self,
        request: ValidateSegmentationResultRequest,
    ) -> ToolResponseEnvelope[SegmentationValidationResult]:
        tool_name = "validate_segmentation_result"
        with self._lock:
            started_at = self._utc_now()
            try:
                self._check_deadline(request.context.deadline_epoch_ms)
                ct_manifest = self._resolve(
                    request.ct_artifact,
                    caller=request.context.caller,
                    purpose=tool_name,
                    allowed_types=(ArtifactType.CT_VOLUME,),
                    field_path="ct_artifact",
                )
                segmentation_manifest = self._resolve(
                    request.segmentation_artifact,
                    caller=request.context.caller,
                    purpose=tool_name,
                    allowed_types=(ArtifactType.SEGMENTATION_MASK, ArtifactType.NIFTI_LABELMAP),
                    field_path="segmentation_artifact",
                )
                fingerprint = _request_fingerprint(request)
                replay = self._replay(request, tool_name, fingerprint, started_at)
                if replay is not None:
                    return replay
                expected_names = {item.name for item in request.expected_labels}
                unknown_thresholds = [
                    item.label_name
                    for item in request.quality_thresholds
                    if item.label_name not in expected_names
                ]
                if unknown_thresholds:
                    raise SegmentationBackendError(
                        ErrorCode.INVALID_ARGUMENT,
                        "quality threshold references an unknown label name",
                        field_path="quality_thresholds",
                        details={"label_name": unknown_thresholds[0]},
                    )
                analysis = self.algorithms.label_statistics(
                    request.segmentation_artifact,
                    request.expected_labels,
                )
                self._check_deadline(request.context.deadline_epoch_ms)
                if not analysis.is_discrete or segmentation_manifest.dtype not in _INTEGER_DTYPES:
                    raise SegmentationBackendError(
                        ErrorCode.UNSUPPORTED_FORMAT,
                        "segmentation labelmap must use a discrete integer dtype",
                        field_path="segmentation_artifact",
                    )
                if analysis.total_voxel_count <= 0:
                    raise SegmentationBackendError(
                        ErrorCode.QUALITY_CHECK_FAILED,
                        "segmentation analysis returned an invalid voxel count",
                        dependency="image_algorithms",
                    )
                expected_total_voxels = _geometry_voxel_count(
                    self._require_geometry(
                        segmentation_manifest.geometry,
                        "segmentation_artifact.geometry",
                    )
                )
                if analysis.total_voxel_count != expected_total_voxels:
                    raise SegmentationBackendError(
                        ErrorCode.QUALITY_CHECK_FAILED,
                        "segmentation analysis voxel count differs from image geometry",
                        dependency="image_algorithms",
                    )
                if analysis.elapsed_ms < 0 or not isfinite(analysis.elapsed_ms):
                    raise SegmentationBackendError(
                        ErrorCode.QUALITY_CHECK_FAILED,
                        "segmentation analysis returned invalid timing metadata",
                        dependency="image_algorithms",
                    )
                if (
                    len(analysis.observed_label_values)
                    != len(set(analysis.observed_label_values))
                    or any(value < 0 for value in analysis.observed_label_values)
                ):
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "image analysis returned invalid observed label values",
                        dependency="image_algorithms",
                    )
                geometry_matches = _geometry_matches(
                    ct_manifest.geometry,
                    segmentation_manifest.geometry,
                )
                issues: list[ValidationIssue] = []
                if request.require_geometry_match and not geometry_matches:
                    issues.append(
                        ValidationIssue(
                            "GEOMETRY_MISMATCH",
                            ValidationSeverity.ERROR,
                            "segmentation geometry differs from the reference CT",
                            artifact_id=request.segmentation_artifact.artifact_id,
                            field_path="segmentation_artifact.geometry",
                        )
                    )
                expected_values = {item.value for item in request.expected_labels}
                unknown_values = sorted(
                    value for value in analysis.observed_label_values if value not in expected_values
                )
                for value in unknown_values:
                    issues.append(
                        ValidationIssue(
                            "UNKNOWN_LABEL_VALUE",
                            ValidationSeverity.ERROR,
                            f"observed label value {value} is not in the expected schema",
                            artifact_id=request.segmentation_artifact.artifact_id,
                            field_path="segmentation_artifact",
                        )
                    )
                stats_by_value = {item.label_value: item for item in analysis.statistics}
                if len(stats_by_value) != len(analysis.statistics):
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "image analysis returned duplicate label statistics",
                        dependency="image_algorithms",
                    )
                if any(
                    item.voxel_count < 0
                    or item.voxel_count > analysis.total_voxel_count
                    or item.component_count < 0
                    or item.volume_ml < 0
                    or not isfinite(item.volume_ml)
                    for item in analysis.statistics
                ):
                    raise SegmentationBackendError(
                        ErrorCode.QUALITY_CHECK_FAILED,
                        "image analysis returned invalid label statistics",
                        dependency="image_algorithms",
                    )
                missing_statistics = sorted(
                    set(analysis.observed_label_values) - set(stats_by_value)
                )
                if missing_statistics:
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "image analysis omitted statistics for an observed label",
                        dependency="image_algorithms",
                        details={"label_value": str(missing_statistics[0])},
                    )
                non_background_voxels = sum(
                    item.voxel_count
                    for item in analysis.statistics
                    if item.label_value != 0
                )
                if non_background_voxels > analysis.total_voxel_count:
                    raise SegmentationBackendError(
                        ErrorCode.QUALITY_CHECK_FAILED,
                        "non-background label counts exceed the image volume",
                        dependency="image_algorithms",
                    )
                thresholds = {item.label_name: item for item in request.quality_thresholds}
                label_results = []
                voxel_volume_ml = _voxel_volume_ml(segmentation_manifest.geometry)
                for label in request.expected_labels:
                    observed = label.value in analysis.observed_label_values
                    measured = stats_by_value.get(label.value)
                    voxel_count = measured.voxel_count if measured is not None else 0
                    component_count = measured.component_count if measured is not None else 0
                    touches_border = measured.touches_volume_border if measured is not None else False
                    statistics = LabelStatistics(
                        label_name=label.name,
                        label_value=label.value,
                        voxel_count=voxel_count,
                        volume_ml=voxel_count * voxel_volume_ml,
                        component_count=component_count,
                        touches_volume_border=touches_border,
                    )
                    label_issue_codes: list[str] = []
                    label_has_error = False
                    if not observed:
                        if label.required:
                            label_has_error = True
                            label_issue_codes.append("REQUIRED_LABEL_MISSING")
                            issues.append(
                                ValidationIssue(
                                    "REQUIRED_LABEL_MISSING",
                                    ValidationSeverity.ERROR,
                                    f"required label {label.name} is missing",
                                    artifact_id=request.segmentation_artifact.artifact_id,
                                    field_path=f"expected_labels.{label.name}",
                                )
                            )
                        else:
                            label_issue_codes.append("OPTIONAL_LABEL_MISSING")
                            issues.append(
                                ValidationIssue(
                                    "OPTIONAL_LABEL_MISSING",
                                    ValidationSeverity.WARNING,
                                    f"optional label {label.name} is missing",
                                    artifact_id=request.segmentation_artifact.artifact_id,
                                    field_path=f"expected_labels.{label.name}",
                                )
                            )
                    threshold = thresholds.get(label.name)
                    if observed and threshold is not None:
                        if voxel_count < threshold.min_voxel_count:
                            label_has_error = True
                            label_issue_codes.append("VOXEL_COUNT_TOO_LOW")
                            issues.append(
                                ValidationIssue(
                                    "VOXEL_COUNT_TOO_LOW",
                                    ValidationSeverity.ERROR,
                                    f"label {label.name} is below its minimum voxel count",
                                    artifact_id=request.segmentation_artifact.artifact_id,
                                    field_path=f"quality_thresholds.{label.name}.min_voxel_count",
                                )
                            )
                        if (
                            statistics.volume_ml < threshold.min_volume_ml
                            or (
                                threshold.max_volume_ml is not None
                                and statistics.volume_ml > threshold.max_volume_ml
                            )
                        ):
                            label_has_error = True
                            label_issue_codes.append("VOLUME_OUT_OF_RANGE")
                            issues.append(
                                ValidationIssue(
                                    "VOLUME_OUT_OF_RANGE",
                                    ValidationSeverity.ERROR,
                                    f"label {label.name} volume is outside the approved range",
                                    artifact_id=request.segmentation_artifact.artifact_id,
                                    field_path=f"quality_thresholds.{label.name}",
                                )
                            )
                        if component_count > threshold.max_component_count:
                            label_has_error = True
                            label_issue_codes.append("TOO_MANY_COMPONENTS")
                            issues.append(
                                ValidationIssue(
                                    "TOO_MANY_COMPONENTS",
                                    ValidationSeverity.ERROR,
                                    f"label {label.name} has too many connected components",
                                    artifact_id=request.segmentation_artifact.artifact_id,
                                    field_path=f"quality_thresholds.{label.name}.max_component_count",
                                )
                            )
                    if (
                        observed
                        and label.value != 0
                        and voxel_count >= analysis.total_voxel_count
                    ):
                        label_has_error = True
                        label_issue_codes.append("MASK_IMPLAUSIBLY_FULL")
                        issues.append(
                            ValidationIssue(
                                "MASK_IMPLAUSIBLY_FULL",
                                ValidationSeverity.ERROR,
                                f"label {label.name} occupies the entire image volume",
                                artifact_id=request.segmentation_artifact.artifact_id,
                                field_path=f"expected_labels.{label.name}",
                            )
                        )
                    if observed and label.name.lower() in {"skin", "body"} and not touches_border:
                        label_issue_codes.append("EXPECTED_BORDER_CONTACT_MISSING")
                        issues.append(
                            ValidationIssue(
                                "EXPECTED_BORDER_CONTACT_MISSING",
                                ValidationSeverity.WARNING,
                                f"label {label.name} does not touch the image border",
                                artifact_id=request.segmentation_artifact.artifact_id,
                                field_path=f"expected_labels.{label.name}",
                            )
                        )
                    label_results.append(
                        LabelValidationResult(
                            statistics=statistics,
                            passed=not label_has_error,
                            issue_codes=tuple(label_issue_codes),
                        )
                    )
                valid = not any(item.severity is ValidationSeverity.ERROR for item in issues)
                result = SegmentationValidationResult(
                    valid=valid,
                    geometry_matches_ct=geometry_matches,
                    label_results=tuple(label_results),
                    issues=tuple(issues),
                    recommended_action=(
                        RecommendedAction.CONTINUE if valid else RecommendedAction.MANUAL_REVIEW
                    ),
                )
                metrics = (MetricValue("validation_time", analysis.elapsed_ms, "ms"),)
                warnings = _simulation_warnings(analysis.runtime_name)
                response = self._success(
                    request,
                    tool_name,
                    result,
                    started_at,
                    metrics=metrics,
                    warnings=warnings,
                )
                self._remember(request, tool_name, fingerprint, response)
                self._trace(
                    request,
                    tool_name,
                    "success",
                    {
                        "artifact_id": request.segmentation_artifact.artifact_id,
                        "valid": result.valid,
                        "geometry_matches_ct": geometry_matches,
                        "issue_count": len(issues),
                        "validation_time_ms": analysis.elapsed_ms,
                    },
                )
                return response
            except SegmentationBackendError as exc:
                return self._failure(request, tool_name, started_at, exc)
            except Exception:
                return self._failure(
                    request,
                    tool_name,
                    started_at,
                    SegmentationBackendError(
                        ErrorCode.INTERNAL_ERROR,
                        "unexpected segmentation validation failure",
                    ),
                )

    def extract_skin_surface(
        self,
        request: ExtractSkinSurfaceRequest,
    ) -> ToolResponseEnvelope[SkinSurfaceExtractionResult]:
        tool_name = "extract_skin_surface"
        with self._lock:
            started_at = self._utc_now()
            try:
                self._check_deadline(request.context.deadline_epoch_ms)
                source_manifest = self._resolve(
                    request.skin_mask_artifact,
                    caller=request.context.caller,
                    purpose=tool_name,
                    allowed_types=(ArtifactType.SEGMENTATION_MASK, ArtifactType.NIFTI_LABELMAP),
                    field_path="skin_mask_artifact",
                )
                fingerprint = _request_fingerprint(request)
                replay = self._replay(request, tool_name, fingerprint, started_at)
                if replay is not None:
                    return replay
                if source_manifest.dtype not in _INTEGER_DTYPES:
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "skin mask must use a discrete integer label dtype",
                        field_path="skin_mask_artifact",
                    )
                source_stats = {
                    item.label_value: item for item in source_manifest.label_statistics
                }
                if len(source_stats) != len(source_manifest.label_statistics):
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "skin-mask manifest contains duplicate label statistics",
                        field_path="skin_mask_artifact",
                    )
                skin_value = source_manifest.primary_label_value
                if skin_value is None:
                    raise SegmentationBackendError(
                        ErrorCode.LABEL_SCHEMA_ERROR,
                        "skin-mask manifest does not identify the configured skin label",
                        field_path="skin_mask_artifact",
                    )
                source_skin = source_stats.get(skin_value)
                if (
                    skin_value not in source_manifest.observed_label_values
                    or source_skin is None
                    or source_skin.voxel_count <= 0
                ):
                    raise SegmentationBackendError(
                        ErrorCode.EMPTY_SEGMENTATION,
                        "configured skin label is empty",
                        field_path="skin_mask_artifact",
                        details={"skin_label_value": str(skin_value)},
                    )
                kernel = self.algorithms.extract_external_skin_surface(
                    request.skin_mask_artifact,
                    request.thickness_mm,
                    request.method,
                    request.connectivity,
                    request.keep_largest_component,
                )
                self._check_deadline(request.context.deadline_epoch_ms)
                source_geometry = self._require_geometry(
                    source_manifest.geometry,
                    "skin_mask_artifact.geometry",
                )
                self._validate_skin_kernel(
                    kernel,
                    source_geometry,
                    request.thickness_mm,
                    source_skin.voxel_count,
                )
                artifact = _build_output_artifact(
                    case_id=request.context.case_id,
                    suffix="skin-surface",
                    artifact_type=ArtifactType.SKIN_SURFACE_MASK,
                    parent_ids=(request.skin_mask_artifact.artifact_id,),
                    producer_name=tool_name,
                    geometry=kernel.geometry,
                    content={
                        "request": fingerprint,
                        "source_artifact_id": request.skin_mask_artifact.artifact_id,
                        "surface_voxel_count": kernel.surface_voxel_count,
                        "effective_thickness_mm": kernel.effective_thickness_mm,
                    },
                    metadata={
                        "method": request.method.value,
                        "requested_thickness_mm": str(request.thickness_mm),
                        "effective_thickness_mm": ",".join(
                            f"{value:.9g}" for value in kernel.effective_thickness_mm
                        ),
                        "connectivity": str(request.connectivity.value),
                        "keep_largest_component": str(request.keep_largest_component).lower(),
                        "external_surface_policy": "border-connected-background-only",
                        "label_values": "0,1",
                        "runtime_name": kernel.runtime_name,
                    },
                )
                output_stats = LabelStatistics(
                    label_name="skin_surface",
                    label_value=1,
                    voxel_count=kernel.surface_voxel_count,
                    volume_ml=kernel.surface_voxel_count * _voxel_volume_ml(kernel.geometry),
                    component_count=1 if request.keep_largest_component else max(
                        1, kernel.components_removed + 1
                    ),
                    touches_volume_border=True,
                )
                output_manifest = VolumeManifest(
                    artifact_id=artifact.artifact_id,
                    geometry=kernel.geometry,
                    dtype="uint8",
                    observed_label_values=(0, 1),
                    label_statistics=(output_stats,),
                    expected_checksum_sha256=artifact.checksum_sha256,
                    allowed_callers=frozenset({request.context.caller}),
                    primary_label_value=1,
                    total_voxel_count=_geometry_voxel_count(kernel.geometry),
                )
                committed = self.artifacts.commit(
                    artifact,
                    output_manifest,
                    caller=request.context.caller,
                    commit_key=self._commit_key(request, tool_name),
                    request_fingerprint=fingerprint,
                )
                result = SkinSurfaceExtractionResult(
                    surface_artifact=committed,
                    source_voxel_count=kernel.source_voxel_count,
                    surface_voxel_count=kernel.surface_voxel_count,
                    requested_thickness_mm=request.thickness_mm,
                    effective_thickness_mm=kernel.effective_thickness_mm,
                    components_removed=kernel.components_removed,
                    geometry_matches_source=True,
                )
                metrics = (MetricValue("morphology_time", kernel.elapsed_ms, "ms"),)
                warnings = _simulation_warnings(kernel.runtime_name)
                response = self._success(
                    request,
                    tool_name,
                    result,
                    started_at,
                    artifacts=(committed,),
                    metrics=metrics,
                    warnings=warnings,
                )
                self._remember(request, tool_name, fingerprint, response)
                self._trace(
                    request,
                    tool_name,
                    "success",
                    {
                        "artifact_id": committed.artifact_id,
                        "source_artifact_id": request.skin_mask_artifact.artifact_id,
                        "method": request.method.value,
                        "requested_thickness_mm": request.thickness_mm,
                        "surface_voxel_count": kernel.surface_voxel_count,
                        "components_removed": kernel.components_removed,
                        "morphology_time_ms": kernel.elapsed_ms,
                    },
                )
                return response
            except SegmentationBackendError as exc:
                return self._failure(request, tool_name, started_at, exc)
            except Exception:
                return self._failure(
                    request,
                    tool_name,
                    started_at,
                    SegmentationBackendError(
                        ErrorCode.INTERNAL_ERROR,
                        "unexpected skin-surface adapter failure",
                    ),
                )

    def _resolve(
        self,
        artifact: ArtifactRef,
        *,
        caller: str,
        purpose: str,
        allowed_types: tuple[ArtifactType, ...],
        field_path: str,
    ) -> VolumeManifest:
        if artifact.artifact_type not in allowed_types:
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "artifact type is not supported by this tool",
                field_path=f"{field_path}.artifact_type",
                details={"artifact_id": artifact.artifact_id, "artifact_type": artifact.artifact_type.value},
            )
        manifest = self.artifacts.resolve(artifact, caller=caller, purpose=purpose)
        if manifest.artifact_id != artifact.artifact_id:
            raise SegmentationBackendError(
                ErrorCode.CONTRACT_VIOLATION,
                "artifact resolver returned a different artifact identity",
                dependency="artifact_registry",
            )
        geometry = self._require_geometry(manifest.geometry, f"{field_path}.geometry")
        if artifact.geometry is None:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "artifact reference is missing image geometry",
                field_path=f"{field_path}.geometry",
            )
        if geometry.geometry_fingerprint != artifact.geometry.geometry_fingerprint:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "artifact reference and resolved geometry differ",
                field_path=f"{field_path}.geometry",
                details={"artifact_id": artifact.artifact_id},
            )
        if (
            manifest.total_voxel_count is not None
            and manifest.total_voxel_count != _geometry_voxel_count(geometry)
        ):
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "artifact manifest voxel count differs from image geometry",
                dependency="artifact_registry",
                details={"artifact_id": artifact.artifact_id},
            )
        return manifest

    @staticmethod
    def _require_geometry(
        geometry: VolumeGeometry | None,
        field_path: str,
    ) -> VolumeGeometry:
        if geometry is None:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "image geometry is required",
                field_path=field_path,
            )
        return geometry

    @staticmethod
    def _validate_profile_for_request(
        profile: ModelProfile,
        request: RunSegmentationRequest,
        ct_manifest: VolumeManifest,
    ) -> None:
        if (
            profile.model_id != request.model_id
            or profile.model_version != request.model_version
            or profile.precision is not request.precision
        ):
            raise SegmentationBackendError(
                ErrorCode.MODEL_VERSION_MISMATCH,
                "model registry returned a conflicting engine profile",
                dependency="model_registry",
            )
        available_names = {item.name for item in profile.labels}
        unknown = [name for name in request.requested_labels if name not in available_names]
        if unknown:
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "requested label is not present in the registered model schema",
                field_path="requested_labels",
                details={"label_name": unknown[0]},
            )
        if request.output_probability_maps and not profile.supports_probability_maps:
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "the selected profile does not expose probability maps",
                field_path="output_probability_maps",
            )
        if request.device_id not in profile.supported_device_ids:
            raise SegmentationBackendError(
                ErrorCode.INVALID_ARGUMENT,
                "requested device is not approved for this engine profile",
                field_path="device_id",
            )
        if ct_manifest.dtype not in profile.input_dtypes:
            raise SegmentationBackendError(
                ErrorCode.UNSUPPORTED_FORMAT,
                "CT dtype is not supported by the registered preprocessing profile",
                field_path="ct_artifact",
                details={"dtype": ct_manifest.dtype},
            )

    @staticmethod
    def _validate_engine_result(
        output: EngineResult,
        profile: ModelProfile,
        ct_geometry: VolumeGeometry | None,
        requested_labels: tuple[str, ...],
    ) -> tuple[LabelStatistics, ...]:
        if ct_geometry is None or output.geometry.geometry_fingerprint != ct_geometry.geometry_fingerprint:
            raise SegmentationBackendError(
                ErrorCode.GEOMETRY_MISMATCH,
                "inference output was not restored to the original CT geometry",
                dependency="segmentation_engine",
            )
        if output.engine_hash != profile.engine_hash:
            raise SegmentationBackendError(
                ErrorCode.MODEL_VERSION_MISMATCH,
                "runtime engine hash differs from the approved model profile",
                dependency="segmentation_engine",
            )
        if output.output_dtype.lower() not in _INTEGER_DTYPES:
            raise SegmentationBackendError(
                ErrorCode.MODEL_INFERENCE_FAILED,
                "inference output labelmap is not an integer dtype",
                dependency="segmentation_engine",
            )
        if (
            output.inference_time_ms < 0
            or output.peak_gpu_memory_mb < 0
            or not isfinite(output.inference_time_ms)
            or not isfinite(output.peak_gpu_memory_mb)
        ):
            raise SegmentationBackendError(
                ErrorCode.MODEL_INFERENCE_FAILED,
                "runtime returned invalid inference metrics",
                dependency="segmentation_engine",
            )
        if (
            len(output.observed_label_values) != len(set(output.observed_label_values))
            or any(value < 0 for value in output.observed_label_values)
        ):
            raise SegmentationBackendError(
                ErrorCode.LABEL_SCHEMA_ERROR,
                "inference output contains invalid observed label values",
                dependency="segmentation_engine",
            )
        profile_by_value = {item.value: item for item in profile.labels}
        profile_by_name = {item.name: item for item in profile.labels}
        observed_values = set(output.observed_label_values)
        unknown_values = sorted(observed_values - {0, *profile_by_value})
        if unknown_values:
            raise SegmentationBackendError(
                ErrorCode.UNKNOWN_LABEL_VALUE,
                "inference output contains a label value absent from the model profile",
                dependency="segmentation_engine",
                details={"label_value": str(unknown_values[0])},
            )
        stats_by_name = {item.label_name: item for item in output.label_statistics}
        if len(stats_by_name) != len(output.label_statistics):
            raise SegmentationBackendError(
                ErrorCode.LABEL_SCHEMA_ERROR,
                "inference output contains duplicate label statistics",
                dependency="segmentation_engine",
            )
        stat_values = [item.label_value for item in output.label_statistics]
        if len(stat_values) != len(set(stat_values)):
            raise SegmentationBackendError(
                ErrorCode.LABEL_SCHEMA_ERROR,
                "inference output contains duplicate label values",
                dependency="segmentation_engine",
            )
        total_voxels = _geometry_voxel_count(output.geometry)
        for stat in output.label_statistics:
            registered = profile_by_name.get(stat.label_name)
            if registered is None or registered.value != stat.label_value:
                raise SegmentationBackendError(
                    ErrorCode.LABEL_SCHEMA_ERROR,
                    "inference statistics conflict with the registered label map",
                    dependency="segmentation_engine",
                    details={"label_name": stat.label_name},
                )
            if stat.label_value not in observed_values:
                raise SegmentationBackendError(
                    ErrorCode.LABEL_SCHEMA_ERROR,
                    "inference statistics refer to an unobserved label value",
                    dependency="segmentation_engine",
                    details={"label_name": stat.label_name},
                )
            if (
                stat.voxel_count < 0
                or stat.voxel_count > total_voxels
                or stat.component_count < 0
                or stat.volume_ml < 0
                or not isfinite(stat.volume_ml)
            ):
                raise SegmentationBackendError(
                    ErrorCode.MODEL_INFERENCE_FAILED,
                    "inference returned invalid label statistics",
                    dependency="segmentation_engine",
                    details={"label_name": stat.label_name},
                )
            expected_volume = stat.voxel_count * _voxel_volume_ml(output.geometry)
            if not isclose(stat.volume_ml, expected_volume, rel_tol=0.0, abs_tol=1e-6):
                raise SegmentationBackendError(
                    ErrorCode.MODEL_INFERENCE_FAILED,
                    "inference label volume does not match voxel count and spacing",
                    dependency="segmentation_engine",
                    details={"label_name": stat.label_name},
                )
        if sum(item.voxel_count for item in output.label_statistics) > total_voxels:
            raise SegmentationBackendError(
                ErrorCode.MODEL_INFERENCE_FAILED,
                "inference label counts exceed the output image volume",
                dependency="segmentation_engine",
            )
        required_names = (
            {name for name in requested_labels if profile_by_name[name].required}
            if profile.supports_label_selection
            else {item.name for item in profile.labels if item.required}
        )
        for name in required_names:
            stat = stats_by_name.get(name)
            if (
                stat is None
                or stat.label_value not in observed_values
                or stat.voxel_count <= 0
            ):
                raise SegmentationBackendError(
                    ErrorCode.EMPTY_SEGMENTATION,
                    "inference output is missing a required nonempty label",
                    dependency="segmentation_engine",
                    details={"label_name": name},
                )
        selected_names = set(requested_labels)
        ordered = tuple(
            stats_by_name[item.name]
            for item in profile.labels
            if item.name in stats_by_name
            and (not profile.supports_label_selection or item.name in selected_names)
        )
        return ordered

    @staticmethod
    def _validate_skin_kernel(
        kernel: SkinSurfaceKernelResult,
        source_geometry: VolumeGeometry,
        requested_thickness_mm: float,
        expected_source_voxel_count: int,
    ) -> None:
        if kernel.geometry.geometry_fingerprint != source_geometry.geometry_fingerprint:
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface output geometry differs from its source mask",
                dependency="image_algorithms",
            )
        if set(kernel.binary_values) - {0, 1} or 1 not in kernel.binary_values:
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface output is not binary",
                dependency="image_algorithms",
            )
        if not kernel.external_surface_only:
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface output includes non-external cavity boundaries",
                dependency="image_algorithms",
            )
        if kernel.source_voxel_count <= 0:
            raise SegmentationBackendError(
                ErrorCode.EMPTY_SEGMENTATION,
                "skin source mask is empty",
                dependency="image_algorithms",
            )
        if kernel.source_voxel_count != expected_source_voxel_count:
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface kernel source count differs from the artifact manifest",
                dependency="image_algorithms",
            )
        if (
            kernel.surface_voxel_count <= 0
            or kernel.surface_voxel_count > kernel.source_voxel_count
            or not kernel.subset_of_source
        ):
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface output failed nonempty/subset validation",
                dependency="image_algorithms",
            )
        if len(kernel.effective_thickness_mm) != 3 or any(
            value < requested_thickness_mm for value in kernel.effective_thickness_mm
        ):
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface kernel returned an invalid physical thickness",
                dependency="image_algorithms",
            )
        if any(
            value >= requested_thickness_mm + spacing + 1e-9
            for value, spacing in zip(
                kernel.effective_thickness_mm,
                source_geometry.spacing_mm,
            )
        ):
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface physical thickness exceeds the one-voxel approximation bound",
                dependency="image_algorithms",
            )
        if kernel.components_removed < 0 or kernel.elapsed_ms < 0:
            raise SegmentationBackendError(
                ErrorCode.QUALITY_CHECK_FAILED,
                "skin-surface kernel returned invalid metrics",
                dependency="image_algorithms",
            )

    def _check_deadline(self, deadline_epoch_ms: int | None) -> None:
        if deadline_epoch_ms is not None:
            now_ms = int(self._clock().timestamp() * 1000)
            if now_ms >= deadline_epoch_ms:
                raise SegmentationBackendError(
                    ErrorCode.TIMEOUT,
                    "tool deadline expired before execution",
                    retryable=True,
                    field_path="context.deadline_epoch_ms",
                )

    def _replay(
        self,
        request: Any,
        tool_name: str,
        fingerprint: str,
        started_at: str,
    ) -> ToolResponseEnvelope[Any] | None:
        key = self._cache_key(request, tool_name)
        cached = self._cache.get(key)
        if cached is None:
            return None
        if cached.request_fingerprint != fingerprint:
            raise SegmentationBackendError(
                ErrorCode.CONTRACT_VIOLATION,
                "idempotency key was reused with a different request",
                field_path="context.idempotency_key",
            )
        response = self._success(
            request,
            tool_name,
            cached.result,
            started_at,
            artifacts=cached.artifacts,
            metrics=cached.metrics,
            warnings=cached.warnings,
        )
        self._trace(
            request,
            tool_name,
            "idempotency_replay",
            {"idempotency_replay": True},
        )
        return response

    def _remember(
        self,
        request: Any,
        tool_name: str,
        fingerprint: str,
        response: ToolResponseEnvelope[Any],
    ) -> None:
        self._cache[self._cache_key(request, tool_name)] = _CachedSuccess(
            request_fingerprint=fingerprint,
            result=response.result,
            artifacts=response.artifacts,
            metrics=response.metrics,
            warnings=response.warnings,
        )

    @staticmethod
    def _cache_key(request: Any, tool_name: str) -> tuple[str, str, str, str]:
        return (
            tool_name,
            request.context.case_id,
            request.context.caller,
            request.context.idempotency_key,
        )

    @staticmethod
    def _commit_key(request: Any, tool_name: str) -> str:
        return "|".join(SegmentationToolAdapter._cache_key(request, tool_name))

    def _success(
        self,
        request: Any,
        tool_name: str,
        result: _ResultT,
        started_at: str,
        *,
        artifacts: tuple[ArtifactRef, ...] = (),
        metrics: tuple[MetricValue, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> ToolResponseEnvelope[_ResultT]:
        return ToolResponseEnvelope(
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            tool_version=TOOL_VERSION,
            status=ToolExecutionStatus.SUCCESS,
            result=result,
            artifacts=artifacts,
            metrics=metrics,
            warnings=warnings,
            error=None,
            started_at=started_at,
            finished_at=self._utc_now(),
        )

    def _failure(
        self,
        request: Any,
        tool_name: str,
        started_at: str,
        error: SegmentationBackendError,
    ) -> ToolResponseEnvelope[Any]:
        sensitive_values = _request_sensitive_values(request)
        message = _redact(error.public_message, sensitive_values)
        details = {
            key: _redact(str(value), sensitive_values)
            for key, value in error.details.items()
            if not any(fragment in key.lower() for fragment in _SENSITIVE_DETAIL_FRAGMENTS)
        }
        response = ToolResponseEnvelope(
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            tool_version=TOOL_VERSION,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(),
            warnings=(),
            error=ErrorDetail(
                code=error.code,
                message=message,
                retryable=error.retryable,
                field_path=error.field_path,
                dependency=error.dependency,
                details=details,
            ),
            started_at=started_at,
            finished_at=self._utc_now(),
        )
        attributes: dict[str, str | int | float | bool] = {
            "error_code": error.code.value,
            "retryable": error.retryable,
        }
        if error.dependency:
            attributes["dependency"] = error.dependency
        self._trace(request, tool_name, "failure", attributes)
        return response

    def _trace(
        self,
        request: Any,
        tool_name: str,
        outcome: str,
        attributes: Mapping[str, str | int | float | bool],
    ) -> None:
        safe_attributes = {
            key: value
            for key, value in attributes.items()
            if not any(fragment in key.lower() for fragment in _SENSITIVE_DETAIL_FRAGMENTS)
        }
        event = SegmentationTraceEvent(
            timestamp=self._utc_now(),
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            outcome=outcome,
            attributes=safe_attributes,
        )
        try:
            self.trace_sink.record(event)
        except Exception:
            # Telemetry must not alter a clinical algorithm result.
            return

    def _utc_now(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _geometry_voxel_count(geometry: VolumeGeometry) -> int:
    return geometry.size_ijk[0] * geometry.size_ijk[1] * geometry.size_ijk[2]


def _voxel_volume_ml(geometry: VolumeGeometry | None) -> float:
    if geometry is None:
        return 0.0
    return geometry.spacing_mm[0] * geometry.spacing_mm[1] * geometry.spacing_mm[2] / 1000.0


def _geometry_matches(left: VolumeGeometry | None, right: VolumeGeometry | None) -> bool:
    return bool(left is not None and right is not None and left.is_compatible_with(right))


def _request_fingerprint(request: Any) -> str:
    payload = to_primitive(request)
    if isinstance(payload, dict):
        payload.pop("context", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _build_output_artifact(
    *,
    case_id: str,
    suffix: str,
    artifact_type: ArtifactType,
    parent_ids: tuple[str, ...],
    producer_name: str,
    geometry: VolumeGeometry,
    content: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> ArtifactRef:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    checksum = sha256(encoded.encode("utf-8")).hexdigest()
    artifact_id = f"{case_id}-{suffix}-{checksum[:12]}"
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id=case_id,
        artifact_type=artifact_type,
        uri=f"local-manifest://{case_id}/{artifact_id}",
        checksum_sha256=checksum,
        status=ArtifactStatus.AVAILABLE,
        geometry=geometry,
        producer_name=producer_name,
        producer_version=TOOL_VERSION,
        parent_artifact_ids=parent_ids,
        metadata=dict(metadata),
    )


def _simulation_warnings(runtime_name: str) -> tuple[str, ...]:
    if runtime_name == "deterministic-manifest":
        return ("LOCAL_DETERMINISTIC_SIMULATION_NOT_A_HARDWARE_BENCHMARK",)
    return ()


def _csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _csv_ints(raw: str, *, field_name: str) -> tuple[int, ...]:
    try:
        return tuple(sorted(set(int(item) for item in _csv_strings(raw))))
    except ValueError as exc:
        raise SegmentationBackendError(
            ErrorCode.LABEL_SCHEMA_ERROR,
            f"{field_name} metadata is not a comma-separated integer list",
            field_path=f"artifact.metadata.{field_name}",
        ) from exc


def _int_mapping(raw: str, field_name: str) -> dict[int, int]:
    if not raw.strip():
        return {}
    result: dict[int, int] = {}
    try:
        for item in raw.split(","):
            key, value = item.split(":", 1)
            result[int(key.strip())] = int(value.strip())
    except (ValueError, TypeError) as exc:
        raise SegmentationBackendError(
            ErrorCode.LABEL_SCHEMA_ERROR,
            f"{field_name} metadata must use label:value pairs",
            field_path=f"artifact.metadata.{field_name}",
        ) from exc
    return result


def _request_sensitive_values(request: Any) -> tuple[str, ...]:
    values = []
    primitive = to_primitive(request)

    def visit(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                visit(child, str(child_key))
        elif isinstance(node, list):
            for child in node:
                visit(child, key)
        elif isinstance(node, str) and any(
            fragment in key.lower() for fragment in ("uri", "checksum")
        ):
            values.append(node)

    visit(primitive)
    return tuple(item for item in values if item)


def _redact(message: str, sensitive_values: tuple[str, ...]) -> str:
    safe = message
    for value in sensitive_values:
        safe = safe.replace(value, "[redacted]")
    return safe


def build_local_segmentation_adapter(
    *,
    trace_sink: SegmentationTraceSink | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SegmentationToolAdapter:
    """Build the deterministic local adapter without model/GPU dependencies."""

    return SegmentationToolAdapter(trace_sink=trace_sink, clock=clock)


def build_segmentation_handlers(
    backend: SegmentationBackendPort | None = None,
    *,
    engine: SegmentationEnginePort | None = None,
    algorithms: ImageAlgorithmPort | None = None,
    artifacts: SegmentationArtifactPort | None = None,
    trace_sink: SegmentationTraceSink | None = None,
    clock: Callable[[], datetime] | None = None,
) -> tuple[
    SegmentationToolAdapter,
    Mapping[str, Callable[[Any], ToolResponseEnvelope[Any]]],
]:
    """Build adapter plus registry-ready handlers for local or production ports."""

    if backend is not None:
        if engine is not None or algorithms is not None or artifacts is not None:
            raise ValueError("backend cannot be combined with individual port overrides")
    adapter = SegmentationToolAdapter(
        backend,
        engine=engine,
        algorithms=algorithms,
        artifacts=artifacts,
        trace_sink=trace_sink,
        clock=clock,
    )
    return adapter, adapter.handlers()


_DEFAULT_ADAPTER = build_local_segmentation_adapter()


def run_segmentation(
    request: RunSegmentationRequest,
) -> ToolResponseEnvelope[SegmentationInferenceResult]:
    """Module-level handler suitable for later production-registry binding."""

    return _DEFAULT_ADAPTER.run_segmentation(request)


def validate_segmentation_result(
    request: ValidateSegmentationResultRequest,
) -> ToolResponseEnvelope[SegmentationValidationResult]:
    """Module-level handler suitable for later production-registry binding."""

    return _DEFAULT_ADAPTER.validate_segmentation_result(request)


def extract_skin_surface(
    request: ExtractSkinSurfaceRequest,
) -> ToolResponseEnvelope[SkinSurfaceExtractionResult]:
    """Module-level handler suitable for later production-registry binding."""

    return _DEFAULT_ADAPTER.extract_skin_surface(request)


__all__ = [
    "DeterministicSegmentationBackend",
    "EngineResult",
    "ImageAlgorithmPort",
    "InMemorySegmentationTraceSink",
    "LabelAnalysis",
    "ModelLabel",
    "ModelProfile",
    "SegmentationArtifactPort",
    "SegmentationBackendPort",
    "SegmentationBackendError",
    "SegmentationEnginePort",
    "SegmentationToolAdapter",
    "SegmentationTraceEvent",
    "SegmentationTraceSink",
    "SkinSurfaceKernelResult",
    "VolumeManifest",
    "build_local_segmentation_adapter",
    "build_segmentation_handlers",
    "extract_skin_surface",
    "run_segmentation",
    "validate_segmentation_result",
]
