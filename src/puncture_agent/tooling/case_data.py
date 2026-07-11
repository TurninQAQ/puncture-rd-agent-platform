"""Dependency-free local adapters for the three case-data tool contracts.

The real Mimics reader, object store, and artifact registry remain behind
``CaseDataBackendPort``.  ``ManifestCaseDataBackend`` is a deterministic local
implementation used by the Python 3.10 demo/tests: it reads bytes or regular
files plus explicit header/label manifests and never attempts to parse MCS or
NIfTI binary formats.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
import time
from typing import Any, Protocol

from contracts.artifacts import ArtifactRef
from contracts.common import MetricValue, ToolCallContext, ToolResponseEnvelope
from contracts.domain import ValidationIssue
from contracts.enums import (
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
    ErrorCode,
    ToolExecutionStatus,
    ValidationSeverity,
)
from contracts.errors import ErrorDetail
from contracts.geometry import VolumeGeometry
from contracts.tool_inputs import (
    ConvertMcsToNiftiRequest,
    InspectCaseMetadataRequest,
    LabelMappingEntry,
    ValidateLabelSchemaRequest,
)
from contracts.tool_outputs import (
    AppliedLabelMapping,
    ArtifactInspection,
    CaseMetadataResult,
    LabelSchemaValidationResult,
    McsToNiftiResult,
)


TOOL_VERSION = "1.0.0"
_DEFAULT_TIMEOUT_MS = {
    "inspect_case_metadata": 10_000,
    "convert_mcs_to_nifti": 60_000,
    "validate_label_schema": 15_000,
}
_DTYPE_MAX = {"uint8": 255, "uint16": 65_535, "int16": 32_767}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _geometry_dict(geometry: VolumeGeometry) -> dict[str, Any]:
    return {
        "size_ijk": list(geometry.size_ijk),
        "spacing_mm": list(geometry.spacing_mm),
        "origin_mm": list(geometry.origin_mm),
        "direction_cosines": list(geometry.direction_cosines),
        "coordinate_system": geometry.coordinate_system.value,
        "geometry_fingerprint": geometry.geometry_fingerprint,
    }


def _convert_coordinate_system(
    geometry: VolumeGeometry,
    target: CoordinateSystem,
) -> VolumeGeometry:
    if geometry.coordinate_system is target:
        return geometry
    # LPS <-> RAS is the same involution: negate the first two world axes.
    direction = geometry.direction_cosines
    return VolumeGeometry(
        size_ijk=geometry.size_ijk,
        spacing_mm=geometry.spacing_mm,
        origin_mm=(-geometry.origin_mm[0], -geometry.origin_mm[1], geometry.origin_mm[2]),
        direction_cosines=(
            -direction[0],
            -direction[1],
            -direction[2],
            -direction[3],
            -direction[4],
            -direction[5],
            direction[6],
            direction[7],
            direction[8],
        ),
        coordinate_system=target,
    )


@dataclass(frozen=True, slots=True)
class BackendInvocation:
    """Safe correlation/deadline data passed to every backend operation."""

    request_id: str
    trace_id: str
    case_id: str
    caller: str
    idempotency_key: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class BackendCallRecord:
    operation: str
    artifact_id: str | None
    request_id: str
    trace_id: str
    case_id: str
    caller: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class McsSegmentManifest:
    source_name: str
    source_value: int
    voxel_count: int
    overlap_voxels: int = 0

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("MCS segment name is required")
        if self.source_value < 0 or self.voxel_count < 0 or self.overlap_voxels < 0:
            raise ValueError("MCS segment values/counts must be non-negative")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Authoritative metadata plus a payload source for one immutable artifact."""

    artifact: ArtifactRef
    tenant_id: str
    allowed_callers: tuple[str, ...]
    format_name: str
    geometry: VolumeGeometry | None = None
    payload: bytes | None = None
    file_path: str | None = None
    label_value_chunks: tuple[tuple[int | float, ...], ...] = ()
    label_names: tuple[tuple[int, str], ...] = ()
    mcs_segments: tuple[McsSegmentManifest, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.format_name.strip():
            raise ValueError("tenant_id and format_name are required")
        if self.payload is not None and self.file_path is not None:
            raise ValueError("manifest must use payload or file_path, not both")
        object.__setattr__(self, "allowed_callers", tuple(self.allowed_callers))
        object.__setattr__(
            self,
            "label_value_chunks",
            tuple(tuple(chunk) for chunk in self.label_value_chunks),
        )
        object.__setattr__(self, "label_names", tuple(self.label_names))
        object.__setattr__(self, "mcs_segments", tuple(self.mcs_segments))


@dataclass(frozen=True, slots=True)
class ConversionProduct:
    """Backend-rendered pseudo-NIfTI product that the adapter re-verifies."""

    payload: bytes
    geometry: VolumeGeometry
    label_counts: tuple[tuple[int, int], ...]
    format_name: str = "nifti-manifest-v1"

    def __post_init__(self) -> None:
        if not self.payload or not self.format_name:
            raise ValueError("conversion product payload and format are required")
        if any(value < 0 or count < 0 for value, count in self.label_counts):
            raise ValueError("conversion label values/counts must be non-negative")


@dataclass(frozen=True, slots=True)
class ConversionCommit:
    fingerprint: str
    artifact: ArtifactRef
    applied_mappings: tuple[AppliedLabelMapping, ...]
    manifest: ArtifactManifest


class BackendFailure(RuntimeError):
    """Typed dependency failure; raw provider text never reaches the envelope."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        operation: str,
        retryable: bool = False,
        dependency: str = "case_data_backend",
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.operation = operation
        self.retryable = retryable
        self.dependency = dependency


class CaseDataBackendPort(Protocol):
    def resolve(self, artifact_id: str, invocation: BackendInvocation) -> ArtifactManifest: ...

    def iter_payload(
        self,
        manifest: ArtifactManifest,
        invocation: BackendInvocation,
        *,
        chunk_size: int,
    ) -> Iterable[bytes]: ...

    def render_conversion(
        self,
        mcs: ArtifactManifest,
        reference_ct: ArtifactManifest,
        mapping: tuple[LabelMappingEntry, ...],
        output_geometry: VolumeGeometry,
        output_dtype: str,
        invocation: BackendInvocation,
    ) -> ConversionProduct: ...

    def lookup_conversion(
        self,
        scope: str,
        invocation: BackendInvocation,
    ) -> ConversionCommit | None: ...

    def commit_conversion(
        self,
        scope: str,
        candidate: ConversionCommit,
        invocation: BackendInvocation,
    ) -> ConversionCommit: ...


class ManifestCaseDataBackend:
    """Thread-safe deterministic backend for manifests, bytes, and regular files."""

    def __init__(
        self,
        manifests: Iterable[ArtifactManifest] = (),
        *,
        caller_tenants: Mapping[str, str] | None = None,
        operation_hook: Callable[[BackendCallRecord], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._manifests: dict[str, ArtifactManifest] = {}
        self._conversions: dict[str, ConversionCommit] = {}
        self._caller_tenants = dict(caller_tenants or {})
        self._failures: dict[tuple[str, str | None], BackendFailure] = {}
        self._operation_hook = operation_hook
        self.calls: list[BackendCallRecord] = []
        for manifest in manifests:
            self.register_manifest(manifest)

    def register_manifest(self, manifest: ArtifactManifest) -> None:
        with self._lock:
            if manifest.artifact.artifact_id in self._manifests:
                raise ValueError(f"duplicate artifact manifest: {manifest.artifact.artifact_id}")
            self._manifests[manifest.artifact.artifact_id] = manifest

    def inject_failure(
        self,
        operation: str,
        code: ErrorCode,
        *,
        artifact_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        self._failures[(operation, artifact_id)] = BackendFailure(
            code,
            operation=operation,
            retryable=retryable,
        )

    @property
    def committed_conversion_count(self) -> int:
        return len(self._conversions)

    def _record(
        self,
        operation: str,
        invocation: BackendInvocation,
        artifact_id: str | None = None,
    ) -> None:
        record = BackendCallRecord(
            operation=operation,
            artifact_id=artifact_id,
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            case_id=invocation.case_id,
            caller=invocation.caller,
            timeout_seconds=invocation.timeout_seconds,
        )
        self.calls.append(record)
        if self._operation_hook is not None:
            self._operation_hook(record)
        failure = self._failures.get((operation, artifact_id)) or self._failures.get(
            (operation, None)
        )
        if failure is not None:
            raise failure

    def _authorize(self, manifest: ArtifactManifest, invocation: BackendInvocation) -> None:
        caller_tenant = self._caller_tenants.get(invocation.caller)
        allowed = not manifest.allowed_callers or invocation.caller in manifest.allowed_callers
        if (
            caller_tenant != manifest.tenant_id
            or manifest.artifact.case_id != invocation.case_id
            or not allowed
        ):
            raise BackendFailure(
                ErrorCode.PERMISSION_DENIED,
                operation="authorize",
                retryable=False,
            )

    def resolve(self, artifact_id: str, invocation: BackendInvocation) -> ArtifactManifest:
        self._record("resolve", invocation, artifact_id)
        with self._lock:
            manifest = self._manifests.get(artifact_id)
        if manifest is None:
            raise BackendFailure(
                ErrorCode.MISSING_ARTIFACT,
                operation="resolve",
                retryable=False,
            )
        self._authorize(manifest, invocation)
        return manifest

    def iter_payload(
        self,
        manifest: ArtifactManifest,
        invocation: BackendInvocation,
        *,
        chunk_size: int,
    ) -> Iterable[bytes]:
        self._record("read", invocation, manifest.artifact.artifact_id)
        self._authorize(manifest, invocation)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if manifest.payload is not None:
            payload = manifest.payload
            for offset in range(0, len(payload), chunk_size):
                yield payload[offset : offset + chunk_size]
            return
        if manifest.file_path is None:
            raise BackendFailure(
                ErrorCode.ARTIFACT_NOT_AVAILABLE,
                operation="read",
                retryable=False,
            )
        path = Path(manifest.file_path)
        if path.is_symlink() or not path.is_file():
            raise BackendFailure(
                ErrorCode.ARTIFACT_NOT_AVAILABLE,
                operation="read",
                retryable=False,
            )
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except OSError as exc:
            raise BackendFailure(
                ErrorCode.DEPENDENCY_FAILED,
                operation="read",
                retryable=True,
            ) from exc

    def render_conversion(
        self,
        mcs: ArtifactManifest,
        reference_ct: ArtifactManifest,
        mapping: tuple[LabelMappingEntry, ...],
        output_geometry: VolumeGeometry,
        output_dtype: str,
        invocation: BackendInvocation,
    ) -> ConversionProduct:
        self._record("convert", invocation, mcs.artifact.artifact_id)
        self._authorize(mcs, invocation)
        self._authorize(reference_ct, invocation)
        segments = {(segment.source_name.strip().casefold(), segment.source_value): segment for segment in mcs.mcs_segments}
        counts = tuple(
            (entry.target_value, segments[(entry.source_name.strip().casefold(), entry.source_value)].voxel_count)
            for entry in mapping
        )
        payload = _canonical_json(
            {
                "format": "nifti-manifest-v1",
                "geometry": _geometry_dict(output_geometry),
                "dtype": output_dtype,
                "labels": [
                    {"value": value, "voxel_count": count} for value, count in counts
                ],
                "source_mcs": mcs.artifact.artifact_id,
                "reference_ct": reference_ct.artifact.artifact_id,
            }
        )
        return ConversionProduct(payload, output_geometry, counts)

    def lookup_conversion(
        self,
        scope: str,
        invocation: BackendInvocation,
    ) -> ConversionCommit | None:
        self._record("idempotency_lookup", invocation)
        with self._lock:
            return self._conversions.get(scope)

    def commit_conversion(
        self,
        scope: str,
        candidate: ConversionCommit,
        invocation: BackendInvocation,
    ) -> ConversionCommit:
        self._record("commit", invocation, candidate.artifact.artifact_id)
        with self._lock:
            existing = self._conversions.get(scope)
            if existing is not None:
                if existing.fingerprint != candidate.fingerprint:
                    raise BackendFailure(
                        ErrorCode.INVALID_ARGUMENT,
                        operation="commit",
                        retryable=False,
                    )
                return existing
            self._conversions[scope] = candidate
            self._manifests[candidate.artifact.artifact_id] = candidate.manifest
            return candidate


class _AdapterFailure(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        *,
        retryable: bool = False,
        operation: str | None = None,
        dependency: str | None = None,
        field_path: str | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.retryable = retryable
        self.operation = operation
        self.dependency = dependency
        self.field_path = field_path


class _Deadline:
    def __init__(
        self,
        context: ToolCallContext,
        default_timeout_ms: int,
        *,
        started_monotonic: float,
        monotonic: Callable[[], float],
        wall_clock_ms: Callable[[], int],
    ) -> None:
        budget_ms = default_timeout_ms
        if context.deadline_epoch_ms is not None:
            remaining_ms = context.deadline_epoch_ms - wall_clock_ms()
            if remaining_ms <= 0:
                raise _AdapterFailure(ErrorCode.TIMEOUT, retryable=True, operation="deadline")
            budget_ms = min(budget_ms, remaining_ms)
        self._deadline = started_monotonic + budget_ms / 1000.0
        self._monotonic = monotonic

    def check(self, operation: str) -> None:
        if self._monotonic() >= self._deadline:
            raise _AdapterFailure(ErrorCode.TIMEOUT, retryable=True, operation=operation)

    def invocation(self, context: ToolCallContext, operation: str) -> BackendInvocation:
        self.check(operation)
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise _AdapterFailure(ErrorCode.TIMEOUT, retryable=True, operation=operation)
        return BackendInvocation(
            request_id=context.request_id,
            trace_id=context.trace_id,
            case_id=context.case_id,
            caller=context.caller,
            idempotency_key=context.idempotency_key,
            timeout_seconds=remaining,
        )


class CaseDataToolAdapter:
    """Typed handlers suitable for later binding into ``ToolRegistry``/MCP."""

    def __init__(
        self,
        backend: CaseDataBackendPort | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock_ms: Callable[[], int] | None = None,
        now_iso: Callable[[], str] = _utc_now,
        checksum_chunk_size: int = 64 * 1024,
    ) -> None:
        if checksum_chunk_size < 1:
            raise ValueError("checksum_chunk_size must be positive")
        self.backend = backend or ManifestCaseDataBackend()
        self._monotonic = monotonic
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1000))
        self._now_iso = now_iso
        self._checksum_chunk_size = checksum_chunk_size

    def handlers(self) -> dict[str, Callable[[Any], ToolResponseEnvelope[Any]]]:
        return {
            "inspect_case_metadata": self.inspect_case_metadata,
            "convert_mcs_to_nifti": self.convert_mcs_to_nifti,
            "validate_label_schema": self.validate_label_schema,
        }

    def inspect_case_metadata(
        self,
        request: InspectCaseMetadataRequest,
    ) -> ToolResponseEnvelope[CaseMetadataResult]:
        return self._execute(
            request,
            "inspect_case_metadata",
            self._inspect_case_metadata,
        )

    def convert_mcs_to_nifti(
        self,
        request: ConvertMcsToNiftiRequest,
    ) -> ToolResponseEnvelope[McsToNiftiResult]:
        return self._execute(
            request,
            "convert_mcs_to_nifti",
            self._convert_mcs_to_nifti,
        )

    def validate_label_schema(
        self,
        request: ValidateLabelSchemaRequest,
    ) -> ToolResponseEnvelope[LabelSchemaValidationResult]:
        return self._execute(
            request,
            "validate_label_schema",
            self._validate_label_schema,
        )

    def _execute(
        self,
        request: Any,
        tool_name: str,
        operation: Callable[[Any, _Deadline], tuple[Any, tuple[ArtifactRef, ...], tuple[str, ...]]],
    ) -> ToolResponseEnvelope[Any]:
        started_at = self._now_iso()
        started_monotonic = self._monotonic()
        try:
            deadline = _Deadline(
                request.context,
                _DEFAULT_TIMEOUT_MS[tool_name],
                started_monotonic=started_monotonic,
                monotonic=self._monotonic,
                wall_clock_ms=self._wall_clock_ms,
            )
            result, artifacts, warnings = operation(request, deadline)
            deadline.check("complete")
            return ToolResponseEnvelope(
                request_id=request.context.request_id,
                trace_id=request.context.trace_id,
                tool_name=tool_name,
                tool_version=TOOL_VERSION,
                status=ToolExecutionStatus.SUCCESS,
                result=result,
                artifacts=artifacts,
                metrics=(
                    MetricValue(
                        name=f"{tool_name}_elapsed",
                        value=max(0.0, (self._monotonic() - started_monotonic) * 1000.0),
                        unit="ms",
                    ),
                ),
                warnings=warnings,
                error=None,
                started_at=started_at,
                finished_at=self._now_iso(),
            )
        except BackendFailure as exc:
            return self._failed(request, tool_name, started_at, started_monotonic, exc)
        except _AdapterFailure as exc:
            return self._failed(request, tool_name, started_at, started_monotonic, exc)
        except Exception:
            failure = _AdapterFailure(ErrorCode.INTERNAL_ERROR, operation="adapter")
            return self._failed(request, tool_name, started_at, started_monotonic, failure)

    def _failed(
        self,
        request: Any,
        tool_name: str,
        started_at: str,
        started_monotonic: float,
        failure: BackendFailure | _AdapterFailure,
    ) -> ToolResponseEnvelope[Any]:
        safe_messages = {
            ErrorCode.MISSING_ARTIFACT: "referenced artifact was not found",
            ErrorCode.ARTIFACT_NOT_AVAILABLE: "artifact is not available for inspection",
            ErrorCode.PERMISSION_DENIED: "artifact access was denied",
            ErrorCode.CHECKSUM_MISMATCH: "artifact checksum verification failed",
            ErrorCode.UNSUPPORTED_FORMAT: "artifact format is not supported by the configured backend",
            ErrorCode.CONTRACT_VIOLATION: "artifact metadata violates the case-data contract",
            ErrorCode.GEOMETRY_MISMATCH: "artifact geometry is incompatible with the reference",
            ErrorCode.LABEL_SCHEMA_ERROR: "label data violates the configured schema",
            ErrorCode.UNKNOWN_LABEL_VALUE: "source data contains an unmapped label",
            ErrorCode.INVALID_ARGUMENT: "case-data conversion arguments are inconsistent",
            ErrorCode.TIMEOUT: "case-data operation exceeded its deadline",
            ErrorCode.DEPENDENCY_FAILED: "case-data dependency failed",
            ErrorCode.INTERNAL_ERROR: "unexpected case-data adapter failure",
        }
        operation = getattr(failure, "operation", None)
        details = {"operation": operation} if operation else {}
        error = ErrorDetail(
            code=failure.code,
            message=safe_messages.get(failure.code, "case-data operation failed"),
            retryable=failure.retryable,
            field_path=getattr(failure, "field_path", None),
            dependency=getattr(failure, "dependency", None),
            details=details,
        )
        return ToolResponseEnvelope(
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            tool_version=TOOL_VERSION,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(
                MetricValue(
                    name=f"{tool_name}_elapsed",
                    value=max(0.0, (self._monotonic() - started_monotonic) * 1000.0),
                    unit="ms",
                ),
            ),
            warnings=(),
            error=error,
            started_at=started_at,
            finished_at=self._now_iso(),
        )

    def _resolve(
        self,
        requested: ArtifactRef,
        context: ToolCallContext,
        deadline: _Deadline,
    ) -> ArtifactManifest:
        invocation = deadline.invocation(context, "resolve")
        manifest = self.backend.resolve(requested.artifact_id, invocation)
        deadline.check("resolve")
        authoritative = manifest.artifact
        if authoritative.case_id != context.case_id:
            raise _AdapterFailure(ErrorCode.PERMISSION_DENIED, operation="resolve")
        if authoritative.artifact_type is not requested.artifact_type:
            raise _AdapterFailure(ErrorCode.CONTRACT_VIOLATION, operation="resolve")
        if authoritative.checksum_sha256 != requested.checksum_sha256:
            raise _AdapterFailure(ErrorCode.CHECKSUM_MISMATCH, operation="resolve")
        if authoritative.status is not ArtifactStatus.AVAILABLE:
            raise _AdapterFailure(ErrorCode.ARTIFACT_NOT_AVAILABLE, operation="resolve")
        return manifest

    def _verify_checksum(
        self,
        manifest: ArtifactManifest,
        context: ToolCallContext,
        deadline: _Deadline,
    ) -> None:
        invocation = deadline.invocation(context, "read")
        digest = sha256()
        try:
            for chunk in self.backend.iter_payload(
                manifest,
                invocation,
                chunk_size=self._checksum_chunk_size,
            ):
                if not isinstance(chunk, bytes):
                    raise _AdapterFailure(ErrorCode.CONTRACT_VIOLATION, operation="read")
                digest.update(chunk)
                deadline.check("read")
        except BackendFailure:
            raise
        if digest.hexdigest() != manifest.artifact.checksum_sha256:
            raise _AdapterFailure(ErrorCode.CHECKSUM_MISMATCH, operation="read")

    @staticmethod
    def _geometry(manifest: ArtifactManifest) -> VolumeGeometry:
        geometry = manifest.geometry or manifest.artifact.geometry
        if geometry is None:
            raise _AdapterFailure(ErrorCode.UNSUPPORTED_FORMAT, operation="read_header")
        if (
            manifest.artifact.geometry is not None
            and manifest.artifact.geometry.geometry_fingerprint != geometry.geometry_fingerprint
        ):
            raise _AdapterFailure(ErrorCode.CONTRACT_VIOLATION, operation="read_header")
        return geometry

    def _inspect_case_metadata(
        self,
        request: InspectCaseMetadataRequest,
        deadline: _Deadline,
    ) -> tuple[CaseMetadataResult, tuple[ArtifactRef, ...], tuple[str, ...]]:
        requested_artifacts = (request.ct_artifact,) + tuple(request.related_artifacts)
        manifests = tuple(self._resolve(item, request.context, deadline) for item in requested_artifacts)
        if request.ct_artifact.artifact_type is not ArtifactType.CT_VOLUME:
            raise _AdapterFailure(
                ErrorCode.INVALID_ARGUMENT,
                operation="inspect",
                field_path="ct_artifact.artifact_type",
            )
        geometries = tuple(self._geometry(item) for item in manifests)
        ct_geometry = geometries[0]
        inspections: list[ArtifactInspection] = []
        issues: list[ValidationIssue] = []
        all_compatible = True
        for requested, manifest, geometry in zip(requested_artifacts, manifests, geometries):
            if request.verify_checksums:
                self._verify_checksum(manifest, request.context, deadline)
            compatible = ct_geometry.is_compatible_with(geometry)
            all_compatible = all_compatible and compatible
            inspections.append(
                ArtifactInspection(
                    artifact_id=requested.artifact_id,
                    artifact_type=requested.artifact_type.value,
                    available=True,
                    checksum_valid=True,
                    geometry_matches_ct=compatible,
                )
            )
            if not compatible:
                issues.append(
                    ValidationIssue(
                        code="GEOMETRY_MISMATCH",
                        severity=(
                            ValidationSeverity.ERROR
                            if request.require_same_geometry
                            else ValidationSeverity.WARNING
                        ),
                        message="artifact geometry differs from the reference CT",
                        artifact_id=requested.artifact_id,
                    )
                )
        present_types = {item.artifact.artifact_type for item in manifests}
        missing_types = tuple(
            item for item in request.required_artifact_types if item not in present_types
        )
        if missing_types:
            issues.append(
                ValidationIssue(
                    code="MISSING_ARTIFACT_TYPE",
                    severity=ValidationSeverity.ERROR,
                    message="one or more required artifact types are absent",
                    field_path="required_artifact_types",
                )
            )
        ready = not any(issue.severity is ValidationSeverity.ERROR for issue in issues)
        return (
            CaseMetadataResult(
                case_id=request.context.case_id,
                ct_geometry=ct_geometry,
                inspections=tuple(inspections),
                required_types_present=not missing_types,
                all_geometries_compatible=all_compatible,
                ready_for_next_stage=ready,
                issues=tuple(issues),
            ),
            (),
            (),
        )

    def _conversion_fingerprint(
        self,
        request: ConvertMcsToNiftiRequest,
        mcs: ArtifactManifest,
        ct: ArtifactManifest,
    ) -> str:
        payload = {
            "case_id": request.context.case_id,
            "mcs": [mcs.artifact.artifact_id, mcs.artifact.checksum_sha256],
            "ct": [ct.artifact.artifact_id, ct.artifact.checksum_sha256],
            "mapping": [
                [entry.source_name, entry.source_value, entry.target_name, entry.target_value]
                for entry in request.label_mapping
            ],
            "output_coordinate_system": request.output_coordinate_system.value,
            "output_dtype": request.output_dtype,
            "overwrite": request.overwrite,
        }
        return sha256(_canonical_json(payload)).hexdigest()

    @staticmethod
    def _validated_segments(
        manifest: ArtifactManifest,
        mapping: tuple[LabelMappingEntry, ...],
        output_dtype: str,
    ) -> dict[tuple[str, int], McsSegmentManifest]:
        if manifest.format_name != "mcs-manifest-v1" or not manifest.mcs_segments:
            raise _AdapterFailure(ErrorCode.UNSUPPORTED_FORMAT, operation="parse_mcs")
        by_key: dict[tuple[str, int], McsSegmentManifest] = {}
        normalized_names: set[str] = set()
        source_values: set[int] = set()
        for segment in manifest.mcs_segments:
            normalized = segment.source_name.strip().casefold()
            if normalized in normalized_names or segment.source_value in source_values:
                raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="parse_mcs")
            normalized_names.add(normalized)
            source_values.add(segment.source_value)
            by_key[(normalized, segment.source_value)] = segment
            if segment.overlap_voxels:
                raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="merge_labels")
        mapped_keys: set[tuple[str, int]] = set()
        for entry in mapping:
            if entry.target_value > _DTYPE_MAX[output_dtype]:
                raise _AdapterFailure(
                    ErrorCode.INVALID_ARGUMENT,
                    operation="validate_mapping",
                    field_path="label_mapping.target_value",
                )
            key = (entry.source_name.strip().casefold(), entry.source_value)
            if key not in by_key:
                raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="validate_mapping")
            mapped_keys.add(key)
        if any(segment.source_value != 0 and key not in mapped_keys for key, segment in by_key.items()):
            raise _AdapterFailure(ErrorCode.UNKNOWN_LABEL_VALUE, operation="validate_mapping")
        return by_key

    def _convert_mcs_to_nifti(
        self,
        request: ConvertMcsToNiftiRequest,
        deadline: _Deadline,
    ) -> tuple[McsToNiftiResult, tuple[ArtifactRef, ...], tuple[str, ...]]:
        mcs = self._resolve(request.mcs_artifact, request.context, deadline)
        ct = self._resolve(request.reference_ct_artifact, request.context, deadline)
        self._verify_checksum(mcs, request.context, deadline)
        self._verify_checksum(ct, request.context, deadline)
        mcs_geometry = self._geometry(mcs)
        ct_geometry = self._geometry(ct)
        if not ct_geometry.is_compatible_with(mcs_geometry):
            raise _AdapterFailure(ErrorCode.GEOMETRY_MISMATCH, operation="align_geometry")
        segments = self._validated_segments(mcs, request.label_mapping, request.output_dtype)
        fingerprint = self._conversion_fingerprint(request, mcs, ct)
        scope = "|".join(
            (
                mcs.tenant_id,
                request.context.case_id,
                "convert_mcs_to_nifti",
                request.context.idempotency_key,
            )
        )
        existing = self.backend.lookup_conversion(
            scope,
            deadline.invocation(request.context, "idempotency_lookup"),
        )
        deadline.check("idempotency_lookup")
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise _AdapterFailure(ErrorCode.INVALID_ARGUMENT, operation="idempotency_lookup")
            return self._conversion_result(existing, request.output_dtype), (existing.artifact,), ()

        output_geometry = _convert_coordinate_system(ct_geometry, request.output_coordinate_system)
        product = self.backend.render_conversion(
            mcs,
            ct,
            request.label_mapping,
            output_geometry,
            request.output_dtype,
            deadline.invocation(request.context, "convert"),
        )
        deadline.check("convert")
        if product.geometry.geometry_fingerprint != output_geometry.geometry_fingerprint:
            raise _AdapterFailure(ErrorCode.GEOMETRY_MISMATCH, operation="verify_output")
        counts = dict(product.label_counts)
        expected_values = {entry.target_value for entry in request.label_mapping}
        if set(counts) != expected_values:
            raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="verify_output")
        applied = tuple(
            AppliedLabelMapping(
                source_name=entry.source_name,
                source_value=entry.source_value,
                target_name=entry.target_name,
                target_value=entry.target_value,
                voxel_count=counts[entry.target_value],
            )
            for entry in request.label_mapping
        )
        # Ensure the backend did not silently change the authoritative histogram.
        for entry, item in zip(request.label_mapping, applied):
            segment = segments[(entry.source_name.strip().casefold(), entry.source_value)]
            if segment.voxel_count != item.voxel_count:
                raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="verify_output")
        artifact_id = f"{request.context.case_id}-nifti-{sha256((scope + fingerprint).encode()).hexdigest()[:16]}"
        checksum = sha256(product.payload).hexdigest()
        output = ArtifactRef(
            artifact_id=artifact_id,
            case_id=request.context.case_id,
            artifact_type=ArtifactType.NIFTI_LABELMAP,
            uri=f"memory://case-data/{artifact_id}",
            checksum_sha256=checksum,
            status=ArtifactStatus.AVAILABLE,
            geometry=product.geometry,
            producer_name="convert_mcs_to_nifti",
            producer_version=TOOL_VERSION,
            parent_artifact_ids=(mcs.artifact.artifact_id, ct.artifact.artifact_id),
            metadata={
                "format": product.format_name,
                "label_values": ",".join(str(value) for value in sorted({0, *counts})),
                "label_names": json.dumps(
                    {str(entry.target_value): entry.target_name for entry in request.label_mapping},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "idempotency_fingerprint": fingerprint,
            },
        )
        if not mcs.allowed_callers:
            output_allowed_callers = ct.allowed_callers
        elif not ct.allowed_callers:
            output_allowed_callers = mcs.allowed_callers
        else:
            output_allowed_callers = tuple(
                sorted(set(mcs.allowed_callers).intersection(ct.allowed_callers))
            )
        manifest = ArtifactManifest(
            artifact=output,
            tenant_id=mcs.tenant_id,
            allowed_callers=output_allowed_callers,
            format_name=product.format_name,
            geometry=product.geometry,
            payload=product.payload,
            label_value_chunks=(tuple(sorted({0, *counts})),),
            label_names=tuple((entry.target_value, entry.target_name) for entry in request.label_mapping),
        )
        committed = self.backend.commit_conversion(
            scope,
            ConversionCommit(fingerprint, output, applied, manifest),
            deadline.invocation(request.context, "commit"),
        )
        deadline.check("commit")
        return self._conversion_result(committed, request.output_dtype), (committed.artifact,), ()

    @staticmethod
    def _conversion_result(commit: ConversionCommit, output_dtype: str) -> McsToNiftiResult:
        return McsToNiftiResult(
            output_artifact=commit.artifact,
            applied_mappings=commit.applied_mappings,
            geometry_matches_reference=True,
            output_dtype=output_dtype,
            total_nonzero_voxels=sum(
                item.voxel_count for item in commit.applied_mappings if item.target_value != 0
            ),
        )

    def _validate_label_schema(
        self,
        request: ValidateLabelSchemaRequest,
        deadline: _Deadline,
    ) -> tuple[LabelSchemaValidationResult, tuple[ArtifactRef, ...], tuple[str, ...]]:
        manifest = self._resolve(request.labelmap_artifact, request.context, deadline)
        self._verify_checksum(manifest, request.context, deadline)
        if manifest.format_name not in {"labelmap-manifest-v1", "nifti-manifest-v1"}:
            raise _AdapterFailure(ErrorCode.UNSUPPORTED_FORMAT, operation="read_labels")
        observed: set[int] = set()
        for chunk in manifest.label_value_chunks:
            for raw_value in chunk:
                if (
                    not isinstance(raw_value, int)
                    or isinstance(raw_value, bool)
                    or raw_value < 0
                    or raw_value > 2**63 - 1
                ):
                    raise _AdapterFailure(ErrorCode.LABEL_SCHEMA_ERROR, operation="read_labels")
                observed.add(raw_value)
            deadline.check("read_labels")
        observed_values = tuple(sorted(observed))
        expected_by_value = {item.value: item for item in request.expected_labels}
        missing = tuple(
            item.name
            for item in request.expected_labels
            if item.required and item.value not in observed
        )
        unknown = tuple(value for value in observed_values if value not in expected_by_value)
        issues: list[ValidationIssue] = []
        for name in missing:
            issues.append(
                ValidationIssue(
                    code="REQUIRED_LABEL_MISSING",
                    severity=(
                        ValidationSeverity.ERROR
                        if request.require_all_required_labels
                        else ValidationSeverity.WARNING
                    ),
                    message=f"required label is absent: {name}",
                    artifact_id=request.labelmap_artifact.artifact_id,
                )
            )
        for value in unknown:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_LABEL_VALUE",
                    severity=(
                        ValidationSeverity.WARNING
                        if request.allow_unknown_values
                        else ValidationSeverity.ERROR
                    ),
                    message=f"label value is not present in the expected schema: {value}",
                    artifact_id=request.labelmap_artifact.artifact_id,
                )
            )
        for value, actual_name in sorted(manifest.label_names):
            expected = expected_by_value.get(value)
            if expected is None:
                continue
            allowed_names = {expected.name.strip().casefold(), *(alias.strip().casefold() for alias in expected.aliases)}
            if actual_name.strip().casefold() not in allowed_names:
                issues.append(
                    ValidationIssue(
                        code="LABEL_NAME_VALUE_MISMATCH",
                        severity=ValidationSeverity.ERROR,
                        message="label name metadata disagrees with the expected value",
                        artifact_id=request.labelmap_artifact.artifact_id,
                        field_path=f"label_names.{value}",
                    )
                )
        return (
            LabelSchemaValidationResult(
                valid=not any(issue.severity is ValidationSeverity.ERROR for issue in issues),
                observed_label_values=observed_values,
                missing_required_label_names=missing,
                unknown_label_values=unknown,
                issues=tuple(issues),
            ),
            (),
            (),
        )


def build_case_data_handlers(
    backend: CaseDataBackendPort | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Callable[[Any], ToolResponseEnvelope[Any]]]:
    """Return the three stable handlers without mutating the global registry."""

    return CaseDataToolAdapter(backend, **adapter_kwargs).handlers()


__all__ = [
    "ArtifactManifest",
    "BackendCallRecord",
    "BackendFailure",
    "BackendInvocation",
    "CaseDataBackendPort",
    "CaseDataToolAdapter",
    "ConversionCommit",
    "ConversionProduct",
    "ManifestCaseDataBackend",
    "McsSegmentManifest",
    "build_case_data_handlers",
]
