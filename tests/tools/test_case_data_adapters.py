from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.artifacts import ArtifactRef  # noqa: E402
from contracts.common import ToolCallContext, to_json  # noqa: E402
from contracts.domain import LabelDefinition  # noqa: E402
from contracts.enums import (  # noqa: E402
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
    ErrorCode,
    ToolExecutionStatus,
)
from contracts.geometry import VolumeGeometry  # noqa: E402
from contracts.tool_inputs import (  # noqa: E402
    ConvertMcsToNiftiRequest,
    InspectCaseMetadataRequest,
    LabelMappingEntry,
    ValidateLabelSchemaRequest,
)
from contracts.tool_outputs import (  # noqa: E402
    CaseMetadataResult,
    LabelSchemaValidationResult,
    McsToNiftiResult,
)
from puncture_agent.tooling.case_data import (  # noqa: E402
    ArtifactManifest,
    CaseDataToolAdapter,
    ConversionProduct,
    ManifestCaseDataBackend,
    McsSegmentManifest,
    build_case_data_handlers,
)


FIXED_TIME = "2026-07-11T00:00:00Z"


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self.epoch_ms = 1_800_000_000_000

    def monotonic(self) -> float:
        return self.value

    def wall_ms(self) -> int:
        return self.epoch_ms

    def advance(self, seconds: float) -> None:
        self.value += seconds
        self.epoch_ms += int(seconds * 1000)


def geometry(
    *,
    origin_x: float = 10.0,
    coordinate_system: CoordinateSystem = CoordinateSystem.LPS,
) -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(8, 9, 10),
        spacing_mm=(0.8, 1.1, 2.5),
        origin_mm=(origin_x, -20.0, 30.0),
        direction_cosines=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=coordinate_system,
    )


def context(
    *,
    caller: str = "svc-alice",
    idempotency_key: str = "idem-case-data-1",
    deadline_epoch_ms: int | None = None,
) -> ToolCallContext:
    return ToolCallContext(
        request_id="req-case-data-1",
        trace_id="trace-case-data-1",
        case_id="case-001",
        caller=caller,
        idempotency_key=idempotency_key,
        requested_at=FIXED_TIME,
        deadline_epoch_ms=deadline_epoch_ms,
    )


def artifact_ref(
    artifact_id: str,
    artifact_type: ArtifactType,
    payload: bytes,
    *,
    volume_geometry: VolumeGeometry | None = None,
    status: ArtifactStatus = ArtifactStatus.AVAILABLE,
    checksum: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id="case-001",
        artifact_type=artifact_type,
        uri=f"memory://private/{artifact_id}",
        checksum_sha256=checksum or sha256(payload).hexdigest(),
        status=status,
        geometry=volume_geometry,
        producer_name="fixture",
        producer_version="1",
    )


def labels() -> tuple[LabelDefinition, ...]:
    return (
        LabelDefinition("background", 0, True),
        LabelDefinition("skin", 1, True, aliases=("body_surface",)),
        LabelDefinition("lung", 2, True),
        LabelDefinition("heart", 3, True),
        LabelDefinition("optional_vessel", 4, False),
    )


class CaseDataAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.ct_payload = b"deterministic-ct-header-and-pixels"
        self.mcs_payload = b"deterministic-company-mcs-export"
        self.label_payload = b"deterministic-labelmap-payload"
        self.volume_geometry = geometry()
        self.ct_ref = artifact_ref(
            "case-001-ct",
            ArtifactType.CT_VOLUME,
            self.ct_payload,
            volume_geometry=self.volume_geometry,
        )
        self.mcs_ref = artifact_ref(
            "case-001-mcs",
            ArtifactType.MCS_SEGMENTATION,
            self.mcs_payload,
            volume_geometry=self.volume_geometry,
        )
        self.label_ref = artifact_ref(
            "case-001-labels",
            ArtifactType.NIFTI_LABELMAP,
            self.label_payload,
            volume_geometry=self.volume_geometry,
        )
        self.ct_manifest = ArtifactManifest(
            artifact=self.ct_ref,
            tenant_id="tenant-a",
            allowed_callers=("svc-alice", "svc-peer"),
            format_name="ct-manifest-v1",
            geometry=self.volume_geometry,
            payload=self.ct_payload,
        )
        self.mcs_manifest = ArtifactManifest(
            artifact=self.mcs_ref,
            tenant_id="tenant-a",
            allowed_callers=("svc-alice", "svc-peer"),
            format_name="mcs-manifest-v1",
            geometry=self.volume_geometry,
            payload=self.mcs_payload,
            mcs_segments=(
                McsSegmentManifest("Skin", 1, 120),
                McsSegmentManifest("Lung", 2, 340),
            ),
        )
        self.label_manifest = ArtifactManifest(
            artifact=self.label_ref,
            tenant_id="tenant-a",
            allowed_callers=("svc-alice", "svc-peer"),
            format_name="labelmap-manifest-v1",
            geometry=self.volume_geometry,
            payload=self.label_payload,
            label_value_chunks=((0, 1), (2, 3)),
            label_names=((0, "background"), (1, "body_surface"), (2, "lung"), (3, "heart")),
        )
        self.backend = ManifestCaseDataBackend(
            (self.ct_manifest, self.mcs_manifest, self.label_manifest),
            caller_tenants={
                "svc-alice": "tenant-a",
                "svc-peer": "tenant-a",
                "svc-bob": "tenant-b",
            },
        )
        self.adapter = CaseDataToolAdapter(
            self.backend,
            monotonic=self.clock.monotonic,
            wall_clock_ms=self.clock.wall_ms,
            now_iso=lambda: FIXED_TIME,
            checksum_chunk_size=7,
        )

    def inspect_request(self, **changes: object) -> InspectCaseMetadataRequest:
        request = InspectCaseMetadataRequest(
            context=context(),
            ct_artifact=self.ct_ref,
            related_artifacts=(self.label_ref,),
            required_artifact_types=(ArtifactType.CT_VOLUME, ArtifactType.NIFTI_LABELMAP),
            require_same_geometry=True,
            verify_checksums=True,
        )
        return replace(request, **changes)

    def conversion_request(self, **changes: object) -> ConvertMcsToNiftiRequest:
        request = ConvertMcsToNiftiRequest(
            context=context(),
            mcs_artifact=self.mcs_ref,
            reference_ct_artifact=self.ct_ref,
            label_mapping=(
                LabelMappingEntry("Skin", 1, "skin", 1),
                LabelMappingEntry("Lung", 2, "lung", 2),
            ),
            output_coordinate_system=CoordinateSystem.LPS,
            output_dtype="uint16",
        )
        return replace(request, **changes)

    def validation_request(self, **changes: object) -> ValidateLabelSchemaRequest:
        request = ValidateLabelSchemaRequest(
            context=context(),
            labelmap_artifact=self.label_ref,
            expected_labels=labels(),
        )
        return replace(request, **changes)

    def test_handlers_and_success_envelope_are_strongly_typed_and_traceable(self) -> None:
        handlers = build_case_data_handlers(
            self.backend,
            monotonic=self.clock.monotonic,
            wall_clock_ms=self.clock.wall_ms,
            now_iso=lambda: FIXED_TIME,
        )
        self.assertEqual(
            set(handlers),
            {"inspect_case_metadata", "convert_mcs_to_nifti", "validate_label_schema"},
        )
        response = handlers["inspect_case_metadata"](self.inspect_request())
        self.assertEqual(ToolExecutionStatus.SUCCESS, response.status)
        self.assertIsInstance(response.result, CaseMetadataResult)
        self.assertEqual("req-case-data-1", response.request_id)
        self.assertEqual("trace-case-data-1", response.trace_id)
        self.assertEqual("1.0.0", response.tool_version)
        json.loads(to_json(response))
        self.assertTrue(self.backend.calls)
        self.assertTrue(
            all(call.trace_id == "trace-case-data-1" for call in self.backend.calls)
        )
        self.assertTrue(all(call.timeout_seconds > 0 for call in self.backend.calls))

    def test_inspection_streams_checksums_but_header_only_mode_does_not_read_payload(self) -> None:
        response = self.adapter.inspect_case_metadata(self.inspect_request())
        self.assertTrue(response.ok)
        self.assertTrue(response.result.ready_for_next_stage)
        self.assertEqual(
            [item.artifact_id for item in response.result.inspections],
            [self.ct_ref.artifact_id, self.label_ref.artifact_id],
        )
        self.assertEqual(2, sum(call.operation == "read" for call in self.backend.calls))

        self.backend.calls.clear()
        response = self.adapter.inspect_case_metadata(
            self.inspect_request(verify_checksums=False)
        )
        self.assertTrue(response.result.ready_for_next_stage)
        self.assertFalse(any(call.operation == "read" for call in self.backend.calls))

    def test_geometry_policy_and_required_type_issues_are_successful_validation(self) -> None:
        other_geometry = geometry(origin_x=11.0)
        other_payload = b"other-labels"
        other_ref = artifact_ref(
            "case-001-other-labels",
            ArtifactType.NIFTI_LABELMAP,
            other_payload,
            volume_geometry=other_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                artifact=other_ref,
                tenant_id="tenant-a",
                allowed_callers=("svc-alice",),
                format_name="labelmap-manifest-v1",
                geometry=other_geometry,
                payload=other_payload,
                label_value_chunks=((0, 1),),
            )
        )
        strict = self.adapter.inspect_case_metadata(
            self.inspect_request(
                related_artifacts=(other_ref,),
                required_artifact_types=(ArtifactType.DANGER_MASK,),
            )
        )
        self.assertTrue(strict.ok)
        self.assertFalse(strict.result.ready_for_next_stage)
        self.assertFalse(strict.result.required_types_present)
        self.assertEqual(
            [issue.code for issue in strict.result.issues],
            ["GEOMETRY_MISMATCH", "MISSING_ARTIFACT_TYPE"],
        )

        relaxed = self.adapter.inspect_case_metadata(
            self.inspect_request(
                related_artifacts=(other_ref,),
                required_artifact_types=(),
                require_same_geometry=False,
            )
        )
        self.assertTrue(relaxed.result.ready_for_next_stage)
        self.assertFalse(relaxed.result.all_geometries_compatible)

    def test_missing_unavailable_checksum_permission_and_dependency_errors_are_stable(self) -> None:
        missing_payload = b"missing"
        missing_ref = artifact_ref(
            "case-001-missing",
            ArtifactType.NIFTI_LABELMAP,
            missing_payload,
            volume_geometry=self.volume_geometry,
        )
        missing = self.adapter.inspect_case_metadata(
            self.inspect_request(related_artifacts=(missing_ref,))
        )
        self.assertEqual(ErrorCode.MISSING_ARTIFACT, missing.error.code)

        unavailable_payload = b"unavailable"
        unavailable_ref = artifact_ref(
            "case-001-unavailable",
            ArtifactType.NIFTI_LABELMAP,
            unavailable_payload,
            volume_geometry=self.volume_geometry,
            status=ArtifactStatus.INVALID,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                unavailable_ref,
                "tenant-a",
                ("svc-alice",),
                "labelmap-manifest-v1",
                geometry=self.volume_geometry,
                payload=unavailable_payload,
            )
        )
        unavailable = self.adapter.inspect_case_metadata(
            self.inspect_request(related_artifacts=(unavailable_ref,))
        )
        self.assertEqual(ErrorCode.ARTIFACT_NOT_AVAILABLE, unavailable.error.code)

        corrupt_ref = artifact_ref(
            "case-001-corrupt",
            ArtifactType.NIFTI_LABELMAP,
            b"expected",
            volume_geometry=self.volume_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                corrupt_ref,
                "tenant-a",
                ("svc-alice",),
                "labelmap-manifest-v1",
                geometry=self.volume_geometry,
                payload=b"tampered",
            )
        )
        checksum = self.adapter.inspect_case_metadata(
            self.inspect_request(related_artifacts=(corrupt_ref,))
        )
        self.assertEqual(ErrorCode.CHECKSUM_MISMATCH, checksum.error.code)

        denied = self.adapter.inspect_case_metadata(
            self.inspect_request(context=context(caller="svc-bob"))
        )
        self.assertEqual(ErrorCode.PERMISSION_DENIED, denied.error.code)
        self.assertNotIn("tenant-a", to_json(denied))
        self.assertNotIn("memory://", to_json(denied))

        self.backend.inject_failure(
            "resolve", ErrorCode.DEPENDENCY_FAILED, artifact_id=self.ct_ref.artifact_id, retryable=True
        )
        dependency = self.adapter.inspect_case_metadata(self.inspect_request())
        self.assertEqual(ErrorCode.DEPENDENCY_FAILED, dependency.error.code)
        self.assertTrue(dependency.error.retryable)
        self.assertEqual("case_data_backend", dependency.error.dependency)

    def test_file_payload_protocol_reads_regular_files_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "labels.bin"
            payload_path.write_bytes(b"regular-file-labelmap")
            file_ref = artifact_ref(
                "case-001-file-labels",
                ArtifactType.NIFTI_LABELMAP,
                payload_path.read_bytes(),
                volume_geometry=self.volume_geometry,
            )
            self.backend.register_manifest(
                ArtifactManifest(
                    file_ref,
                    "tenant-a",
                    ("svc-alice",),
                    "labelmap-manifest-v1",
                    geometry=self.volume_geometry,
                    file_path=str(payload_path),
                    label_value_chunks=((0, 1),),
                )
            )
            response = self.adapter.inspect_case_metadata(
                self.inspect_request(related_artifacts=(file_ref,))
            )
            self.assertTrue(response.ok)

            symlink = root / "labels-link.bin"
            symlink.symlink_to(payload_path)
            link_ref = artifact_ref(
                "case-001-link-labels",
                ArtifactType.NIFTI_LABELMAP,
                payload_path.read_bytes(),
                volume_geometry=self.volume_geometry,
            )
            self.backend.register_manifest(
                ArtifactManifest(
                    link_ref,
                    "tenant-a",
                    ("svc-alice",),
                    "labelmap-manifest-v1",
                    geometry=self.volume_geometry,
                    file_path=str(symlink),
                    label_value_chunks=((0, 1),),
                )
            )
            rejected = self.adapter.inspect_case_metadata(
                self.inspect_request(related_artifacts=(link_ref,))
            )
            self.assertEqual(ErrorCode.ARTIFACT_NOT_AVAILABLE, rejected.error.code)

    def test_expired_and_slow_operations_return_timeout_with_no_raw_exception(self) -> None:
        expired = self.adapter.inspect_case_metadata(
            self.inspect_request(
                context=context(deadline_epoch_ms=self.clock.wall_ms() - 1)
            )
        )
        self.assertEqual(ErrorCode.TIMEOUT, expired.error.code)
        self.assertTrue(expired.error.retryable)
        self.assertEqual([], self.backend.calls)

        def advance_on_resolve(call: object) -> None:
            if getattr(call, "operation") == "resolve":
                self.clock.advance(11.0)

        slow_backend = ManifestCaseDataBackend(
            (self.ct_manifest, self.label_manifest),
            caller_tenants={"svc-alice": "tenant-a"},
            operation_hook=advance_on_resolve,
        )
        slow_adapter = CaseDataToolAdapter(
            slow_backend,
            monotonic=self.clock.monotonic,
            wall_clock_ms=self.clock.wall_ms,
            now_iso=lambda: FIXED_TIME,
        )
        slow = slow_adapter.inspect_case_metadata(self.inspect_request())
        self.assertEqual(ErrorCode.TIMEOUT, slow.error.code)
        self.assertEqual("resolve", slow.error.details["operation"])

    def test_conversion_preserves_geometry_mapping_lineage_and_ras_world_transform(self) -> None:
        response = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(output_coordinate_system=CoordinateSystem.RAS)
        )
        self.assertTrue(response.ok)
        self.assertIsInstance(response.result, McsToNiftiResult)
        self.assertEqual((120, 340), tuple(item.voxel_count for item in response.result.applied_mappings))
        self.assertEqual(460, response.result.total_nonzero_voxels)
        output = response.result.output_artifact
        self.assertEqual(ArtifactType.NIFTI_LABELMAP, output.artifact_type)
        self.assertEqual(
            (self.mcs_ref.artifact_id, self.ct_ref.artifact_id),
            output.parent_artifact_ids,
        )
        self.assertEqual(CoordinateSystem.RAS, output.geometry.coordinate_system)
        self.assertEqual((-10.0, 20.0, 30.0), output.geometry.origin_mm)
        self.assertEqual((output,), response.artifacts)
        json.loads(to_json(response))

    def test_conversion_is_idempotent_and_concurrent_retries_commit_once(self) -> None:
        request = self.conversion_request()
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: self.adapter.convert_mcs_to_nifti(request), range(6)))
        self.assertTrue(all(response.ok for response in responses))
        self.assertEqual(
            1,
            len({response.result.output_artifact.artifact_id for response in responses}),
        )
        self.assertEqual(1, self.backend.committed_conversion_count)

        # The idempotency namespace is tenant-scoped. A peer service authorized
        # for both source artifacts reuses the same commit and derived access.
        peer_request = replace(
            request,
            context=replace(request.context, caller="svc-peer", request_id="req-convert-peer"),
        )
        peer_response = self.adapter.convert_mcs_to_nifti(peer_request)
        self.assertTrue(peer_response.ok)
        self.assertEqual(
            responses[0].result.output_artifact.artifact_id,
            peer_response.result.output_artifact.artifact_id,
        )
        self.assertEqual(1, self.backend.committed_conversion_count)
        peer_validation = self.adapter.validate_label_schema(
            ValidateLabelSchemaRequest(
                context=replace(
                    request.context,
                    caller="svc-peer",
                    request_id="req-peer-validate",
                    idempotency_key="peer-validate",
                ),
                labelmap_artifact=peer_response.result.output_artifact,
                expected_labels=labels(),
            )
        )
        self.assertTrue(peer_validation.ok)

        conflict = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(output_dtype="uint8")
        )
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, conflict.error.code)
        self.assertEqual(1, self.backend.committed_conversion_count)

    def test_conversion_mapping_overlap_unknown_and_dtype_boundaries_fail_closed(self) -> None:
        unknown_payload = b"mcs-with-unknown"
        unknown_ref = artifact_ref(
            "case-001-mcs-unknown",
            ArtifactType.MCS_SEGMENTATION,
            unknown_payload,
            volume_geometry=self.volume_geometry,
        )
        unknown_manifest = ArtifactManifest(
            unknown_ref,
            "tenant-a",
            ("svc-alice",),
            "mcs-manifest-v1",
            geometry=self.volume_geometry,
            payload=unknown_payload,
            mcs_segments=(
                McsSegmentManifest("Skin", 1, 10),
                McsSegmentManifest("Unmapped", 9, 4),
            ),
        )
        self.backend.register_manifest(unknown_manifest)
        unknown = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(
                mcs_artifact=unknown_ref,
                label_mapping=(LabelMappingEntry("Skin", 1, "skin", 1),),
            )
        )
        self.assertEqual(ErrorCode.UNKNOWN_LABEL_VALUE, unknown.error.code)

        missing_source = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(
                context=context(idempotency_key="missing-source"),
                label_mapping=(LabelMappingEntry("Wrong", 1, "skin", 1),),
            )
        )
        self.assertEqual(ErrorCode.LABEL_SCHEMA_ERROR, missing_source.error.code)

        overflow = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(
                context=context(idempotency_key="overflow"),
                label_mapping=(
                    LabelMappingEntry("Skin", 1, "skin", 300),
                    LabelMappingEntry("Lung", 2, "lung", 2),
                ),
                output_dtype="uint8",
            )
        )
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, overflow.error.code)

        overlap_payload = b"mcs-overlap"
        overlap_ref = artifact_ref(
            "case-001-mcs-overlap",
            ArtifactType.MCS_SEGMENTATION,
            overlap_payload,
            volume_geometry=self.volume_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                overlap_ref,
                "tenant-a",
                ("svc-alice",),
                "mcs-manifest-v1",
                geometry=self.volume_geometry,
                payload=overlap_payload,
                mcs_segments=(McsSegmentManifest("Skin", 1, 10, overlap_voxels=1),),
            )
        )
        overlap = self.adapter.convert_mcs_to_nifti(
            self.conversion_request(
                context=context(idempotency_key="overlap"),
                mcs_artifact=overlap_ref,
                label_mapping=(LabelMappingEntry("Skin", 1, "skin", 1),),
            )
        )
        self.assertEqual(ErrorCode.LABEL_SCHEMA_ERROR, overlap.error.code)

    def test_conversion_dependency_or_partial_commit_failure_never_publishes_output(self) -> None:
        self.backend.inject_failure("convert", ErrorCode.DEPENDENCY_FAILED, retryable=True)
        failed = self.adapter.convert_mcs_to_nifti(self.conversion_request())
        self.assertEqual(ErrorCode.DEPENDENCY_FAILED, failed.error.code)
        self.assertEqual(0, self.backend.committed_conversion_count)
        self.assertEqual((), failed.artifacts)

        backend = ManifestCaseDataBackend(
            (self.ct_manifest, self.mcs_manifest),
            caller_tenants={"svc-alice": "tenant-a"},
        )
        backend.inject_failure("commit", ErrorCode.DEPENDENCY_FAILED, retryable=True)
        adapter = CaseDataToolAdapter(
            backend,
            monotonic=self.clock.monotonic,
            wall_clock_ms=self.clock.wall_ms,
            now_iso=lambda: FIXED_TIME,
        )
        failed_commit = adapter.convert_mcs_to_nifti(self.conversion_request())
        self.assertEqual(ErrorCode.DEPENDENCY_FAILED, failed_commit.error.code)
        self.assertEqual(0, backend.committed_conversion_count)
        self.assertEqual((), failed_commit.artifacts)

    def test_conversion_rejects_backend_geometry_or_histogram_drift(self) -> None:
        class DriftBackend(ManifestCaseDataBackend):
            def __init__(self, *args: object, drift: str, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.drift = drift

            def render_conversion(self, *args: object, **kwargs: object) -> ConversionProduct:
                product = super().render_conversion(*args, **kwargs)
                if self.drift == "geometry":
                    return ConversionProduct(product.payload, geometry(origin_x=99.0), product.label_counts)
                return ConversionProduct(product.payload, product.geometry, ((1, 999), (2, 340)))

        for drift, expected in (
            ("geometry", ErrorCode.GEOMETRY_MISMATCH),
            ("histogram", ErrorCode.LABEL_SCHEMA_ERROR),
        ):
            with self.subTest(drift=drift):
                backend = DriftBackend(
                    (self.ct_manifest, self.mcs_manifest),
                    caller_tenants={"svc-alice": "tenant-a"},
                    drift=drift,
                )
                adapter = CaseDataToolAdapter(
                    backend,
                    monotonic=self.clock.monotonic,
                    wall_clock_ms=self.clock.wall_ms,
                    now_iso=lambda: FIXED_TIME,
                )
                response = adapter.convert_mcs_to_nifti(
                    self.conversion_request(context=context(idempotency_key=f"drift-{drift}"))
                )
                self.assertEqual(expected, response.error.code)
                self.assertEqual(0, backend.committed_conversion_count)

    def test_label_schema_exact_unknown_missing_and_flag_combinations(self) -> None:
        exact = self.adapter.validate_label_schema(self.validation_request())
        self.assertTrue(exact.ok)
        self.assertIsInstance(exact.result, LabelSchemaValidationResult)
        self.assertTrue(exact.result.valid)
        self.assertEqual((0, 1, 2, 3), exact.result.observed_label_values)

        edge_payload = b"edge-labels"
        edge_ref = artifact_ref(
            "case-001-edge-labels",
            ArtifactType.NIFTI_LABELMAP,
            edge_payload,
            volume_geometry=self.volume_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                edge_ref,
                "tenant-a",
                ("svc-alice",),
                "labelmap-manifest-v1",
                geometry=self.volume_geometry,
                payload=edge_payload,
                label_value_chunks=((0, 1, 65_535),),
            )
        )
        strict = self.adapter.validate_label_schema(
            self.validation_request(labelmap_artifact=edge_ref)
        )
        self.assertFalse(strict.result.valid)
        self.assertEqual(("lung", "heart"), strict.result.missing_required_label_names)
        self.assertEqual((65_535,), strict.result.unknown_label_values)

        permissive = self.adapter.validate_label_schema(
            self.validation_request(
                labelmap_artifact=edge_ref,
                allow_unknown_values=True,
                require_all_required_labels=False,
            )
        )
        self.assertTrue(permissive.result.valid)
        self.assertEqual((65_535,), permissive.result.unknown_label_values)

    def test_label_name_mismatch_invalid_values_and_unsupported_format_are_stable(self) -> None:
        mismatch_payload = b"name-mismatch"
        mismatch_ref = artifact_ref(
            "case-001-name-mismatch",
            ArtifactType.NIFTI_LABELMAP,
            mismatch_payload,
            volume_geometry=self.volume_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                mismatch_ref,
                "tenant-a",
                ("svc-alice",),
                "labelmap-manifest-v1",
                geometry=self.volume_geometry,
                payload=mismatch_payload,
                label_value_chunks=((0, 1, 2, 3),),
                label_names=((1, "wrong-name"),),
            )
        )
        mismatch = self.adapter.validate_label_schema(
            self.validation_request(labelmap_artifact=mismatch_ref)
        )
        self.assertFalse(mismatch.result.valid)
        self.assertIn("LABEL_NAME_VALUE_MISMATCH", {issue.code for issue in mismatch.result.issues})

        for index, raw in enumerate((1.5, float("nan"), -1, True)):
            payload = f"invalid-{index}".encode()
            invalid_ref = artifact_ref(
                f"case-001-invalid-{index}",
                ArtifactType.NIFTI_LABELMAP,
                payload,
                volume_geometry=self.volume_geometry,
            )
            self.backend.register_manifest(
                ArtifactManifest(
                    invalid_ref,
                    "tenant-a",
                    ("svc-alice",),
                    "labelmap-manifest-v1",
                    geometry=self.volume_geometry,
                    payload=payload,
                    label_value_chunks=((raw,),),
                )
            )
            with self.subTest(raw=raw):
                response = self.adapter.validate_label_schema(
                    self.validation_request(labelmap_artifact=invalid_ref)
                )
                self.assertEqual(ErrorCode.LABEL_SCHEMA_ERROR, response.error.code)

        opaque_payload = b"opaque"
        opaque_ref = artifact_ref(
            "case-001-opaque",
            ArtifactType.NIFTI_LABELMAP,
            opaque_payload,
            volume_geometry=self.volume_geometry,
        )
        self.backend.register_manifest(
            ArtifactManifest(
                opaque_ref,
                "tenant-a",
                ("svc-alice",),
                "opaque-binary",
                geometry=self.volume_geometry,
                payload=opaque_payload,
            )
        )
        unsupported = self.adapter.validate_label_schema(
            self.validation_request(labelmap_artifact=opaque_ref)
        )
        self.assertEqual(ErrorCode.UNSUPPORTED_FORMAT, unsupported.error.code)

    def test_label_validation_is_repeatable_read_only_and_checksum_guarded(self) -> None:
        first = self.adapter.validate_label_schema(self.validation_request())
        committed_before = self.backend.committed_conversion_count
        second = self.adapter.validate_label_schema(self.validation_request())
        self.assertEqual(first.result, second.result)
        self.assertEqual(committed_before, self.backend.committed_conversion_count)

        forged = replace(self.label_ref, checksum_sha256="f" * 64)
        guarded = self.adapter.validate_label_schema(
            self.validation_request(labelmap_artifact=forged)
        )
        self.assertEqual(ErrorCode.CHECKSUM_MISMATCH, guarded.error.code)


if __name__ == "__main__":
    unittest.main()
