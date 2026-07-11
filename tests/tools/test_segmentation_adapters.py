from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.common import ToolCallContext, to_primitive  # noqa: E402
from contracts.domain import LabelQualityThreshold  # noqa: E402
from contracts.enums import (  # noqa: E402
    ArtifactStatus,
    ArtifactType,
    ErrorCode,
    InferencePrecision,
    RecommendedAction,
    ToolExecutionStatus,
)
from contracts.geometry import VolumeGeometry  # noqa: E402
from contracts.tool_outputs import LabelStatistics  # noqa: E402
from puncture_agent.tooling.segmentation import (  # noqa: E402
    DeterministicSegmentationBackend,
    EngineResult,
    InMemorySegmentationTraceSink,
    LabelAnalysis,
    SegmentationBackendError,
    SegmentationToolAdapter,
    SkinSurfaceKernelResult,
    VolumeManifest,
    build_segmentation_handlers,
)
from tools.helpers import (  # noqa: E402
    artifact,
    geometry,
    labels,
    segmentation_request,
    segmentation_validation_request,
    skin_request,
)


FIXED_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


def _stats(
    name: str,
    value: int,
    voxels: int,
    *,
    volume_geometry: VolumeGeometry,
    components: int = 1,
    border: bool = False,
) -> LabelStatistics:
    spacing = volume_geometry.spacing_mm
    volume_ml = voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0
    return LabelStatistics(name, value, voxels, volume_ml, components, border)


def _manifest(
    ref,
    *,
    dtype: str = "uint8",
    values: tuple[int, ...] = (0, 1),
    statistics: tuple[LabelStatistics, ...] = (),
    primary_label_value: int | None = 1,
    surface_voxels: int | None = None,
    checksum_valid: bool = True,
) -> VolumeManifest:
    total = None
    if ref.geometry is not None:
        total = ref.geometry.size_ijk[0] * ref.geometry.size_ijk[1] * ref.geometry.size_ijk[2]
    return VolumeManifest(
        artifact_id=ref.artifact_id,
        geometry=ref.geometry,
        dtype=dtype,
        observed_label_values=values,
        label_statistics=statistics,
        checksum_valid=checksum_valid,
        expected_checksum_sha256=ref.checksum_sha256,
        primary_label_value=primary_label_value,
        total_voxel_count=total,
        deterministic_surface_voxel_count=surface_voxels,
    )


class SegmentationAdapterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = DeterministicSegmentationBackend()
        self.traces = InMemorySegmentationTraceSink()
        self.adapter = SegmentationToolAdapter(
            engine=self.backend,
            algorithms=self.backend,
            artifacts=self.backend,
            trace_sink=self.traces,
            clock=lambda: FIXED_NOW,
        )

    def _new_context(
        self,
        source: ToolCallContext,
        *,
        request_id: str,
        trace_id: str,
        idempotency_key: str,
        deadline_epoch_ms: int | None = None,
    ) -> ToolCallContext:
        return replace(
            source,
            request_id=request_id,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
            deadline_epoch_ms=deadline_epoch_ms,
        )


class RunSegmentationAdapterTests(SegmentationAdapterTestCase):
    def test_success_preserves_registered_mapping_lineage_contract_and_trace(self) -> None:
        request = segmentation_request(requested_labels=("heart", "skin", "lung"))

        response = self.adapter.run_segmentation(request)

        self.assertEqual(ToolExecutionStatus.SUCCESS, response.status)
        self.assertEqual("1.0.0", response.tool_version)
        self.assertEqual(
            (("skin", 1), ("lung", 2), ("heart", 3)),
            tuple((item.label_name, item.label_value) for item in response.result.produced_labels),
        )
        output = response.result.segmentation_artifact
        self.assertEqual(ArtifactType.SEGMENTATION_MASK, output.artifact_type)
        self.assertEqual((request.ct_artifact.artifact_id,), output.parent_artifact_ids)
        self.assertEqual(
            request.ct_artifact.geometry.geometry_fingerprint,
            output.geometry.geometry_fingerprint,
        )
        self.assertEqual(
            {"inference_time", "peak_gpu_memory"},
            {item.name for item in response.metrics},
        )
        self.assertIn("LOCAL_DETERMINISTIC_SIMULATION", response.warnings[0])
        self.assertIsNotNone(self.backend.committed_manifest(output.artifact_id))
        json.loads(json.dumps(to_primitive(response)))
        event = self.traces.events[-1]
        self.assertEqual("success", event.outcome)
        self.assertEqual(request.context.trace_id, event.trace_id)
        self.assertEqual(output.artifact_id, event.attributes["artifact_id"])
        self.assertIn("engine_hash", event.attributes)

    def test_permission_is_checked_before_engine_use_without_secret_leakage(self) -> None:
        request = segmentation_request()
        self.backend.set_artifact_permissions(request.ct_artifact.artifact_id, {"another-service"})

        response = self.adapter.run_segmentation(request)

        self.assertEqual(ErrorCode.PERMISSION_DENIED, response.error.code)
        self.assertFalse(response.error.retryable)
        self.assertEqual(0, self.backend.inference_count)
        rendered = json.dumps(to_primitive(response))
        self.assertNotIn(request.ct_artifact.uri, rendered)
        self.assertNotIn(request.ct_artifact.checksum_sha256, rendered)

    def test_same_idempotency_key_reuses_one_inference_and_one_commit(self) -> None:
        first_request = segmentation_request()
        second_request = replace(
            first_request,
            context=self._new_context(
                first_request.context,
                request_id="req-retry",
                trace_id="trace-retry",
                idempotency_key=first_request.context.idempotency_key,
            ),
        )

        first = self.adapter.run_segmentation(first_request)
        second = self.adapter.run_segmentation(second_request)

        self.assertEqual(first.result, second.result)
        self.assertEqual("req-retry", second.request_id)
        self.assertEqual("trace-retry", second.trace_id)
        self.assertEqual(1, self.backend.inference_count)
        self.assertEqual(1, self.backend.commit_count)
        self.assertEqual("idempotency_replay", self.traces.events[-1].outcome)

    def test_idempotency_key_conflict_is_non_retryable_contract_error(self) -> None:
        request = segmentation_request()
        self.assertTrue(self.adapter.run_segmentation(request).ok)
        conflicting = replace(request, requested_labels=("skin",))

        response = self.adapter.run_segmentation(conflicting)

        self.assertEqual(ErrorCode.CONTRACT_VIOLATION, response.error.code)
        self.assertFalse(response.error.retryable)
        self.assertEqual(1, self.backend.inference_count)
        self.assertEqual(1, self.backend.commit_count)

    def test_engine_is_reused_across_distinct_calls(self) -> None:
        first = segmentation_request()
        second_ct = artifact(ArtifactType.CT_VOLUME, "ct-second")
        second = replace(
            first,
            context=self._new_context(
                first.context,
                request_id="req-002",
                trace_id="trace-002",
                idempotency_key="idem-002",
            ),
            ct_artifact=second_ct,
        )

        self.assertTrue(self.adapter.run_segmentation(first).ok)
        self.assertTrue(self.adapter.run_segmentation(second).ok)

        self.assertEqual(1, self.backend.engine_load_count)
        self.assertEqual(2, self.backend.inference_count)
        self.assertEqual(2, self.backend.commit_count)

    def test_model_profile_and_argument_errors_are_stable(self) -> None:
        cases = (
            (replace(segmentation_request(), model_id="absent"), ErrorCode.MODEL_NOT_FOUND),
            (
                replace(segmentation_request(), model_version="v999"),
                ErrorCode.MODEL_VERSION_MISMATCH,
            ),
            (
                replace(segmentation_request(), precision=InferencePrecision.INT8),
                ErrorCode.INVALID_ARGUMENT,
            ),
            (
                replace(segmentation_request(), requested_labels=("not-trained",)),
                ErrorCode.INVALID_ARGUMENT,
            ),
            (
                replace(segmentation_request(), output_probability_maps=True),
                ErrorCode.INVALID_ARGUMENT,
            ),
        )
        for index, (request, expected_code) in enumerate(cases):
            request = replace(
                request,
                context=self._new_context(
                    request.context,
                    request_id=f"req-error-{index}",
                    trace_id=f"trace-error-{index}",
                    idempotency_key=f"idem-error-{index}",
                ),
            )
            with self.subTest(expected_code=expected_code):
                response = self.adapter.run_segmentation(request)
                self.assertEqual(expected_code, response.error.code)
                self.assertFalse(response.error.retryable)

    def test_retryable_oom_releases_runtime_state_and_can_retry(self) -> None:
        request = segmentation_request()
        self.backend.fail_next(
            "infer",
            SegmentationBackendError(
                ErrorCode.GPU_OUT_OF_MEMORY,
                "GPU allocation failed",
                retryable=True,
                dependency="cuda",
            ),
        )

        failed = self.adapter.run_segmentation(request)
        retried = self.adapter.run_segmentation(request)

        self.assertEqual(ErrorCode.GPU_OUT_OF_MEMORY, failed.error.code)
        self.assertTrue(failed.error.retryable)
        self.assertEqual(0, self.backend.active_inferences)
        self.assertTrue(retried.ok)
        self.assertEqual(1, self.backend.inference_count)
        self.assertEqual(1, self.backend.commit_count)

    def test_retryable_commit_failure_does_not_leave_a_partial_artifact(self) -> None:
        request = segmentation_request()
        self.backend.fail_next(
            "commit",
            SegmentationBackendError(
                ErrorCode.DEPENDENCY_FAILED,
                "atomic artifact commit failed",
                retryable=True,
                dependency="artifact_store",
            ),
        )

        failed = self.adapter.run_segmentation(request)
        self.assertEqual(ErrorCode.DEPENDENCY_FAILED, failed.error.code)
        self.assertEqual(0, self.backend.commit_count)

        retried = self.adapter.run_segmentation(request)
        self.assertTrue(retried.ok)
        self.assertEqual(2, self.backend.inference_count)
        self.assertEqual(1, self.backend.commit_count)

    def test_malformed_geometry_and_unknown_output_label_are_rejected(self) -> None:
        request = segmentation_request()
        profile = self.backend.describe(request.model_id, request.model_version, request.precision)
        shifted = geometry(origin_x=5.0)
        bad_geometry = EngineResult(
            geometry=shifted,
            output_dtype="uint8",
            observed_label_values=(0, 1, 2, 3),
            label_statistics=tuple(
                _stats(
                    label.name,
                    label.value,
                    label.deterministic_voxel_count,
                    volume_geometry=shifted,
                    border=label.touches_volume_border,
                )
                for label in profile.labels
            ),
            inference_time_ms=1.0,
            peak_gpu_memory_mb=1.0,
            engine_hash=profile.engine_hash,
        )
        self.backend.set_inference_override(
            request.model_id, request.model_version, request.precision, bad_geometry
        )
        response = self.adapter.run_segmentation(request)
        self.assertEqual(ErrorCode.GEOMETRY_MISMATCH, response.error.code)
        self.assertEqual(0, self.backend.commit_count)

        backend = DeterministicSegmentationBackend()
        adapter = SegmentationToolAdapter(engine=backend, algorithms=backend, artifacts=backend)
        profile = backend.describe(request.model_id, request.model_version, request.precision)
        unknown = EngineResult(
            geometry=request.ct_artifact.geometry,
            output_dtype="uint8",
            observed_label_values=(0, 1, 2, 3, 99),
            label_statistics=tuple(
                _stats(
                    label.name,
                    label.value,
                    label.deterministic_voxel_count,
                    volume_geometry=request.ct_artifact.geometry,
                    border=label.touches_volume_border,
                )
                for label in profile.labels
            ),
            inference_time_ms=1.0,
            peak_gpu_memory_mb=1.0,
            engine_hash=profile.engine_hash,
        )
        backend.set_inference_override(request.model_id, request.model_version, request.precision, unknown)
        response = adapter.run_segmentation(request)
        self.assertEqual(ErrorCode.UNKNOWN_LABEL_VALUE, response.error.code)
        self.assertEqual(0, backend.commit_count)

    def test_expired_deadline_is_retryable_and_short_circuits_dependencies(self) -> None:
        request = segmentation_request()
        request = replace(
            request,
            context=replace(request.context, deadline_epoch_ms=int(FIXED_NOW.timestamp() * 1000)),
        )

        response = self.adapter.run_segmentation(request)

        self.assertEqual(ErrorCode.TIMEOUT, response.error.code)
        self.assertTrue(response.error.retryable)
        self.assertEqual(0, self.backend.resolve_count)
        self.assertEqual(0, self.backend.inference_count)


class ValidateSegmentationAdapterTests(SegmentationAdapterTestCase):
    def test_analytic_statistics_use_spacing_and_preserve_schema_order(self) -> None:
        request = segmentation_validation_request(
            quality_thresholds=(
                LabelQualityThreshold("skin", 100, 2, min_volume_ml=1.0, max_volume_ml=2.0),
                LabelQualityThreshold("lung", 100, 2),
                LabelQualityThreshold("heart", 100, 2),
            )
        )
        ref = request.segmentation_artifact
        geo = ref.geometry
        statistics = (
            _stats("background", 0, 1_000, volume_geometry=geo, border=True),
            _stats("skin", 1, 1_000, volume_geometry=geo, border=True),
            _stats("lung", 2, 200, volume_geometry=geo, components=2),
            _stats("heart", 3, 300, volume_geometry=geo),
        )
        self.backend.register_manifest(
            _manifest(ref, values=(0, 1, 2, 3), statistics=statistics)
        )

        response = self.adapter.validate_segmentation_result(request)

        self.assertTrue(response.result.valid)
        self.assertEqual(RecommendedAction.CONTINUE, response.result.recommended_action)
        self.assertEqual(
            ("background", "skin", "lung", "heart"),
            tuple(item.statistics.label_name for item in response.result.label_results),
        )
        skin = response.result.label_results[1].statistics
        self.assertAlmostEqual(1.5, skin.volume_ml, places=9)
        self.assertEqual(2, response.result.label_results[2].statistics.component_count)
        self.assertTrue(skin.touches_volume_border)

    def test_quality_findings_are_results_not_failed_tool_calls(self) -> None:
        request = segmentation_validation_request()
        ref = request.segmentation_artifact
        stats = (
            _stats("background", 0, 10, volume_geometry=ref.geometry, border=True),
            _stats("skin", 1, 50, volume_geometry=ref.geometry, components=20, border=False),
            _stats("label_7", 7, 10, volume_geometry=ref.geometry),
        )
        self.backend.register_manifest(
            _manifest(ref, values=(0, 1, 7), statistics=stats)
        )

        response = self.adapter.validate_segmentation_result(request)

        self.assertEqual(ToolExecutionStatus.SUCCESS, response.status)
        self.assertFalse(response.result.valid)
        self.assertEqual(RecommendedAction.MANUAL_REVIEW, response.result.recommended_action)
        codes = [item.code for item in response.result.issues]
        self.assertIn("UNKNOWN_LABEL_VALUE", codes)
        self.assertIn("VOXEL_COUNT_TOO_LOW", codes)
        self.assertIn("TOO_MANY_COMPONENTS", codes)
        self.assertEqual(2, codes.count("REQUIRED_LABEL_MISSING"))
        self.assertIn(
            "EXPECTED_BORDER_CONTACT_MISSING",
            response.result.label_results[1].issue_codes,
        )

    def test_geometry_mismatch_is_reported_even_when_shape_matches(self) -> None:
        shifted_ref = artifact(
            ArtifactType.SEGMENTATION_MASK,
            "seg-shifted",
            metadata={"label_values": "0,1,2,3"},
            volume_geometry=geometry(origin_x=2.0),
        )
        request = segmentation_validation_request(segmentation_artifact=shifted_ref)

        response = self.adapter.validate_segmentation_result(request)

        self.assertEqual(ToolExecutionStatus.SUCCESS, response.status)
        self.assertFalse(response.result.geometry_matches_ct)
        self.assertFalse(response.result.valid)
        self.assertIn("GEOMETRY_MISMATCH", {item.code for item in response.result.issues})

        second = replace(
            request,
            context=self._new_context(
                request.context,
                request_id="req-ignore-geometry",
                trace_id="trace-ignore-geometry",
                idempotency_key="idem-ignore-geometry",
            ),
            require_geometry_match=False,
        )
        ignored = self.adapter.validate_segmentation_result(second)
        self.assertFalse(ignored.result.geometry_matches_ct)
        self.assertTrue(ignored.result.valid)

    def test_fractional_labelmap_is_a_failed_unsupported_format(self) -> None:
        request = segmentation_validation_request()
        ref = request.segmentation_artifact
        self.backend.register_manifest(
            _manifest(ref, dtype="float32", values=(0, 1, 2, 3))
        )

        response = self.adapter.validate_segmentation_result(request)

        self.assertEqual(ErrorCode.UNSUPPORTED_FORMAT, response.error.code)
        self.assertFalse(response.error.retryable)

    def test_permission_checksum_and_pending_state_are_mapped(self) -> None:
        request = segmentation_validation_request()
        self.backend.set_artifact_permissions(request.segmentation_artifact.artifact_id, {"other"})
        denied = self.adapter.validate_segmentation_result(request)
        self.assertEqual(ErrorCode.PERMISSION_DENIED, denied.error.code)

        backend = DeterministicSegmentationBackend()
        adapter = SegmentationToolAdapter(engine=backend, algorithms=backend, artifacts=backend)
        backend.register_manifest(
            _manifest(request.segmentation_artifact, checksum_valid=False)
        )
        checksum = adapter.validate_segmentation_result(request)
        self.assertEqual(ErrorCode.CHECKSUM_MISMATCH, checksum.error.code)

        pending_ref = replace(request.segmentation_artifact, status=ArtifactStatus.PENDING)
        pending_request = replace(request, segmentation_artifact=pending_ref)
        backend = DeterministicSegmentationBackend()
        adapter = SegmentationToolAdapter(engine=backend, algorithms=backend, artifacts=backend)
        pending = adapter.validate_segmentation_result(pending_request)
        self.assertEqual(ErrorCode.ARTIFACT_NOT_AVAILABLE, pending.error.code)
        self.assertTrue(pending.error.retryable)

    def test_repeated_validation_is_result_equivalent_and_does_not_mutate_input(self) -> None:
        request = segmentation_validation_request()
        original_metadata = dict(request.segmentation_artifact.metadata)

        first = self.adapter.validate_segmentation_result(request)
        second = self.adapter.validate_segmentation_result(request)

        self.assertEqual(first.result, second.result)
        self.assertEqual(original_metadata, request.segmentation_artifact.metadata)
        self.assertEqual("idempotency_replay", self.traces.events[-1].outcome)

    def test_unknown_threshold_label_is_rejected(self) -> None:
        request = segmentation_validation_request(
            quality_thresholds=(LabelQualityThreshold("not-in-schema", 1, 1),)
        )

        response = self.adapter.validate_segmentation_result(request)

        self.assertEqual(ErrorCode.INVALID_ARGUMENT, response.error.code)
        self.assertEqual("quality_thresholds", response.error.field_path)

    def test_duplicate_statistics_from_native_port_are_rejected(self) -> None:
        request = segmentation_validation_request()
        duplicate = _stats(
            "skin",
            1,
            100,
            volume_geometry=request.segmentation_artifact.geometry,
            border=True,
        )
        self.backend.set_analysis_override(
            request.segmentation_artifact.artifact_id,
            LabelAnalysis(
                observed_label_values=(0, 1, 2, 3),
                statistics=(duplicate, duplicate),
                total_voxel_count=(
                    request.segmentation_artifact.geometry.size_ijk[0]
                    * request.segmentation_artifact.geometry.size_ijk[1]
                    * request.segmentation_artifact.geometry.size_ijk[2]
                ),
                is_discrete=True,
                elapsed_ms=1.0,
            ),
        )

        response = self.adapter.validate_segmentation_result(request)

        self.assertEqual(ErrorCode.LABEL_SCHEMA_ERROR, response.error.code)


class ExtractSkinSurfaceAdapterTests(SegmentationAdapterTestCase):
    def test_anisotropic_physical_thickness_geometry_lineage_and_cleanup(self) -> None:
        geo = VolumeGeometry(
            size_ijk=(80, 70, 60),
            spacing_mm=(0.7, 1.2, 2.5),
            origin_mm=(1.0, 2.0, 3.0),
            direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            coordinate_system=geometry().coordinate_system,
        )
        ref = artifact(
            ArtifactType.SEGMENTATION_MASK,
            "skin-anisotropic",
            volume_geometry=geo,
        )
        skin_stat = _stats("skin", 1, 20_000, volume_geometry=geo, components=4, border=True)
        self.backend.register_manifest(
            _manifest(
                ref,
                values=(0, 1),
                statistics=(skin_stat,),
                surface_voxels=1_234,
            )
        )
        request = skin_request(skin_mask_artifact=ref, thickness_mm=2.0)

        response = self.adapter.extract_skin_surface(request)

        self.assertTrue(response.ok)
        self.assertEqual((2.1, 2.4, 2.5), response.result.effective_thickness_mm)
        self.assertEqual(20_000, response.result.source_voxel_count)
        self.assertEqual(1_234, response.result.surface_voxel_count)
        self.assertEqual(3, response.result.components_removed)
        output = response.result.surface_artifact
        self.assertEqual((ref.artifact_id,), output.parent_artifact_ids)
        self.assertEqual(geo.geometry_fingerprint, output.geometry.geometry_fingerprint)
        self.assertEqual(ArtifactType.SKIN_SURFACE_MASK, output.artifact_type)
        output_manifest = self.backend.committed_manifest(output.artifact_id)
        self.assertEqual("uint8", output_manifest.dtype)
        self.assertEqual((0, 1), output_manifest.observed_label_values)

    def test_component_cleanup_can_be_disabled(self) -> None:
        request = skin_request(keep_largest_component=False)
        ref = request.skin_mask_artifact
        skin_stat = _stats("skin", 1, 10_000, volume_geometry=ref.geometry, components=4, border=True)
        self.backend.register_manifest(
            _manifest(ref, statistics=(skin_stat,), surface_voxels=1_000)
        )

        response = self.adapter.extract_skin_surface(request)

        self.assertEqual(0, response.result.components_removed)

    def test_same_request_is_idempotent_and_commits_once(self) -> None:
        request = skin_request()

        first = self.adapter.extract_skin_surface(request)
        second = self.adapter.extract_skin_surface(request)

        self.assertEqual(first.result, second.result)
        self.assertEqual(1, self.backend.commit_count)
        self.assertEqual("idempotency_replay", self.traces.events[-1].outcome)

    def test_empty_or_fractional_skin_mask_is_rejected(self) -> None:
        empty_request = skin_request()
        empty_ref = empty_request.skin_mask_artifact
        self.backend.register_manifest(
            _manifest(empty_ref, values=(0,), statistics=(), primary_label_value=1)
        )
        empty = self.adapter.extract_skin_surface(empty_request)
        self.assertEqual(ErrorCode.EMPTY_SEGMENTATION, empty.error.code)

        backend = DeterministicSegmentationBackend()
        adapter = SegmentationToolAdapter(engine=backend, algorithms=backend, artifacts=backend)
        fractional_request = skin_request()
        backend.register_manifest(
            _manifest(fractional_request.skin_mask_artifact, dtype="float32")
        )
        fractional = adapter.extract_skin_surface(fractional_request)
        self.assertEqual(ErrorCode.LABEL_SCHEMA_ERROR, fractional.error.code)

    def test_one_voxel_thick_source_may_equal_its_surface(self) -> None:
        request = skin_request()
        ref = request.skin_mask_artifact
        one = _stats("skin", 1, 1, volume_geometry=ref.geometry, border=True)
        self.backend.register_manifest(
            _manifest(ref, statistics=(one,), surface_voxels=1)
        )

        response = self.adapter.extract_skin_surface(request)

        self.assertTrue(response.ok)
        self.assertEqual(1, response.result.source_voxel_count)
        self.assertEqual(1, response.result.surface_voxel_count)

    def test_malformed_native_outputs_fail_quality_checks(self) -> None:
        request = skin_request()
        ref = request.skin_mask_artifact
        cases = (
            SkinSurfaceKernelResult(
                geometry=ref.geometry,
                source_voxel_count=100,
                surface_voxel_count=101,
                effective_thickness_mm=(2.0, 2.0, 3.0),
                components_removed=0,
            ),
            SkinSurfaceKernelResult(
                geometry=geometry(origin_x=3.0),
                source_voxel_count=100,
                surface_voxel_count=20,
                effective_thickness_mm=(2.0, 2.0, 3.0),
                components_removed=0,
            ),
            SkinSurfaceKernelResult(
                geometry=ref.geometry,
                source_voxel_count=100,
                surface_voxel_count=20,
                effective_thickness_mm=(2.0, 2.0, 3.0),
                components_removed=0,
                binary_values=(0, 1, 2),
            ),
            SkinSurfaceKernelResult(
                geometry=ref.geometry,
                source_voxel_count=100,
                surface_voxel_count=20,
                effective_thickness_mm=(2.0, 2.0, 3.0),
                components_removed=0,
                subset_of_source=False,
            ),
            SkinSurfaceKernelResult(
                geometry=ref.geometry,
                source_voxel_count=100,
                surface_voxel_count=20,
                effective_thickness_mm=(2.0, 2.0, 3.0),
                components_removed=0,
                external_surface_only=False,
            ),
        )
        for index, output in enumerate(cases):
            backend = DeterministicSegmentationBackend()
            adapter = SegmentationToolAdapter(engine=backend, algorithms=backend, artifacts=backend)
            current = replace(
                request,
                context=self._new_context(
                    request.context,
                    request_id=f"req-malformed-{index}",
                    trace_id=f"trace-malformed-{index}",
                    idempotency_key=f"idem-malformed-{index}",
                ),
            )
            backend.register_manifest(
                _manifest(
                    ref,
                    statistics=(
                        _stats(
                            "skin",
                            1,
                            100,
                            volume_geometry=ref.geometry,
                            border=True,
                        ),
                    ),
                )
            )
            backend.set_skin_override(ref.artifact_id, output)
            with self.subTest(index=index):
                response = adapter.extract_skin_surface(current)
                self.assertEqual(ErrorCode.QUALITY_CHECK_FAILED, response.error.code)
                self.assertEqual(0, backend.commit_count)

    def test_retryable_native_timeout_and_permission_failure_are_mapped(self) -> None:
        request = skin_request()
        self.backend.fail_next(
            "extract_skin_surface",
            SegmentationBackendError(
                ErrorCode.TIMEOUT,
                "native morphology timed out",
                retryable=True,
                dependency="image_algorithms",
            ),
        )
        timeout = self.adapter.extract_skin_surface(request)
        self.assertEqual(ErrorCode.TIMEOUT, timeout.error.code)
        self.assertTrue(timeout.error.retryable)
        self.assertEqual(0, self.backend.commit_count)

        backend = DeterministicSegmentationBackend()
        traces = InMemorySegmentationTraceSink()
        adapter = SegmentationToolAdapter(
            engine=backend, algorithms=backend, artifacts=backend, trace_sink=traces
        )
        backend.set_artifact_permissions(request.skin_mask_artifact.artifact_id, {"other"})
        denied = adapter.extract_skin_surface(request)
        self.assertEqual(ErrorCode.PERMISSION_DENIED, denied.error.code)
        self.assertEqual(0, backend.commit_count)


class TraceAndFailureSafetyTests(SegmentationAdapterTestCase):
    def test_builder_returns_adapter_and_read_only_three_handler_mapping(self) -> None:
        adapter, handlers = build_segmentation_handlers(
            self.backend,
            trace_sink=self.traces,
            clock=lambda: FIXED_NOW,
        )

        self.assertIsInstance(adapter, SegmentationToolAdapter)
        self.assertEqual(
            {
                "run_segmentation",
                "validate_segmentation_result",
                "extract_skin_surface",
            },
            set(handlers),
        )
        self.assertTrue(handlers["run_segmentation"](segmentation_request()).ok)
        with self.assertRaises(TypeError):
            handlers["new_tool"] = adapter.run_segmentation

    def test_trace_and_errors_never_include_artifact_uri_or_checksum(self) -> None:
        request = segmentation_request()
        backend_message = (
            f"reader failed for {request.ct_artifact.uri} "
            f"with checksum {request.ct_artifact.checksum_sha256}"
        )
        self.backend.fail_next(
            "resolve",
            SegmentationBackendError(
                ErrorCode.DEPENDENCY_FAILED,
                backend_message,
                retryable=True,
                dependency="artifact_store",
                details={
                    "uri": request.ct_artifact.uri,
                    "safe_operation": "resolve",
                },
            ),
        )

        response = self.adapter.run_segmentation(request)

        rendered_response = json.dumps(to_primitive(response))
        rendered_trace = json.dumps([to_primitive(item) for item in self.traces.events])
        for secret in (request.ct_artifact.uri, request.ct_artifact.checksum_sha256):
            self.assertNotIn(secret, rendered_response)
            self.assertNotIn(secret, rendered_trace)
        self.assertIn("[redacted]", response.error.message)
        self.assertNotIn("uri", response.error.details)
        self.assertEqual("resolve", response.error.details["safe_operation"])

    def test_success_trace_contains_model_profile_but_no_storage_location(self) -> None:
        request = segmentation_request()
        response = self.adapter.run_segmentation(request)

        event = self.traces.events[-1]
        self.assertEqual(response.result.model_id, event.attributes["model_id"])
        self.assertEqual(response.result.model_version, event.attributes["model_version"])
        self.assertEqual(response.result.precision.value, event.attributes["precision"])
        self.assertFalse(any("uri" in key.lower() for key in event.attributes))
        self.assertFalse(any("checksum" in key.lower() for key in event.attributes))


if __name__ == "__main__":
    unittest.main()
