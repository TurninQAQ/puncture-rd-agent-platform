"""Offline contract, safety, idempotency, and failure tests for planning adapters."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from contracts.common import to_json  # noqa: E402
from contracts.domain import DangerMaskSpec, SafetyMargin  # noqa: E402
from contracts.enums import (  # noqa: E402
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
from contracts.geometry import VolumeGeometry, WorldPoint  # noqa: E402
from puncture_agent.tooling.catalog import TOOL_DEFINITIONS  # noqa: E402
from puncture_agent.tooling.planning import (  # noqa: E402
    DeterministicPlanningBackend,
    InMemoryPlanningTraceSink,
    NativeCandidate,
    NativeCandidateBatch,
    NativeClearance,
    NativePathAssessment,
    NativeRiskFlag,
    NativeRiskState,
    NativeSafetyBatch,
    PlanningBackendInvalidArgument,
    PlanningBackendNoCandidate,
    PlanningBackendPenetrationUndetermined,
    PlanningBackendRiskFailure,
    PlanningBackendSafetyFailure,
    PlanningBackendTimeout,
    PlanningBackendUnavailable,
    PlanningKernelManifest,
    PlanningToolAdapters,
    generate_candidate_paths as local_generate_candidate_paths,
)
from puncture_agent.tooling.registry import ToolRegistry  # noqa: E402
from tools.helpers import (  # noqa: E402
    artifact,
    candidate_request,
    danger_specs,
    geometry,
    penetration_request,
    risk_request,
    safety_request,
)


FIXED_TIME = "2026-07-11T00:00:00Z"


def build_adapter(backend=None, *, trace_sink=None, epoch_ms=None):
    backend = backend or DeterministicPlanningBackend()
    sink = trace_sink or InMemoryPlanningTraceSink()
    adapter = PlanningToolAdapters(
        backend,
        trace_sink=sink,
        clock=lambda: FIXED_TIME,
        epoch_ms=epoch_ms or (lambda: 1_700_000_000_000),
    )
    return adapter, backend, sink


def build_registry(adapter: PlanningToolAdapters) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in adapter.handlers().items():
        registry.register(TOOL_DEFINITIONS[name], handler)
    return registry


def request_with_context(request, *, request_id=None, trace_id=None, idempotency_key=None, caller=None, deadline=None):
    context = replace(
        request.context,
        request_id=request_id or request.context.request_id,
        trace_id=trace_id or request.context.trace_id,
        idempotency_key=idempotency_key or request.context.idempotency_key,
        caller=caller or request.context.caller,
        deadline_epoch_ms=deadline,
    )
    return replace(request, context=context)


def danger_with_large_vessel():
    return (
        *danger_specs(),
        DangerMaskSpec(
            structure=RiskStructure.LARGE_VESSEL,
            artifact=artifact(ArtifactType.DANGER_MASK, "large-vessel"),
            safety_margin=SafetyMargin(warning_mm=6.0, stop_mm=2.0),
            vessel_core_erosion_mm=1.5,
        ),
    )


class CapturingBackend(DeterministicPlanningBackend):
    def __init__(self):
        super().__init__()
        self.last_generate_command = None

    def generate(self, command):
        self.last_generate_command = command
        return super().generate(command)


class FailingTraceSink:
    def record(self, trace):
        del trace
        raise RuntimeError("telemetry unavailable")


class PlanningAdapterTests(unittest.TestCase):
    def test_trace_sink_failure_does_not_change_planning_result(self) -> None:
        adapter, _, _ = build_adapter(trace_sink=FailingTraceSink())
        response = adapter.generate_candidate_paths(candidate_request(max_candidates=1))
        self.assertTrue(response.ok, response.error)
        self.assertEqual("path-001", response.result.candidates[0].candidate_id)

    def test_catalog_handlers_generate_typed_json_and_sanitized_backend_command(self) -> None:
        backend = CapturingBackend()
        adapter, _, sink = build_adapter(backend)
        registry = build_registry(adapter)

        response = registry.execute("generate_candidate_paths", candidate_request(max_candidates=2))

        self.assertEqual(ToolExecutionStatus.SUCCESS, response.status)
        self.assertEqual("1.0.0", response.tool_version)
        self.assertEqual(["path-001", "path-002"], [item.candidate_id for item in response.result.candidates])
        self.assertEqual([1, 2], [item.rank_hint for item in response.result.candidates])
        self.assertEqual("trace-001", response.trace_id)
        self.assertEqual("planning_time", response.metrics[0].name)
        serialized = to_json(response)
        self.assertNotIn("mock://", serialized)
        self.assertNotIn("mock_", serialized)
        command = backend.last_generate_command
        self.assertEqual("case-001-ct", command.ct.artifact_id)
        self.assertFalse(hasattr(command.ct, "uri"))
        self.assertFalse(hasattr(command.ct, "metadata"))
        self.assertEqual("generate_candidate_paths", sink.records[-1].tool_name)
        self.assertEqual({"candidate_count": 2}, dict(sink.records[-1].result_summary))
        local_response = local_generate_candidate_paths(
            request_with_context(candidate_request(max_candidates=1), idempotency_key="module-local")
        )
        self.assertTrue(local_response.ok, local_response.error)
        self.assertEqual(1, len(local_response.result.candidates))

    def test_rotated_anisotropic_world_coordinates_are_validated_without_identity_assumption(self) -> None:
        rotated = VolumeGeometry(
            size_ijk=(16, 16, 16),
            spacing_mm=(2.0, 3.0, 4.0),
            origin_mm=(10.0, 20.0, 30.0),
            direction_cosines=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            coordinate_system=CoordinateSystem.LPS,
        )

        def world(i, j, k):
            return WorldPoint(10.0 - j * 3.0, 20.0 + i * 2.0, 30.0 + k * 4.0)

        entry = world(1, 2, 3)
        target = world(4, 5, 6)
        length = math.dist(entry.as_tuple(), target.as_tuple())
        batch = NativeCandidateBatch(
            candidates=(
                NativeCandidate(
                    "rotated-path-001",
                    entry,
                    target,
                    length,
                    12.0,
                    "LOCAL_SURFACE_NORMAL",
                    1,
                ),
            ),
            sampled_entry_point_count=1,
            rejected_by_length_count=0,
            rejected_by_angle_count=0,
            elapsed_ms=1.0,
        )
        adapter, _, _ = build_adapter(DeterministicPlanningBackend(candidate_batch=batch))
        request = candidate_request(
            ct_artifact=artifact(ArtifactType.CT_VOLUME, "rotated-ct", volume_geometry=rotated),
            skin_surface_artifact=artifact(
                ArtifactType.SKIN_SURFACE_MASK, "rotated-skin", volume_geometry=rotated
            ),
            target_artifact=artifact(ArtifactType.TARGET_MASK, "rotated-target", volume_geometry=rotated),
            target_point_world_mm=target,
            max_needle_length_mm=100.0,
            max_candidates=1,
        )
        response = adapter.generate_candidate_paths(request)
        self.assertTrue(response.ok, response.error)
        self.assertEqual(entry, response.result.candidates[0].entry_point_world_mm)
        self.assertAlmostEqual(length, response.result.candidates[0].length_mm)

    def test_geometry_mismatch_and_out_of_bounds_target_fail_before_backend(self) -> None:
        adapter, backend, _ = build_adapter()
        mismatch = candidate_request(
            target_artifact=artifact(
                ArtifactType.TARGET_MASK,
                "shifted-target",
                volume_geometry=geometry(origin_x=2.0),
            )
        )
        mismatch_response = adapter.generate_candidate_paths(mismatch)
        self.assertEqual(ErrorCode.GEOMETRY_MISMATCH, mismatch_response.error.code)

        outside = request_with_context(
            candidate_request(target_point_world_mm=WorldPoint(999.0, 999.0, 999.0)),
            idempotency_key="outside-target",
        )
        outside_response = adapter.generate_candidate_paths(outside)
        self.assertEqual(ErrorCode.TARGET_OUT_OF_BOUNDS, outside_response.error.code)
        self.assertEqual(0, backend.call_counts["generate"])

    def test_permission_and_expired_deadline_fail_closed_and_are_traced(self) -> None:
        manifest = PlanningKernelManifest(
            "restricted-kernel",
            "v1",
            ("planner-v1",),
            ("risk-v1",),
            ("approved-service",),
        )
        adapter, backend, sink = build_adapter(DeterministicPlanningBackend(manifest=manifest))
        denied = adapter.generate_candidate_paths(candidate_request())
        self.assertEqual(ErrorCode.PERMISSION_DENIED, denied.error.code)
        self.assertEqual(0, backend.call_counts["generate"])
        self.assertEqual(ErrorCode.PERMISSION_DENIED, sink.records[-1].error_code)

        allowed_manifest = replace(manifest, allowed_callers=("unit-test",))
        deadline_adapter, deadline_backend, deadline_sink = build_adapter(
            DeterministicPlanningBackend(manifest=allowed_manifest),
            epoch_ms=lambda: 2000,
        )
        expired = request_with_context(candidate_request(), deadline=2000)
        response = deadline_adapter.generate_candidate_paths(expired)
        self.assertEqual(ErrorCode.TIMEOUT, response.error.code)
        self.assertTrue(response.error.retryable)
        self.assertEqual(0, deadline_backend.call_counts["generate"])
        self.assertEqual("trace-001", deadline_sink.records[-1].trace_id)

    def test_idempotent_retry_reuses_backend_result_with_new_trace(self) -> None:
        adapter, backend, sink = build_adapter()
        first_request = candidate_request(max_candidates=2)
        second_request = request_with_context(
            first_request,
            request_id="req-retry",
            trace_id="trace-retry",
        )
        first = adapter.generate_candidate_paths(first_request)
        second = adapter.generate_candidate_paths(second_request)
        self.assertEqual(1, backend.call_counts["generate"])
        self.assertEqual(first.result, second.result)
        self.assertEqual("req-retry", second.request_id)
        self.assertEqual("trace-retry", second.trace_id)
        self.assertFalse(sink.records[-2].idempotent_replay)
        self.assertTrue(sink.records[-1].idempotent_replay)
        self.assertEqual(64, len(sink.records[-1].idempotency_key_sha256))

    def test_same_idempotency_key_with_changed_payload_is_rejected(self) -> None:
        adapter, backend, sink = build_adapter()
        first = adapter.generate_candidate_paths(candidate_request(max_candidates=2))
        changed = adapter.generate_candidate_paths(candidate_request(max_candidates=3))
        self.assertTrue(first.ok)
        self.assertEqual(ErrorCode.CONTRACT_VIOLATION, changed.error.code)
        self.assertEqual(1, backend.call_counts["generate"])
        self.assertEqual(ErrorCode.CONTRACT_VIOLATION, sink.records[-1].error_code)

    def test_concurrent_idempotent_calls_execute_backend_once(self) -> None:
        adapter, backend, sink = build_adapter()
        request = candidate_request(max_candidates=2)
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = tuple(executor.map(lambda _: adapter.generate_candidate_paths(request), range(4)))
        self.assertTrue(all(response.ok for response in responses))
        self.assertTrue(all(response.result == responses[0].result for response in responses))
        self.assertEqual(1, backend.call_counts["generate"])
        self.assertEqual(3, sum(record.idempotent_replay for record in sink.records))

    def test_empty_candidate_and_backend_failure_mapping(self) -> None:
        empty = NativeCandidateBatch((), 5, 5, 0, 1.0)
        adapter, _, _ = build_adapter(DeterministicPlanningBackend(candidate_batch=empty))
        response = adapter.generate_candidate_paths(candidate_request())
        self.assertEqual(ErrorCode.NO_CANDIDATE_PATH, response.error.code)

        cases = (
            (PlanningBackendTimeout("timeout"), ErrorCode.TIMEOUT, True),
            (PlanningBackendUnavailable("down"), ErrorCode.DEPENDENCY_FAILED, True),
            (PlanningBackendInvalidArgument("bad"), ErrorCode.INVALID_ARGUMENT, False),
            (PlanningBackendNoCandidate("none"), ErrorCode.NO_CANDIDATE_PATH, False),
            (RuntimeError("native crash with secret detail"), ErrorCode.DEPENDENCY_FAILED, False),
        )
        for index, (failure, code, retryable) in enumerate(cases):
            backend = DeterministicPlanningBackend(failures={"generate": failure})
            failed_adapter, _, _ = build_adapter(backend)
            request = request_with_context(candidate_request(), idempotency_key=f"failure-{index}")
            failed = failed_adapter.generate_candidate_paths(request)
            with self.subTest(code=code):
                self.assertEqual(code, failed.error.code)
                self.assertEqual(retryable, failed.error.retryable)
                self.assertNotIn("secret detail", failed.error.message)

    def test_safety_stop_warning_policy_and_all_rejected_are_explicit(self) -> None:
        clearances = {
            ("path-001", RiskStructure.HEART): 4.0,
            ("path-001", RiskStructure.BONE): 10.0,
            ("path-002", RiskStructure.HEART): 2.0,
            ("path-002", RiskStructure.BONE): 10.0,
        }
        adapter, _, sink = build_adapter(
            DeterministicPlanningBackend(safety_clearances=clearances)
        )
        response = adapter.evaluate_path_safety(safety_request())
        by_id = {item.candidate_id: item for item in response.result.assessments}
        self.assertEqual(PathDisposition.ACCEPTED_WITH_WARNING, by_id["path-001"].disposition)
        self.assertEqual(PathDisposition.REJECTED, by_id["path-002"].disposition)
        self.assertEqual(("path-001",), response.result.accepted_candidate_ids)
        self.assertEqual(("path-002",), response.result.rejected_candidate_ids)
        self.assertEqual("path-001", response.result.safest_candidate_id)
        self.assertEqual(("PATH_WARNING_INTERSECTION",), response.warnings)
        self.assertEqual(1, sink.records[-1].result_summary["accepted_count"])

        rejecting_adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(safety_clearances=clearances)
        )
        all_rejected = rejecting_adapter.evaluate_path_safety(
            safety_request(reject_warning_intersection=True)
        )
        self.assertEqual((), all_rejected.result.accepted_candidate_ids)
        self.assertEqual(("path-001", "path-002"), all_rejected.result.rejected_candidate_ids)
        self.assertIsNone(all_rejected.result.safest_candidate_id)

    def test_safety_boundary_downgrade_and_required_mask_failure_map_safely(self) -> None:
        corrupt = NativeSafetyBatch(
            assessments=(
                NativePathAssessment(
                    "path-001",
                    (
                        NativeClearance(RiskStructure.HEART, 3.0, False, True),
                        NativeClearance(RiskStructure.BONE, 10.0, False, False),
                    ),
                ),
                NativePathAssessment(
                    "path-002",
                    (
                        NativeClearance(RiskStructure.HEART, 10.0, False, False),
                        NativeClearance(RiskStructure.BONE, 10.0, False, False),
                    ),
                ),
            ),
            elapsed_ms=1.0,
        )
        adapter, _, _ = build_adapter(DeterministicPlanningBackend(safety_batch=corrupt))
        response = adapter.evaluate_path_safety(safety_request())
        self.assertEqual(ErrorCode.SAFETY_CHECK_FAILED, response.error.code)

        missing = danger_specs()
        missing_heart = replace(
            missing[0],
            artifact=artifact(
                ArtifactType.DANGER_MASK,
                "missing-heart",
                status=ArtifactStatus.MISSING,
            ),
        )
        missing_adapter, missing_backend, _ = build_adapter()
        missing_response = missing_adapter.evaluate_path_safety(
            safety_request(danger_masks=(missing_heart, missing[1]))
        )
        self.assertEqual(ErrorCode.REQUIRED_DANGER_MASK_MISSING, missing_response.error.code)
        self.assertEqual(0, missing_backend.call_counts["path_clearance"])

        optional = DangerMaskSpec(
            structure=RiskStructure.OTHER,
            artifact=artifact(
                ArtifactType.DANGER_MASK,
                "optional-missing",
                status=ArtifactStatus.MISSING,
            ),
            safety_margin=SafetyMargin(warning_mm=4.0, stop_mm=1.0),
            required=False,
        )
        optional_adapter, optional_backend, _ = build_adapter()
        optional_response = optional_adapter.evaluate_path_safety(
            request_with_context(
                safety_request(danger_masks=(*danger_specs(), optional)),
                idempotency_key="optional-danger",
            )
        )
        self.assertTrue(optional_response.ok, optional_response.error)
        self.assertIn("OPTIONAL_OTHER_DANGER_MASK_UNAVAILABLE", optional_response.warnings)
        self.assertEqual(1, optional_backend.call_counts["path_clearance"])

    def test_safety_backend_exception_has_operation_specific_error(self) -> None:
        backend = DeterministicPlanningBackend(
            failures={"path_clearance": PlanningBackendSafetyFailure("distance cache corrupt")}
        )
        adapter, _, sink = build_adapter(backend)
        response = adapter.evaluate_path_safety(safety_request())
        self.assertEqual(ErrorCode.SAFETY_CHECK_FAILED, response.error.code)
        self.assertEqual(ErrorCode.SAFETY_CHECK_FAILED, sink.records[-1].error_code)

    def test_risk_precedence_warning_stop_vessel_and_lung_are_preserved(self) -> None:
        backend = DeterministicPlanningBackend(
            risk_distances={
                RiskStructure.HEART: 6.0,
                RiskStructure.BONE: 12.0,
                RiskStructure.LARGE_VESSEL: 12.0,
            },
            large_vessel_penetration=True,
            needle_in_lung=True,
        )
        adapter, _, sink = build_adapter(backend)
        response = adapter.evaluate_intraoperative_risk(
            risk_request(danger_masks=danger_with_large_vessel())
        )
        self.assertEqual(RiskLevel.STOP, response.result.overall_level)
        self.assertTrue(response.result.requires_manual_review)
        self.assertTrue(response.result.needle_in_lung)
        self.assertTrue(response.result.large_vessel_penetration)
        levels = {flag.structure: flag.level for flag in response.result.flags}
        self.assertEqual(RiskLevel.WARNING, levels[RiskStructure.HEART])
        self.assertEqual(RiskLevel.SAFE, levels[RiskStructure.BONE])
        self.assertEqual(RiskLevel.STOP, levels[RiskStructure.LARGE_VESSEL])
        self.assertEqual("STOP", sink.records[-1].result_summary["overall_level"])

    def test_optional_lung_unavailable_is_unknown_and_requires_review(self) -> None:
        adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(
                risk_distances={RiskStructure.HEART: 12.0, RiskStructure.BONE: 12.0},
                needle_in_lung=True,
            )
        )
        request = risk_request(
            lung_mask_artifact=artifact(
                ArtifactType.SEGMENTATION_MASK,
                "missing-lung",
                status=ArtifactStatus.MISSING,
            )
        )
        response = adapter.evaluate_intraoperative_risk(request)
        self.assertTrue(response.ok, response.error)
        self.assertEqual(RiskLevel.SAFE, response.result.overall_level)
        self.assertIsNone(response.result.needle_in_lung)
        self.assertTrue(response.result.requires_manual_review)
        self.assertIn("LUNG_MASK_ARTIFACT_UNAVAILABLE", response.warnings)

    def test_risk_stop_downgrade_missing_mask_and_backend_failure_fail_closed(self) -> None:
        flags = (
            NativeRiskFlag(
                RiskStructure.HEART,
                RiskLevel.SAFE,
                "HEART_SAFE",
                "invalid downgrade",
                1.0,
                ("case-001-heart",),
            ),
            NativeRiskFlag(
                RiskStructure.BONE,
                RiskLevel.SAFE,
                "BONE_SAFE",
                "safe",
                12.0,
                ("case-001-bone",),
            ),
        )
        corrupt_adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(risk_state=NativeRiskState(flags, True, None, 1.0))
        )
        corrupt = corrupt_adapter.evaluate_intraoperative_risk(risk_request())
        self.assertEqual(ErrorCode.RISK_EVALUATION_FAILED, corrupt.error.code)

        specs = danger_specs()
        missing = replace(
            specs[0],
            artifact=artifact(ArtifactType.DANGER_MASK, "missing-heart", status=ArtifactStatus.MISSING),
        )
        missing_adapter, backend, _ = build_adapter()
        response = missing_adapter.evaluate_intraoperative_risk(
            risk_request(danger_masks=(missing, specs[1]))
        )
        self.assertEqual(ErrorCode.REQUIRED_DANGER_MASK_MISSING, response.error.code)
        self.assertEqual(0, backend.call_counts["tip_risk"])

        failure_adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(failures={"tip_risk": PlanningBackendRiskFailure("bad EDT")})
        )
        failed = failure_adapter.evaluate_intraoperative_risk(risk_request())
        self.assertEqual(ErrorCode.RISK_EVALUATION_FAILED, failed.error.code)

    def test_penetration_crossing_not_crossed_and_slip_threshold_equality(self) -> None:
        penetrated_adapter, _, _ = build_adapter(DeterministicPlanningBackend(crossed_skin=True))
        penetrated = penetrated_adapter.verify_skin_penetration(penetration_request())
        self.assertEqual(PenetrationStatus.PENETRATED, penetrated.result.status)
        self.assertTrue(penetrated.result.crossed_skin)
        self.assertIsNotNone(penetrated.result.crossing_point_world_mm)

        not_adapter, _, _ = build_adapter(DeterministicPlanningBackend(crossed_skin=False))
        not_penetrated = not_adapter.verify_skin_penetration(
            penetration_request(insertion_depth_mm=4.999, min_depth_for_slip_mm=5.0)
        )
        self.assertEqual(PenetrationStatus.NOT_PENETRATED, not_penetrated.result.status)

        slip_adapter, _, sink = build_adapter(DeterministicPlanningBackend(crossed_skin=False))
        slipped = slip_adapter.verify_skin_penetration(
            penetration_request(insertion_depth_mm=5.0, min_depth_for_slip_mm=5.0)
        )
        self.assertEqual(PenetrationStatus.SUSPECTED_SLIP, slipped.result.status)
        self.assertEqual(("SUSPECTED_SKIN_SLIP",), slipped.warnings)
        self.assertEqual("SUSPECTED_SLIP", sink.records[-1].result_summary["penetration_status"])

    def test_penetration_missing_label_outside_segment_and_backend_failure_are_mapped(self) -> None:
        label_adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(crossed_skin=False, skin_label_present=False)
        )
        missing_label = label_adapter.verify_skin_penetration(penetration_request())
        self.assertEqual(ErrorCode.REQUIRED_LABEL_MISSING, missing_label.error.code)

        outside_adapter, outside_backend, _ = build_adapter()
        outside = outside_adapter.verify_skin_penetration(
            penetration_request(current_tip_world_mm=WorldPoint(999.0, 999.0, 999.0))
        )
        self.assertEqual(ErrorCode.SKIN_PENETRATION_UNDETERMINED, outside.error.code)
        self.assertEqual(0, outside_backend.call_counts["traverse_skin"])

        zero_adapter, _, _ = build_adapter()
        zero = zero_adapter.verify_skin_penetration(
            penetration_request(current_tip_world_mm=WorldPoint(20.0, 40.0, 10.0))
        )
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, zero.error.code)

        failure_adapter, _, _ = build_adapter(
            DeterministicPlanningBackend(
                failures={"traverse_skin": PlanningBackendPenetrationUndetermined("ambiguous")}
            )
        )
        failed = failure_adapter.verify_skin_penetration(penetration_request())
        self.assertEqual(ErrorCode.SKIN_PENETRATION_UNDETERMINED, failed.error.code)

    def test_unsupported_manifest_versions_are_rejected_without_backend_execution(self) -> None:
        adapter, backend, _ = build_adapter()
        planner = adapter.generate_candidate_paths(candidate_request(planner_version="unknown-planner"))
        risk = adapter.evaluate_intraoperative_risk(
            request_with_context(
                risk_request(risk_rule_version="unknown-risk"),
                idempotency_key="unknown-risk-idem",
            )
        )
        device_axis = adapter.generate_candidate_paths(
            request_with_context(
                candidate_request(angle_reference=AngleReference.NEEDLE_DEVICE_AXIS),
                idempotency_key="uncalibrated-device-axis",
            )
        )
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, planner.error.code)
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, risk.error.code)
        self.assertEqual(ErrorCode.INVALID_ARGUMENT, device_axis.error.code)
        self.assertEqual(0, backend.call_counts["generate"])
        self.assertEqual(0, backend.call_counts["tip_risk"])


if __name__ == "__main__":
    unittest.main()
