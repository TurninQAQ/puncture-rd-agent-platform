from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from contracts.artifacts import ArtifactRef  # noqa: E402
from contracts.enums import ArtifactStatus, ArtifactType, CoordinateSystem  # noqa: E402
from contracts.geometry import VolumeGeometry  # noqa: E402
from contracts.tool_inputs import TOOL_REQUEST_TYPES  # noqa: E402
from puncture_agent.agent.artifact_validation import (  # noqa: E402
    RegistryToolArtifactValidator,
)
from puncture_agent.agent.nodes import build_mock_handlers  # noqa: E402
from puncture_agent.agent.runtime import NodeContext  # noqa: E402
from puncture_agent.agent.state import AgentState  # noqa: E402
from puncture_agent.agent.tool_bridge import (  # noqa: E402
    DEFAULT_TOOL_BRIDGE_POLICY,
    McpToolExecutor,
    ToolBridgeContextError,
    ToolBridgeContractError,
    ToolBridgeResponseError,
    ToolBridgeTransportError,
)
from puncture_agent.artifacts import (  # noqa: E402
    ArtifactRegistryError,
    InMemoryArtifactRegistry,
)
from puncture_agent.mcp import (  # noqa: E402
    InMemoryArtifactResolver,
    McpPrincipal,
    McpToolRuntime,
    decode_tool_request,
)
from puncture_agent.tooling import build_mock_registry  # noqa: E402
from puncture_agent.tooling.catalog import TOOL_DEFINITIONS  # noqa: E402


FIXED_NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
CASE_ID = "case-001"


def geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0, 1.0, 1.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        coordinate_system=CoordinateSystem.LPS,
    )


def artifact(artifact_id: str, artifact_type: ArtifactType) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id=CASE_ID,
        artifact_type=artifact_type,
        uri=f"mock://private/{artifact_id}",
        checksum_sha256="a" * 64,
        status=ArtifactStatus.AVAILABLE,
        geometry=geometry(),
        producer_name="tool-bridge-test",
        producer_version="1",
    )


def register_artifact(
    registry: InMemoryArtifactRegistry,
    value: ArtifactRef,
    *,
    producer_name: str | None = None,
    producer_version: str | None = None,
    parent_artifact_ids: tuple[str, ...] = (),
) -> None:
    registry.begin_registration(
        case_id=value.case_id,
        artifact_type=value.artifact_type,
        internal_uri=value.uri,
        created_by="tool-bridge-test",
        idempotency_key=f"seed:{value.artifact_id}",
        producer_name=producer_name or value.producer_name,
        producer_version=producer_version or value.producer_version,
        parent_artifact_ids=parent_artifact_ids,
        geometry=value.geometry,
        artifact_id=value.artifact_id,
    )
    registry.finalize(value.artifact_id, value.checksum_sha256, 1)


class CapturingCaller:
    def __init__(self, delegate: McpToolRuntime) -> None:
        self.delegate = delegate
        self.tool_names = delegate.tool_names
        self.calls: list[tuple[str, dict[str, Any], McpPrincipal]] = []

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> Any:
        copied = json.loads(json.dumps(arguments))
        self.calls.append((name, copied, principal))
        return self.delegate.call_tool(name, arguments, principal=principal)


class StaticCaller:
    tool_names = ("inspect_case_metadata",)

    def __init__(self, structured_content: Mapping[str, Any]) -> None:
        self.structured_content = dict(structured_content)
        self.calls: list[tuple[str, Mapping[str, Any], McpPrincipal]] = []

    def call_tool(self, name, arguments, *, principal):
        self.calls.append((name, arguments, principal))
        payload = dict(self.structured_content)
        payload.setdefault("request_id", arguments["context"]["request_id"])
        payload.setdefault("trace_id", arguments["context"]["trace_id"])
        payload.setdefault("tool_name", name)
        payload.setdefault("tool_version", "1.0.0")
        payload.setdefault("artifacts", [])
        payload.setdefault("metrics", [])
        payload.setdefault("warnings", [])
        payload.setdefault("started_at", "2026-07-11T12:00:00Z")
        payload.setdefault("finished_at", "2026-07-11T12:00:00Z")
        return {
            "structured_content": payload,
            "is_error": payload.get("status") == "FAILED",
        }


class NeverCalledCaller:
    tool_names = ("inspect_case_metadata",)

    def __init__(self) -> None:
        self.call_count = 0

    def call_tool(self, name, arguments, *, principal):
        del name, arguments, principal
        self.call_count += 1
        raise AssertionError("build_arguments must not invoke the MCP caller")


class ToolBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ct = artifact("case-001-ct", ArtifactType.CT_VOLUME)
        self.skin_surface = artifact(
            "case-001-skin-surface", ArtifactType.SKIN_SURFACE_MASK
        )
        self.target = artifact("case-001-target", ArtifactType.TARGET_MASK)
        self.segmentation = artifact(
            "case-001-segmentation", ArtifactType.SEGMENTATION_MASK
        )
        self.mcs = artifact("case-001-mcs", ArtifactType.MCS_SEGMENTATION)
        self.labelmap = artifact("case-001-labelmap", ArtifactType.NIFTI_LABELMAP)
        self.skin = artifact("case-001-skin", ArtifactType.SEGMENTATION_MASK)
        self.lung = artifact("case-001-lung", ArtifactType.SEGMENTATION_MASK)
        self.heart = artifact("case-001-heart", ArtifactType.DANGER_MASK)
        self.bone = artifact("case-001-bone", ArtifactType.DANGER_MASK)
        self.bronchus = artifact("case-001-bronchus", ArtifactType.DANGER_MASK)
        self.vessel = artifact("case-001-vessel", ArtifactType.DANGER_MASK)
        self.resolver = InMemoryArtifactResolver(
            (
                self.ct,
                self.skin_surface,
                self.target,
                self.segmentation,
                self.mcs,
                self.labelmap,
                self.skin,
                self.lung,
                self.heart,
                self.bone,
                self.bronchus,
                self.vessel,
            )
        )
        registry = build_mock_registry()
        self.callers = {
            name: CapturingCaller(
                McpToolRuntime(registry, self.resolver, server_name=name)
            )
            for name in ("case-data", "segmentation", "planning-safety")
        }
        self.principal = McpPrincipal("agent-runtime", (CASE_ID,))
        self.executor = McpToolExecutor(
            self.callers,
            principal=self.principal,
            session_id="session-001",
            trace_id="trace-agent-001",
            clock=lambda: FIXED_NOW,
        )

    def test_inspect_request_maps_to_opaque_frozen_wire_schema(self) -> None:
        response = self.executor.execute(
            "inspect_case_metadata",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "input_format": "NIFTI",
            },
        )

        self.assertEqual("SUCCESS", response["status"])
        self.assertIn("checksum_valid", response["result"]["inspections"][0])
        _, arguments, _ = self.callers["case-data"].calls[-1]
        self.assertEqual(
            {"artifact_id": self.ct.artifact_id}, arguments["ct_artifact"]
        )
        self.assertEqual([], arguments["related_artifacts"])
        self.assertTrue(arguments["require_same_geometry"])
        encoded = json.dumps(arguments, sort_keys=True)
        self.assertNotIn("mock://", encoded)
        self.assertNotIn("checksum_sha256", encoded)

    def test_segmentation_request_uses_immutable_model_policy(self) -> None:
        response = self.executor.execute(
            "run_segmentation",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "model_version": "v1",
            },
        )

        self.assertEqual("SUCCESS", response["status"])
        _, arguments, _ = self.callers["segmentation"].calls[-1]
        self.assertEqual("nnunet-puncture", arguments["model_id"])
        self.assertEqual(["skin", "lung", "heart"], arguments["requested_labels"])
        self.assertEqual("FP16", arguments["precision"])
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_TOOL_BRIDGE_POLICY.model_id = "untrusted-model"  # type: ignore[misc]

    def test_planning_request_maps_to_frozen_planner_schema(self) -> None:
        response = self.executor.execute(
            "generate_candidate_paths",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "skin_surface_artifact_id": self.skin_surface.artifact_id,
                "target_artifact_id": self.target.artifact_id,
                "max_needle_length_mm": 120.0,
                "max_insertion_angle_deg": 45.0,
                "safety_radius_mm": 5.0,
                "top_k": 3,
            },
        )

        self.assertEqual("SUCCESS", response["status"])
        self.assertEqual(3, len(response["result"]["candidates"]))
        _, arguments, _ = self.callers["planning-safety"].calls[-1]
        self.assertEqual("planner-v1", arguments["planner_version"])
        self.assertEqual("LOCAL_SURFACE_NORMAL", arguments["angle_reference"])
        self.assertEqual({"artifact_id": self.target.artifact_id}, arguments["target_artifact"])

    def test_same_logical_input_reuses_stable_request_and_idempotency_identity(self) -> None:
        request = {
            "case_id": CASE_ID,
            "ct_artifact_id": self.ct.artifact_id,
            "model_version": "v1",
        }
        self.executor.execute("run_segmentation", request)
        self.executor.execute("run_segmentation", dict(reversed(tuple(request.items()))))

        first = self.callers["segmentation"].calls[-2][1]["context"]
        second = self.callers["segmentation"].calls[-1][1]["context"]
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["idempotency_key"], second["idempotency_key"])
        self.assertEqual(first["trace_id"], second["trace_id"])

    def test_nodes_running_audit_entry_keeps_retry_identity_stable(self) -> None:
        request = {
            "case_id": CASE_ID,
            "ct_artifact_id": self.ct.artifact_id,
            "model_version": "v1",
        }
        state = AgentState(
            user_query="segment",
            session_id="retry-session",
            case_id=CASE_ID,
        )
        state.current_node = "data_model_subgraph.run_segmentation"
        state.tool_calls.append(
            {
                "call_id": "call-0001",
                "tool_name": "run_segmentation",
                "request": dict(request),
                "node_id": state.current_node,
                "status": "RUNNING",
            }
        )
        with self.executor.bind_state(state, trace_id="retry-trace"):
            self.executor.execute("run_segmentation", request)
        first = self.callers["segmentation"].calls[-1][1]["context"]

        state.tool_calls.append(
            {
                "call_id": "call-0002",
                "tool_name": "run_segmentation",
                "request": dict(request),
                "node_id": state.current_node,
                "status": "RUNNING",
            }
        )
        with self.executor.bind_state(state, trace_id="retry-trace"):
            self.executor.execute("run_segmentation", request)
        second = self.callers["segmentation"].calls[-1][1]["context"]

        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["idempotency_key"], second["idempotency_key"])
        self.assertEqual("retry-trace", second["trace_id"])

    def test_concurrent_argument_builds_do_not_invoke_or_replace_caller(self) -> None:
        caller = NeverCalledCaller()
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="audit-session",
            clock=lambda: FIXED_NOW,
        )
        request = {
            "case_id": CASE_ID,
            "ct_artifact_id": self.ct.artifact_id,
            "input_format": "NIFTI",
        }

        with ThreadPoolExecutor(max_workers=8) as pool:
            arguments = list(
                pool.map(
                    lambda _: executor.build_arguments(
                        "inspect_case_metadata", dict(request)
                    ),
                    range(32),
                )
            )

        self.assertEqual(0, caller.call_count)
        self.assertTrue(
            all(
                item["context"]["request_id"]
                == arguments[0]["context"]["request_id"]
                for item in arguments
            )
        )
        self.assertTrue(
            all(
                item["context"]["idempotency_key"]
                == arguments[0]["context"]["idempotency_key"]
                for item in arguments
            )
        )
        # A normal execute still reaches the exact caller configured before
        # the concurrent audit builds; no shared route was swapped.
        with self.assertRaisesRegex(AssertionError, "must not invoke"):
            executor.execute("inspect_case_metadata", request)
        self.assertEqual(1, caller.call_count)

    def test_bound_state_supplies_session_trace_history_and_context_artifacts(self) -> None:
        state = AgentState(
            user_query="evaluate safety",
            session_id="bound-session",
            case_id=CASE_ID,
            artifacts={"ct": self.ct.artifact_id},
            metadata={"trace_id": "trace-from-agent-state"},
        )
        request = {
            "case_id": CASE_ID,
            "candidate_paths": [
                {
                    "candidate_id": "path-001",
                    "entry_point_world_mm": [20.0, 40.0, 10.0],
                    "target_point_world_mm": [55.0, 65.0, 60.0],
                    "length_mm": 65.6,
                    "insertion_angle_deg": 18.0,
                    "angle_reference": "LOCAL_SURFACE_NORMAL",
                    "rank_hint": 1,
                }
            ],
            "danger_mask_artifact_ids": {
                "heart": self.heart.artifact_id,
                "bone": self.bone.artifact_id,
                "bronchus": self.bronchus.artifact_id,
                "vessel": self.vessel.artifact_id,
                "lung": self.lung.artifact_id,
            },
            "safety_radius_mm": 5.0,
        }
        # Argument construction stops before resolver access, so the danger
        # artifact need not exist in this focused context/identity test.
        with self.executor.bind_state(state):
            arguments = self.executor.build_arguments("evaluate_path_safety", request)

        self.assertEqual("trace-from-agent-state", arguments["context"]["trace_id"])
        self.assertEqual({"artifact_id": self.ct.artifact_id}, arguments["ct_artifact"])
        self.assertEqual(
            {"artifact_id": "case-001-heart"},
            arguments["danger_masks"][0]["artifact"],
        )

    def test_unknown_tool_fails_before_calling_runtime(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown MCP tool"):
            self.executor.execute("not_a_tool", {"case_id": CASE_ID})
        self.assertFalse(any(caller.calls for caller in self.callers.values()))

    def test_raw_bytes_are_rejected_before_calling_runtime(self) -> None:
        with self.assertRaisesRegex(ToolBridgeContractError, "byte payloads"):
            self.executor.execute(
                "run_segmentation",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": b"raw-ct-volume",
                    "model_version": "v1",
                },
            )
        self.assertFalse(self.callers["segmentation"].calls)

    def test_storage_uri_cannot_masquerade_as_an_artifact_id(self) -> None:
        for storage_uri in (
            "s3://private-bucket/case-001/ct.nii.gz",
            "file:/private/case-001/ct.nii.gz",
        ):
            with self.subTest(storage_uri=storage_uri):
                with self.assertRaisesRegex(
                    ToolBridgeContractError,
                    "not a storage URI",
                ):
                    self.executor.execute(
                        "inspect_case_metadata",
                        {
                            "case_id": CASE_ID,
                            "ct_artifact_id": storage_uri,
                            "input_format": "NIFTI",
                        },
                    )
        self.assertFalse(self.callers["case-data"].calls)

    def test_transport_principal_is_forwarded_unchanged(self) -> None:
        self.executor.execute(
            "inspect_case_metadata",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "input_format": "NIFTI",
            },
        )
        _, arguments, principal = self.callers["case-data"].calls[-1]
        self.assertIs(self.principal, principal)
        self.assertEqual(self.principal.subject, arguments["context"]["caller"])

        caller = NeverCalledCaller()
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            clock=lambda: FIXED_NOW,
        )
        state = AgentState(
            user_query="inspect",
            session_id="principal-identity-session",
            case_id=CASE_ID,
        )
        request = {
            "case_id": CASE_ID,
            "ct_artifact_id": self.ct.artifact_id,
            "input_format": "NIFTI",
        }
        with executor.bind_state(
            state,
            principal=McpPrincipal("operator-a", (CASE_ID,)),
        ):
            first = executor.build_arguments("inspect_case_metadata", request)
        with executor.bind_state(
            state,
            principal=McpPrincipal("operator-b", (CASE_ID,)),
        ):
            second = executor.build_arguments("inspect_case_metadata", request)
        self.assertNotEqual(
            first["context"]["request_id"], second["context"]["request_id"]
        )
        self.assertNotEqual(
            first["context"]["idempotency_key"],
            second["context"]["idempotency_key"],
        )

        denied_caller = NeverCalledCaller()
        denied_executor = McpToolExecutor(
            denied_caller,
            principal=McpPrincipal("operator-denied", ("case-other",)),
            session_id="principal-denied-session",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaises(ToolBridgeContextError):
            denied_executor.execute("inspect_case_metadata", request)
        self.assertEqual(0, denied_caller.call_count)

    def test_tool_version_is_part_of_replay_identity(self) -> None:
        caller = NeverCalledCaller()
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="versioned-identity-session",
            clock=lambda: FIXED_NOW,
        )
        request = {
            "case_id": CASE_ID,
            "ct_artifact_id": self.ct.artifact_id,
            "input_format": "NIFTI",
        }
        first = executor.build_arguments("inspect_case_metadata", request)
        definition = TOOL_DEFINITIONS["inspect_case_metadata"]
        with patch.dict(
            TOOL_DEFINITIONS,
            {"inspect_case_metadata": replace(definition, version="2.0.0")},
        ):
            second = executor.build_arguments("inspect_case_metadata", request)
        self.assertNotEqual(
            first["context"]["request_id"],
            second["context"]["request_id"],
        )
        self.assertNotEqual(
            first["context"]["idempotency_key"],
            second["context"]["idempotency_key"],
        )

    def test_error_envelope_is_returned_and_storage_details_are_redacted(self) -> None:
        checksum = "b" * 64
        caller = StaticCaller(
            {
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "TIMEOUT",
                    "message": f"backend mock://private/volume failed checksum={checksum}",
                    "retryable": True,
                    "details": {
                        "uri": "mock://private/volume",
                        "checksum_sha256": checksum,
                    },
                },
            }
        )
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="error-session",
            clock=lambda: FIXED_NOW,
        )

        response = executor.execute(
            "inspect_case_metadata",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "input_format": "NIFTI",
            },
        )

        self.assertEqual("FAILED", response["status"])
        self.assertEqual("TIMEOUT", response["error"]["code"])
        encoded = json.dumps(response, sort_keys=True)
        self.assertNotIn("mock://private", encoded)
        self.assertNotIn(checksum, encoded)
        self.assertNotIn("uri", response["error"]["details"])
        self.assertNotIn("checksum_sha256", response["error"]["details"])

        class FailingTransportCaller:
            tool_names = ("inspect_case_metadata",)

            def call_tool(self, name, arguments, *, principal):
                del name, arguments, principal
                raise OSError("private DNS detail")

        transport_executor = McpToolExecutor(
            FailingTransportCaller(),
            principal=self.principal,
            session_id="transport-error-session",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaises(ToolBridgeTransportError) as raised:
            transport_executor.execute(
                "inspect_case_metadata",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "input_format": "NIFTI",
                },
            )
        self.assertEqual("DEPENDENCY_FAILED", raised.exception.code)
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn("private DNS detail", str(raised.exception))

    def test_remote_response_identity_and_is_error_must_match(self) -> None:
        for payload in (
            {
                "request_id": "wrong-request",
                "status": "SUCCESS",
                "result": {"ready": True},
                "error": None,
            },
            {
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "TIMEOUT",
                    "message": "timeout",
                    "retryable": True,
                },
            },
        ):
            with self.subTest(payload=payload):
                caller = StaticCaller(payload)
                if payload["status"] == "FAILED":
                    original = caller.call_tool

                    def inconsistent(name, arguments, *, principal):
                        result = original(name, arguments, principal=principal)
                        result["is_error"] = False
                        return result

                    caller.call_tool = inconsistent  # type: ignore[method-assign]
                executor = McpToolExecutor(
                    caller,
                    principal=self.principal,
                    session_id="remote-contract-test",
                    clock=lambda: FIXED_NOW,
                )
                with self.assertRaises(ToolBridgeContractError):
                    executor.execute(
                        "inspect_case_metadata",
                        {
                            "case_id": CASE_ID,
                            "ct_artifact_id": self.ct.artifact_id,
                            "input_format": "NIFTI",
                        },
                    )

        delegate = self.callers["case-data"].delegate

        class ConflictingAliasCaller:
            tool_names = ("inspect_case_metadata",)

            def call_tool(inner_self, name, arguments, *, principal):
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                protocol["is_error"] = protocol["isError"]
                return protocol

        executor = McpToolExecutor(
            ConflictingAliasCaller(),
            principal=self.principal,
            session_id="remote-alias-conflict",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaisesRegex(ToolBridgeContractError, "conflicting isError"):
            executor.execute(
                "inspect_case_metadata",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "input_format": "NIFTI",
                },
            )

        malformed_status = McpToolExecutor(
            StaticCaller(
                {
                    "status": [],
                    "result": {"ready": True},
                    "error": None,
                }
            ),
            principal=self.principal,
            session_id="remote-status-type",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaisesRegex(
            ToolBridgeResponseError,
            "status is not canonical",
        ):
            malformed_status.execute(
                "inspect_case_metadata",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "input_format": "NIFTI",
                },
            )

    def test_remote_frozen_result_rejects_string_boolean(self) -> None:
        delegate = self.callers["case-data"].delegate

        class MutatingCaller:
            tool_names = ("inspect_case_metadata",)

            def __init__(inner_self, mutator):
                inner_self.mutator = mutator

            def call_tool(inner_self, name, arguments, *, principal):
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                inner_self.mutator(protocol["structuredContent"]["result"])
                return protocol

        def string_boolean(result):
            result["ready_for_next_stage"] = "false"

        def invalid_geometry(result):
            result["ct_geometry"]["size_ijk"] = [0, 0, 0]

        def cross_case(result):
            result["case_id"] = "case-OTHER"

        scenarios = (
            (
                string_boolean,
                r"\$\.result\.ready_for_next_stage must be a boolean",
            ),
            (
                invalid_geometry,
                r"\$\.result\.ct_geometry violates frozen contract invariants",
            ),
            (cross_case, r"\$\.result\.case_id does not match the request case"),
        )
        for mutator, expected in scenarios:
            with self.subTest(expected=expected):
                executor = McpToolExecutor(
                    MutatingCaller(mutator),
                    principal=self.principal,
                    session_id="remote-schema-test",
                    clock=lambda: FIXED_NOW,
                )
                with self.assertRaisesRegex(ToolBridgeContractError, expected):
                    executor.execute(
                        "inspect_case_metadata",
                        {
                            "case_id": CASE_ID,
                            "ct_artifact_id": self.ct.artifact_id,
                            "input_format": "NIFTI",
                        },
                    )

        def missing_fingerprint(result):
            result["ct_geometry"]["geometry_fingerprint"] = ""

        normalized_executor = McpToolExecutor(
            MutatingCaller(missing_fingerprint),
            principal=self.principal,
            session_id="remote-normalization-test",
            clock=lambda: FIXED_NOW,
        )
        normalized = normalized_executor.execute(
            "inspect_case_metadata",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "input_format": "NIFTI",
            },
        )
        self.assertEqual(
            geometry().geometry_fingerprint,
            normalized["result"]["ct_geometry"]["geometry_fingerprint"],
        )

        def uri_identity(result):
            result["inspections"][0]["artifact_id"] = (
                "file:/private/case-001/ct.nii.gz"
            )

        uri_executor = McpToolExecutor(
            MutatingCaller(uri_identity),
            principal=self.principal,
            session_id="remote-uri-identity-test",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaisesRegex(
            ToolBridgeResponseError,
            "not a storage URI",
        ):
            uri_executor.execute(
                "inspect_case_metadata",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "input_format": "NIFTI",
                },
            )

    def test_result_artifact_must_match_the_envelope_artifact(self) -> None:
        delegate = self.callers["segmentation"].delegate

        class MutatingSegmentationCaller:
            tool_names = ("run_segmentation",)

            def call_tool(inner_self, name, arguments, *, principal):
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                artifact_result = protocol["structuredContent"]["result"][
                    "segmentation_artifact"
                ]
                artifact_result["artifact_id"] = "case-001-unregistered-output"
                artifact_result["artifact_type"] = "CT_VOLUME"
                return protocol

        executor = McpToolExecutor(
            MutatingSegmentationCaller(),
            principal=self.principal,
            session_id="response-artifact-binding",
            clock=lambda: FIXED_NOW,
        )
        with self.assertRaisesRegex(
            ToolBridgeResponseError,
            "not bound to a request or envelope artifact",
        ):
            executor.execute(
                "run_segmentation",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "model_version": "v1",
                },
            )

    def test_trusted_registry_accepts_an_authoritative_remote_output(self) -> None:
        registry = InMemoryArtifactRegistry()
        register_artifact(registry, self.ct)
        output = artifact(f"{CASE_ID}-seg-v1", ArtifactType.SEGMENTATION_MASK)
        register_artifact(
            registry,
            output,
            producer_name="run_segmentation",
            producer_version="1.0.0",
            parent_artifact_ids=(self.ct.artifact_id,),
        )
        delegate = self.callers["segmentation"].delegate

        class AuthoritativeSegmentationCaller:
            tool_names = ("run_segmentation",)

            def call_tool(inner_self, name, arguments, *, principal):
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                structured = protocol["structuredContent"]
                structured["result"]["segmentation_artifact"][
                    "producer_version"
                ] = "1.0.0"
                structured["artifacts"][0]["producer_version"] = "1.0.0"
                return protocol

        executor = McpToolExecutor(
            AuthoritativeSegmentationCaller(),
            principal=self.principal,
            session_id="registry-valid-output",
            clock=lambda: FIXED_NOW,
            artifact_validator=RegistryToolArtifactValidator(registry),
        )

        response = executor.execute(
            "run_segmentation",
            {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "model_version": "v1",
            },
        )

        self.assertEqual(output.artifact_id, response["artifacts"][0]["artifact_id"])

    def test_trusted_registry_rejects_consistent_remote_self_declaration(self) -> None:
        registry = InMemoryArtifactRegistry()
        register_artifact(registry, self.ct)
        delegate = self.callers["segmentation"].delegate

        class ConsistentlyForgingSegmentationCaller:
            tool_names = ("run_segmentation",)

            def __init__(inner_self) -> None:
                inner_self.call_count = 0

            def call_tool(inner_self, name, arguments, *, principal):
                inner_self.call_count += 1
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                structured = protocol["structuredContent"]
                forged = dict(structured["artifacts"][0])
                forged["artifact_id"] = "case-001-unregistered-output"
                structured["result"]["segmentation_artifact"] = dict(forged)
                structured["artifacts"] = [dict(forged)]
                return protocol

        caller = ConsistentlyForgingSegmentationCaller()
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="registry-forged-output",
            clock=lambda: FIXED_NOW,
            artifact_validator=RegistryToolArtifactValidator(registry),
        )

        with self.assertRaisesRegex(
            ToolBridgeResponseError,
            "trusted Artifact Registry rejected",
        ):
            executor.execute(
                "run_segmentation",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "model_version": "v1",
                },
            )
        self.assertEqual(1, caller.call_count)

    def test_rejected_remote_artifact_never_mutates_artifact_state_or_runs_validation(self) -> None:
        registry = InMemoryArtifactRegistry()
        register_artifact(registry, self.ct)
        delegate = self.callers["segmentation"].delegate

        class ConsistentlyForgingSegmentationCaller:
            tool_names = ("run_segmentation", "validate_segmentation_result")

            def __init__(inner_self) -> None:
                inner_self.call_count = 0

            def call_tool(inner_self, name, arguments, *, principal):
                inner_self.call_count += 1
                result = delegate.call_tool(name, arguments, principal=principal)
                protocol = result.to_protocol_result()
                structured = protocol["structuredContent"]
                forged = dict(structured["artifacts"][0])
                forged["artifact_id"] = "case-001-unregistered-output"
                structured["result"]["segmentation_artifact"] = dict(forged)
                structured["artifacts"] = [dict(forged)]
                return protocol

        caller = ConsistentlyForgingSegmentationCaller()
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="registry-state-guard",
            clock=lambda: FIXED_NOW,
            artifact_validator=RegistryToolArtifactValidator(registry),
        )
        handlers = build_mock_handlers(executor)
        state = AgentState(
            user_query="segment",
            session_id="registry-state-guard",
            case_id=CASE_ID,
            artifacts={"ct": self.ct.artifact_id},
            metadata={"model_version": "v1"},
        )
        state.current_node = "data_model_subgraph.run_segmentation"
        node_context = NodeContext(
            "data_model_subgraph",
            "run_segmentation",
            state.current_node,
            {},
        )

        handlers["run_segmentation"](state, node_context)

        self.assertEqual({"ct": self.ct.artifact_id}, state.artifacts)
        self.assertEqual("CONTRACT_VIOLATION", state.metadata["last_tool_error"]["code"])
        self.assertEqual(1, caller.call_count)

        state.current_node = "data_model_subgraph.validate_segmentation_result"
        handlers["validate_segmentation_result"](
            state,
            NodeContext(
                "data_model_subgraph",
                "validate_segmentation_result",
                state.current_node,
                {},
            ),
        )
        self.assertFalse(state.metadata["segmentation_valid"])
        self.assertEqual(1, caller.call_count)

    def test_trusted_registry_rejects_input_before_remote_execution(self) -> None:
        caller = self.callers["segmentation"]
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="registry-invalid-input",
            clock=lambda: FIXED_NOW,
            artifact_validator=RegistryToolArtifactValidator(
                InMemoryArtifactRegistry()
            ),
        )

        with self.assertRaisesRegex(
            ToolBridgeContractError,
            "trusted Artifact Registry rejected",
        ):
            executor.execute(
                "run_segmentation",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "model_version": "v1",
                },
            )
        self.assertEqual([], caller.calls)

    def test_trusted_registry_unavailability_is_retryable_and_fail_closed(self) -> None:
        class UnavailableRegistry:
            def get_validation_record(inner_self, artifact_id):
                del artifact_id
                raise ArtifactRegistryError(
                    "STORAGE_ERROR",
                    "private database details",
                    retryable=True,
                )

        caller = self.callers["segmentation"]
        executor = McpToolExecutor(
            caller,
            principal=self.principal,
            session_id="registry-unavailable",
            clock=lambda: FIXED_NOW,
            artifact_validator=RegistryToolArtifactValidator(UnavailableRegistry()),
        )

        with self.assertRaises(ToolBridgeTransportError) as caught:
            executor.execute(
                "run_segmentation",
                {
                    "case_id": CASE_ID,
                    "ct_artifact_id": self.ct.artifact_id,
                    "model_version": "v1",
                },
            )
        self.assertEqual("DEPENDENCY_FAILED", caught.exception.code)
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private database", str(caught.exception))
        self.assertEqual([], caller.calls)

    def test_ambiguous_multiple_segmentation_artifacts_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(
            ToolBridgeContractError, "multiple per-label mask artifacts"
        ):
            self.executor.execute(
                "validate_segmentation_result",
                {
                    "case_id": CASE_ID,
                    "mask_artifact_ids": {
                        "skin": "case-001-skin-mask",
                        "heart": "case-001-heart-mask",
                    },
                    "reference_ct_artifact_id": self.ct.artifact_id,
                },
            )
        self.assertFalse(self.callers["segmentation"].calls)

    def test_all_ten_legacy_shapes_decode_as_the_frozen_request_contracts(self) -> None:
        candidate = {
            "candidate_id": "path-001",
            "entry_point_world_mm": [20.0, 40.0, 10.0],
            "target_point_world_mm": [55.0, 65.0, 60.0],
            "length_mm": 65.6,
            "insertion_angle_deg": 18.0,
            "angle_reference": "LOCAL_SURFACE_NORMAL",
            "rank_hint": 1,
        }
        requests = {
            "inspect_case_metadata": {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "input_format": "NIFTI",
            },
            "convert_mcs_to_nifti": {
                "case_id": CASE_ID,
                "source_artifact_id": self.mcs.artifact_id,
                "reference_ct_artifact_id": self.ct.artifact_id,
            },
            "validate_label_schema": {
                "case_id": CASE_ID,
                "label_artifact_id": self.labelmap.artifact_id,
                "schema_version": "1.4",
            },
            "run_segmentation": {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "model_version": "v1",
            },
            "validate_segmentation_result": {
                "case_id": CASE_ID,
                "mask_artifact_ids": {"all": self.segmentation.artifact_id},
                "reference_ct_artifact_id": self.ct.artifact_id,
            },
            "extract_skin_surface": {
                "case_id": CASE_ID,
                "skin_mask_artifact_id": self.skin.artifact_id,
                "thickness_voxels": 2,
            },
            "generate_candidate_paths": {
                "case_id": CASE_ID,
                "ct_artifact_id": self.ct.artifact_id,
                "skin_surface_artifact_id": self.skin_surface.artifact_id,
                "target_artifact_id": self.target.artifact_id,
                "max_needle_length_mm": 120.0,
                "max_insertion_angle_deg": 45.0,
                "safety_radius_mm": 5.0,
                "top_k": 3,
            },
            "evaluate_path_safety": {
                "case_id": CASE_ID,
                "candidate_paths": [candidate],
                "danger_mask_artifact_ids": {
                    "heart": self.heart.artifact_id,
                    "bone": self.bone.artifact_id,
                    "bronchus": self.bronchus.artifact_id,
                    "vessel": self.vessel.artifact_id,
                    "lung": self.lung.artifact_id,
                },
                "safety_radius_mm": 5.0,
            },
            "evaluate_intraoperative_risk": {
                "case_id": CASE_ID,
                "planned_entry_point_world_mm": [20.0, 40.0, 10.0],
                "needle_tip_world_mm": [30.0, 45.0, 25.0],
                "danger_mask_artifact_ids": {
                    "heart": self.heart.artifact_id,
                    "bone": self.bone.artifact_id,
                    "bronchus": self.bronchus.artifact_id,
                    "vessel": self.vessel.artifact_id,
                    "lung": self.lung.artifact_id,
                },
            },
            "verify_skin_penetration": {
                "case_id": CASE_ID,
                "skin_mask_artifact_id": self.skin.artifact_id,
                "planned_entry_point_world_mm": [20.0, 40.0, 10.0],
                "needle_tip_world_mm": [30.0, 45.0, 25.0],
                "sample_step_voxel": 0.5,
            },
        }
        state = AgentState(
            user_query="contract audit",
            session_id="contract-audit-session",
            case_id=CASE_ID,
            artifacts={
                "ct": self.ct.artifact_id,
                "skin": self.skin.artifact_id,
                "segmentation_masks": {"lung": self.lung.artifact_id},
            },
        )

        with self.executor.bind_state(state):
            for tool_name, loose_request in requests.items():
                with self.subTest(tool_name=tool_name):
                    arguments = self.executor.build_arguments(tool_name, loose_request)
                    decoded = decode_tool_request(
                        TOOL_REQUEST_TYPES[tool_name],
                        arguments,
                        artifact_resolver=self.resolver,
                    )
                    self.assertEqual(CASE_ID, decoded.context.case_id)


if __name__ == "__main__":
    unittest.main()
