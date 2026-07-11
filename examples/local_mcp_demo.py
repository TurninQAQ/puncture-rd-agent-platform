#!/usr/bin/env python3
"""Run all ten strongly typed tools through three local MCP runtimes."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import pathlib
import sys
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts.artifacts import ArtifactRef  # noqa: E402
from contracts.common import ToolCallContext  # noqa: E402
from contracts.domain import (  # noqa: E402
    CandidatePath,
    DangerMaskSpec,
    LabelDefinition,
    LabelQualityThreshold,
    SafetyMargin,
)
from contracts.enums import (  # noqa: E402
    AngleReference,
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
    RiskStructure,
)
from contracts.geometry import VolumeGeometry, WorldPoint  # noqa: E402
from contracts.tool_inputs import (  # noqa: E402
    ConvertMcsToNiftiRequest,
    EvaluateIntraoperativeRiskRequest,
    EvaluatePathSafetyRequest,
    ExtractSkinSurfaceRequest,
    GenerateCandidatePathsRequest,
    InspectCaseMetadataRequest,
    LabelMappingEntry,
    RunSegmentationRequest,
    ValidateLabelSchemaRequest,
    ValidateSegmentationResultRequest,
    VerifySkinPenetrationRequest,
)
from puncture_agent.mcp import (  # noqa: E402
    InMemoryArtifactResolver,
    McpPrincipal,
    McpToolRuntime,
    to_mcp_arguments,
)
from puncture_agent.tooling import build_adapter_registry  # noqa: E402
from puncture_agent.tooling.case_data import (  # noqa: E402
    ArtifactManifest,
    ManifestCaseDataBackend,
    McsSegmentManifest,
)


CASE_ID = "demo-case-001"
CALLER = "local-demo"
NOW = "2026-07-11T00:00:00Z"


def _geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0, 1.0, 1.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=CoordinateSystem.LPS,
    )


def _artifact(
    artifact_type: ArtifactType,
    suffix: str,
    payload: bytes,
    *,
    metadata: dict[str, str] | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"{CASE_ID}-{suffix}",
        case_id=CASE_ID,
        artifact_type=artifact_type,
        uri=f"memory://private/{CASE_ID}/{suffix}",
        checksum_sha256=sha256(payload).hexdigest(),
        status=ArtifactStatus.AVAILABLE,
        geometry=_geometry(),
        producer_name="local-demo-fixture",
        producer_version="1",
        metadata=metadata or {},
    )


def _context(tool_name: str) -> ToolCallContext:
    return ToolCallContext(
        request_id=f"demo-{tool_name}-request",
        trace_id="demo-mcp-trace-001",
        case_id=CASE_ID,
        caller=CALLER,
        idempotency_key=f"demo-{tool_name}-idempotency",
        requested_at=NOW,
    )


def _labels() -> tuple[LabelDefinition, ...]:
    return (
        LabelDefinition("background", 0, True),
        LabelDefinition("skin", 1, True),
        LabelDefinition("lung", 2, True),
        LabelDefinition("heart", 3, True),
    )


def _collect_artifacts(value: Any) -> tuple[ArtifactRef, ...]:
    found: dict[str, ArtifactRef] = {}

    def visit(item: Any) -> None:
        if isinstance(item, ArtifactRef):
            found[item.artifact_id] = item
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def _build_requests() -> tuple[dict[str, Any], ManifestCaseDataBackend]:
    ct_payload = b"local-demo-ct-volume"
    mcs_payload = b"local-demo-mcs-segments"
    labels_payload = b"local-demo-labelmap"
    segmentation_payload = b"local-demo-segmentation"
    skin_payload = b"local-demo-skin-mask"

    ct = _artifact(ArtifactType.CT_VOLUME, "ct", ct_payload)
    mcs = _artifact(ArtifactType.MCS_SEGMENTATION, "mcs", mcs_payload)
    labelmap = _artifact(
        ArtifactType.NIFTI_LABELMAP,
        "labels",
        labels_payload,
        metadata={"label_values": "0,1,2,3"},
    )
    segmentation = _artifact(
        ArtifactType.SEGMENTATION_MASK,
        "segmentation",
        segmentation_payload,
        metadata={
            "label_values": "0,1,2,3",
            "label_voxel_counts": "1:480000,2:180000,3:60000",
            "label_component_counts": "1:1,2:2,3:1",
            "border_label_values": "1",
        },
    )
    skin_mask = _artifact(
        ArtifactType.SEGMENTATION_MASK,
        "skin-mask",
        skin_payload,
        metadata={
            "label_values": "0,1",
            "label_voxel_counts": "1:480000",
            "primary_label_value": "1",
            "surface_voxel_count": "85000",
        },
    )
    skin_surface = _artifact(ArtifactType.SKIN_SURFACE_MASK, "skin-surface", b"skin-surface")
    target = _artifact(ArtifactType.TARGET_MASK, "target", b"target")
    heart = _artifact(ArtifactType.DANGER_MASK, "heart", b"heart")
    bone = _artifact(ArtifactType.DANGER_MASK, "bone", b"bone")
    lung = _artifact(ArtifactType.SEGMENTATION_MASK, "lung", b"lung")

    danger_masks = (
        DangerMaskSpec(
            RiskStructure.HEART,
            heart,
            SafetyMargin(warning_mm=8.0, stop_mm=3.0),
        ),
        DangerMaskSpec(
            RiskStructure.BONE,
            bone,
            SafetyMargin(warning_mm=5.0, stop_mm=2.0),
        ),
    )
    candidates = (
        CandidatePath(
            "path-001",
            WorldPoint(20.0, 40.0, 10.0),
            WorldPoint(55.0, 65.0, 60.0),
            65.6,
            18.0,
            "LOCAL_SURFACE_NORMAL",
            1,
        ),
        CandidatePath(
            "path-002",
            WorldPoint(23.0, 37.0, 11.0),
            WorldPoint(55.0, 65.0, 60.0),
            64.8,
            24.0,
            "LOCAL_SURFACE_NORMAL",
            2,
        ),
    )

    requests: dict[str, Any] = {
        "inspect_case_metadata": InspectCaseMetadataRequest(
            _context("inspect_case_metadata"),
            ct,
            (labelmap,),
            (ArtifactType.NIFTI_LABELMAP,),
        ),
        "convert_mcs_to_nifti": ConvertMcsToNiftiRequest(
            _context("convert_mcs_to_nifti"),
            mcs,
            ct,
            (
                LabelMappingEntry("Skin", 1, "skin", 1),
                LabelMappingEntry("Lung", 2, "lung", 2),
            ),
            CoordinateSystem.LPS,
        ),
        "validate_label_schema": ValidateLabelSchemaRequest(
            _context("validate_label_schema"),
            labelmap,
            _labels(),
        ),
        "run_segmentation": RunSegmentationRequest(
            _context("run_segmentation"),
            ct,
            "nnunet-puncture",
            "v1",
            ("skin", "lung", "heart"),
        ),
        "validate_segmentation_result": ValidateSegmentationResultRequest(
            _context("validate_segmentation_result"),
            ct,
            segmentation,
            _labels(),
            (
                LabelQualityThreshold("skin", 100, 10),
                LabelQualityThreshold("lung", 100, 10),
                LabelQualityThreshold("heart", 100, 5),
            ),
        ),
        "extract_skin_surface": ExtractSkinSurfaceRequest(
            _context("extract_skin_surface"),
            skin_mask,
        ),
        "generate_candidate_paths": GenerateCandidatePathsRequest(
            context=_context("generate_candidate_paths"),
            ct_artifact=ct,
            skin_surface_artifact=skin_surface,
            target_artifact=target,
            lesion_artifact=None,
            target_point_world_mm=WorldPoint(55.0, 65.0, 60.0),
            max_needle_length_mm=120.0,
            max_insertion_angle_deg=45.0,
            angle_reference=AngleReference.LOCAL_SURFACE_NORMAL,
            max_candidates=3,
            entry_sampling_step_mm=2.0,
            planner_version="planner-v1",
        ),
        "evaluate_path_safety": EvaluatePathSafetyRequest(
            _context("evaluate_path_safety"),
            ct,
            candidates,
            danger_masks,
            1.0,
        ),
        "evaluate_intraoperative_risk": EvaluateIntraoperativeRiskRequest(
            context=_context("evaluate_intraoperative_risk"),
            ct_artifact=ct,
            planned_entry_world_mm=WorldPoint(20.0, 40.0, 10.0),
            current_tip_world_mm=WorldPoint(30.0, 45.0, 25.0),
            insertion_depth_mm=20.0,
            danger_masks=danger_masks,
            lung_mask_artifact=lung,
            skin_mask_artifact=skin_mask,
            risk_rule_version="risk-v1",
        ),
        "verify_skin_penetration": VerifySkinPenetrationRequest(
            _context("verify_skin_penetration"),
            skin_surface,
            WorldPoint(20.0, 40.0, 10.0),
            WorldPoint(30.0, 45.0, 25.0),
            20.0,
        ),
    }

    case_backend = ManifestCaseDataBackend(
        (
            ArtifactManifest(
                ct,
                "demo-tenant",
                (CALLER,),
                "ct-manifest-v1",
                geometry=ct.geometry,
                payload=ct_payload,
            ),
            ArtifactManifest(
                mcs,
                "demo-tenant",
                (CALLER,),
                "mcs-manifest-v1",
                geometry=mcs.geometry,
                payload=mcs_payload,
                mcs_segments=(
                    McsSegmentManifest("Skin", 1, 120_000),
                    McsSegmentManifest("Lung", 2, 340_000),
                ),
            ),
            ArtifactManifest(
                labelmap,
                "demo-tenant",
                (CALLER,),
                "labelmap-manifest-v1",
                geometry=labelmap.geometry,
                payload=labels_payload,
                label_value_chunks=((0, 1), (2, 3)),
                label_names=((0, "background"), (1, "skin"), (2, "lung"), (3, "heart")),
            ),
        ),
        caller_tenants={CALLER: "demo-tenant"},
    )
    return requests, case_backend


def _result_summary(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"] or {}
    selected_fields = {
        "inspect_case_metadata": ("ready_for_next_stage", "required_types_present"),
        "convert_mcs_to_nifti": ("output_dtype", "total_nonzero_voxels"),
        "validate_label_schema": ("valid", "observed_label_values"),
        "run_segmentation": ("model_id", "model_version", "inference_time_ms"),
        "validate_segmentation_result": ("valid", "recommended_action"),
        "extract_skin_surface": ("surface_voxel_count", "effective_thickness_mm"),
        "generate_candidate_paths": ("sampled_entry_point_count", "planner_version"),
        "evaluate_path_safety": ("accepted_candidate_ids", "safest_candidate_id"),
        "evaluate_intraoperative_risk": ("overall_level", "requires_manual_review"),
        "verify_skin_penetration": ("status", "crossed_skin", "samples_evaluated"),
    }[tool_name]
    summary = {field: result.get(field) for field in selected_fields}
    for artifact_field in ("output_artifact", "segmentation_artifact", "surface_artifact"):
        if isinstance(result.get(artifact_field), dict):
            summary[artifact_field] = result[artifact_field]["artifact_id"]
    return summary


def run_demo() -> dict[str, Any]:
    requests, case_backend = _build_requests()
    bundle = build_adapter_registry(case_data_backend=case_backend)
    artifacts: dict[str, ArtifactRef] = {}
    for request in requests.values():
        for artifact in _collect_artifacts(request):
            artifacts[artifact.artifact_id] = artifact
    resolver = InMemoryArtifactResolver(tuple(artifacts.values()))
    principal = McpPrincipal(CALLER, (CASE_ID,))

    server_for_tool = {
        name: server_name
        for server_name, names in {
            "case-data": (
                "inspect_case_metadata",
                "convert_mcs_to_nifti",
                "validate_label_schema",
            ),
            "segmentation": (
                "run_segmentation",
                "validate_segmentation_result",
                "extract_skin_surface",
            ),
            "planning-safety": (
                "generate_candidate_paths",
                "evaluate_path_safety",
                "evaluate_intraoperative_risk",
                "verify_skin_penetration",
            ),
        }.items()
        for name in names
    }
    runtimes = {
        name: McpToolRuntime(bundle.registry, resolver, server_name=name)
        for name in ("case-data", "segmentation", "planning-safety")
    }
    calls = []
    for tool_name, request in requests.items():
        call = runtimes[server_for_tool[tool_name]].call_tool(
            tool_name,
            to_mcp_arguments(request),
            principal=principal,
        )
        if call.is_error:
            raise RuntimeError(
                f"local MCP demo failed at {tool_name}: "
                f"{call.structured_content['error']['code']}"
            )
        encoded = json.dumps(call.structured_content, sort_keys=True)
        if any(secret in encoded for secret in ("memory://private", "checksum_sha256", '"uri"')):
            raise RuntimeError(f"MCP result leaked private artifact fields: {tool_name}")
        calls.append(
            {
                "server": server_for_tool[tool_name],
                "tool": tool_name,
                "status": call.structured_content["status"],
                "trace_id": call.structured_content["trace_id"],
                "summary": _result_summary(tool_name, call.structured_content),
            }
        )

    return {
        "demo": "local-strongly-typed-mcp-tools",
        "protocol_baseline": "2025-11-25",
        "runtime": {
            "python": "3.10+",
            "external_services": False,
            "third_party_dependencies": False,
            "company_algorithms_reimplemented": False,
        },
        "servers": {
            name: {"tool_count": len(runtime.list_tools()), "tools": list(runtime.tool_names)}
            for name, runtime in runtimes.items()
        },
        "calls": calls,
        "security": {
            "artifact_wire_format": "opaque-id-handle",
            "uri_or_checksum_visible_to_model": False,
            "principal_case_tool_policy": True,
        },
    }


def main() -> int:
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
