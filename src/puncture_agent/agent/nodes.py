"""Deterministic mock nodes and tool adapter used to exercise the full graph."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Protocol

from .runtime import NodeContext, NodeOutcome
from .state import AgentState, AgentStatus, TaskType, VerificationStatus
from .verifier import verify_agent_state


class ToolExecutor(Protocol):
    """Minimal adapter boundary for the future MCP/ToolRegistry implementation."""

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> Any:
        ...


def _success(result: Any, **extra: Any) -> dict[str, Any]:
    return {"status": "SUCCESS", "result": result, "error": None, **extra}


class DeterministicMockToolExecutor:
    """Returns stable fake outputs without performing medical algorithms."""

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(request.get("case_id", "unknown"))
        responses: dict[str, Any] = {
            "inspect_case_metadata": {
                "case_id": case_id,
                "shape": [512, 512, 320],
                "spacing_mm": [0.75, 0.75, 1.0],
                "coordinate_system": "LPS",
                "input_format": request.get("input_format", "NIFTI"),
            },
            "convert_mcs_to_nifti": {
                "output_artifact_id": f"artifact-{case_id}-labels-nifti",
                "converted_label_count": 8,
                "geometry_preserved": True,
            },
            "validate_label_schema": {
                "valid": True,
                "present_labels": ["skin", "lung", "heart", "bone", "bronchus"],
                "missing_labels": [],
                "unexpected_values": [],
            },
            "run_segmentation": {
                "mask_artifact_ids": {
                    "skin": f"artifact-{case_id}-skin",
                    "lung": f"artifact-{case_id}-lung",
                    "heart": f"artifact-{case_id}-heart",
                    "bone": f"artifact-{case_id}-bone",
                    "bronchus": f"artifact-{case_id}-bronchus",
                },
                "latency_ms": 158.0,
                "gpu_memory_mb": 2795.0,
                "model_version": request.get("model_version", "mock-nnunet-v1"),
            },
            "validate_segmentation_result": {
                "valid": True,
                "empty_labels": [],
                "out_of_bounds_labels": [],
                "quality_warnings": [],
            },
            "extract_skin_surface": {
                "skin_surface_artifact_id": f"artifact-{case_id}-skin-surface",
                "method": "erosion_difference",
                "thickness_voxels": 2,
            },
            "generate_candidate_paths": {
                "candidates": [
                    {
                        "candidate_id": "path-001",
                        "entry_point_world_mm": [112.5, 84.0, 46.5],
                        "target_point_world_mm": [154.0, 113.0, 108.0],
                        "length_mm": 79.8,
                        "insertion_angle_deg": 21.4,
                    },
                    {
                        "candidate_id": "path-002",
                        "entry_point_world_mm": [118.0, 80.0, 50.0],
                        "target_point_world_mm": [154.0, 113.0, 108.0],
                        "length_mm": 76.5,
                        "insertion_angle_deg": 27.1,
                    },
                    {
                        "candidate_id": "path-003",
                        "entry_point_world_mm": [106.0, 92.0, 44.0],
                        "target_point_world_mm": [154.0, 113.0, 108.0],
                        "length_mm": 82.7,
                        "insertion_angle_deg": 32.0,
                    },
                ]
            },
            "evaluate_path_safety": {
                "accepted_candidate_ids": ["path-001", "path-002"],
                "rejected_candidates": [
                    {
                        "candidate_id": "path-003",
                        "reason": "SAFETY_ENVELOPE_INTERSECTION",
                        "intersected_structure": "bone",
                    }
                ],
                "minimum_clearance_mm": {"path-001": 8.2, "path-002": 6.4},
            },
            "evaluate_intraoperative_risk": {
                "risk_flags": {
                    "bone_warning": False,
                    "bone_stop": False,
                    "heart_warning": False,
                    "heart_stop": False,
                    "bronchus_warning": False,
                    "bronchus_stop": False,
                    "large_vessel_entry": False,
                    "needle_tip_in_lung": True,
                    "contradictory": False,
                }
            },
            "verify_skin_penetration": {
                "penetrated": True,
                "sample_step_voxel": 0.5,
                "skin_voxel_hits": 3,
                "slippage_suspected": False,
                "skin_not_punctured_suspected": False,
            },
        }
        if tool_name not in responses:
            return {
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"No mock response for {tool_name}",
                    "retryable": False,
                },
            }
        return _success(responses[tool_name], metrics={"mock": True})


def _to_mapping(response: Any) -> dict[str, Any]:
    """Normalize dict, dataclass, or Pydantic-like tool envelopes."""

    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if is_dataclass(response):
        return asdict(response)
    payload: dict[str, Any] = {}
    for name in ("status", "result", "data", "error", "metrics"):
        if hasattr(response, name):
            value = getattr(response, name)
            payload[name] = value.value if hasattr(value, "value") else value
    if not payload:
        raise TypeError(f"Unsupported tool response type: {type(response).__name__}")
    return payload


def _response_status(response: Mapping[str, Any]) -> str:
    value = response.get("status", "FAILED")
    if hasattr(value, "value"):
        value = value.value
    return str(value).upper()


def _response_result(response: Mapping[str, Any]) -> Any:
    if "result" in response:
        return response["result"]
    return response.get("data")


def _execute_tool(
    state: AgentState,
    executor: ToolExecutor,
    tool_name: str,
    request: Mapping[str, Any],
) -> tuple[bool, Any]:
    call_id = f"call-{len(state.tool_calls) + 1:04d}"
    state.tool_calls.append(
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "request": dict(request),
            "node_id": state.current_node,
            "status": "RUNNING",
        }
    )

    fail_once = set(state.metadata.get("fail_tool_once", []))
    fail_always = set(state.metadata.get("fail_tool_always", []))
    fail_non_retryable = set(state.metadata.get("fail_tool_non_retryable", []))
    already_failed = set(state.metadata.setdefault("failed_tools_once", []))
    if tool_name in fail_non_retryable:
        raw_response = {
            "status": "FAILED",
            "result": None,
            "error": {
                "code": "PERMISSION_DENIED",
                "message": f"Injected non-retryable failure for {tool_name}",
                "retryable": False,
            },
        }
    elif tool_name in fail_always:
        raw_response = {
            "status": "FAILED",
            "result": None,
            "error": {
                "code": "TIMEOUT",
                "message": f"Injected persistent timeout for {tool_name}",
                "retryable": True,
            },
        }
    elif tool_name in fail_once and tool_name not in already_failed:
        already_failed.add(tool_name)
        state.metadata["failed_tools_once"] = sorted(already_failed)
        raw_response: Any = {
            "status": "FAILED",
            "result": None,
            "error": {
                "code": "TIMEOUT",
                "message": f"Injected one-time timeout for {tool_name}",
                "retryable": True,
            },
        }
    else:
        raw_response = executor.execute(tool_name, request)

    response = _to_mapping(raw_response)
    status = _response_status(response)
    success = status in {"SUCCESS", "SUCCEEDED", "OK"}
    state.tool_calls[-1]["status"] = "SUCCESS" if success else "FAILED"
    state.tool_results.append(
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "response": response,
        }
    )

    if not success:
        error = response.get("error") or {}
        if not isinstance(error, Mapping):
            error = {"code": "TOOL_FAILED", "message": str(error)}
        code = str(error.get("code", "TOOL_FAILED"))
        message = str(error.get("message", f"Tool {tool_name} failed"))
        retryable = bool(error.get("retryable", code in {"TIMEOUT", "RETRYABLE_ERROR"}))
        state.metadata["last_tool_error"] = {
            "tool_name": tool_name,
            "code": code,
            "message": message,
            "retryable": retryable,
        }
        state.add_error(
            code,
            message,
            retryable=retryable,
            details={"tool_name": tool_name, "call_id": call_id},
        )
        return False, None
    return True, _response_result(response)


def _noop(_: AgentState, __: NodeContext) -> NodeOutcome:
    return NodeOutcome()


def _parse_request(state: AgentState, _: NodeContext) -> NodeOutcome:
    query = state.user_query
    lowered = query.lower()
    if not state.case_id:
        match = re.search(
            r"(?<![A-Za-z0-9])case[-_\s]?\d+(?![A-Za-z0-9])",
            query,
            flags=re.IGNORECASE,
        )
        if match:
            state.case_id = match.group(0).replace("_", "-").replace(" ", "-")

    if state.task_type == TaskType.UNKNOWN:
        planning_keywords = (
            "路径",
            "针道",
            "入针",
            "安全评估",
            "皮肤穿透",
            "planning",
            "trajectory",
            "risk",
        )
        data_keywords = (
            "数据",
            "标签",
            "分割",
            "mcs",
            "nifti",
            "nii.gz",
            "spacing",
            "模型验证",
            "segmentation",
        )
        if any(keyword in lowered for keyword in planning_keywords):
            state.task_type = TaskType.PLANNING_SAFETY
        elif any(keyword in lowered for keyword in data_keywords):
            state.task_type = TaskType.DATA_MODEL_VALIDATION

    if "mcs" in lowered:
        state.metadata.setdefault("input_format", "MCS")
    state.metadata.setdefault("run_segmentation", True)
    state.metadata.setdefault("extract_skin_surface", True)

    if state.task_type == TaskType.PLANNING_SAFETY:
        state.tool_plan = [
            "generate_candidate_paths",
            "evaluate_path_safety",
            "evaluate_intraoperative_risk",
            "verify_skin_penetration",
        ]
    elif state.task_type == TaskType.DATA_MODEL_VALIDATION:
        state.tool_plan = [
            "inspect_case_metadata",
            "validate_label_schema",
            "run_segmentation",
            "validate_segmentation_result",
            "extract_skin_surface",
        ]
    return NodeOutcome(output={"task_type": state.task_type, "case_id": state.case_id})


def _retrieve_project_knowledge(state: AgentState, _: NodeContext) -> NodeOutcome:
    if state.task_type == TaskType.PLANNING_SAFETY:
        docs = [
            {
                "document_id": "planning-rule-v2",
                "version": "2.0",
                "section": "Needle constraints",
                "score": 0.94,
            },
            {
                "document_id": "safety-rule-v3",
                "version": "3.1",
                "section": "Warning and stop boundaries",
                "score": 0.91,
            },
        ]
    else:
        docs = [
            {
                "document_id": "label-schema-v1",
                "version": "1.4",
                "section": "Label values",
                "score": 0.95,
            },
            {
                "document_id": "segmentation-interface-v2",
                "version": "2.2",
                "section": "Input geometry",
                "score": 0.90,
            },
        ]
    state.retrieved_documents = docs
    state.citations = [
        {
            "document_id": item["document_id"],
            "version": item["version"],
            "section": item["section"],
        }
        for item in docs
    ]
    return NodeOutcome(output={"document_count": len(docs)})


def _resolve_case_context(state: AgentState, _: NodeContext) -> NodeOutcome:
    ready = bool(state.case_id) and not state.metadata.get("force_case_missing", False)
    state.metadata["case_context_ready"] = ready
    if ready and state.metadata.get("use_mock_artifacts", True):
        case_id = state.case_id
        defaults = {
            "ct": f"artifact-{case_id}-ct",
            "raw_labels": f"artifact-{case_id}-raw-labels",
            "skin": f"artifact-{case_id}-skin",
            "skin_surface": f"artifact-{case_id}-skin-surface",
            "target": f"artifact-{case_id}-target",
            "danger_masks": {
                "heart": f"artifact-{case_id}-heart",
                "bone": f"artifact-{case_id}-bone",
                "bronchus": f"artifact-{case_id}-bronchus",
                "vessel": f"artifact-{case_id}-vessel",
                "lung": f"artifact-{case_id}-lung",
            },
        }
        for key, value in defaults.items():
            state.artifacts.setdefault(key, value)
    return NodeOutcome(output={"case_context_ready": ready})


def _request_missing_data(state: AgentState, _: NodeContext) -> NodeOutcome:
    missing = []
    if not state.case_id:
        missing.append("case_id")
    missing.extend(state.metadata.get("missing_required_artifacts", []))
    if state.task_type == TaskType.UNKNOWN:
        missing.append("unambiguous_task_type")
    state.status = AgentStatus.AWAITING_INPUT
    state.final_report = {
        "report_version": "1.0",
        "status": AgentStatus.AWAITING_INPUT,
        "case_id": state.case_id,
        "missing_fields": sorted(set(missing)),
        "message": "Additional input is required before the workflow can continue.",
        "citations": state.citations,
    }
    return NodeOutcome(output=state.final_report)


def _inspect_case_metadata(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "inspect_case_metadata",
        {
            "case_id": state.case_id,
            "ct_artifact_id": state.artifacts.get("ct"),
            "input_format": state.metadata.get("input_format", "NIFTI"),
        },
    )
    if ok:
        state.metadata["case_geometry"] = result
    return NodeOutcome(output=result)


def _validate_geometry(state: AgentState, _: NodeContext) -> NodeOutcome:
    geometry = state.metadata.get("case_geometry")
    valid = bool(geometry) and not state.metadata.get("force_geometry_mismatch", False)
    state.metadata["geometry_valid"] = valid
    state.metadata["requires_conversion"] = (
        str(state.metadata.get("input_format", "NIFTI")).upper() == "MCS"
    )
    if not valid:
        state.add_error(
            "GEOMETRY_MISMATCH",
            "CT and label geometry are missing or inconsistent",
            retryable=False,
        )
    return NodeOutcome(output={"valid": valid})


def _convert_mcs_to_nifti(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "convert_mcs_to_nifti",
        {
            "case_id": state.case_id,
            "source_artifact_id": state.artifacts.get("raw_labels"),
            "reference_ct_artifact_id": state.artifacts.get("ct"),
        },
    )
    if ok and result:
        state.artifacts["labels_nifti"] = result.get("output_artifact_id")
    return NodeOutcome(output=result)


def _validate_label_schema(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "validate_label_schema",
        {
            "case_id": state.case_id,
            "label_artifact_id": state.artifacts.get("labels_nifti")
            or state.artifacts.get("raw_labels"),
            "schema_version": state.metadata.get("label_schema_version", "1.4"),
        },
    )
    valid = bool(ok and result and result.get("valid", False))
    if state.metadata.get("force_label_schema_error"):
        valid = False
    state.metadata["label_schema_valid"] = valid
    if not valid and ok:
        state.add_error(
            "LABEL_SCHEMA_ERROR",
            "Label names or values do not match the configured schema",
            retryable=False,
        )
    return NodeOutcome(output=result)


def _run_segmentation(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "run_segmentation",
        {
            "case_id": state.case_id,
            "ct_artifact_id": state.artifacts.get("ct"),
            "model_version": state.metadata.get("model_version", "mock-nnunet-v1"),
        },
    )
    if ok and result:
        state.artifacts.setdefault("segmentation_masks", {}).update(
            result.get("mask_artifact_ids", {})
        )
        state.metadata["segmentation_metrics"] = {
            "latency_ms": result.get("latency_ms"),
            "gpu_memory_mb": result.get("gpu_memory_mb"),
        }
    return NodeOutcome(output=result)


def _validate_segmentation_result(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "validate_segmentation_result",
        {
            "case_id": state.case_id,
            "mask_artifact_ids": state.artifacts.get("segmentation_masks", {}),
            "reference_ct_artifact_id": state.artifacts.get("ct"),
        },
    )
    valid = bool(ok and result and result.get("valid", False))
    if state.metadata.get("force_empty_segmentation"):
        valid = False
    state.metadata["segmentation_valid"] = valid
    if not valid and ok:
        state.add_error(
            "SEGMENTATION_VALIDATION_FAILED",
            "Segmentation output is empty or geometrically invalid",
            retryable=False,
        )
    return NodeOutcome(output=result)


def _extract_skin_surface(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    masks = state.artifacts.get("segmentation_masks", {})
    ok, result = _execute_tool(
        state,
        executor,
        "extract_skin_surface",
        {
            "case_id": state.case_id,
            "skin_mask_artifact_id": masks.get("skin") or state.artifacts.get("skin"),
            "thickness_voxels": 2,
        },
    )
    if ok and result:
        state.artifacts["skin_surface"] = result.get("skin_surface_artifact_id")
    return NodeOutcome(output=result)


def _finalize_data_model(state: AgentState, _: NodeContext) -> NodeOutcome:
    if state.metadata.get("last_tool_error"):
        status = "ERROR"
        reasons = [state.metadata["last_tool_error"]["message"]]
    elif not state.metadata.get("geometry_valid", False):
        status = "VALIDATION_FAILED"
        reasons = ["geometry validation failed"]
    elif not state.metadata.get("label_schema_valid", False):
        status = "VALIDATION_FAILED"
        reasons = ["label schema validation failed"]
    elif state.metadata.get("run_segmentation") and not state.metadata.get(
        "segmentation_valid", False
    ):
        status = "VALIDATION_FAILED"
        reasons = ["segmentation validation failed"]
    else:
        status = "SUCCESS"
        reasons = []
    state.subgraph_result = {
        "subgraph": "data_model_validation",
        "status": status,
        "reasons": reasons,
    }
    return NodeOutcome(output=state.subgraph_result)


def _ensure_required_artifacts(state: AgentState, _: NodeContext) -> NodeOutcome:
    required = ["ct", "skin_surface", "target", "danger_masks"]
    forced_missing = list(state.metadata.get("missing_required_artifacts", []))
    missing = [name for name in required if not state.artifacts.get(name)] + forced_missing
    state.metadata["missing_required_artifacts"] = sorted(set(missing))
    state.metadata["required_artifacts_ready"] = not missing
    return NodeOutcome(output={"missing": state.metadata["missing_required_artifacts"]})


def _resolve_planning_constraints(state: AgentState, _: NodeContext) -> NodeOutcome:
    defaults = {
        "max_needle_length_mm": 120.0,
        "max_insertion_angle_deg": 45.0,
        "safety_radius_mm": 5.0,
        "top_k": 3,
    }
    defaults.update(state.planning_constraints)
    state.planning_constraints = defaults
    return NodeOutcome(output=defaults)


def _generate_candidate_paths(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    if state.metadata.get("force_no_feasible_path"):
        state.candidate_paths = []
        return NodeOutcome(output={"candidates": []})
    ok, result = _execute_tool(
        state,
        executor,
        "generate_candidate_paths",
        {
            "case_id": state.case_id,
            "ct_artifact_id": state.artifacts.get("ct"),
            "skin_surface_artifact_id": state.artifacts.get("skin_surface"),
            "target_artifact_id": state.artifacts.get("target"),
            **state.planning_constraints,
        },
    )
    state.candidate_paths = list(result.get("candidates", [])) if ok and result else []
    return NodeOutcome(output=result)


def _evaluate_path_safety(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "evaluate_path_safety",
        {
            "case_id": state.case_id,
            "candidate_paths": state.candidate_paths,
            "danger_mask_artifact_ids": state.artifacts.get("danger_masks", {}),
            "safety_radius_mm": state.planning_constraints.get("safety_radius_mm"),
        },
    )
    state.safety_result = dict(result or {}) if ok else {}
    return NodeOutcome(output=result)


def _evaluate_intraoperative_risk(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "evaluate_intraoperative_risk",
        {
            "case_id": state.case_id,
            "planned_entry_point_world_mm": state.candidate_paths[0][
                "entry_point_world_mm"
            ],
            "needle_tip_world_mm": state.candidate_paths[0]["target_point_world_mm"],
            "danger_mask_artifact_ids": state.artifacts.get("danger_masks", {}),
        },
    )
    state.risk_flags = dict((result or {}).get("risk_flags", {})) if ok else {}
    return NodeOutcome(output=result)


def _verify_skin_penetration(
    state: AgentState, _: NodeContext, executor: ToolExecutor
) -> NodeOutcome:
    ok, result = _execute_tool(
        state,
        executor,
        "verify_skin_penetration",
        {
            "case_id": state.case_id,
            "skin_mask_artifact_id": state.artifacts.get("skin"),
            "planned_entry_point_world_mm": state.candidate_paths[0][
                "entry_point_world_mm"
            ],
            "needle_tip_world_mm": state.candidate_paths[0]["target_point_world_mm"],
            "sample_step_voxel": 0.5,
        },
    )
    state.skin_penetration_result = dict(result or {}) if ok else {}
    return NodeOutcome(output=result)


def _finalize_planning(state: AgentState, _: NodeContext) -> NodeOutcome:
    if not state.metadata.get("required_artifacts_ready", False):
        status = "MISSING_INPUT"
        reasons = [
            "missing artifacts: "
            + ", ".join(state.metadata.get("missing_required_artifacts", []))
        ]
    elif state.metadata.get("last_tool_error"):
        status = "ERROR"
        reasons = [state.metadata["last_tool_error"]["message"]]
    elif not state.candidate_paths:
        status = "NO_FEASIBLE_PATH"
        reasons = ["no candidate satisfies geometric constraints"]
    else:
        status = "SUCCESS"
        reasons = []
    state.subgraph_result = {
        "subgraph": "planning_safety",
        "status": status,
        "reasons": reasons,
    }
    return NodeOutcome(output=state.subgraph_result)


def _result_verifier(state: AgentState, _: NodeContext) -> NodeOutcome:
    result = verify_agent_state(state)
    state.verification_status = result.status
    state.metadata["verification_reasons"] = list(result.reasons)
    state.metadata["verification_evidence"] = result.evidence
    return NodeOutcome(
        output={
            "status": result.status,
            "reasons": list(result.reasons),
            "evidence": result.evidence,
        }
    )


def _error_recovery(state: AgentState, _: NodeContext) -> NodeOutcome:
    state.retry_count += 1
    state.metadata.pop("last_tool_error", None)
    state.subgraph_result = {}
    state.verification_status = VerificationStatus.NOT_RUN
    return NodeOutcome(output={"retry_count": state.retry_count})


def _report_generator(state: AgentState, _: NodeContext) -> NodeOutcome:
    if state.verification_status == VerificationStatus.PASS:
        state.status = AgentStatus.SUCCEEDED
    elif state.verification_status == VerificationStatus.NO_FEASIBLE_PATH:
        state.status = AgentStatus.COMPLETED_WITH_NO_RESULT
    elif state.verification_status == VerificationStatus.MANUAL_REVIEW:
        state.status = AgentStatus.MANUAL_REVIEW
    else:
        state.status = AgentStatus.FAILED

    state.final_report = {
        "report_version": "1.0",
        "session_id": state.session_id,
        "case_id": state.case_id,
        "task_type": state.task_type,
        "status": state.status,
        "verification_status": state.verification_status,
        "verification_reasons": state.metadata.get("verification_reasons", []),
        "citations": state.citations,
        "candidate_paths": state.candidate_paths,
        "safety_result": state.safety_result,
        "risk_flags": state.risk_flags,
        "skin_penetration_result": state.skin_penetration_result,
        "data_validation": {
            "geometry_valid": state.metadata.get("geometry_valid"),
            "label_schema_valid": state.metadata.get("label_schema_valid"),
            "segmentation_valid": state.metadata.get("segmentation_valid"),
        },
        "retry_count": state.retry_count,
        "error_count": len(state.errors),
    }
    return NodeOutcome(output=state.final_report)


def build_mock_handlers(
    tool_executor: ToolExecutor | None = None,
) -> dict[str, Any]:
    """Build all handlers required by the three checked-in JSON graphs."""

    executor = tool_executor or DeterministicMockToolExecutor()

    def bind(function: Any) -> Any:
        return lambda state, context: function(state, context, executor)

    return {
        "parse_request": _parse_request,
        "retrieve_project_knowledge": _retrieve_project_knowledge,
        "resolve_case_context": _resolve_case_context,
        "task_router": _noop,
        "request_missing_data": _request_missing_data,
        "result_verifier": _result_verifier,
        "error_recovery": _error_recovery,
        "report_generator": _report_generator,
        "inspect_case_metadata": bind(_inspect_case_metadata),
        "validate_geometry": _validate_geometry,
        "conversion_router": _noop,
        "convert_mcs_to_nifti": bind(_convert_mcs_to_nifti),
        "validate_label_schema": bind(_validate_label_schema),
        "segmentation_router": _noop,
        "run_segmentation": bind(_run_segmentation),
        "validate_segmentation_result": bind(_validate_segmentation_result),
        "skin_processing_router": _noop,
        "extract_skin_surface": bind(_extract_skin_surface),
        "finalize_data_model": _finalize_data_model,
        "ensure_required_artifacts": _ensure_required_artifacts,
        "artifact_router": _noop,
        "resolve_planning_constraints": _resolve_planning_constraints,
        "generate_candidate_paths": bind(_generate_candidate_paths),
        "candidate_router": _noop,
        "evaluate_path_safety": bind(_evaluate_path_safety),
        "evaluate_intraoperative_risk": bind(_evaluate_intraoperative_risk),
        "verify_skin_penetration": bind(_verify_skin_penetration),
        "finalize_planning": _finalize_planning,
    }
