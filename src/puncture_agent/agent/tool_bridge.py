"""Strict bridge from legacy Agent node requests to frozen MCP contracts.

The handlers in :mod:`puncture_agent.agent.nodes` intentionally expose a very
small ``execute(tool_name, mapping)`` boundary.  The algorithm tools, however,
accept the frozen request dataclasses from ``contracts.tool_inputs`` over MCP.
This module is the only compatibility layer between those two shapes.

Large artifacts remain opaque IDs.  Storage URIs, checksums, bytes, and
best-effort guesses are rejected at this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from math import dist, isfinite
import re
from threading import Lock
from types import UnionType
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from contracts.artifacts import ArtifactPublicView, ArtifactRef
from contracts.common import MetricValue, to_primitive
from contracts.errors import ErrorDetail
from puncture_agent.mcp import McpPrincipal
from puncture_agent.tooling.catalog import TOOL_DEFINITIONS


TOOL_NAMES: tuple[str, ...] = (
    "inspect_case_metadata",
    "convert_mcs_to_nifti",
    "validate_label_schema",
    "run_segmentation",
    "validate_segmentation_result",
    "extract_skin_surface",
    "generate_candidate_paths",
    "evaluate_path_safety",
    "evaluate_intraoperative_risk",
    "verify_skin_penetration",
)


class ToolBridgeError(ValueError):
    """Base error raised before an unsafe request can cross the MCP boundary."""


class ToolBridgeContextError(ToolBridgeError):
    """The executor lacks required session, case, or state context."""


class ToolBridgeContractError(ToolBridgeError):
    """A legacy request cannot be represented by the frozen tool contract."""


class ToolBridgeResponseError(ToolBridgeContractError):
    """A remote MCP response violated the frozen response contract."""


class ToolBridgeTransportError(RuntimeError):
    """A remote MCP transport failed before a contract response was available."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class ToolCaller(Protocol):
    """Minimal MCP caller used by the bridge.

    ``McpToolRuntime`` satisfies this protocol directly.  Remote clients may
    also implement it as long as their result exposes ``structured_content``.
    """

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    name: str
    value: int
    required: bool = True
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LabelMappingPolicy:
    source_name: str
    source_value: int
    target_name: str
    target_value: int


@dataclass(frozen=True, slots=True)
class LabelQualityPolicy:
    label_name: str
    min_voxel_count: int
    max_component_count: int
    min_volume_ml: float = 0.0
    max_volume_ml: float | None = None


@dataclass(frozen=True, slots=True)
class DangerMaskPolicy:
    """Approved interpretation of one legacy danger-mask key."""

    structure: str
    request_keys: tuple[str, ...]
    warning_mm: float
    stop_mm: float
    required: bool = True
    vessel_core_erosion_mm: float = 0.0


DEFAULT_LABELS: tuple[LabelPolicy, ...] = (
    LabelPolicy("background", 0),
    LabelPolicy("skin", 1),
    LabelPolicy("lung", 2),
    LabelPolicy("heart", 3),
)

DEFAULT_LABEL_MAPPING: tuple[LabelMappingPolicy, ...] = (
    LabelMappingPolicy("Skin", 1, "skin", 1),
    LabelMappingPolicy("Lung", 2, "lung", 2),
    LabelMappingPolicy("Heart", 3, "heart", 3),
)

DEFAULT_QUALITY_THRESHOLDS: tuple[LabelQualityPolicy, ...] = (
    LabelQualityPolicy("skin", 100, 10),
    LabelQualityPolicy("heart", 100, 5),
)

DEFAULT_DANGER_MASKS: tuple[DangerMaskPolicy, ...] = (
    DangerMaskPolicy("HEART", ("heart",), 8.0, 3.0),
    DangerMaskPolicy("BONE", ("bone",), 5.0, 2.0),
    DangerMaskPolicy("BRONCHUS", ("bronchus",), 5.0, 2.0),
    DangerMaskPolicy("LARGE_VESSEL", ("vessel", "large_vessel"), 8.0, 3.0),
    DangerMaskPolicy("LUNG", ("lung",), 5.0, 2.0),
    DangerMaskPolicy("SKIN", ("skin",), 2.0, 0.0, required=False),
)


@dataclass(frozen=True, slots=True)
class ToolBridgePolicy:
    """Immutable, versionable defaults needed by the legacy node requests.

    The legacy mappings omit these frozen-contract fields.  Keeping them in a
    frozen policy prevents request data or model output from silently changing
    label, model, planner, and risk behavior.
    """

    label_schema_version: str = "1.4"
    labels: tuple[LabelPolicy, ...] = DEFAULT_LABELS
    label_mapping: tuple[LabelMappingPolicy, ...] = DEFAULT_LABEL_MAPPING
    quality_thresholds: tuple[LabelQualityPolicy, ...] = DEFAULT_QUALITY_THRESHOLDS

    output_coordinate_system: str = "LPS"
    output_dtype: str = "uint16"
    overwrite_conversion: bool = False

    model_id: str = "nnunet-puncture"
    requested_labels: tuple[str, ...] = ("skin", "lung", "heart")
    inference_precision: str = "FP16"
    device_id: int = 0
    output_probability_maps: bool = False

    inspect_require_same_geometry: bool = True
    inspect_verify_checksums: bool = True
    segmentation_require_geometry_match: bool = True

    skin_surface_method: str = "EROSION_DIFFERENCE"
    skin_surface_thickness_mm: float = 2.0
    accepted_legacy_thickness_voxels: int = 2
    skin_surface_connectivity: int = 26
    keep_largest_skin_component: bool = True

    planner_angle_reference: str = "LOCAL_SURFACE_NORMAL"
    planner_entry_sampling_step_mm: float = 2.0
    planner_version: str = "planner-v1"

    danger_masks: tuple[DangerMaskPolicy, ...] = DEFAULT_DANGER_MASKS
    accepted_legacy_safety_radius_mm: float = 5.0
    needle_radius_mm: float = 1.0
    path_sampling_step_mm: float = 0.5
    reject_warning_intersection: bool = False
    risk_rule_version: str = "risk-v1"
    skin_sampling_step_voxel: float = 0.5
    min_depth_for_slip_mm: float = 5.0
    skin_label_value: int = 1

    def __post_init__(self) -> None:
        if not self.label_schema_version.strip():
            raise ValueError("label_schema_version is required")
        if not self.labels or not self.label_mapping or not self.requested_labels:
            raise ValueError("label, mapping, and model-label policies must not be empty")
        if not self.danger_masks:
            raise ValueError("danger-mask policy must not be empty")
        keys = [key for rule in self.danger_masks for key in rule.request_keys]
        if len(keys) != len(set(keys)):
            raise ValueError("danger-mask request keys must be unique")
        structures = [rule.structure for rule in self.danger_masks]
        if len(structures) != len(set(structures)):
            raise ValueError("danger-mask structures must be unique")
        if self.accepted_legacy_thickness_voxels < 1:
            raise ValueError("accepted legacy thickness must be positive")
        positive = (
            self.skin_surface_thickness_mm,
            self.planner_entry_sampling_step_mm,
            self.accepted_legacy_safety_radius_mm,
            self.needle_radius_mm,
            self.path_sampling_step_mm,
            self.skin_sampling_step_voxel,
            self.min_depth_for_slip_mm,
        )
        if any(value <= 0 or not isfinite(value) for value in positive):
            raise ValueError("physical policy values must be positive and finite")


DEFAULT_TOOL_BRIDGE_POLICY = ToolBridgePolicy()


@dataclass(frozen=True, slots=True)
class _Binding:
    session_id: str
    trace_id: str | None
    state: Any | None
    principal: McpPrincipal | None


_FORBIDDEN_STORAGE_KEYS = {
    "uri",
    "url",
    "storage_uri",
    "internal_uri",
    "signed_uri",
    "checksum",
    "checksum_sha256",
    "sha256_checksum",
}
_URI_IDENTITY_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_URI_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*:[^\s,;)\]}]+")
_CHECKSUM_VALUE_PATTERN = re.compile(
    r"(?i)(checksum(?:_sha256)?\s*(?:=|:)\s*)[0-9a-f]{64}"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class McpToolExecutor:
    """Expose one or more local MCP runtimes as the Agent ``ToolExecutor``.

    A production graph should bind its current :class:`AgentState` around node
    execution.  For small direct tests, ``session_id`` may instead be supplied
    to the constructor.
    """

    def __init__(
        self,
        callers: ToolCaller | Mapping[str, ToolCaller] | Iterable[ToolCaller],
        *,
        principal: McpPrincipal | Callable[[str], McpPrincipal],
        policy: ToolBridgePolicy = DEFAULT_TOOL_BRIDGE_POLICY,
        session_id: str | None = None,
        trace_id: str | None = None,
        deadline_ms: int | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id must be non-empty when provided")
        if trace_id is not None and not trace_id.strip():
            raise ValueError("trace_id must be non-empty when provided")
        if deadline_ms is not None and deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when provided")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._callers = self._route_callers(callers)
        self._principal_source = principal
        self.policy = policy
        self._session_id = session_id
        self._trace_id = trace_id
        self._deadline_ms = deadline_ms
        self._clock = clock
        self._binding: ContextVar[_Binding | None] = ContextVar(
            f"mcp_tool_executor_binding_{id(self)}", default=None
        )
        self._ordinal_lock = Lock()
        self._ordinal_by_key: dict[tuple[str, str, str, str, str], int] = {}
        self._next_ordinal: dict[tuple[str, str], int] = {}

    @contextmanager
    def bind_state(
        self,
        state: Any,
        *,
        trace_id: str | None = None,
        principal: McpPrincipal | None = None,
    ) -> Iterator["McpToolExecutor"]:
        """Bind AgentState identity/history for all calls in this context."""

        session_id = _state_value(state, "session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ToolBridgeContextError("bound state requires a non-empty session_id")
        selected_trace = trace_id or _state_trace_id(state) or self._trace_id
        if selected_trace is not None and not selected_trace.strip():
            raise ToolBridgeContextError("bound trace_id must be non-empty")
        token = self._binding.set(
            _Binding(session_id.strip(), selected_trace, state, principal)
        )
        try:
            yield self
        finally:
            self._binding.reset(token)

    def execute(self, tool_name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        """Map one legacy node request, invoke MCP, and return structured content."""

        caller, principal, arguments = self._prepare_call(tool_name, request)
        try:
            raw_result = caller.call_tool(tool_name, arguments, principal=principal)
        except TimeoutError as exc:
            raise ToolBridgeTransportError(
                "TIMEOUT",
                "MCP transport exceeded its bounded deadline",
                retryable=True,
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise ToolBridgeTransportError(
                "DEPENDENCY_FAILED",
                "MCP transport dependency is temporarily unavailable",
                retryable=True,
            ) from exc
        try:
            structured = _structured_content(raw_result)
            normalized = _validate_response_envelope(
                tool_name,
                arguments,
                raw_result,
                structured,
            )
            sanitized = _sanitize_response(normalized)
            if isinstance(sanitized, Mapping):
                sanitized = _validate_response_envelope(
                    tool_name,
                    arguments,
                    raw_result,
                    sanitized,
                )
        except ToolBridgeResponseError:
            raise
        except ToolBridgeContractError as exc:
            raise ToolBridgeResponseError(str(exc)) from exc
        if not isinstance(sanitized, dict):  # defensive; root is checked above
            raise ToolBridgeResponseError("MCP structured content must be an object")
        return sanitized

    def build_arguments(
        self, tool_name: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build wire arguments without invoking a caller (primarily for audits/tests)."""

        _, _, arguments = self._prepare_call(tool_name, request)
        return arguments

    def _prepare_call(
        self, tool_name: str, request: Mapping[str, Any]
    ) -> tuple[ToolCaller, McpPrincipal, dict[str, Any]]:
        if tool_name not in TOOL_NAMES:
            raise KeyError(f"unknown MCP tool: {tool_name}")
        caller = self._callers.get(tool_name)
        if caller is None:
            raise KeyError(f"no MCP caller configured for tool: {tool_name}")
        if not isinstance(request, Mapping):
            raise ToolBridgeContractError("legacy tool request must be an object")
        _reject_unsafe_value(request, path="request")

        binding = self._effective_binding()
        case_id = _required_string(request, "case_id", tool_name)
        state_case_id = _state_value(binding.state, "case_id")
        if state_case_id is not None and state_case_id != case_id:
            raise ToolBridgeContextError(
                "legacy request case_id does not match the bound AgentState"
            )
        principal = self._principal(case_id, binding)
        body = self._map_request(tool_name, request, binding)
        request_digest = _canonical_digest(body)
        ordinal = self._logical_call_ordinal(
            binding, case_id, tool_name, request, request_digest
        )
        identity = _canonical_digest(
            {
                "session_id": binding.session_id,
                "case_id": case_id,
                "caller": principal.subject,
                "tool_name": tool_name,
                "tool_version": TOOL_DEFINITIONS[tool_name].version,
                "call_ordinal": ordinal,
                "request_sha256": request_digest,
            }
        )
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ToolBridgeContextError("clock must return a timezone-aware datetime")
        now_utc = now.astimezone(timezone.utc)
        requested_at = now_utc.isoformat().replace("+00:00", "Z")
        deadline_epoch_ms = None
        if self._deadline_ms is not None:
            deadline_epoch_ms = int(now_utc.timestamp() * 1000) + self._deadline_ms
        trace_id = binding.trace_id or self._generated_trace_id(
            binding.session_id, case_id
        )
        arguments = {
            "context": {
                "request_id": f"req-{identity[:32]}",
                "trace_id": trace_id,
                "case_id": case_id,
                "caller": principal.subject,
                "idempotency_key": f"idem-{identity}",
                "requested_at": requested_at,
                "deadline_epoch_ms": deadline_epoch_ms,
            },
            **body,
        }
        return caller, principal, arguments

    def _effective_binding(self) -> _Binding:
        bound = self._binding.get()
        if bound is not None:
            return bound
        if self._session_id is None:
            raise ToolBridgeContextError(
                "bind AgentState or configure session_id before executing tools"
            )
        return _Binding(self._session_id, self._trace_id, None, None)

    def _principal(self, case_id: str, binding: _Binding) -> McpPrincipal:
        if binding.principal is not None:
            principal = binding.principal
        elif isinstance(self._principal_source, McpPrincipal):
            principal = self._principal_source
        else:
            principal = self._principal_source(case_id)
        if not isinstance(principal, McpPrincipal):
            raise ToolBridgeContextError("principal provider must return McpPrincipal")
        return principal

    @staticmethod
    def _generated_trace_id(session_id: str, case_id: str) -> str:
        digest = _canonical_digest({"session_id": session_id, "case_id": case_id})
        return f"trace-{digest[:32]}"

    def _logical_call_ordinal(
        self,
        binding: _Binding,
        case_id: str,
        tool_name: str,
        loose_request: Mapping[str, Any],
        request_digest: str,
    ) -> int:
        if binding.state is None:
            # The constructor-only convenience mode has no durable tool-call
            # history.  A process-local sequence would reset after restart and
            # change idempotency identity based on which earlier tools happened
            # to run.  Tool name + semantic request digest already distinguish
            # independent direct calls, so a stable ordinal of one is correct.
            return 1
        node_id = _state_value(binding.state, "current_node")
        node_key = node_id if isinstance(node_id, str) else ""
        loose_digest = _canonical_digest(loose_request)
        history = _state_value(binding.state, "tool_calls")
        seen: dict[tuple[str, str, str], int] = {}
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, Mapping):
                    continue
                historical_request = item.get("request")
                if not isinstance(historical_request, Mapping):
                    continue
                try:
                    historical_digest = _canonical_digest(historical_request)
                except (TypeError, ValueError):
                    continue
                key = (
                    str(item.get("node_id") or ""),
                    str(item.get("tool_name") or ""),
                    historical_digest,
                )
                if key not in seen:
                    seen[key] = len(seen) + 1
        current = (node_key, tool_name, loose_digest)
        if current in seen:
            return seen[current]

        cache_key = (
            binding.session_id,
            case_id,
            node_key,
            tool_name,
            request_digest,
        )
        scope = (binding.session_id, case_id)
        with self._ordinal_lock:
            cached = self._ordinal_by_key.get(cache_key)
            if cached is not None:
                return cached
            next_value = max(self._next_ordinal.get(scope, 0), len(seen)) + 1
            self._next_ordinal[scope] = next_value
            self._ordinal_by_key[cache_key] = next_value
            return next_value

    def _map_request(
        self,
        tool_name: str,
        request: Mapping[str, Any],
        binding: _Binding,
    ) -> dict[str, Any]:
        mapper = {
            "inspect_case_metadata": self._map_inspect,
            "convert_mcs_to_nifti": self._map_conversion,
            "validate_label_schema": self._map_label_validation,
            "run_segmentation": self._map_segmentation,
            "validate_segmentation_result": self._map_segmentation_validation,
            "extract_skin_surface": self._map_skin_surface,
            "generate_candidate_paths": self._map_candidate_generation,
            "evaluate_path_safety": self._map_path_safety,
            "evaluate_intraoperative_risk": self._map_risk,
            "verify_skin_penetration": self._map_penetration,
        }[tool_name]
        return mapper(request, binding)

    def _map_inspect(self, request: Mapping[str, Any], _: _Binding) -> dict[str, Any]:
        tool = "inspect_case_metadata"
        required = {"case_id", "ct_artifact_id", "input_format"}
        optional = {"related_artifact_ids"}
        missing = sorted(required - set(request))
        unknown = sorted(set(request) - required - optional)
        if missing or unknown:
            raise ToolBridgeContractError(
                f"{tool} legacy request does not match the approved fields"
            )
        input_format = _required_string(request, "input_format", tool).upper()
        if input_format not in {"NIFTI", "MCS"}:
            raise ToolBridgeContractError("inspect input_format must be NIFTI or MCS")
        related_ids = request.get("related_artifact_ids", ())
        if isinstance(related_ids, (str, bytes)) or not isinstance(
            related_ids, (list, tuple)
        ):
            raise ToolBridgeContractError("related_artifact_ids must be an array")
        return {
            "ct_artifact": _artifact_handle(request["ct_artifact_id"], "ct_artifact_id"),
            "related_artifacts": [
                _artifact_handle(value, f"related_artifact_ids[{index}]")
                for index, value in enumerate(related_ids)
            ],
            "required_artifact_types": [
                "MCS_SEGMENTATION" if input_format == "MCS" else "NIFTI_LABELMAP"
            ],
            "require_same_geometry": self.policy.inspect_require_same_geometry,
            "verify_checksums": self.policy.inspect_verify_checksums,
        }

    def _map_conversion(self, request: Mapping[str, Any], _: _Binding) -> dict[str, Any]:
        tool = "convert_mcs_to_nifti"
        _require_fields(
            tool,
            request,
            {"case_id", "source_artifact_id", "reference_ct_artifact_id"},
        )
        return {
            "mcs_artifact": _artifact_handle(
                request["source_artifact_id"], "source_artifact_id"
            ),
            "reference_ct_artifact": _artifact_handle(
                request["reference_ct_artifact_id"], "reference_ct_artifact_id"
            ),
            "label_mapping": [
                {
                    "source_name": item.source_name,
                    "source_value": item.source_value,
                    "target_name": item.target_name,
                    "target_value": item.target_value,
                }
                for item in self.policy.label_mapping
            ],
            "output_coordinate_system": self.policy.output_coordinate_system,
            "output_dtype": self.policy.output_dtype,
            "overwrite": self.policy.overwrite_conversion,
        }

    def _map_label_validation(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "validate_label_schema"
        _require_fields(
            tool,
            request,
            {"case_id", "label_artifact_id", "schema_version"},
        )
        version = _required_string(request, "schema_version", tool)
        if version != self.policy.label_schema_version:
            raise ToolBridgeContractError(
                "requested label schema version is not present in the active bridge policy"
            )
        return {
            "labelmap_artifact": _artifact_handle(
                request["label_artifact_id"], "label_artifact_id"
            ),
            "expected_labels": _wire_labels(self.policy.labels),
            "allow_unknown_values": False,
            "require_all_required_labels": True,
        }

    def _map_segmentation(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "run_segmentation"
        _require_fields(
            tool, request, {"case_id", "ct_artifact_id", "model_version"}
        )
        return {
            "ct_artifact": _artifact_handle(request["ct_artifact_id"], "ct_artifact_id"),
            "model_id": self.policy.model_id,
            "model_version": _required_string(request, "model_version", tool),
            "requested_labels": list(self.policy.requested_labels),
            "precision": self.policy.inference_precision,
            "device_id": self.policy.device_id,
            "output_probability_maps": self.policy.output_probability_maps,
        }

    def _map_segmentation_validation(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "validate_segmentation_result"
        _require_fields(
            tool,
            request,
            {"case_id", "mask_artifact_ids", "reference_ct_artifact_id"},
        )
        masks = request["mask_artifact_ids"]
        if not isinstance(masks, Mapping) or not masks:
            raise ToolBridgeContractError("mask_artifact_ids must be a non-empty object")
        artifact_ids = []
        for key, value in masks.items():
            if not isinstance(key, str):
                raise ToolBridgeContractError("mask_artifact_ids keys must be strings")
            artifact_ids.append(_artifact_id(value, f"mask_artifact_ids.{key}"))
        unique_ids = tuple(dict.fromkeys(artifact_ids))
        if len(unique_ids) != 1:
            raise ToolBridgeContractError(
                "multiple per-label mask artifacts cannot be represented as the frozen "
                "single segmentation_artifact"
            )
        return {
            "ct_artifact": _artifact_handle(
                request["reference_ct_artifact_id"], "reference_ct_artifact_id"
            ),
            "segmentation_artifact": {"artifact_id": unique_ids[0]},
            "expected_labels": _wire_labels(self.policy.labels),
            "quality_thresholds": [
                {
                    "label_name": item.label_name,
                    "min_voxel_count": item.min_voxel_count,
                    "max_component_count": item.max_component_count,
                    "min_volume_ml": item.min_volume_ml,
                    "max_volume_ml": item.max_volume_ml,
                }
                for item in self.policy.quality_thresholds
            ],
            "require_geometry_match": self.policy.segmentation_require_geometry_match,
        }

    def _map_skin_surface(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "extract_skin_surface"
        _require_fields(
            tool,
            request,
            {"case_id", "skin_mask_artifact_id", "thickness_voxels"},
        )
        thickness = _required_int(request, "thickness_voxels", tool)
        if thickness != self.policy.accepted_legacy_thickness_voxels:
            raise ToolBridgeContractError(
                "legacy voxel thickness has no safe physical-mm conversion in the bridge"
            )
        return {
            "skin_mask_artifact": _artifact_handle(
                request["skin_mask_artifact_id"], "skin_mask_artifact_id"
            ),
            "method": self.policy.skin_surface_method,
            "thickness_mm": self.policy.skin_surface_thickness_mm,
            "connectivity": self.policy.skin_surface_connectivity,
            "keep_largest_component": self.policy.keep_largest_skin_component,
        }

    def _map_candidate_generation(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "generate_candidate_paths"
        _require_fields(
            tool,
            request,
            {
                "case_id",
                "ct_artifact_id",
                "skin_surface_artifact_id",
                "target_artifact_id",
                "max_needle_length_mm",
                "max_insertion_angle_deg",
                "safety_radius_mm",
                "top_k",
            },
        )
        maximum_length = _required_number(request, "max_needle_length_mm", tool)
        maximum_angle = _required_number(request, "max_insertion_angle_deg", tool)
        safety_radius = _required_number(request, "safety_radius_mm", tool)
        maximum_candidates = _required_int(request, "top_k", tool)
        if maximum_length <= 0 or not 0 < maximum_angle <= 90:
            raise ToolBridgeContractError("invalid candidate length or insertion angle")
        if safety_radius <= 0:
            raise ToolBridgeContractError("safety_radius_mm must be positive")
        if maximum_candidates < 1:
            raise ToolBridgeContractError("top_k must be at least 1")
        return {
            "ct_artifact": _artifact_handle(request["ct_artifact_id"], "ct_artifact_id"),
            "skin_surface_artifact": _artifact_handle(
                request["skin_surface_artifact_id"], "skin_surface_artifact_id"
            ),
            "target_artifact": _artifact_handle(
                request["target_artifact_id"], "target_artifact_id"
            ),
            "lesion_artifact": None,
            "target_point_world_mm": None,
            "max_needle_length_mm": maximum_length,
            "max_insertion_angle_deg": maximum_angle,
            "angle_reference": self.policy.planner_angle_reference,
            "max_candidates": maximum_candidates,
            "entry_sampling_step_mm": self.policy.planner_entry_sampling_step_mm,
            "planner_version": self.policy.planner_version,
        }

    def _map_path_safety(
        self, request: Mapping[str, Any], binding: _Binding
    ) -> dict[str, Any]:
        tool = "evaluate_path_safety"
        _require_fields(
            tool,
            request,
            {
                "case_id",
                "candidate_paths",
                "danger_mask_artifact_ids",
                "safety_radius_mm",
            },
        )
        radius = _required_number(request, "safety_radius_mm", tool)
        if radius != self.policy.accepted_legacy_safety_radius_mm:
            raise ToolBridgeContractError(
                "legacy safety_radius_mm cannot override frozen needle/risk policy"
            )
        ct_id = _bound_artifact_id(binding.state, ("ct",), "evaluate_path_safety CT")
        return {
            "ct_artifact": {"artifact_id": ct_id},
            "candidate_paths": _candidate_paths(request["candidate_paths"]),
            "danger_masks": self._danger_masks(request["danger_mask_artifact_ids"]),
            "needle_radius_mm": self.policy.needle_radius_mm,
            "path_sampling_step_mm": self.policy.path_sampling_step_mm,
            "reject_warning_intersection": self.policy.reject_warning_intersection,
        }

    def _map_risk(self, request: Mapping[str, Any], binding: _Binding) -> dict[str, Any]:
        tool = "evaluate_intraoperative_risk"
        _require_fields(
            tool,
            request,
            {
                "case_id",
                "planned_entry_point_world_mm",
                "needle_tip_world_mm",
                "danger_mask_artifact_ids",
            },
        )
        planned = _world_point(
            request["planned_entry_point_world_mm"], "planned_entry_point_world_mm"
        )
        current = _world_point(request["needle_tip_world_mm"], "needle_tip_world_mm")
        ct_id = _bound_artifact_id(
            binding.state, ("ct",), "evaluate_intraoperative_risk CT"
        )
        lung_id = _optional_bound_artifact_id(
            binding.state, ("segmentation_masks", "lung")
        )
        skin_id = _optional_bound_artifact_id(binding.state, ("skin",))
        return {
            "ct_artifact": {"artifact_id": ct_id},
            "planned_entry_world_mm": planned,
            "current_tip_world_mm": current,
            "insertion_depth_mm": _world_distance(planned, current),
            "danger_masks": self._danger_masks(request["danger_mask_artifact_ids"]),
            "lung_mask_artifact": None if lung_id is None else {"artifact_id": lung_id},
            "skin_mask_artifact": None if skin_id is None else {"artifact_id": skin_id},
            "risk_rule_version": self.policy.risk_rule_version,
        }

    def _map_penetration(
        self, request: Mapping[str, Any], _: _Binding
    ) -> dict[str, Any]:
        tool = "verify_skin_penetration"
        _require_fields(
            tool,
            request,
            {
                "case_id",
                "skin_mask_artifact_id",
                "planned_entry_point_world_mm",
                "needle_tip_world_mm",
                "sample_step_voxel",
            },
        )
        requested_step = _required_number(request, "sample_step_voxel", tool)
        if requested_step != self.policy.skin_sampling_step_voxel:
            raise ToolBridgeContractError(
                "sample_step_voxel differs from the frozen penetration policy"
            )
        planned = _world_point(
            request["planned_entry_point_world_mm"], "planned_entry_point_world_mm"
        )
        current = _world_point(request["needle_tip_world_mm"], "needle_tip_world_mm")
        return {
            "skin_mask_artifact": _artifact_handle(
                request["skin_mask_artifact_id"], "skin_mask_artifact_id"
            ),
            "planned_entry_world_mm": planned,
            "current_tip_world_mm": current,
            "insertion_depth_mm": _world_distance(planned, current),
            "sampling_step_voxel": requested_step,
            "min_depth_for_slip_mm": self.policy.min_depth_for_slip_mm,
            "skin_label_value": self.policy.skin_label_value,
        }

    def _danger_masks(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, Mapping) or not value:
            raise ToolBridgeContractError(
                "danger_mask_artifact_ids must be a non-empty object"
            )
        if any(not isinstance(key, str) for key in value):
            raise ToolBridgeContractError("danger mask keys must be strings")
        known_keys = {
            key for rule in self.policy.danger_masks for key in rule.request_keys
        }
        unknown = sorted(set(value) - known_keys)
        if unknown:
            raise ToolBridgeContractError(
                "danger mask keys lack an approved risk-policy mapping: "
                + ", ".join(unknown)
            )
        result: list[dict[str, Any]] = []
        missing_required: list[str] = []
        for rule in self.policy.danger_masks:
            present = [key for key in rule.request_keys if key in value]
            if len(present) > 1:
                raise ToolBridgeContractError(
                    f"multiple aliases supplied for danger structure {rule.structure}"
                )
            if not present:
                if rule.required:
                    missing_required.append(rule.structure)
                continue
            artifact_id = _artifact_id(value[present[0]], f"danger mask {present[0]}")
            result.append(
                {
                    "structure": rule.structure,
                    "artifact": {"artifact_id": artifact_id},
                    "safety_margin": {
                        "warning_mm": rule.warning_mm,
                        "stop_mm": rule.stop_mm,
                    },
                    "required": rule.required,
                    "vessel_core_erosion_mm": rule.vessel_core_erosion_mm,
                }
            )
        if missing_required:
            raise ToolBridgeContractError(
                "required danger masks are missing: " + ", ".join(missing_required)
            )
        if not result:
            raise ToolBridgeContractError("no approved danger masks were supplied")
        return result

    @staticmethod
    def _route_callers(
        callers: ToolCaller | Mapping[str, ToolCaller] | Iterable[ToolCaller],
    ) -> dict[str, ToolCaller]:
        routes: dict[str, ToolCaller] = {}

        def register(caller: ToolCaller, names: Iterable[str]) -> None:
            if not hasattr(caller, "call_tool"):
                raise TypeError("MCP caller must define call_tool")
            for name in names:
                if name not in TOOL_NAMES:
                    continue
                existing = routes.get(name)
                if existing is not None and existing is not caller:
                    raise ValueError(f"multiple MCP callers configured for tool: {name}")
                routes[name] = caller

        if isinstance(callers, Mapping):
            for key, caller in callers.items():
                if key in TOOL_NAMES:
                    register(caller, (key,))
                    continue
                advertised = getattr(caller, "tool_names", ())
                if not advertised:
                    raise ValueError(
                        f"caller mapping key {key!r} is not a tool and caller advertises no tool_names"
                    )
                register(caller, advertised)
        elif hasattr(callers, "call_tool"):
            advertised = getattr(callers, "tool_names", ())
            if not advertised:
                raise ValueError("a single MCP caller must advertise tool_names")
            register(callers, advertised)
        else:
            for caller in callers:
                advertised = getattr(caller, "tool_names", ())
                if not advertised:
                    raise ValueError("each MCP caller must advertise tool_names")
                register(caller, advertised)
        if not routes:
            raise ValueError("at least one known MCP tool caller is required")
        return routes


McpToolExecutorAdapter = McpToolExecutor
LocalMcpToolExecutor = McpToolExecutor


def _require_fields(tool: str, request: Mapping[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(request))
    unknown = sorted(set(request) - required)
    if missing:
        raise ToolBridgeContractError(
            f"{tool} legacy request is missing fields: {', '.join(missing)}"
        )
    if unknown:
        if any(_is_forbidden_storage_key(key) for key in unknown):
            raise ToolBridgeContractError("legacy request contains forbidden storage fields")
        raise ToolBridgeContractError(
            f"{tool} legacy request has unknown fields: {', '.join(unknown)}"
        )


def _required_string(request: Mapping[str, Any], field: str, tool: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ToolBridgeContractError(f"{tool}.{field} must be a non-empty string")
    return value.strip()


def _required_number(request: Mapping[str, Any], field: str, tool: str) -> float:
    value = request.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolBridgeContractError(f"{tool}.{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ToolBridgeContractError(f"{tool}.{field} must be a finite number")
    return result


def _required_int(request: Mapping[str, Any], field: str, tool: str) -> int:
    value = request.get(field)
    if type(value) is not int:
        raise ToolBridgeContractError(f"{tool}.{field} must be an integer")
    return value


def _artifact_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolBridgeContractError(f"{field} must be an opaque artifact ID")
    normalized = value.strip()
    if _URI_IDENTITY_PATTERN.match(normalized):
        raise ToolBridgeContractError(
            f"{field} must be an opaque artifact ID, not a storage URI"
        )
    return normalized


def _artifact_handle(value: Any, field: str) -> dict[str, str]:
    return {"artifact_id": _artifact_id(value, field)}


def _wire_labels(labels: tuple[LabelPolicy, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "value": item.value,
            "required": item.required,
            "aliases": list(item.aliases),
        }
        for item in labels
    ]


def _world_point(value: Any, field: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        expected = {"x_mm", "y_mm", "z_mm"}
        if set(value) != expected:
            raise ToolBridgeContractError(
                f"{field} must contain exactly x_mm, y_mm, and z_mm"
            )
        raw = (value["x_mm"], value["y_mm"], value["z_mm"])
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw = tuple(value)
    else:
        raise ToolBridgeContractError(f"{field} must be a three-number world point")
    coordinates = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ToolBridgeContractError(f"{field} must contain finite numbers")
        number = float(item)
        if not isfinite(number):
            raise ToolBridgeContractError(f"{field} must contain finite numbers")
        coordinates.append(number)
    return {"x_mm": coordinates[0], "y_mm": coordinates[1], "z_mm": coordinates[2]}


def _world_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return dist(
        (first["x_mm"], first["y_mm"], first["z_mm"]),
        (second["x_mm"], second["y_mm"], second["z_mm"]),
    )


def _candidate_paths(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ToolBridgeContractError("candidate_paths must be a non-empty array")
    result: list[dict[str, Any]] = []
    required = {
        "candidate_id",
        "entry_point_world_mm",
        "target_point_world_mm",
        "length_mm",
        "insertion_angle_deg",
        "angle_reference",
        "rank_hint",
    }
    optional = {"path_artifact_id"}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ToolBridgeContractError(f"candidate_paths[{index}] must be an object")
        missing = required - set(item)
        unknown = set(item) - required - optional
        if missing or unknown:
            raise ToolBridgeContractError(
                f"candidate_paths[{index}] does not match the frozen candidate schema"
            )
        candidate_id = item["candidate_id"]
        angle_reference = item["angle_reference"]
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ToolBridgeContractError("candidate_id must be a non-empty string")
        if not isinstance(angle_reference, str) or not angle_reference.strip():
            raise ToolBridgeContractError("candidate angle_reference must be a string")
        length = _finite_number(item["length_mm"], "candidate length_mm")
        angle = _finite_number(item["insertion_angle_deg"], "candidate insertion_angle_deg")
        rank = item["rank_hint"]
        if length <= 0 or not 0 <= angle <= 180 or type(rank) is not int or rank < 1:
            raise ToolBridgeContractError("candidate length, angle, or rank is invalid")
        path_artifact_id = item.get("path_artifact_id")
        if path_artifact_id is not None:
            path_artifact_id = _artifact_id(path_artifact_id, "path_artifact_id")
        result.append(
            {
                "candidate_id": candidate_id.strip(),
                "entry_point_world_mm": _world_point(
                    item["entry_point_world_mm"], "entry_point_world_mm"
                ),
                "target_point_world_mm": _world_point(
                    item["target_point_world_mm"], "target_point_world_mm"
                ),
                "length_mm": length,
                "insertion_angle_deg": angle,
                "angle_reference": angle_reference.strip(),
                "rank_hint": rank,
                "path_artifact_id": path_artifact_id,
            }
        )
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolBridgeContractError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ToolBridgeContractError(f"{field} must be a finite number")
    return result


def _state_value(state: Any | None, name: str) -> Any:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _state_trace_id(state: Any) -> str | None:
    direct = _state_value(state, "trace_id")
    if isinstance(direct, str):
        return direct
    metadata = _state_value(state, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("trace_id") or metadata.get("run_trace_id")
        if isinstance(value, str):
            return value
    return None


def _optional_bound_artifact_id(state: Any | None, path: tuple[str, ...]) -> str | None:
    artifacts = _state_value(state, "artifacts")
    current: Any = artifacts
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    return _artifact_id(current, "bound state artifact")


def _bound_artifact_id(state: Any | None, path: tuple[str, ...], label: str) -> str:
    value = _optional_bound_artifact_id(state, path)
    if value is None:
        raise ToolBridgeContextError(f"bound AgentState is required to resolve {label}")
    return value


def _canonical_digest(value: Any) -> str:
    primitive = _canonical_value(value, path="$")
    encoded = json.dumps(
        primitive,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ToolBridgeContractError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolBridgeContractError(f"{path} contains a non-string object key")
            result[key] = _canonical_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ToolBridgeContractError("raw byte payloads are forbidden at the tool boundary")
    raise ToolBridgeContractError(f"{path} is not JSON-compatible")


def _reject_unsafe_value(value: Any, *, path: str) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ToolBridgeContractError("raw byte payloads are forbidden at the tool boundary")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolBridgeContractError(f"{path} contains a non-string object key")
            if _is_forbidden_storage_key(key):
                raise ToolBridgeContractError("legacy request contains forbidden storage fields")
            _reject_unsafe_value(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unsafe_value(item, path=f"{path}[{index}]")


def _is_forbidden_storage_key(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered == "checksum_valid":
        return False
    return (
        lowered in _FORBIDDEN_STORAGE_KEYS
        or lowered.endswith("_uri")
        or lowered.startswith("uri_")
        or lowered.endswith("_checksum")
        or lowered.endswith("_checksum_sha256")
    )


def _structured_content(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        if "structured_content" in result and "structuredContent" in result:
            raise ToolBridgeResponseError(
                "MCP result contains conflicting structured-content aliases"
            )
        if "structured_content" in result:
            structured = result["structured_content"]
        elif "structuredContent" in result:
            structured = result["structuredContent"]
        elif "status" in result:
            structured = result
        else:
            raise ToolBridgeContractError("MCP result has no structured content")
    else:
        structured = getattr(result, "structured_content", None)
    if not isinstance(structured, Mapping):
        raise ToolBridgeContractError("MCP structured content must be an object")
    return structured


def _outer_is_error(result: Any) -> bool | None:
    if isinstance(result, Mapping):
        if "is_error" in result and "isError" in result:
            raise ToolBridgeResponseError(
                "MCP result contains conflicting isError aliases"
            )
        if "is_error" in result:
            value = result["is_error"]
        elif "isError" in result:
            value = result["isError"]
        else:
            return None
    else:
        value = getattr(result, "is_error", None)
        if value is None:
            return None
    if not isinstance(value, bool):
        raise ToolBridgeContractError("MCP isError must be a boolean")
    return value


def _validate_response_envelope(
    tool_name: str,
    arguments: Mapping[str, Any],
    raw_result: Any,
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    context = arguments.get("context")
    if not isinstance(context, Mapping):  # constructed internally; defensive only
        raise ToolBridgeContractError("tool request context is missing")
    expected_case_id = context.get("case_id")
    if not isinstance(expected_case_id, str) or not expected_case_id:
        raise ToolBridgeContractError("tool request context case_id is missing")
    envelope_fields = {
        "request_id",
        "trace_id",
        "tool_name",
        "tool_version",
        "status",
        "result",
        "artifacts",
        "metrics",
        "warnings",
        "error",
        "started_at",
        "finished_at",
    }
    missing = sorted(envelope_fields - set(structured))
    unknown = sorted(set(structured) - envelope_fields)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ToolBridgeContractError(
            "MCP response fields do not match the frozen envelope: " + "; ".join(details)
        )
    expected = {
        "request_id": context.get("request_id"),
        "trace_id": context.get("trace_id"),
        "tool_name": tool_name,
        "tool_version": TOOL_DEFINITIONS[tool_name].version,
    }
    for field, value in expected.items():
        if structured.get(field) != value:
            raise ToolBridgeContractError(
                f"MCP response {field} does not match the request identity"
            )
    status = structured.get("status")
    if not isinstance(status, str) or status not in {
        "SUCCESS",
        "PARTIAL",
        "FAILED",
    }:
        raise ToolBridgeContractError("MCP response status is not canonical")
    result = structured.get("result")
    error = structured.get("error")
    if status == "SUCCESS" and (result is None or error is not None):
        raise ToolBridgeContractError("MCP SUCCESS envelope is internally inconsistent")
    if status == "FAILED" and not isinstance(error, Mapping):
        raise ToolBridgeContractError("MCP FAILED envelope requires a structured error")
    if status == "PARTIAL" and result is None:
        raise ToolBridgeContractError("MCP PARTIAL envelope requires a consumable result")
    if status == "FAILED" and result is not None:
        raise ToolBridgeContractError("MCP FAILED envelope must not expose a result")
    normalized_result = None
    if result is not None:
        normalized_result = _validate_frozen_value(
            TOOL_DEFINITIONS[tool_name].result_type,
            result,
            path="$.result",
        )
        _validate_response_case_scope(
            to_primitive(normalized_result),
            expected_case_id=expected_case_id,
            path="$.result",
        )
    normalized_artifacts = _validate_frozen_value(
        tuple[ArtifactRef, ...], structured["artifacts"], path="$.artifacts"
    )
    _validate_response_case_scope(
        to_primitive(normalized_artifacts),
        expected_case_id=expected_case_id,
        path="$.artifacts",
    )
    normalized_metrics = _validate_frozen_value(
        tuple[MetricValue, ...], structured["metrics"], path="$.metrics"
    )
    normalized_warnings = _validate_frozen_value(
        tuple[str, ...], structured["warnings"], path="$.warnings"
    )
    normalized_error = _validate_frozen_value(
        ErrorDetail | None,
        error,
        path="$.error",
    )
    for timestamp_field in ("started_at", "finished_at"):
        timestamp = structured[timestamp_field]
        _validate_frozen_value(str, timestamp, path=f"$.{timestamp_field}")
        if not timestamp.endswith("Z"):
            raise ToolBridgeContractError(
                f"MCP response {timestamp_field} must be an ISO-8601 UTC string ending in Z"
            )
    outer_error = _outer_is_error(raw_result)
    if outer_error is not None and outer_error != (status == "FAILED"):
        raise ToolBridgeContractError("MCP isError conflicts with structured status")
    normalized = dict(structured)
    normalized["result"] = to_primitive(normalized_result)
    normalized["artifacts"] = to_primitive(normalized_artifacts)
    normalized["metrics"] = to_primitive(normalized_metrics)
    normalized["warnings"] = to_primitive(normalized_warnings)
    normalized["error"] = to_primitive(normalized_error)
    primitive = to_primitive(normalized)
    if not isinstance(primitive, dict):  # defensive: the envelope root is fixed
        raise ToolBridgeContractError("MCP response normalization failed")
    _validate_response_artifact_lineage(
        arguments,
        primitive["result"],
        primitive["artifacts"],
    )
    return primitive


def _validate_response_artifact_lineage(
    arguments: Mapping[str, Any],
    result: Any,
    artifacts: Any,
) -> None:
    """Bind response artifact identities to request inputs or envelope outputs.

    This prevents a result field from silently naming an unrelated artifact or
    disagreeing with the corresponding frozen public view.  A production
    deployment must additionally resolve envelope outputs against its trusted
    registry; the bridge cannot infer registry ownership from a remote claim.
    """

    request_ids = _collect_request_artifact_ids(arguments)
    if not isinstance(artifacts, list):
        raise ToolBridgeContractError("$.artifacts must be an array")
    envelope_by_id: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ToolBridgeContractError(f"$.artifacts[{index}] must be an object")
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ToolBridgeContractError(
                f"$.artifacts[{index}].artifact_id must be an opaque artifact ID"
            )
        if artifact_id in envelope_by_id:
            raise ToolBridgeContractError(
                "MCP response contains duplicate envelope artifact IDs"
            )
        envelope_by_id[artifact_id] = artifact
    allowed_ids = request_ids | set(envelope_by_id)

    def visit(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            artifact_id = value.get("artifact_id")
            if isinstance(artifact_id, str):
                normalized_artifact_id = _artifact_id(
                    artifact_id,
                    f"{path}.artifact_id",
                )
                if normalized_artifact_id != artifact_id:
                    raise ToolBridgeResponseError(
                        f"{path}.artifact_id must not contain surrounding whitespace"
                    )
                if artifact_id not in allowed_ids:
                    raise ToolBridgeResponseError(
                        f"{path}.artifact_id is not bound to a request or envelope artifact"
                    )
                public_fields = {
                    "artifact_id",
                    "case_id",
                    "artifact_type",
                    "status",
                    "producer_name",
                    "producer_version",
                    "geometry_fingerprint",
                }
                if public_fields <= set(value):
                    envelope = envelope_by_id.get(artifact_id)
                    if envelope is None or dict(value) != dict(envelope):
                        raise ToolBridgeResponseError(
                            f"{path} does not match its envelope artifact"
                        )
            for key, item in value.items():
                child_path = f"{path}.{key}"
                if _is_artifact_identity_key(key):
                    for identity in _identity_strings(item, path=child_path):
                        if identity not in allowed_ids:
                            raise ToolBridgeResponseError(
                                f"{child_path} is not bound to a request or envelope artifact"
                            )
                visit(item, path=child_path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, path=f"{path}[{index}]")

    if result is not None:
        visit(result, path="$.result")


def _collect_request_artifact_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        artifact_id = value.get("artifact_id")
        if isinstance(artifact_id, str):
            result.add(artifact_id)
        for item in value.values():
            result.update(_collect_request_artifact_ids(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_collect_request_artifact_ids(item))
    return result


def _identity_strings(value: Any, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = _artifact_id(value, path)
        if normalized != value:
            raise ToolBridgeResponseError(
                f"{path} must not contain surrounding whitespace"
            )
        return (normalized,)
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ToolBridgeContractError(f"{path} must contain artifact ID strings")
        normalized = tuple(
            _artifact_id(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
        if normalized != tuple(value):
            raise ToolBridgeResponseError(
                f"{path} must not contain surrounding whitespace"
            )
        return normalized
    raise ToolBridgeContractError(f"{path} must be an artifact ID or array")


def _is_artifact_identity_key(key: str) -> bool:
    lowered = key.strip().lower()
    return (
        lowered == "artifact_id"
        or lowered.endswith("_artifact_id")
        or lowered.endswith("_artifact_ids")
    )


def _validate_response_case_scope(
    value: Any,
    *,
    expected_case_id: str,
    path: str,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == "case_id" and item != expected_case_id:
                raise ToolBridgeResponseError(
                    f"{child_path} does not match the request case"
                )
            _validate_response_case_scope(
                item,
                expected_case_id=expected_case_id,
                path=child_path,
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_response_case_scope(
                item,
                expected_case_id=expected_case_id,
                path=f"{path}[{index}]",
            )


def _validate_frozen_value(expected_type: Any, value: Any, *, path: str) -> Any:
    """Validate and reconstruct one value using its frozen contract type."""

    if expected_type is Any:
        return _canonical_value(value, path=path)
    if expected_type is ArtifactRef:
        expected_type = ArtifactPublicView

    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if origin in (Union, UnionType):
        for option in arguments:
            try:
                return _validate_frozen_value(option, value, path=path)
            except ToolBridgeContractError:
                continue
        raise ToolBridgeContractError(f"{path} does not match any frozen contract type")

    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ToolBridgeContractError(f"{path} must be an array")
        if origin is tuple and arguments and not (
            len(arguments) == 2 and arguments[1] is Ellipsis
        ):
            if len(value) != len(arguments):
                raise ToolBridgeContractError(
                    f"{path} must contain exactly {len(arguments)} items"
                )
            item_types = arguments
        else:
            item_type = arguments[0] if arguments else Any
            item_types = (item_type,) * len(value)
        converted = [
            _validate_frozen_value(item_type, item, path=f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(item_types, value))
        ]
        return tuple(converted) if origin is tuple else converted

    if origin in (dict, Mapping, MappingABC):
        if not isinstance(value, Mapping):
            raise ToolBridgeContractError(f"{path} must be an object")
        key_type, item_type = arguments if len(arguments) == 2 else (str, Any)
        if key_type is not str:
            raise ToolBridgeContractError(f"{path} uses an unsupported non-string key type")
        converted_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolBridgeContractError(f"{path} contains a non-string object key")
            converted_mapping[key] = _validate_frozen_value(
                item_type,
                item,
                path=f"{path}.{key}",
            )
        return converted_mapping

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        for member in expected_type:
            if value == member.value and type(value) is type(member.value):
                return member
        raise ToolBridgeContractError(f"{path} is not a canonical enum value")

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise ToolBridgeContractError(f"{path} must be an object")
        hints = get_type_hints(expected_type)
        contract_fields = {item.name: item for item in fields(expected_type)}
        unknown = sorted(set(value) - set(contract_fields))
        if unknown:
            raise ToolBridgeContractError(
                f"{path} contains unknown frozen-contract fields: {', '.join(unknown)}"
            )
        converted_fields: dict[str, Any] = {}
        for name, field in contract_fields.items():
            if name not in value:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise ToolBridgeContractError(f"{path}.{name} is required")
                continue
            converted_fields[name] = _validate_frozen_value(
                hints.get(name, Any),
                value[name],
                path=f"{path}.{name}",
            )
        try:
            return expected_type(**converted_fields)
        except Exception as exc:
            raise ToolBridgeContractError(
                f"{path} violates frozen contract invariants"
            ) from exc

    if expected_type is str:
        if not isinstance(value, str):
            raise ToolBridgeContractError(f"{path} must be a string")
        return value
    if expected_type is bool:
        if type(value) is not bool:
            raise ToolBridgeContractError(f"{path} must be a boolean")
        return value
    if expected_type is int:
        if type(value) is not int:
            raise ToolBridgeContractError(f"{path} must be an integer")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolBridgeContractError(f"{path} must be a finite number")
        if not isfinite(float(value)):
            raise ToolBridgeContractError(f"{path} must be a finite number")
        return float(value)
    if expected_type is type(None):
        if value is not None:
            raise ToolBridgeContractError(f"{path} must be null")
        return None
    raise ToolBridgeContractError(f"{path} uses an unsupported frozen contract type")


def _sanitize_response(value: Any, *, identity_field: bool = False) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ToolBridgeContractError("MCP response contains a non-finite number")
        return value
    if isinstance(value, str):
        if identity_field and _URI_IDENTITY_PATTERN.match(value):
            raise ToolBridgeContractError(
                "MCP response artifact identity must not contain a storage URI"
            )
        redacted = _URI_PATTERN.sub("[REDACTED_URI]", value)
        return _CHECKSUM_VALUE_PATTERN.sub(r"\1[REDACTED_CHECKSUM]", redacted)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolBridgeContractError("MCP response contains a non-string key")
            if _is_forbidden_storage_key(key):
                continue
            result[key] = _sanitize_response(
                item,
                identity_field=_is_artifact_identity_key(key),
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_response(item, identity_field=identity_field)
            for item in value
        ]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ToolBridgeContractError("MCP response contains a forbidden byte payload")
    raise ToolBridgeContractError("MCP response is not JSON-compatible")


__all__ = [
    "DEFAULT_DANGER_MASKS",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_LABELS",
    "DEFAULT_QUALITY_THRESHOLDS",
    "DEFAULT_TOOL_BRIDGE_POLICY",
    "DangerMaskPolicy",
    "LabelMappingPolicy",
    "LabelPolicy",
    "LabelQualityPolicy",
    "LocalMcpToolExecutor",
    "McpToolExecutor",
    "McpToolExecutorAdapter",
    "TOOL_NAMES",
    "ToolBridgeContextError",
    "ToolBridgeContractError",
    "ToolBridgeError",
    "ToolBridgeResponseError",
    "ToolBridgeTransportError",
    "ToolBridgePolicy",
    "ToolCaller",
]
