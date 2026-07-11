"""Trusted Registry validation for MCP Artifact handles and public outputs.

The MCP response contract can prove that a result and its envelope agree with
each other, but a remote service can forge both values.  This module adds the
independent authority boundary: every Artifact is resolved through one atomic
Registry record before it can be sent to a tool or accepted into Agent state.

The validation record deliberately contains no storage URI or checksum.  It is
the smallest snapshot needed to prove public identity, full geometry and direct
lineage without exposing storage details to the Agent or model context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from contracts.common import to_primitive
from contracts.enums import ArtifactStatus, ArtifactType
from puncture_agent.artifacts.registry import (
    ArtifactRegistryError,
    ArtifactValidationRecord,
)
from puncture_agent.mcp import McpPrincipal
from puncture_agent.tooling.catalog import TOOL_DEFINITIONS


class ArtifactValidationError(RuntimeError):
    """Base class for fail-closed Artifact authority errors."""

    code = "CONTRACT_VIOLATION"
    retryable = False


class ArtifactValidationRejected(ArtifactValidationError):
    """The request or response disagrees with authoritative Registry state."""


class ArtifactValidationUnavailable(ArtifactValidationError):
    """The trusted Registry could not provide an authoritative decision."""

    code = "DEPENDENCY_FAILED"
    retryable = True


@runtime_checkable
class ArtifactValidationRegistry(Protocol):
    """Registry port that returns metadata, geometry and parents atomically."""

    def get_validation_record(self, artifact_id: str) -> ArtifactValidationRecord:
        ...


@dataclass(frozen=True, slots=True)
class _InputArtifact:
    path: str
    handle: Mapping[str, Any]
    allowed_types: tuple[ArtifactType, ...]
    geometry_group: str | None = None
    geometry_anchor: bool = False


@dataclass(frozen=True, slots=True)
class _ValidatedInput:
    spec: _InputArtifact
    artifact_id: str
    record: ArtifactValidationRecord


@dataclass(frozen=True, slots=True)
class _OutputPolicy:
    result_field: str
    artifact_type: ArtifactType
    parent_roles: tuple[str, ...]
    geometry_role: str
    exact_geometry: bool


_ALL_ARTIFACT_TYPES = tuple(ArtifactType)
_LABELMAP_TYPES = (
    ArtifactType.NIFTI_LABELMAP,
    ArtifactType.SEGMENTATION_MASK,
)
_SEGMENTATION_INPUT_TYPES = (
    ArtifactType.SEGMENTATION_MASK,
    ArtifactType.NIFTI_LABELMAP,
)
_LUNG_MASK_TYPES = (
    ArtifactType.SEGMENTATION_MASK,
    ArtifactType.DANGER_MASK,
)
_SKIN_MASK_TYPES = (
    ArtifactType.SEGMENTATION_MASK,
    ArtifactType.SKIN_SURFACE_MASK,
)

_OUTPUT_POLICIES: dict[str, _OutputPolicy] = {
    "convert_mcs_to_nifti": _OutputPolicy(
        result_field="output_artifact",
        artifact_type=ArtifactType.NIFTI_LABELMAP,
        parent_roles=("mcs_artifact", "reference_ct_artifact"),
        geometry_role="reference_ct_artifact",
        exact_geometry=False,
    ),
    "run_segmentation": _OutputPolicy(
        result_field="segmentation_artifact",
        artifact_type=ArtifactType.SEGMENTATION_MASK,
        parent_roles=("ct_artifact",),
        geometry_role="ct_artifact",
        exact_geometry=True,
    ),
    "extract_skin_surface": _OutputPolicy(
        result_field="surface_artifact",
        artifact_type=ArtifactType.SKIN_SURFACE_MASK,
        parent_roles=("skin_mask_artifact",),
        geometry_role="skin_mask_artifact",
        exact_geometry=True,
    ),
}


class RegistryToolArtifactValidator:
    """Validate tool requests and normalized responses against a trusted Registry.

    ``validate_request`` must run after the bridge has built canonical MCP
    arguments but before transport.  ``validate_response`` must run after the
    bridge's final normalization/sanitization pass and before the response is
    recorded in ``AgentState``.
    """

    def __init__(self, registry: ArtifactValidationRegistry) -> None:
        if not isinstance(registry, ArtifactValidationRegistry):
            raise TypeError(
                "registry must implement get_validation_record(artifact_id)"
            )
        self._registry = registry

    def validate_request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        principal: McpPrincipal,
    ) -> None:
        """Prove current ACL, availability, type and geometry of every input."""

        case_id = self._authorize(tool_name, arguments, principal)
        self._validated_inputs(tool_name, arguments, case_id=case_id)

    def validate_response(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        normalized_response: Mapping[str, Any],
        principal: McpPrincipal,
    ) -> None:
        """Prove all remotely returned Artifact claims before state mutation."""

        case_id = self._authorize(tool_name, arguments, principal)
        if not isinstance(normalized_response, Mapping):
            raise ArtifactValidationRejected("normalized tool response must be an object")
        if normalized_response.get("tool_name") != tool_name:
            raise ArtifactValidationRejected("tool response identity is inconsistent")
        expected_version = self._tool_version(tool_name)
        if normalized_response.get("tool_version") != expected_version:
            raise ArtifactValidationRejected("tool response version is inconsistent")

        artifacts = normalized_response.get("artifacts")
        if not isinstance(artifacts, list):
            raise ArtifactValidationRejected("tool response artifacts must be an array")
        status = normalized_response.get("status")
        if status not in {"SUCCESS", "PARTIAL", "FAILED"}:
            raise ArtifactValidationRejected("tool response status is not canonical")
        if status == "FAILED":
            if artifacts:
                raise ArtifactValidationRejected(
                    "failed tool responses must not expose artifacts"
                )
            return

        if tool_name == "generate_candidate_paths":
            inputs = self._validated_inputs(tool_name, arguments, case_id=case_id)
            self._validate_candidate_path_outputs(
                inputs=inputs,
                result=normalized_response.get("result"),
                artifacts=artifacts,
                case_id=case_id,
                expected_version=expected_version,
            )
            return

        policy = _OUTPUT_POLICIES.get(tool_name)
        if policy is None:
            if artifacts:
                raise ArtifactValidationRejected(
                    "this tool is not allowed to publish response artifacts"
                )
            return

        inputs = self._validated_inputs(tool_name, arguments, case_id=case_id)
        result = normalized_response.get("result")
        if not isinstance(result, Mapping):
            raise ArtifactValidationRejected(
                "artifact-producing response requires a structured result"
            )
        claimed = result.get(policy.result_field)
        if not isinstance(claimed, Mapping):
            raise ArtifactValidationRejected(
                f"result.{policy.result_field} must be an artifact public view"
            )
        artifact_id = self._artifact_id(
            claimed,
            path=f"result.{policy.result_field}",
        )
        if artifact_id in {item.artifact_id for item in inputs.values()}:
            raise ArtifactValidationRejected(
                "a produced artifact must not reuse an input artifact identity"
            )

        envelope_by_id = self._response_artifacts_by_id(artifacts)
        if set(envelope_by_id) != {artifact_id}:
            raise ArtifactValidationRejected(
                "response artifact envelope must exactly describe the produced artifact"
            )

        record = self._load_record(artifact_id)
        self._validate_record_common(
            record,
            artifact_id=artifact_id,
            case_id=case_id,
            allowed_types=(policy.artifact_type,),
            path=f"result.{policy.result_field}",
        )
        if record.public_view.producer_name != tool_name:
            raise ArtifactValidationRejected(
                "produced artifact has an unexpected authoritative producer"
            )
        if record.public_view.producer_version != expected_version:
            raise ArtifactValidationRejected(
                "produced artifact has an unexpected authoritative producer version"
            )

        canonical_view = to_primitive(record.public_view)
        if dict(claimed) != canonical_view:
            raise ArtifactValidationRejected(
                "result artifact does not match the authoritative Registry view"
            )
        if dict(envelope_by_id[artifact_id]) != canonical_view:
            raise ArtifactValidationRejected(
                "envelope artifact does not match the authoritative Registry view"
            )

        expected_parents = tuple(
            sorted(inputs[role].artifact_id for role in policy.parent_roles)
        )
        actual_parents = record.parent_artifact_ids
        if len(actual_parents) != len(set(actual_parents)) or tuple(
            sorted(actual_parents)
        ) != expected_parents:
            raise ArtifactValidationRejected(
                "produced artifact direct lineage does not match tool inputs"
            )

        expected_geometry = inputs[policy.geometry_role].record.geometry
        actual_geometry = record.geometry
        if expected_geometry is None or actual_geometry is None:
            raise ArtifactValidationRejected(
                "produced artifact geometry is unavailable in the Registry"
            )
        geometry_matches = (
            actual_geometry == expected_geometry
            if policy.exact_geometry
            else actual_geometry.is_compatible_with(expected_geometry)
        )
        if not geometry_matches:
            qualifier = "exact" if policy.exact_geometry else "compatible"
            raise ArtifactValidationRejected(
                f"produced artifact geometry is not {qualifier} with its source"
            )

    def _validate_candidate_path_outputs(
        self,
        *,
        inputs: Mapping[str, _ValidatedInput],
        result: Any,
        artifacts: list[Any],
        case_id: str,
        expected_version: str,
    ) -> None:
        if not isinstance(result, Mapping):
            raise ArtifactValidationRejected(
                "candidate generation response requires a structured result"
            )
        candidates = result.get("candidates")
        if not isinstance(candidates, list):
            raise ArtifactValidationRejected("result.candidates must be an array")

        path_ids: list[str] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ArtifactValidationRejected(
                    f"result.candidates[{index}] must be an object"
                )
            path_id = candidate.get("path_artifact_id")
            if path_id is None:
                continue
            path_ids.append(
                self._artifact_id(
                    {"artifact_id": path_id},
                    path=f"result.candidates[{index}].path_artifact_id",
                )
            )
        if len(path_ids) != len(set(path_ids)):
            raise ArtifactValidationRejected(
                "candidate path artifact IDs must be unique"
            )

        envelope_by_id = self._response_artifacts_by_id(artifacts)
        if set(envelope_by_id) != set(path_ids):
            raise ArtifactValidationRejected(
                "response artifact envelope must exactly match candidate path artifacts"
            )
        if not path_ids:
            return

        input_ids = {item.artifact_id for item in inputs.values()}
        expected_parents = tuple(sorted(input_ids))
        ct_input = inputs.get("ct_artifact")
        if ct_input is None or ct_input.record.geometry is None:
            raise ArtifactValidationRejected(
                "candidate path validation requires authoritative CT geometry"
            )
        expected_geometry = ct_input.record.geometry

        for index, artifact_id in enumerate(path_ids):
            if artifact_id in input_ids:
                raise ArtifactValidationRejected(
                    "a candidate path artifact must not reuse an input identity"
                )
            record = self._load_record(artifact_id)
            path = f"result.candidates[{index}].path_artifact_id"
            self._validate_record_common(
                record,
                artifact_id=artifact_id,
                case_id=case_id,
                allowed_types=(ArtifactType.PATH_MASK,),
                path=path,
            )
            if record.public_view.producer_name != "generate_candidate_paths":
                raise ArtifactValidationRejected(
                    "candidate path artifact has an unexpected authoritative producer"
                )
            if record.public_view.producer_version != expected_version:
                raise ArtifactValidationRejected(
                    "candidate path artifact has an unexpected authoritative producer version"
                )
            canonical_view = to_primitive(record.public_view)
            if dict(envelope_by_id[artifact_id]) != canonical_view:
                raise ArtifactValidationRejected(
                    "candidate path artifact does not match the authoritative Registry view"
                )
            actual_parents = record.parent_artifact_ids
            if len(actual_parents) != len(set(actual_parents)) or tuple(
                sorted(actual_parents)
            ) != expected_parents:
                raise ArtifactValidationRejected(
                    "candidate path direct lineage does not match all request artifacts"
                )
            if record.geometry is None or record.geometry != expected_geometry:
                raise ArtifactValidationRejected(
                    "candidate path artifact geometry is not exact with the CT"
                )

    def _response_artifacts_by_id(
        self,
        artifacts: list[Any],
    ) -> dict[str, Mapping[str, Any]]:
        envelope_by_id: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(artifacts):
            if not isinstance(item, Mapping):
                raise ArtifactValidationRejected(
                    f"response artifacts[{index}] must be an object"
                )
            envelope_id = self._artifact_id(item, path=f"artifacts[{index}]")
            if envelope_id in envelope_by_id:
                raise ArtifactValidationRejected(
                    "response artifact identities must be unique"
                )
            envelope_by_id[envelope_id] = item
        return envelope_by_id

    def _authorize(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        principal: McpPrincipal,
    ) -> str:
        self._tool_version(tool_name)
        if not isinstance(arguments, Mapping):
            raise ArtifactValidationRejected("tool arguments must be an object")
        if not isinstance(principal, McpPrincipal):
            raise ArtifactValidationRejected("authenticated MCP principal is required")
        context = arguments.get("context")
        if not isinstance(context, Mapping):
            raise ArtifactValidationRejected("tool arguments require call context")
        case_id = context.get("case_id")
        caller = context.get("caller")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ArtifactValidationRejected("tool context requires a case ID")
        if not isinstance(caller, str) or not caller.strip():
            raise ArtifactValidationRejected("tool context requires a caller")
        if not principal.permits(
            tool_name=tool_name,
            case_id=case_id,
            caller=caller,
        ):
            raise ArtifactValidationRejected(
                "authenticated principal is not permitted for this tool and case"
            )
        return case_id

    def _validated_inputs(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        case_id: str,
    ) -> dict[str, _ValidatedInput]:
        specs = self._input_specs(tool_name, arguments)
        result: dict[str, _ValidatedInput] = {}
        records_by_id: dict[str, ArtifactValidationRecord] = {}
        for spec in specs:
            artifact_id = self._artifact_id(spec.handle, path=spec.path)
            record = records_by_id.get(artifact_id)
            if record is None:
                record = self._load_record(artifact_id)
                records_by_id[artifact_id] = record
            self._validate_handle_assertions(
                spec.handle,
                record,
                case_id=case_id,
                path=spec.path,
            )
            self._validate_record_common(
                record,
                artifact_id=artifact_id,
                case_id=case_id,
                allowed_types=spec.allowed_types,
                path=spec.path,
            )
            result[spec.path] = _ValidatedInput(spec, artifact_id, record)

        groups: dict[str, list[_ValidatedInput]] = {}
        for item in result.values():
            if item.spec.geometry_group is not None:
                groups.setdefault(item.spec.geometry_group, []).append(item)
        for group_items in groups.values():
            anchors = [item for item in group_items if item.spec.geometry_anchor]
            if len(anchors) != 1:
                raise ArtifactValidationRejected(
                    "tool artifact geometry policy has no unique anchor"
                )
            anchor_geometry = anchors[0].record.geometry
            if anchor_geometry is None:
                raise ArtifactValidationRejected(
                    f"{anchors[0].spec.path} has no authoritative geometry"
                )
            for item in group_items:
                geometry = item.record.geometry
                if geometry is None:
                    raise ArtifactValidationRejected(
                        f"{item.spec.path} has no authoritative geometry"
                    )
                if not geometry.is_compatible_with(anchor_geometry):
                    raise ArtifactValidationRejected(
                        f"{item.spec.path} geometry is incompatible with the request"
                    )
        return result

    def _load_record(self, artifact_id: str) -> ArtifactValidationRecord:
        try:
            record = self._registry.get_validation_record(artifact_id)
        except ArtifactRegistryError as exc:
            if exc.code in {
                "NOT_FOUND",
                "PERMISSION_DENIED",
                "ARTIFACT_NOT_AVAILABLE",
            }:
                raise ArtifactValidationRejected(
                    "artifact is not authorized and registered in the trusted Registry"
                ) from exc
            raise ArtifactValidationUnavailable(
                "trusted Artifact Registry is unavailable"
            ) from exc
        except KeyError as exc:
            raise ArtifactValidationRejected(
                "artifact is not registered in the trusted Registry"
            ) from exc
        except Exception as exc:
            raise ArtifactValidationUnavailable(
                "trusted Artifact Registry is unavailable"
            ) from exc
        if record is None:
            raise ArtifactValidationRejected(
                "artifact is not registered in the trusted Registry"
            )
        if not isinstance(record, ArtifactValidationRecord):
            raise ArtifactValidationRejected(
                "trusted Registry returned an invalid Artifact validation record"
            )
        if record.public_view.artifact_id != artifact_id:
            raise ArtifactValidationRejected(
                "trusted Registry returned a different Artifact identity"
            )
        return record

    @staticmethod
    def _validate_record_common(
        record: ArtifactValidationRecord,
        *,
        artifact_id: str,
        case_id: str,
        allowed_types: tuple[ArtifactType, ...],
        path: str,
    ) -> None:
        view = record.public_view
        if view.artifact_id != artifact_id:
            raise ArtifactValidationRejected(f"{path} Registry identity is inconsistent")
        if view.case_id != case_id:
            raise ArtifactValidationRejected(f"{path} belongs to another case")
        if view.status is not ArtifactStatus.AVAILABLE:
            raise ArtifactValidationRejected(f"{path} is not AVAILABLE")
        if view.artifact_type not in allowed_types:
            raise ArtifactValidationRejected(f"{path} has an unexpected Artifact type")

    @staticmethod
    def _validate_handle_assertions(
        handle: Mapping[str, Any],
        record: ArtifactValidationRecord,
        *,
        case_id: str,
        path: str,
    ) -> None:
        asserted_case = handle.get("case_id")
        if asserted_case is not None and asserted_case != case_id:
            raise ArtifactValidationRejected(f"{path}.case_id disagrees with call context")
        if asserted_case is not None and asserted_case != record.public_view.case_id:
            raise ArtifactValidationRejected(f"{path}.case_id disagrees with Registry")
        asserted_type = handle.get("artifact_type")
        if asserted_type is not None:
            try:
                canonical_type = ArtifactType(asserted_type)
            except (TypeError, ValueError) as exc:
                raise ArtifactValidationRejected(
                    f"{path}.artifact_type is not canonical"
                ) from exc
            if canonical_type is not record.public_view.artifact_type:
                raise ArtifactValidationRejected(
                    f"{path}.artifact_type disagrees with Registry"
                )

    @staticmethod
    def _artifact_id(handle: Mapping[str, Any], *, path: str) -> str:
        artifact_id = handle.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ArtifactValidationRejected(f"{path}.artifact_id is required")
        if artifact_id != artifact_id.strip():
            raise ArtifactValidationRejected(
                f"{path}.artifact_id must not contain surrounding whitespace"
            )
        return artifact_id

    @staticmethod
    def _tool_version(tool_name: str) -> str:
        try:
            return TOOL_DEFINITIONS[tool_name].version
        except KeyError as exc:
            raise ArtifactValidationRejected(f"unknown tool: {tool_name}") from exc

    @classmethod
    def _input_specs(
        cls,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> tuple[_InputArtifact, ...]:
        one = cls._required_handle
        optional = cls._optional_handle
        if tool_name == "inspect_case_metadata":
            specs = [
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                )
            ]
            related = arguments.get("related_artifacts", [])
            if not isinstance(related, list):
                raise ArtifactValidationRejected("related_artifacts must be an array")
            require_same = arguments.get("require_same_geometry") is True
            for index, handle in enumerate(related):
                specs.append(
                    _InputArtifact(
                        f"related_artifacts[{index}]",
                        cls._mapping_handle(handle, f"related_artifacts[{index}]"),
                        _ALL_ARTIFACT_TYPES,
                        "image" if require_same else None,
                    )
                )
            return tuple(specs)
        if tool_name == "convert_mcs_to_nifti":
            return (
                _InputArtifact(
                    "mcs_artifact",
                    one(arguments, "mcs_artifact"),
                    (ArtifactType.MCS_SEGMENTATION,),
                    "image",
                ),
                _InputArtifact(
                    "reference_ct_artifact",
                    one(arguments, "reference_ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                ),
            )
        if tool_name == "validate_label_schema":
            return (
                _InputArtifact(
                    "labelmap_artifact",
                    one(arguments, "labelmap_artifact"),
                    _LABELMAP_TYPES,
                    "image",
                    True,
                ),
            )
        if tool_name == "run_segmentation":
            return (
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                ),
            )
        if tool_name == "validate_segmentation_result":
            return (
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                ),
                _InputArtifact(
                    "segmentation_artifact",
                    one(arguments, "segmentation_artifact"),
                    _SEGMENTATION_INPUT_TYPES,
                    "image",
                ),
            )
        if tool_name == "extract_skin_surface":
            return (
                _InputArtifact(
                    "skin_mask_artifact",
                    one(arguments, "skin_mask_artifact"),
                    _SEGMENTATION_INPUT_TYPES,
                    "image",
                    True,
                ),
            )
        if tool_name == "generate_candidate_paths":
            specs = [
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                ),
                _InputArtifact(
                    "skin_surface_artifact",
                    one(arguments, "skin_surface_artifact"),
                    (ArtifactType.SKIN_SURFACE_MASK,),
                    "image",
                ),
                _InputArtifact(
                    "target_artifact",
                    one(arguments, "target_artifact"),
                    (ArtifactType.TARGET_MASK,),
                    "image",
                ),
            ]
            lesion = optional(arguments, "lesion_artifact")
            if lesion is not None:
                specs.append(
                    _InputArtifact(
                        "lesion_artifact",
                        lesion,
                        (ArtifactType.LESION_MASK,),
                        "image",
                    )
                )
            return tuple(specs)
        if tool_name == "evaluate_path_safety":
            specs = [
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                )
            ]
            specs.extend(cls._danger_specs(arguments, geometry_group="image"))
            candidates = arguments.get("candidate_paths", [])
            if not isinstance(candidates, list):
                raise ArtifactValidationRejected("candidate_paths must be an array")
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    raise ArtifactValidationRejected(
                        f"candidate_paths[{index}] must be an object"
                    )
                path_id = candidate.get("path_artifact_id")
                if path_id is not None:
                    specs.append(
                        _InputArtifact(
                            f"candidate_paths[{index}].path_artifact_id",
                            {"artifact_id": path_id},
                            (ArtifactType.PATH_MASK,),
                            "image",
                        )
                    )
            return tuple(specs)
        if tool_name == "evaluate_intraoperative_risk":
            specs = [
                _InputArtifact(
                    "ct_artifact",
                    one(arguments, "ct_artifact"),
                    (ArtifactType.CT_VOLUME,),
                    "image",
                    True,
                )
            ]
            specs.extend(cls._danger_specs(arguments, geometry_group="image"))
            for name in ("lung_mask_artifact", "skin_mask_artifact"):
                handle = optional(arguments, name)
                if handle is not None:
                    specs.append(
                        _InputArtifact(
                            name,
                            handle,
                            _LUNG_MASK_TYPES
                            if name == "lung_mask_artifact"
                            else _SKIN_MASK_TYPES,
                            "image",
                        )
                    )
            return tuple(specs)
        if tool_name == "verify_skin_penetration":
            return (
                _InputArtifact(
                    "skin_mask_artifact",
                    one(arguments, "skin_mask_artifact"),
                    _SKIN_MASK_TYPES,
                    "image",
                    True,
                ),
            )
        raise ArtifactValidationRejected(f"unknown tool: {tool_name}")

    @classmethod
    def _danger_specs(
        cls,
        arguments: Mapping[str, Any],
        *,
        geometry_group: str,
    ) -> list[_InputArtifact]:
        dangers = arguments.get("danger_masks")
        if not isinstance(dangers, list):
            raise ArtifactValidationRejected("danger_masks must be an array")
        specs: list[_InputArtifact] = []
        for index, item in enumerate(dangers):
            if not isinstance(item, Mapping):
                raise ArtifactValidationRejected(f"danger_masks[{index}] must be an object")
            specs.append(
                _InputArtifact(
                    f"danger_masks[{index}].artifact",
                    cls._required_handle(item, "artifact"),
                    (ArtifactType.DANGER_MASK,),
                    geometry_group,
                )
            )
        return specs

    @classmethod
    def _required_handle(
        cls,
        value: Mapping[str, Any],
        name: str,
    ) -> Mapping[str, Any]:
        if name not in value:
            raise ArtifactValidationRejected(f"{name} is required")
        return cls._mapping_handle(value[name], name)

    @classmethod
    def _optional_handle(
        cls,
        value: Mapping[str, Any],
        name: str,
    ) -> Mapping[str, Any] | None:
        handle = value.get(name)
        if handle is None:
            return None
        return cls._mapping_handle(handle, name)

    @staticmethod
    def _mapping_handle(value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ArtifactValidationRejected(f"{path} must be an Artifact handle")
        return value


__all__ = [
    "ArtifactValidationError",
    "ArtifactValidationRecord",
    "ArtifactValidationRegistry",
    "ArtifactValidationRejected",
    "ArtifactValidationUnavailable",
    "RegistryToolArtifactValidator",
]
