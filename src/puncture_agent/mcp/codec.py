"""Strict JSON-to-contract codec with artifact-handle resolution.

MCP arguments never carry storage URIs, checksums, voxels or masks.  An
``ArtifactRef`` is represented on the wire by a small handle and is resolved
against a trusted registry before a tool handler sees it.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from math import isfinite
from types import UnionType
from typing import Any, Mapping, Protocol, Union, get_args, get_origin, get_type_hints

from contracts.artifacts import ArtifactRef
from contracts.common import ToolCallContext
from contracts.enums import ArtifactType


class ContractDecodeError(ValueError):
    """Raised when untrusted MCP arguments do not match a frozen contract."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class ArtifactResolver(Protocol):
    def resolve(
        self,
        artifact_id: str,
        *,
        expected_case_id: str | None = None,
        expected_type: ArtifactType | None = None,
    ) -> ArtifactRef: ...


class InMemoryArtifactResolver:
    """Deterministic resolver used by the local demo and protocol tests."""

    def __init__(self, artifacts: tuple[ArtifactRef, ...] = ()) -> None:
        self._artifacts: dict[str, ArtifactRef] = {}
        for artifact in artifacts:
            self.register(artifact)

    def register(self, artifact: ArtifactRef) -> None:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be ArtifactRef")
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise ValueError(f"artifact ID conflict: {artifact.artifact_id}")
        self._artifacts[artifact.artifact_id] = artifact

    def resolve(
        self,
        artifact_id: str,
        *,
        expected_case_id: str | None = None,
        expected_type: ArtifactType | None = None,
    ) -> ArtifactRef:
        try:
            artifact = self._artifacts[artifact_id]
        except KeyError as exc:
            raise ContractDecodeError("$.artifact_id", "artifact is not registered") from exc
        if expected_case_id is not None and artifact.case_id != expected_case_id:
            raise ContractDecodeError("$.artifact_id", "artifact belongs to another case")
        if expected_type is not None and artifact.artifact_type is not expected_type:
            raise ContractDecodeError("$.artifact_type", "artifact type does not match")
        return artifact

    def list_artifacts(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._artifacts[key] for key in sorted(self._artifacts))


def decode_tool_request(
    request_type: type[Any],
    payload: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
) -> Any:
    """Decode one MCP argument object into the exact request dataclass."""

    if not isinstance(payload, Mapping):
        raise ContractDecodeError("$", "arguments must be an object")
    context_payload = payload.get("context")
    expected_case_id = None
    if isinstance(context_payload, Mapping):
        candidate = context_payload.get("case_id")
        if isinstance(candidate, str) and candidate.strip():
            expected_case_id = candidate
    return _decode_value(
        request_type,
        payload,
        path="$",
        artifact_resolver=artifact_resolver,
        expected_case_id=expected_case_id,
    )


def decode_tool_context(payload: Mapping[str, Any]) -> ToolCallContext:
    """Decode only trusted routing/authentication context before artifact I/O."""

    if not isinstance(payload, Mapping):
        raise ContractDecodeError("$", "arguments must be an object")
    context_payload = payload.get("context")
    if context_payload is None:
        raise ContractDecodeError("$.context", "required field is missing")
    return _decode_value(
        ToolCallContext,
        context_payload,
        path="$.context",
        artifact_resolver=_ContextOnlyResolver(),
        expected_case_id=None,
    )


class _ContextOnlyResolver:
    def resolve(
        self,
        artifact_id: str,
        *,
        expected_case_id: str | None = None,
        expected_type: ArtifactType | None = None,
    ) -> ArtifactRef:
        del artifact_id, expected_case_id, expected_type
        raise ContractDecodeError("$.context", "artifacts are not allowed in call context")


def _decode_value(
    expected_type: Any,
    value: Any,
    *,
    path: str,
    artifact_resolver: ArtifactResolver,
    expected_case_id: str | None,
) -> Any:
    if expected_type is Any:
        return _validate_json_value(value, path)

    if expected_type is ArtifactRef:
        return _decode_artifact_handle(
            value,
            path=path,
            artifact_resolver=artifact_resolver,
            expected_case_id=expected_case_id,
        )

    origin = get_origin(expected_type)
    arguments = get_args(expected_type)

    if origin in (Union, UnionType):
        if value is None and type(None) in arguments:
            return None
        failures: list[str] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _decode_value(
                    option,
                    value,
                    path=path,
                    artifact_resolver=artifact_resolver,
                    expected_case_id=expected_case_id,
                )
            except ContractDecodeError as exc:
                failures.append(exc.message)
        raise ContractDecodeError(path, "value does not match any allowed type: " + "; ".join(failures))

    if origin in (tuple, list):
        if not isinstance(value, list):
            raise ContractDecodeError(path, "expected an array")
        if origin is list:
            item_type = arguments[0] if arguments else Any
            return [
                _decode_value(
                    item_type,
                    item,
                    path=f"{path}[{index}]",
                    artifact_resolver=artifact_resolver,
                    expected_case_id=expected_case_id,
                )
                for index, item in enumerate(value)
            ]
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item_types = [arguments[0]] * len(value)
        else:
            if arguments and len(arguments) != len(value):
                raise ContractDecodeError(path, f"expected exactly {len(arguments)} items")
            item_types = list(arguments) if arguments else [Any] * len(value)
        return tuple(
            _decode_value(
                item_type,
                item,
                path=f"{path}[{index}]",
                artifact_resolver=artifact_resolver,
                expected_case_id=expected_case_id,
            )
            for index, (item_type, item) in enumerate(zip(item_types, value))
        )

    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ContractDecodeError(path, "expected an object")
        key_type, item_type = arguments if len(arguments) == 2 else (str, Any)
        if key_type is not str:
            raise ContractDecodeError(path, "only string-keyed objects are supported")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractDecodeError(path, "object keys must be strings")
            result[key] = _decode_value(
                item_type,
                item,
                path=f"{path}.{key}",
                artifact_resolver=artifact_resolver,
                expected_case_id=expected_case_id,
            )
        return result

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        try:
            return expected_type(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(repr(member.value) for member in expected_type)
            raise ContractDecodeError(path, f"expected one of: {allowed}") from exc

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise ContractDecodeError(path, "expected an object")
        hints = get_type_hints(expected_type)
        dataclass_fields = {field.name: field for field in fields(expected_type)}
        unknown = sorted(set(value) - set(dataclass_fields))
        if unknown:
            raise ContractDecodeError(path, "unknown fields: " + ", ".join(unknown))
        kwargs: dict[str, Any] = {}
        for name, field in dataclass_fields.items():
            if name not in value:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise ContractDecodeError(f"{path}.{name}", "required field is missing")
                continue
            kwargs[name] = _decode_value(
                hints.get(name, Any),
                value[name],
                path=f"{path}.{name}",
                artifact_resolver=artifact_resolver,
                expected_case_id=expected_case_id,
            )
        try:
            return expected_type(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ContractDecodeError(path, str(exc)) from exc

    if expected_type is str:
        if not isinstance(value, str):
            raise ContractDecodeError(path, "expected a string")
        return value
    if expected_type is bool:
        if type(value) is not bool:
            raise ContractDecodeError(path, "expected a boolean")
        return value
    if expected_type is int:
        if type(value) is not int:
            raise ContractDecodeError(path, "expected an integer")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractDecodeError(path, "expected a number")
        result = float(value)
        if not isfinite(result):
            raise ContractDecodeError(path, "number must be finite")
        return result
    if expected_type is type(None):
        if value is not None:
            raise ContractDecodeError(path, "expected null")
        return None

    raise ContractDecodeError(path, f"unsupported contract type: {expected_type!r}")


def _decode_artifact_handle(
    value: Any,
    *,
    path: str,
    artifact_resolver: ArtifactResolver,
    expected_case_id: str | None,
) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ContractDecodeError(path, "artifact must be an object handle")
    allowed = {"artifact_id", "case_id", "artifact_type"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractDecodeError(path, "artifact handle contains forbidden fields: " + ", ".join(unknown))
    artifact_id = value.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ContractDecodeError(f"{path}.artifact_id", "non-empty artifact_id is required")
    case_id = value.get("case_id", expected_case_id)
    if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
        raise ContractDecodeError(f"{path}.case_id", "case_id must be a non-empty string")
    artifact_type_value = value.get("artifact_type")
    artifact_type = None
    if artifact_type_value is not None:
        try:
            artifact_type = ArtifactType(artifact_type_value)
        except (TypeError, ValueError) as exc:
            raise ContractDecodeError(f"{path}.artifact_type", "unknown artifact type") from exc
    try:
        return artifact_resolver.resolve(
            artifact_id,
            expected_case_id=case_id,
            expected_type=artifact_type,
        )
    except ContractDecodeError as exc:
        raise ContractDecodeError(path, exc.message) from exc


def to_mcp_arguments(value: Any) -> Any:
    """Serialize a request while replacing internal artifacts with safe handles."""

    if isinstance(value, ArtifactRef):
        return {
            "artifact_id": value.artifact_id,
            "case_id": value.case_id,
            "artifact_type": value.artifact_type.value,
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_mcp_arguments(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_mcp_arguments(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_mcp_arguments(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("MCP arguments cannot contain non-finite numbers")
    return value


def to_mcp_safe_primitive(value: Any) -> Any:
    """Serialize results without exposing storage URI, checksum or metadata."""

    if isinstance(value, ArtifactRef):
        return to_mcp_safe_primitive(value.to_public_view())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: to_mcp_safe_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_mcp_safe_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [to_mcp_safe_primitive(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("MCP results cannot contain non-finite numbers")
    return value


def _validate_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ContractDecodeError(path, "number must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractDecodeError(path, "object keys must be strings")
            result[key] = _validate_json_value(item, f"{path}.{key}")
        return result
    raise ContractDecodeError(path, "value is not JSON-compatible")
