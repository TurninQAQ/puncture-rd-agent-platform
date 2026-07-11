"""Dependency-free JSON Schema generation for frozen dataclass contracts."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

from contracts.artifacts import ArtifactPublicView, ArtifactRef
from contracts.common import MetricValue
from contracts.enums import ArtifactType, ErrorCode, ToolExecutionStatus
from contracts.errors import ErrorDetail


SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class DataclassSchemaBuilder:
    """Generate deterministic schemas without importing Pydantic at runtime."""

    def __init__(self, *, artifact_mode: str) -> None:
        if artifact_mode not in {"handle", "public"}:
            raise ValueError("artifact_mode must be 'handle' or 'public'")
        self._artifact_mode = artifact_mode
        self._definitions: dict[str, dict[str, Any]] = {}
        self._building: set[str] = set()

    def build(self, value_type: Any) -> dict[str, Any]:
        root = self._schema_for(value_type)
        result = {"$schema": SCHEMA_DIALECT, **root}
        if self._definitions:
            result["$defs"] = {key: self._definitions[key] for key in sorted(self._definitions)}
        return result

    def _schema_for(self, value_type: Any) -> dict[str, Any]:
        if value_type is Any:
            return {}
        if value_type is ArtifactRef:
            if self._artifact_mode == "handle":
                return self._artifact_handle_schema()
            return self._schema_for(ArtifactPublicView)

        origin = get_origin(value_type)
        arguments = get_args(value_type)
        if origin in (Union, UnionType):
            return {"anyOf": [self._schema_for(option) for option in arguments]}
        if origin in (tuple, list):
            schema: dict[str, Any] = {"type": "array"}
            if origin is tuple and arguments and not (
                len(arguments) == 2 and arguments[1] is Ellipsis
            ):
                schema["prefixItems"] = [self._schema_for(option) for option in arguments]
                schema["minItems"] = len(arguments)
                schema["maxItems"] = len(arguments)
            else:
                item_type = arguments[0] if arguments else Any
                schema["items"] = self._schema_for(item_type)
            return schema
        if origin in (dict, Mapping):
            item_type = arguments[1] if len(arguments) == 2 else Any
            return {"type": "object", "additionalProperties": self._schema_for(item_type)}

        if isinstance(value_type, type) and issubclass(value_type, Enum):
            values = [member.value for member in value_type]
            primitive_type = "integer" if values and all(type(value) is int for value in values) else "string"
            return {"type": primitive_type, "enum": values}
        if isinstance(value_type, type) and is_dataclass(value_type):
            return self._dataclass_reference(value_type)
        if value_type is str:
            return {"type": "string"}
        if value_type is bool:
            return {"type": "boolean"}
        if value_type is int:
            return {"type": "integer"}
        if value_type is float:
            return {"type": "number"}
        if value_type is type(None):
            return {"type": "null"}
        raise TypeError(f"unsupported JSON Schema type: {value_type!r}")

    def _dataclass_reference(self, value_type: type[Any]) -> dict[str, Any]:
        name = value_type.__name__
        if name not in self._definitions and name not in self._building:
            self._building.add(name)
            hints = get_type_hints(value_type)
            properties: dict[str, Any] = {}
            required: list[str] = []
            for field in fields(value_type):
                properties[field.name] = self._schema_for(hints.get(field.name, Any))
                if field.default is MISSING and field.default_factory is MISSING:
                    required.append(field.name)
                elif field.default is not MISSING:
                    default = field.default.value if isinstance(field.default, Enum) else field.default
                    if default is None or isinstance(default, (str, int, float, bool)):
                        properties[field.name]["default"] = default
            definition: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            }
            if required:
                definition["required"] = required
            self._definitions[name] = definition
            self._building.remove(name)
        return {"$ref": f"#/$defs/{name}"}

    def _artifact_handle_schema(self) -> dict[str, Any]:
        name = "ArtifactHandle"
        if name not in self._definitions:
            self._definitions[name] = {
                "type": "object",
                "description": "Opaque artifact identity; storage URI and payload never enter model context.",
                "properties": {
                    "artifact_id": {"type": "string", "minLength": 1},
                    "case_id": {"type": "string", "minLength": 1},
                    "artifact_type": {
                        "type": "string",
                        "enum": [member.value for member in ArtifactType],
                        "description": "Optional assertion checked against the trusted artifact registry.",
                    },
                },
                "required": ["artifact_id"],
                "additionalProperties": False,
            }
        return {"$ref": f"#/$defs/{name}"}


def request_schema(request_type: type[Any]) -> dict[str, Any]:
    return DataclassSchemaBuilder(artifact_mode="handle").build(request_type)


def result_schema(result_type: type[Any]) -> dict[str, Any]:
    return DataclassSchemaBuilder(artifact_mode="public").build(result_type)


def envelope_schema(result_type: type[Any]) -> dict[str, Any]:
    """Build the model-visible, URI-redacted ToolResponseEnvelope schema."""

    builder = DataclassSchemaBuilder(artifact_mode="public")
    result_value_schema = builder._schema_for(result_type)
    artifact_schema = builder._schema_for(ArtifactRef)
    metric_schema = builder._schema_for(MetricValue)
    error_schema = builder._schema_for(ErrorDetail)
    schema: dict[str, Any] = {
        "$schema": SCHEMA_DIALECT,
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "trace_id": {"type": "string"},
            "tool_name": {"type": "string"},
            "tool_version": {"type": "string"},
            "status": {"type": "string", "enum": [member.value for member in ToolExecutionStatus]},
            "result": {"anyOf": [result_value_schema, {"type": "null"}]},
            "artifacts": {"type": "array", "items": artifact_schema},
            "metrics": {"type": "array", "items": metric_schema},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "error": {"anyOf": [error_schema, {"type": "null"}]},
            "started_at": {"type": "string"},
            "finished_at": {"type": "string"},
        },
        "required": [
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
        ],
        "additionalProperties": False,
    }
    if builder._definitions:
        schema["$defs"] = {
            key: builder._definitions[key] for key in sorted(builder._definitions)
        }
    return schema


def error_code_schema() -> dict[str, Any]:
    return {"type": "string", "enum": [member.value for member in ErrorCode]}
