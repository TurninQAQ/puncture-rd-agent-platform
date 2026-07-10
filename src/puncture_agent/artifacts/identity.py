"""Stable serialization and idempotency identities for artifact-producing tools."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Any, Mapping, Sequence


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("canonical JSON does not allow NaN or infinite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize equivalent values to identical UTF-8 JSON text.

    This representation is an internal identity format.  Changing it requires
    an idempotency schema-version migration because persisted keys depend on it.
    """

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_artifact_idempotency_key(
    *,
    tool_name: str,
    tool_version: str,
    input_artifact_ids: Sequence[str],
    parameters: Mapping[str, Any],
    geometry_fingerprints: Sequence[str] = (),
    schema_version: str = "1",
) -> str:
    """Return the stable SHA-256 key for one deterministic tool invocation."""

    required = (tool_name, tool_version, schema_version)
    if any(not value.strip() for value in required):
        raise ValueError("tool_name, tool_version, and schema_version are required")
    if any(not artifact_id.strip() for artifact_id in input_artifact_ids):
        raise ValueError("input artifact IDs must not be empty")
    if any(not fingerprint.strip() for fingerprint in geometry_fingerprints):
        raise ValueError("geometry fingerprints must not be empty")

    payload = {
        "schema_version": schema_version,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "input_artifact_ids": sorted(input_artifact_ids),
        "geometry_fingerprints": sorted(geometry_fingerprints),
        "parameters": parameters,
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()
