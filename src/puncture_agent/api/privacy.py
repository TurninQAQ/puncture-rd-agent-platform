"""Framework-neutral validation and redaction for public API values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
import re
from typing import Any


REDACTED = "[REDACTED]"
REDACTED_BINARY = "[REDACTED_BINARY]"

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "bearer_token",
    "client_secret",
    "internal_path",
    "internal_uri",
    "mrn",
    "password",
    "patient_id",
    "patient_name",
    "prompt",
    "raw_image",
    "refresh_token",
    "secret",
    "storage_path",
    "storage_uri",
    "token",
    "uri",
}
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_internal_path",
    "_password",
    "_prompt",
    "_secret",
    "_storage_path",
    "_token",
    "_uri",
)
_SENSITIVE_PARTS = {"authorization", "password", "secret", "token"}
_FORBIDDEN_RAW_KEYS = {
    "image_bytes",
    "image_data",
    "pixel_array",
    "pixel_data",
    "pixels",
    "raw_image",
    "raw_volume",
    "volume_data",
    "voxel_array",
    "voxel_data",
    "voxels",
}
_URI_PATTERN = re.compile(r"(?i)\b[a-z][a-z0-9+.-]{0,31}://[^\s]+")
_JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])"
)
_CREDENTIAL_TEXT_PATTERN = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|authorization|password|secret)\b"
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?i)\btoken(?:\s*(?:=|:)\s*|\s+)[A-Za-z0-9._-]{6,}"
)


class PublicValueValidationError(ValueError):
    """A public request contains a value that cannot cross the API boundary."""


def _normalized_key(value: object) -> str:
    raw = str(value).strip().replace("-", "_")
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words)
    return expanded.lower()


def normalize_public_key(value: object) -> str:
    """Normalize snake, kebab and camel-case public keys for policy checks."""

    return _normalized_key(value)


def _is_forbidden_raw_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return (
        normalized in _FORBIDDEN_RAW_KEYS
        or normalized.endswith(
            (
                "_image_bytes",
                "_image_data",
                "_pixel_array",
                "_pixel_data",
                "_pixels",
                "_volume_data",
                "_voxel_array",
                "_voxel_data",
                "_voxels",
            )
        )
        or normalized.startswith(("raw_image_", "raw_volume_"))
    )


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    parts = set(normalized.split("_"))
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith(_SENSITIVE_SUFFIXES)
        or bool(parts.intersection(_SENSITIVE_PARTS))
        or "mrn" in parts
        or (
            "patient" in parts
            and bool(parts.intersection({"id", "name", "identifier"}))
        )
    )


def _is_sensitive_string(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        "bearer " in lowered
        or bool(_URI_PATTERN.search(stripped))
        or bool(_JWT_PATTERN.search(stripped))
        or bool(_CREDENTIAL_TEXT_PATTERN.search(stripped))
        or bool(_TOKEN_VALUE_PATTERN.search(stripped))
    )


def redact_public_value(value: Any) -> Any:
    """Return a detached JSON-safe public view with secrets and binary removed."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            public_key = str(key)
            redacted[public_key] = (
                REDACTED if _is_sensitive_key(key) else redact_public_value(item)
            )
        return redacted
    if isinstance(value, (bytes, bytearray, memoryview)):
        return REDACTED_BINARY
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_public_value(item) for item in value]
    if isinstance(value, str) and _is_sensitive_string(value):
        return REDACTED
    if isinstance(value, float) and not isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return REDACTED


def validate_public_json_input(
    value: Any,
    *,
    max_depth: int = 32,
    max_nodes: int = 10_000,
    forbidden_keys: Iterable[str] = (),
) -> None:
    """Validate bounded JSON input and reject raw medical-image payload keys."""

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    normalized_forbidden_keys = frozenset(
        normalize_public_key(key) for key in forbidden_keys
    )

    nodes_seen = 0

    def visit(item: Any, *, path: str, depth: int) -> None:
        nonlocal nodes_seen
        nodes_seen += 1
        if nodes_seen > max_nodes:
            raise PublicValueValidationError("request JSON exceeds the node limit")
        if depth > max_depth:
            raise PublicValueValidationError("request JSON exceeds the depth limit")

        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, str):
            if _is_sensitive_string(item):
                raise PublicValueValidationError(
                    f"{path} contains a forbidden credential or internal location"
                )
            return
        if isinstance(item, float):
            if not isfinite(item):
                raise PublicValueValidationError(f"{path} contains a non-finite number")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PublicValueValidationError(f"{path} contains a non-string key")
                normalized = _normalized_key(key)
                child_path = f"{path}.{key}"
                if normalized in normalized_forbidden_keys:
                    raise PublicValueValidationError(
                        f"{child_path} is a server-owned field"
                    )
                if _is_sensitive_key(key):
                    raise PublicValueValidationError(
                        f"{child_path} is a forbidden sensitive field"
                    )
                if _is_forbidden_raw_key(normalized):
                    raise PublicValueValidationError(
                        f"{child_path} is a forbidden raw-image field"
                    )
                visit(child, path=child_path, depth=depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, path=f"{path}[{index}]", depth=depth + 1)
            return
        raise PublicValueValidationError(
            f"{path} contains a non-JSON value of type {type(item).__name__}"
        )

    visit(value, path="$", depth=0)


__all__ = [
    "PublicValueValidationError",
    "REDACTED",
    "REDACTED_BINARY",
    "normalize_public_key",
    "redact_public_value",
    "validate_public_json_input",
]
