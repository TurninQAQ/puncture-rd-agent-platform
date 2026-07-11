"""Bounded JSON copying for durable Run snapshots and executor outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import json
from math import isfinite
from typing import Any


class RuntimeJsonBoundaryError(ValueError):
    """A runtime value cannot be represented safely in a durable JSON record."""


def copy_json_value(
    value: Any,
    *,
    max_depth: int = 32,
    max_nodes: int = 10_000,
    max_bytes: int = 1024 * 1024,
) -> Any:
    if any(
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        for limit in (max_depth, max_nodes, max_bytes)
    ):
        raise ValueError("JSON boundary limits must be positive integers")

    nodes_seen = 0
    active_containers: set[int] = set()

    def visit(item: Any, *, path: str, depth: int) -> Any:
        nonlocal nodes_seen
        nodes_seen += 1
        if nodes_seen > max_nodes:
            raise RuntimeJsonBoundaryError("runtime JSON exceeds the node limit")
        if depth > max_depth:
            raise RuntimeJsonBoundaryError("runtime JSON exceeds the depth limit")

        if isinstance(item, Enum):
            return visit(item.value, path=path, depth=depth)
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not isfinite(item):
                raise RuntimeJsonBoundaryError(f"{path} contains a non-finite number")
            return item
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active_containers:
                raise RuntimeJsonBoundaryError(f"{path} contains a cycle")
            active_containers.add(identity)
            try:
                copied: dict[str, Any] = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise RuntimeJsonBoundaryError(
                            f"{path} contains a non-string key"
                        )
                    copied[key] = visit(
                        child,
                        path=f"{path}.{key}",
                        depth=depth + 1,
                    )
                return copied
            finally:
                active_containers.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active_containers:
                raise RuntimeJsonBoundaryError(f"{path} contains a cycle")
            active_containers.add(identity)
            try:
                return [
                    visit(child, path=f"{path}[{index}]", depth=depth + 1)
                    for index, child in enumerate(item)
                ]
            finally:
                active_containers.remove(identity)
        raise RuntimeJsonBoundaryError(
            f"{path} contains a non-JSON value of type {type(item).__name__}"
        )

    copied = visit(value, path="$", depth=0)
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeJsonBoundaryError(
            "runtime JSON cannot be encoded as UTF-8"
        ) from exc
    if len(encoded) > max_bytes:
        raise RuntimeJsonBoundaryError("runtime JSON exceeds the byte limit")
    return copied


def copy_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeJsonBoundaryError("runtime value must be a mapping")
    copied = copy_json_value(value)
    if not isinstance(copied, dict):
        raise RuntimeJsonBoundaryError("runtime value must remain a mapping")
    return copied


__all__ = [
    "RuntimeJsonBoundaryError",
    "copy_json_mapping",
    "copy_json_value",
]
