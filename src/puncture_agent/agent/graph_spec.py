"""Loader and semantic validator for the framework-neutral JSON graph format."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class GraphSpecError(ValueError):
    """Raised when a JSON graph violates the locked graph contract."""


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: str
    handler: str | None = None
    graph: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    condition: Mapping[str, Any]


@dataclass(frozen=True)
class GraphSpec:
    schema_version: str
    graph_id: str
    description: str
    start: str
    end: str
    max_steps: int
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    source_path: Path | None = None

    @property
    def node_map(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    def outgoing(self, source: str) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.source == source)


def _parse_graph(payload: Mapping[str, Any], source_path: Path | None) -> GraphSpec:
    try:
        nodes = tuple(
            NodeSpec(
                node_id=item["id"],
                kind=item["kind"],
                handler=item.get("handler"),
                graph=item.get("graph"),
                config={
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "kind", "handler", "graph"}
                },
            )
            for item in payload["nodes"]
        )
        edges = tuple(
            EdgeSpec(
                source=item["source"],
                target=item["target"],
                condition=item.get("condition", {"operator": "always"}),
            )
            for item in payload["edges"]
        )
        return GraphSpec(
            schema_version=str(payload["schema_version"]),
            graph_id=str(payload["graph_id"]),
            description=str(payload.get("description", "")),
            start=str(payload.get("start", "START")),
            end=str(payload.get("end", "END")),
            max_steps=int(payload.get("max_steps", 100)),
            nodes=nodes,
            edges=edges,
            source_path=source_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphSpecError(f"Malformed graph specification: {exc}") from exc


def load_graph_spec(path: str | Path, *, validate: bool = True) -> GraphSpec:
    source_path = Path(path).resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphSpecError(f"Cannot load graph spec {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GraphSpecError("Graph specification root must be a JSON object")
    spec = _parse_graph(payload, source_path)
    if validate:
        validate_graph_spec(spec, graph_root=source_path.parent)
    return spec


def _validate_condition(condition: Mapping[str, Any], *, location: str) -> None:
    if not isinstance(condition, Mapping):
        raise GraphSpecError(f"{location}: condition must be an object")
    composite_keys = [key for key in ("all", "any", "not") if key in condition]
    if composite_keys:
        if len(composite_keys) != 1:
            raise GraphSpecError(f"{location}: use exactly one composite operator")
        key = composite_keys[0]
        children = condition[key]
        if key == "not":
            _validate_condition(children, location=f"{location}.not")
            return
        if not isinstance(children, list) or not children:
            raise GraphSpecError(f"{location}.{key}: must be a non-empty list")
        for index, child in enumerate(children):
            _validate_condition(child, location=f"{location}.{key}[{index}]")
        return

    supported = {
        "always",
        "eq",
        "ne",
        "in",
        "not_in",
        "truthy",
        "falsy",
        "gt",
        "gte",
        "lt",
        "lte",
        "eq_field",
        "lte_field",
    }
    operator = condition.get("operator")
    if operator not in supported:
        raise GraphSpecError(f"{location}: unsupported operator {operator!r}")
    if operator != "always" and not condition.get("field"):
        raise GraphSpecError(f"{location}: operator {operator!r} requires field")
    if operator not in {"always", "truthy", "falsy"} and "value" not in condition:
        raise GraphSpecError(f"{location}: operator {operator!r} requires value")


def validate_graph_spec(spec: GraphSpec, *, graph_root: str | Path | None = None) -> None:
    """Validate structure, deterministic fallbacks, reachability and subgraphs."""

    if spec.schema_version != "1.0":
        raise GraphSpecError(f"Unsupported graph schema_version: {spec.schema_version}")
    if spec.start == spec.end:
        raise GraphSpecError("start and end sentinels must differ")
    if spec.max_steps <= 0:
        raise GraphSpecError("max_steps must be positive")

    ids = [node.node_id for node in spec.nodes]
    if len(ids) != len(set(ids)):
        raise GraphSpecError("Node IDs must be unique")
    if spec.start in ids or spec.end in ids:
        raise GraphSpecError("START/END sentinels cannot also be node IDs")

    node_ids = set(ids)
    allowed_sources = node_ids | {spec.start}
    allowed_targets = node_ids | {spec.end}
    outgoing: dict[str, list[EdgeSpec]] = defaultdict(list)

    for node in spec.nodes:
        if node.kind not in {"action", "router", "subgraph"}:
            raise GraphSpecError(f"Node {node.node_id}: unsupported kind {node.kind!r}")
        if node.kind == "subgraph":
            if not node.graph or node.handler:
                raise GraphSpecError(
                    f"Node {node.node_id}: subgraph requires graph and no handler"
                )
            if graph_root is not None:
                root = Path(graph_root).resolve()
                child_path = (root / node.graph).resolve()
                if root not in child_path.parents:
                    raise GraphSpecError(f"Node {node.node_id}: subgraph escapes graph root")
                if not child_path.is_file():
                    raise GraphSpecError(
                        f"Node {node.node_id}: missing subgraph file {child_path}"
                    )
        elif not node.handler or node.graph:
            raise GraphSpecError(
                f"Node {node.node_id}: action/router requires handler and no graph"
            )

    for index, edge in enumerate(spec.edges):
        if edge.source not in allowed_sources:
            raise GraphSpecError(f"Edge {index}: unknown source {edge.source!r}")
        if edge.target not in allowed_targets:
            raise GraphSpecError(f"Edge {index}: unknown target {edge.target!r}")
        if edge.source == spec.end:
            raise GraphSpecError("END cannot have outgoing edges")
        if edge.target == spec.start:
            raise GraphSpecError("No edge may target START")
        _validate_condition(edge.condition, location=f"edge[{index}]")
        outgoing[edge.source].append(edge)

    if not outgoing.get(spec.start):
        raise GraphSpecError("START must have at least one outgoing edge")
    for node_id in node_ids:
        if not outgoing.get(node_id):
            raise GraphSpecError(f"Node {node_id}: missing outgoing edge")

    for source, edges in outgoing.items():
        fallback_indexes = [
            index
            for index, edge in enumerate(edges)
            if edge.condition.get("operator") == "always"
        ]
        if len(fallback_indexes) > 1:
            raise GraphSpecError(f"Node {source}: multiple unconditional fallbacks")
        if fallback_indexes and fallback_indexes[0] != len(edges) - 1:
            raise GraphSpecError(f"Node {source}: unconditional fallback must be last")

    visited = {spec.start}
    queue = deque([spec.start])
    adjacency = defaultdict(list)
    for edge in spec.edges:
        adjacency[edge.source].append(edge.target)
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if target not in visited:
                visited.add(target)
                queue.append(target)
    unreachable = node_ids - visited
    if unreachable:
        raise GraphSpecError(f"Unreachable nodes: {sorted(unreachable)}")
    if spec.end not in visited:
        raise GraphSpecError("END is unreachable from START")
