"""Small deterministic executor used before the real LangGraph dependency exists."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .graph_spec import EdgeSpec, GraphSpec, NodeSpec, load_graph_spec
from .state import AgentState, AgentStatus


class GraphExecutionError(RuntimeError):
    """Raised for graph-contract or node-runtime failures."""


@dataclass
class NodeOutcome:
    """Optional result returned by a node handler.

    Handlers may mutate state directly, but returning explicit updates makes a
    future migration to LangGraph nodes straightforward.
    """

    updates: Mapping[str, Any] = field(default_factory=dict)
    output: Any = None
    events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class NodeContext:
    graph_id: str
    node_id: str
    qualified_node_id: str
    config: Mapping[str, Any]


class NodeHandler(Protocol):
    def __call__(self, state: AgentState, context: NodeContext) -> NodeOutcome | None:
        ...


class TracerLike(Protocol):
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Any:
        ...


def _normalize(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def evaluate_condition(condition: Mapping[str, Any], state: AgentState) -> bool:
    """Evaluate the deliberately small, non-executable JSON condition DSL."""

    if "all" in condition:
        return all(evaluate_condition(item, state) for item in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(item, state) for item in condition["any"])
    if "not" in condition:
        return not evaluate_condition(condition["not"], state)

    operator = condition.get("operator")
    if operator == "always":
        return True

    actual = _normalize(state.get_path(str(condition.get("field"))))
    expected = condition.get("value")
    if operator in {"eq_field", "lte_field"}:
        expected = state.get_path(str(expected))
    expected = _normalize(expected)

    if operator in {"eq", "eq_field"}:
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator in {"lte", "lte_field"}:
        return actual <= expected
    raise GraphExecutionError(f"Unsupported condition operator at runtime: {operator!r}")


class GraphRuntime:
    """Execute a validated JSON graph against :class:`AgentState`.

    This runtime is not intended to compete with LangGraph. It proves that the
    node contracts and branch semantics are complete before infrastructure is
    added, which lets another model replace one layer at a time.
    """

    def __init__(
        self,
        graph: GraphSpec | str | Path,
        handlers: Mapping[str, NodeHandler],
        *,
        graph_root: str | Path | None = None,
        tracer: TracerLike | None = None,
    ) -> None:
        self.graph = load_graph_spec(graph) if not isinstance(graph, GraphSpec) else graph
        if graph_root is not None:
            self.graph_root = Path(graph_root).resolve()
        elif self.graph.source_path is not None:
            self.graph_root = self.graph.source_path.parent
        else:
            raise ValueError("graph_root is required for an in-memory GraphSpec")
        self.handlers = dict(handlers)
        self.tracer = tracer

    def run(self, state: AgentState) -> AgentState:
        state.status = AgentStatus.RUNNING
        graph_context = (
            self.tracer.start_span(
                "agent.graph",
                attributes={
                    "agent.graph_id": self.graph.graph_id,
                    "agent.session_id": state.session_id,
                },
            )
            if self.tracer
            else nullcontext()
        )
        with graph_context:
            self._run_spec(self.graph, state, prefix="")
        return state

    def _run_spec(self, spec: GraphSpec, state: AgentState, *, prefix: str) -> None:
        current = spec.start
        steps = 0
        while current != spec.end:
            steps += 1
            if steps > spec.max_steps:
                raise GraphExecutionError(
                    f"Graph {spec.graph_id} exceeded max_steps={spec.max_steps}"
                )
            edge = self._select_edge(spec, current, state)
            if edge is None:
                raise GraphExecutionError(
                    f"Graph {spec.graph_id}: no condition matched after {current}"
                )
            current = edge.target
            if current == spec.end:
                break
            node = spec.node_map[current]
            qualified = f"{prefix}{node.node_id}"
            state.current_node = qualified
            state.visited_nodes.append(qualified)

            node_context = (
                self.tracer.start_span(
                    "agent.node",
                    attributes={
                        "agent.graph_id": spec.graph_id,
                        "agent.node_id": qualified,
                        "agent.node_kind": node.kind,
                        "agent.retry_count": state.retry_count,
                    },
                )
                if self.tracer
                else nullcontext()
            )
            try:
                with node_context:
                    self._execute_node(spec, node, qualified, state)
            except GraphExecutionError:
                raise
            except Exception as exc:  # pragma: no cover - defensive boundary
                state.add_error(
                    "UNHANDLED_NODE_ERROR",
                    str(exc),
                    node_id=qualified,
                    retryable=False,
                )
                raise GraphExecutionError(f"Node {qualified} failed: {exc}") from exc

    def _execute_node(
        self,
        spec: GraphSpec,
        node: NodeSpec,
        qualified: str,
        state: AgentState,
    ) -> None:
        if node.kind == "subgraph":
            if not node.graph:
                raise GraphExecutionError(f"Subgraph node {qualified} has no graph path")
            child_path = (self.graph_root / node.graph).resolve()
            if self.graph_root not in child_path.parents:
                raise GraphExecutionError(f"Subgraph node {qualified} escapes graph root")
            child = load_graph_spec(child_path)
            self._run_spec(child, state, prefix=f"{qualified}.")
            state.node_outputs[qualified] = {"subgraph_id": child.graph_id}
            return

        if not node.handler or node.handler not in self.handlers:
            raise GraphExecutionError(
                f"No handler registered for {qualified}: {node.handler!r}"
            )
        context = NodeContext(
            graph_id=spec.graph_id,
            node_id=node.node_id,
            qualified_node_id=qualified,
            config=node.config,
        )
        outcome = self.handlers[node.handler](state, context)
        if outcome is None:
            outcome = NodeOutcome()
        if not isinstance(outcome, NodeOutcome):
            raise GraphExecutionError(
                f"Handler {node.handler} must return NodeOutcome or None"
            )
        state.apply_updates(outcome.updates)
        state.node_outputs[qualified] = outcome.output

    @staticmethod
    def _select_edge(
        spec: GraphSpec,
        source: str,
        state: AgentState,
    ) -> EdgeSpec | None:
        for edge in spec.outgoing(source):
            if evaluate_condition(edge.condition, state):
                return edge
        return None
