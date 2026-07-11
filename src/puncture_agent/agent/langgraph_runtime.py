"""Production LangGraph orchestration for the checked-in JSON graph specs.

The repository keeps :class:`~puncture_agent.agent.runtime.GraphRuntime` as a
dependency-free reference executor.  This module is the production adapter: it
compiles the same JSON topology into LangGraph, preserves ``AgentState`` as the
checkpoint wire shape, and keeps every deterministic edge decision outside the
language model.

LangGraph is an optional implementation dependency.  Importing this module is
safe without it; constructing :class:`LangGraphRuntime` reports an actionable
dependency error instead of silently falling back to the mock executor.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

from .graph_spec import EdgeSpec, GraphSpec, NodeSpec, load_graph_spec
from .langgraph_state import (
    LangGraphAgentState,
    StateConversionError,
    state_from_mapping,
    state_to_mapping,
)
from .runtime import (
    GraphExecutionError,
    NodeContext,
    NodeHandler,
    NodeOutcome,
    TracerLike,
    evaluate_condition,
)
from .state import AgentState, AgentStatus


_MISSING = object()


class LangGraphDependencyError(GraphExecutionError):
    """Raised when the optional production runtime packages are unavailable."""


class LangGraphCheckpointError(GraphExecutionError):
    """Raised when a checkpoint cannot be loaded or resumed safely."""


class LangGraphConcurrencyError(GraphExecutionError):
    """Raised when the same thread is executed concurrently in one runtime."""


class _NodeHandlerFailure(GraphExecutionError):
    def __init__(self, node_id: str, state: Mapping[str, Any]) -> None:
        super().__init__(f"Node {node_id} failed")
        self.node_id = node_id
        self.state = dict(state)


@dataclass(frozen=True)
class _LangGraphApi:
    state_graph: type[Any]
    start: Any
    end: Any
    in_memory_saver: type[Any]
    graph_bubble_up: type[BaseException]
    graph_recursion_error: type[BaseException]
    command: type[Any]


@dataclass(frozen=True)
class GraphStreamEvent:
    """Framework-neutral event consumed by the API/SSE layer."""

    event_type: str
    session_id: str
    node_id: str | None
    sequence: int
    state: Mapping[str, Any]


def _load_langgraph_api() -> _LangGraphApi:
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.errors import GraphBubbleUp, GraphRecursionError
        from langgraph.types import Command
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:  # compatibility with older 1.x-compatible releases
            from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    except (ImportError, ModuleNotFoundError) as exc:
        raise LangGraphDependencyError(
            "LangGraph runtime requires the 'implementation' dependencies; "
            "install langgraph>=1.2.9,<2 before constructing LangGraphRuntime"
        ) from exc
    return _LangGraphApi(
        StateGraph,
        START,
        END,
        InMemorySaver,
        GraphBubbleUp,
        GraphRecursionError,
        Command,
    )


def langgraph_available() -> bool:
    """Return whether the real LangGraph runtime can be imported."""

    try:
        _load_langgraph_api()
    except LangGraphDependencyError:
        return False
    return True


def _qualified(prefix: str, node_id: str) -> str:
    return f"{prefix}{node_id}"


def _copy_state(target: AgentState, source: AgentState) -> AgentState:
    for state_field in fields(AgentState):
        setattr(target, state_field.name, deepcopy(getattr(source, state_field.name)))
    return target


def _trusted_state_from_mapping(payload: Mapping[str, Any]) -> AgentState:
    """Rebuild state already validated at an invoke/node/checkpoint boundary."""

    if not isinstance(payload, Mapping):
        raise GraphExecutionError("LangGraph node input must be a state mapping")
    try:
        return AgentState.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise GraphExecutionError(f"invalid trusted graph state: {exc}") from exc


def _normalize_outcome(value: NodeOutcome | None, *, handler_name: str) -> NodeOutcome:
    if value is None:
        return NodeOutcome()
    if not isinstance(value, NodeOutcome):
        raise GraphExecutionError(
            f"Handler {handler_name} must return NodeOutcome or None"
        )
    return value


class LangGraphRuntime:
    """Compile and execute the locked JSON topology with real LangGraph.

    ``handlers`` use the same mutation-friendly adapter contract as the
    dependency-free runtime.  Each LangGraph node rebuilds an ``AgentState``,
    invokes exactly one handler, validates the resulting checkpoint mapping,
    and returns a full state update.  Returning the full mapping avoids reducer
    ambiguity for audit lists and keeps checkpoint round trips exact.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        graph: GraphSpec | str | Path,
        handlers: Mapping[str, NodeHandler],
        *,
        graph_root: str | Path | None = None,
        checkpointer: Any | None = None,
        tracer: TracerLike | None = None,
        langgraph_api: _LangGraphApi | None = None,
    ) -> None:
        self.graph = load_graph_spec(graph) if not isinstance(graph, GraphSpec) else graph
        if graph_root is not None:
            self.graph_root = Path(graph_root).resolve()
        elif self.graph.source_path is not None:
            self.graph_root = self.graph.source_path.parent
        else:
            raise ValueError("graph_root is required for an in-memory GraphSpec")
        self.handlers = MappingProxyType(dict(handlers))
        self.tracer = tracer
        self._api = langgraph_api or _load_langgraph_api()
        self.checkpointer = (
            self._api.in_memory_saver() if checkpointer is None else checkpointer
        )
        self._active_threads: set[str] = set()
        self._uncertain_threads: set[str] = set()
        self._active_threads_lock = Lock()
        self._compiled_children: dict[str, Any] = {}
        self.compiled_graph = self._compile_spec(
            self.graph,
            prefix="",
            checkpointer=self.checkpointer,
            container_node_id=None,
        )

    def run(
        self,
        state: AgentState,
        *,
        thread_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> AgentState:
        """Run a state to a terminal node using a checkpoint-isolated thread."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")
        self._validate_runtime_state(state)
        invocation_config = self._invocation_config(state, thread_id, config)
        normalized_thread = str(
            invocation_config["configurable"]["thread_id"]
        )
        with self._thread_execution(normalized_thread):
            return self._run_configured(state, invocation_config)

    def _run_configured(
        self,
        state: AgentState,
        invocation_config: Mapping[str, Any],
    ) -> AgentState:
        original_status = state.status
        original_trace = state.metadata.get("trace_id", _MISSING)
        state.status = AgentStatus.RUNNING
        graph_context = self._graph_span_context(state, resumed=False)
        try:
            with graph_context as graph_span:
                self._persist_trace_id(state, graph_span)
                raw_result = self.compiled_graph.invoke(
                    state_to_mapping(state),
                    config=invocation_config,
                    durability="sync",
                    version="v2",
                )
        except _NodeHandlerFailure as exc:
            failed_state = state_from_mapping(exc.state)
            _copy_state(state, failed_state)
            try:
                self._persist_failure_checkpoint(invocation_config, exc)
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=state,
                )
                raise LangGraphCheckpointError(
                    "node failed and its terminal checkpoint could not be persisted"
                ) from persist_exc
            raise GraphExecutionError(f"Node {exc.node_id} failed") from exc
        except self._api.graph_recursion_error as exc:
            requested_limit = int(invocation_config["recursion_limit"])
            try:
                terminal = self._terminalize_recursion_failure(
                    invocation_config,
                    fallback=state,
                    requested_limit=requested_limit,
                )
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=state,
                )
                raise LangGraphCheckpointError(
                    "recursion failure could not be persisted durably"
                ) from persist_exc
            _copy_state(state, terminal)
            raise GraphExecutionError(
                f"Graph {self.graph.graph_id} exceeded recursion_limit={requested_limit}"
            ) from exc
        except StateConversionError:
            state.status = original_status
            if original_trace is _MISSING:
                state.metadata.pop("trace_id", None)
            else:
                state.metadata["trace_id"] = original_trace
            raise
        except GraphExecutionError:
            raise
        except Exception as exc:
            self._terminalize_checkpoint_uncertainty(
                invocation_config,
                fallback=state,
            )
            raise LangGraphCheckpointError(
                "graph execution stopped because checkpoint durability is uncertain"
            ) from exc
        result, interrupts = self._unwrap_graph_output(raw_result)
        final_state = state_from_mapping(self._require_mapping(result, "invoke result"))
        self._apply_interrupts(final_state, interrupts)
        return _copy_state(state, final_state)

    def resume(
        self,
        *,
        thread_id: str,
        resume_value: Any = _MISSING,
        updates: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> AgentState:
        """Resume a pending LangGraph checkpoint without submitting new input."""

        normalized_thread = self._require_thread_id(thread_id)
        invocation_config = self._config_for_thread(
            normalized_thread,
            config,
            recursion_limit=self.graph.max_steps,
        )
        with self._thread_execution(normalized_thread):
            return self._resume_configured(
                normalized_thread=normalized_thread,
                invocation_config=invocation_config,
                resume_value=resume_value,
                updates=updates,
            )

    def _resume_configured(
        self,
        *,
        normalized_thread: str,
        invocation_config: Mapping[str, Any],
        resume_value: Any,
        updates: Mapping[str, Any] | None,
    ) -> AgentState:
        checkpoint = self.checkpoint_state(thread_id=normalized_thread)
        if updates is not None:
            if resume_value is not _MISSING:
                raise ValueError("resume_value and updates are mutually exclusive")
            if checkpoint.metadata.get("pending_interrupts"):
                raise ValueError(
                    "updates cannot replace a pending LangGraph interrupt; "
                    "provide resume_value instead"
                )
            if checkpoint.status != AgentStatus.AWAITING_INPUT:
                raise ValueError(
                    "updates are allowed only for an AWAITING_INPUT checkpoint"
                )
            if not isinstance(updates, Mapping) or not updates:
                raise ValueError("updates must be a non-empty mapping")
            checkpoint.apply_updates(updates)
            checkpoint.status = AgentStatus.CREATED
            checkpoint.current_node = None
            checkpoint.subgraph_result = {}
            checkpoint.verification_status = "NOT_RUN"
            checkpoint.final_report = {}
            checkpoint.metadata.pop("pending_interrupts", None)
            self._validate_runtime_state(checkpoint)
            return self._run_configured(checkpoint, invocation_config)
        if (
            checkpoint.metadata.get("pending_interrupts")
            and resume_value is _MISSING
        ):
            raise ValueError(
                "resume_value is required for a pending LangGraph interrupt"
            )
        if (
            resume_value is not _MISSING
            and not checkpoint.metadata.get("pending_interrupts")
        ):
            raise ValueError(
                "resume_value is allowed only for a pending LangGraph interrupt"
            )
        if resume_value is _MISSING and checkpoint.status in {
            AgentStatus.SUCCEEDED,
            AgentStatus.COMPLETED_WITH_NO_RESULT,
            AgentStatus.AWAITING_INPUT,
            AgentStatus.MANUAL_REVIEW,
            AgentStatus.FAILED,
        }:
            return checkpoint
        graph_input = (
            None
            if resume_value is _MISSING
            else self._api.command(resume=resume_value)
        )
        graph_context = self._graph_span_context(checkpoint, resumed=True)
        try:
            with graph_context as graph_span:
                self._persist_trace_id(checkpoint, graph_span)
                raw_result = self.compiled_graph.invoke(
                    graph_input,
                    config=invocation_config,
                    durability="sync",
                    version="v2",
                )
        except _NodeHandlerFailure as exc:
            failed_state = state_from_mapping(exc.state)
            try:
                self._persist_failure_checkpoint(invocation_config, exc)
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=failed_state,
                )
                raise LangGraphCheckpointError(
                    "resumed node failed and its terminal checkpoint was not persisted"
                ) from persist_exc
            raise GraphExecutionError(f"Node {exc.node_id} failed") from exc
        except self._api.graph_recursion_error as exc:
            requested_limit = int(invocation_config["recursion_limit"])
            try:
                self._terminalize_recursion_failure(
                    invocation_config,
                    fallback=checkpoint,
                    requested_limit=requested_limit,
                )
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=checkpoint,
                )
                raise LangGraphCheckpointError(
                    "resumed recursion failure was not persisted durably"
                ) from persist_exc
            raise GraphExecutionError(
                f"Graph {self.graph.graph_id} exceeded recursion_limit={requested_limit}"
            ) from exc
        except (GraphExecutionError, StateConversionError):
            raise
        except Exception as exc:
            self._terminalize_checkpoint_uncertainty(
                invocation_config,
                fallback=checkpoint,
            )
            raise LangGraphCheckpointError(
                "graph resume stopped because checkpoint durability is uncertain"
            ) from exc
        result, interrupts = self._unwrap_graph_output(raw_result)
        if result is None:
            raise LangGraphCheckpointError(
                "checkpoint resume returned no state for the requested thread"
            )
        resumed = state_from_mapping(self._require_mapping(result, "resume result"))
        self._apply_interrupts(resumed, interrupts)
        return resumed

    def checkpoint_state(self, *, thread_id: str) -> AgentState:
        """Read the latest durable values for one thread without executing it."""

        normalized_thread = self._require_thread_id(thread_id)
        config = self._config_for_thread(
            normalized_thread,
            None,
            recursion_limit=self.graph.max_steps,
        )
        try:
            snapshot = self.compiled_graph.get_state(config, subgraphs=True)
            values = self._deepest_snapshot_values(snapshot)
            if not values:
                raise LangGraphCheckpointError(
                    "no checkpoint exists for the requested thread"
                )
            state = state_from_mapping(
                self._require_mapping(values, "checkpoint values")
            )
            self._apply_interrupts(state, self._snapshot_interrupts(snapshot))
            return state
        except LangGraphCheckpointError:
            raise
        except Exception as exc:
            raise LangGraphCheckpointError(
                "failed to load a valid checkpoint for the requested thread"
            ) from exc

    def stream(
        self,
        state: AgentState,
        *,
        thread_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Iterator[GraphStreamEvent]:
        """Yield sanitized state transitions suitable for an API event stream."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")
        self._validate_runtime_state(state)
        invocation_config = self._invocation_config(state, thread_id, config)
        normalized_thread = str(
            invocation_config["configurable"]["thread_id"]
        )
        with self._thread_execution(normalized_thread):
            events = tuple(self._stream_configured(state, invocation_config))
        yield from events

    def _stream_configured(
        self,
        state: AgentState,
        invocation_config: Mapping[str, Any],
    ) -> Iterator[GraphStreamEvent]:
        original_status = state.status
        original_trace = state.metadata.get("trace_id", _MISSING)
        state.status = AgentStatus.RUNNING
        seen_nodes = len(state.visited_nodes)
        sequence = 0
        final_state = state
        final_interrupts: tuple[Any, ...] = ()
        buffered_events: list[GraphStreamEvent] = []
        deferred_containers: list[str] = []
        graph_context = self._graph_span_context(
            state,
            resumed=False,
            streaming=True,
        )
        try:
            with graph_context as graph_span:
                self._persist_trace_id(state, graph_span)
                chunks = self.compiled_graph.stream(
                    state_to_mapping(state),
                    config=invocation_config,
                    stream_mode="values",
                    durability="sync",
                    subgraphs=True,
                    version="v2",
                )
                for raw_chunk in chunks:
                    envelope = self._require_mapping(raw_chunk, "stream envelope")
                    if envelope.get("type") != "values":
                        continue
                    chunk = self._require_mapping(
                        envelope.get("data"), "stream value"
                    )
                    final_state = state_from_mapping(chunk)
                    interrupts = envelope.get("interrupts") or ()
                    if not isinstance(interrupts, (list, tuple)):
                        raise GraphExecutionError(
                            "stream interrupts must be a list or tuple"
                        )
                    if interrupts:
                        final_interrupts = tuple(interrupts)
                    for node_id in final_state.visited_nodes[seen_nodes:]:
                        if (
                            node_id in self._compiled_children
                            and node_id not in final_state.node_outputs
                        ):
                            if node_id not in deferred_containers:
                                deferred_containers.append(node_id)
                            continue
                        sequence += 1
                        buffered_events.append(
                            GraphStreamEvent(
                                event_type="NODE_COMPLETED",
                                session_id=final_state.session_id,
                                node_id=node_id,
                                sequence=sequence,
                                state=state_to_mapping(final_state),
                            )
                        )
                    seen_nodes = len(final_state.visited_nodes)
                    completed_containers = [
                        node_id
                        for node_id in deferred_containers
                        if node_id in final_state.node_outputs
                    ]
                    for node_id in completed_containers:
                        sequence += 1
                        buffered_events.append(
                            GraphStreamEvent(
                                event_type="NODE_COMPLETED",
                                session_id=final_state.session_id,
                                node_id=node_id,
                                sequence=sequence,
                                state=state_to_mapping(final_state),
                            )
                        )
                        deferred_containers.remove(node_id)
        except _NodeHandlerFailure as exc:
            failed_state = state_from_mapping(exc.state)
            _copy_state(state, failed_state)
            try:
                self._persist_failure_checkpoint(invocation_config, exc)
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=state,
                )
                raise LangGraphCheckpointError(
                    "streamed node failed and its terminal checkpoint was not persisted"
                ) from persist_exc
            raise GraphExecutionError(f"Node {exc.node_id} failed") from exc
        except self._api.graph_recursion_error as exc:
            requested_limit = int(invocation_config["recursion_limit"])
            try:
                terminal = self._terminalize_recursion_failure(
                    invocation_config,
                    fallback=state,
                    requested_limit=requested_limit,
                )
            except LangGraphCheckpointError as persist_exc:
                self._terminalize_checkpoint_uncertainty(
                    invocation_config,
                    fallback=state,
                )
                raise LangGraphCheckpointError(
                    "stream recursion failure was not persisted durably"
                ) from persist_exc
            _copy_state(state, terminal)
            raise GraphExecutionError(
                f"Graph {self.graph.graph_id} exceeded recursion_limit={requested_limit}"
            ) from exc
        except StateConversionError:
            state.status = original_status
            if original_trace is _MISSING:
                state.metadata.pop("trace_id", None)
            else:
                state.metadata["trace_id"] = original_trace
            raise
        except GraphExecutionError:
            raise
        except Exception as exc:
            self._terminalize_checkpoint_uncertainty(
                invocation_config,
                fallback=state,
            )
            raise LangGraphCheckpointError(
                "graph stream stopped because checkpoint durability is uncertain"
            ) from exc
        self._apply_interrupts(final_state, final_interrupts)
        _copy_state(state, final_state)
        for event in buffered_events:
            yield event
        sequence += 1
        yield GraphStreamEvent(
            event_type=("RUN_INTERRUPTED" if final_interrupts else "RUN_COMPLETED"),
            session_id=state.session_id,
            node_id=None,
            sequence=sequence,
            state=state_to_mapping(state),
        )

    def _compile_spec(
        self,
        spec: GraphSpec,
        *,
        prefix: str,
        checkpointer: Any | None,
        container_node_id: str | None,
    ) -> Any:
        builder = self._api.state_graph(LangGraphAgentState)
        entry_nodes = {
            edge.target for edge in spec.outgoing(spec.start) if edge.target != spec.end
        }
        exit_nodes = {
            edge.source for edge in spec.edges if edge.target == spec.end
        }
        for node in spec.nodes:
            if node.kind == "subgraph":
                if not node.graph:
                    raise GraphExecutionError(
                        f"Subgraph node {node.node_id} has no graph path"
                    )
                child_path = (self.graph_root / node.graph).resolve()
                if self.graph_root not in child_path.parents:
                    raise GraphExecutionError(
                        f"Subgraph node {node.node_id} escapes graph root"
                    )
                child = load_graph_spec(child_path)
                qualified = _qualified(prefix, node.node_id)
                compiled_child = self._compile_spec(
                    child,
                    prefix=f"{qualified}.",
                    checkpointer=None,
                    container_node_id=qualified,
                )
                self._compiled_children[qualified] = compiled_child
                builder.add_node(node.node_id, compiled_child)
            else:
                builder.add_node(
                    node.node_id,
                    self._build_handler_node(
                        spec,
                        node,
                        prefix=prefix,
                        enter_container=(
                            container_node_id if node.node_id in entry_nodes else None
                        ),
                        exit_container=(
                            container_node_id if node.node_id in exit_nodes else None
                        ),
                    ),
                )
        self._wire_edges(builder, spec)
        if checkpointer is None:
            return builder.compile()
        return builder.compile(checkpointer=checkpointer)

    def _build_handler_node(
        self,
        spec: GraphSpec,
        node: NodeSpec,
        *,
        prefix: str,
        enter_container: str | None,
        exit_container: str | None,
    ) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        if not node.handler or node.handler not in self.handlers:
            raise GraphExecutionError(
                f"No handler registered for {_qualified(prefix, node.node_id)}: "
                f"{node.handler!r}"
            )
        handler = self.handlers[node.handler]
        qualified = _qualified(prefix, node.node_id)

        def execute(payload: Mapping[str, Any]) -> Mapping[str, Any]:
            working = _trusted_state_from_mapping(payload)

            def failed_mapping(
                code: str,
                message: str,
                *,
                error_type: str,
            ) -> Mapping[str, Any]:
                failure_state = working
                failure_state.status = AgentStatus.FAILED
                failure_state.add_error(
                    code,
                    message,
                    node_id=qualified,
                    retryable=False,
                )
                failure_state.node_outputs[qualified] = {
                    "error_code": code,
                    "error_type": error_type,
                }
                try:
                    return state_to_mapping(failure_state)
                except StateConversionError:
                    # The handler may itself have inserted bytes/non-JSON data.
                    # Rebuild from the last trusted input and keep only a safe
                    # deterministic failure record for durable recovery.
                    failure_state = _trusted_state_from_mapping(payload)
                    if enter_container:
                        failure_state.visited_nodes.append(enter_container)
                    failure_state.current_node = qualified
                    failure_state.visited_nodes.append(qualified)
                    failure_state.status = AgentStatus.FAILED
                    failure_state.add_error(
                        code,
                        message,
                        node_id=qualified,
                        retryable=False,
                    )
                    failure_state.node_outputs[qualified] = {
                        "error_code": code,
                        "error_type": error_type,
                    }
                    return state_to_mapping(failure_state)

            if enter_container:
                counters = working.metadata.setdefault(
                    "_langgraph_subgraph_steps", {}
                )
                if not isinstance(counters, dict):
                    raise GraphExecutionError(
                        "reserved subgraph step metadata must be an object"
                    )
                counters[enter_container] = 0
            if prefix:
                container = prefix[:-1]
                counters = working.metadata.setdefault(
                    "_langgraph_subgraph_steps", {}
                )
                if not isinstance(counters, dict):
                    raise GraphExecutionError(
                        "reserved subgraph step metadata must be an object"
                    )
                count = counters.get(container, 0)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise GraphExecutionError("invalid subgraph step counter")
                count += 1
                if count > spec.max_steps:
                    raise GraphExecutionError(
                        f"Graph {spec.graph_id} exceeded max_steps={spec.max_steps}"
                    )
                counters[container] = count
            if enter_container:
                working.current_node = enter_container
                working.visited_nodes.append(enter_container)
            working.current_node = qualified
            working.visited_nodes.append(qualified)
            context = NodeContext(
                graph_id=spec.graph_id,
                node_id=node.node_id,
                qualified_node_id=qualified,
                config=node.config,
            )
            node_context = (
                self.tracer.start_span(
                    "agent.node",
                    attributes={
                        "agent.graph_id": spec.graph_id,
                        "agent.node_id": qualified,
                        "agent.node_kind": node.kind,
                        "agent.retry_count": working.retry_count,
                        "agent.runtime": "langgraph",
                    },
                )
                if self.tracer
                else nullcontext()
            )
            graph_bubble: BaseException | None = None
            successful_mapping: Mapping[str, Any] | None = None
            try:
                with node_context:
                    try:
                        outcome = _normalize_outcome(
                            handler(working, context),
                            handler_name=node.handler or qualified,
                        )
                        working.apply_updates(outcome.updates)
                        working.node_outputs[qualified] = outcome.output
                        if exit_container:
                            working.node_outputs[exit_container] = {
                                "subgraph_id": spec.graph_id
                            }
                            counters = working.metadata.get(
                                "_langgraph_subgraph_steps", {}
                            )
                            if isinstance(counters, dict):
                                counters.pop(exit_container, None)
                                if not counters:
                                    working.metadata.pop(
                                        "_langgraph_subgraph_steps", None
                                    )
                        add_event = getattr(self.tracer, "add_event", None)
                        if callable(add_event):
                            for event in outcome.events:
                                add_event(
                                    str(event.get("name", "agent.node.event")),
                                    attributes={
                                        key: value
                                        for key, value in event.items()
                                        if key != "name"
                                    },
                                )
                        successful_mapping = state_to_mapping(working)
                    except self._api.graph_bubble_up as exc:
                        bubble_interrupts = self._interrupts_from_bubble(exc)
                        if bubble_interrupts:
                            # LangGraph persists dynamic interrupts after this
                            # exception leaves the node.  Validate the payload
                            # first so bytes/cycles/oversized values cannot
                            # create a checkpoint that neither checkpoint_state
                            # nor resume can safely decode.
                            interrupt_state = deepcopy(working)
                            self._apply_interrupts(
                                interrupt_state,
                                bubble_interrupts,
                            )
                        graph_bubble = exc
                        add_event = getattr(self.tracer, "add_event", None)
                        if callable(add_event):
                            add_event(
                                "agent.node.interrupted",
                                attributes={
                                    "agent.graph_id": spec.graph_id,
                                    "agent.node_id": qualified,
                                },
                            )
                if graph_bubble is not None:
                    raise graph_bubble
            except self._api.graph_bubble_up:
                raise
            except GraphExecutionError as exc:
                raise _NodeHandlerFailure(
                    qualified,
                    failed_mapping(
                        "NODE_CONTRACT_ERROR",
                        "node violated the runtime handler contract",
                        error_type=type(exc).__name__,
                    ),
                ) from exc
            except StateConversionError as exc:
                raise _NodeHandlerFailure(
                    qualified,
                    failed_mapping(
                        "STATE_BOUNDARY_ERROR",
                        "node produced state forbidden by the checkpoint contract",
                        error_type=type(exc).__name__,
                    ),
                ) from exc
            except Exception as exc:
                raise _NodeHandlerFailure(
                    qualified,
                    failed_mapping(
                        "UNHANDLED_NODE_ERROR",
                        f"node rejected an unexpected {type(exc).__name__}",
                        error_type=type(exc).__name__,
                    ),
                ) from exc
            if successful_mapping is None:
                raise GraphExecutionError(
                    f"Node {qualified} completed without a validated state update"
                )
            return successful_mapping

        execute.__name__ = f"langgraph_node_{qualified.replace('.', '_')}"
        return execute

    def _wire_edges(self, builder: Any, spec: GraphSpec) -> None:
        sources = [spec.start, *(node.node_id for node in spec.nodes)]
        for source in sources:
            outgoing = spec.outgoing(source)
            if not outgoing:
                continue
            translated_source = self._translate_sentinel(spec, source)
            if len(outgoing) == 1 and outgoing[0].condition.get("operator") == "always":
                builder.add_edge(
                    translated_source,
                    self._translate_sentinel(spec, outgoing[0].target),
                )
                continue

            route = self._build_route(spec, source, outgoing)
            path_map: dict[Any, Any] = {}
            for edge in outgoing:
                target = self._translate_sentinel(spec, edge.target)
                path_map[target] = target
            builder.add_conditional_edges(translated_source, route, path_map)

    def _build_route(
        self,
        spec: GraphSpec,
        source: str,
        outgoing: tuple[EdgeSpec, ...],
    ) -> Callable[[Mapping[str, Any]], Any]:
        def route(payload: Mapping[str, Any]) -> Any:
            state = _trusted_state_from_mapping(payload)
            for edge in outgoing:
                if evaluate_condition(edge.condition, state):
                    return self._translate_sentinel(spec, edge.target)
            raise GraphExecutionError(
                f"Graph {spec.graph_id}: no condition matched after {source}"
            )

        route.__name__ = f"route_{spec.graph_id}_{source.lower()}"
        return route

    def _translate_sentinel(self, spec: GraphSpec, node_id: str) -> Any:
        if node_id == spec.start:
            return self._api.start
        if node_id == spec.end:
            return self._api.end
        return node_id

    def _invocation_config(
        self,
        state: AgentState,
        thread_id: str | None,
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_thread = self._require_thread_id(thread_id or state.session_id)
        if normalized_thread != state.session_id:
            raise ValueError("thread_id must match AgentState.session_id")
        return self._config_for_thread(
            normalized_thread,
            config,
            recursion_limit=self.graph.max_steps,
        )

    @staticmethod
    def _config_for_thread(
        thread_id: str,
        config: Mapping[str, Any] | None,
        *,
        recursion_limit: int,
    ) -> dict[str, Any]:
        result = deepcopy(dict(config or {}))
        configurable = result.get("configurable", {})
        if not isinstance(configurable, Mapping):
            raise ValueError("config.configurable must be an object")
        configurable = dict(configurable)
        existing = configurable.get("thread_id")
        if existing is not None and existing != thread_id:
            raise ValueError("configurable.thread_id conflicts with the runtime thread")
        configurable["thread_id"] = thread_id
        result["configurable"] = configurable
        requested_limit = result.get("recursion_limit", recursion_limit)
        if (
            isinstance(requested_limit, bool)
            or not isinstance(requested_limit, int)
            or requested_limit <= 0
        ):
            raise ValueError("recursion_limit must be a positive integer")
        if requested_limit > recursion_limit:
            raise ValueError(
                f"recursion_limit cannot exceed locked max_steps={recursion_limit}"
            )
        result["recursion_limit"] = requested_limit
        return result

    @contextmanager
    def _thread_execution(self, thread_id: str) -> Iterator[None]:
        with self._active_threads_lock:
            if thread_id in self._uncertain_threads:
                raise LangGraphCheckpointError(
                    "this thread requires manual checkpoint reconciliation"
                )
            if thread_id in self._active_threads:
                raise LangGraphConcurrencyError(
                    "the same LangGraph thread cannot execute concurrently"
                )
            self._active_threads.add(thread_id)
        try:
            yield
        finally:
            with self._active_threads_lock:
                self._active_threads.discard(thread_id)

    @classmethod
    def _validate_runtime_state(cls, state: AgentState) -> None:
        if "_langgraph_subgraph_steps" in state.metadata:
            raise ValueError(
                "metadata._langgraph_subgraph_steps is reserved for the runtime"
            )
        if (
            isinstance(state.max_retries, bool)
            or not isinstance(state.max_retries, int)
            or not 0 <= state.max_retries <= cls.MAX_RETRIES
        ):
            raise ValueError(f"max_retries must be between 0 and {cls.MAX_RETRIES}")
        if (
            isinstance(state.retry_count, bool)
            or not isinstance(state.retry_count, int)
            or not 0 <= state.retry_count <= state.max_retries
        ):
            raise ValueError("retry_count must be between 0 and max_retries")

    @classmethod
    def _deepest_snapshot_values(cls, snapshot: Any) -> Mapping[str, Any] | None:
        candidates: list[Mapping[str, Any]] = []

        def visit(item: Any) -> None:
            values = getattr(item, "values", None)
            if isinstance(values, Mapping) and values:
                candidates.append(values)
            for task in getattr(item, "tasks", ()) or ():
                child = getattr(task, "state", None)
                if hasattr(child, "values"):
                    visit(child)

        visit(snapshot)
        if not candidates:
            return None
        candidates.sort(
            key=lambda value: len(value.get("visited_nodes", ()))
            if isinstance(value.get("visited_nodes", ()), (list, tuple))
            else -1
        )
        return candidates[-1]

    @classmethod
    def _snapshot_interrupts(cls, snapshot: Any) -> tuple[Any, ...]:
        """Collect current interrupts from root/tasks/child snapshots once by ID."""

        by_id: dict[str, Any] = {}

        def add(items: Any) -> None:
            for item in items or ():
                interrupt_id = (
                    item.get("id")
                    if isinstance(item, Mapping)
                    else getattr(item, "id", None)
                )
                if not isinstance(interrupt_id, str) or not interrupt_id:
                    raise LangGraphCheckpointError(
                        "checkpoint contains an interrupt without a stable id"
                    )
                previous = by_id.get(interrupt_id)
                if previous is None:
                    by_id[interrupt_id] = item
                    continue
                previous_value = (
                    previous.get("value")
                    if isinstance(previous, Mapping)
                    else getattr(previous, "value", None)
                )
                current_value = (
                    item.get("value")
                    if isinstance(item, Mapping)
                    else getattr(item, "value", None)
                )
                if previous_value != current_value:
                    raise LangGraphCheckpointError(
                        "checkpoint contains conflicting values for one interrupt id"
                    )

        def visit(item: Any) -> None:
            add(getattr(item, "interrupts", ()))
            for task in getattr(item, "tasks", ()) or ():
                add(getattr(task, "interrupts", ()))
                child = getattr(task, "state", None)
                if hasattr(child, "values"):
                    visit(child)

        visit(snapshot)
        return tuple(by_id.values())

    def _persist_failure_checkpoint(
        self,
        invocation_config: Mapping[str, Any],
        failure: _NodeHandlerFailure,
    ) -> None:
        """Write a terminal failed state and clear the pending LangGraph task."""

        self._persist_terminal_checkpoint(
            invocation_config,
            failure.state,
            node_id=failure.node_id,
        )

    def _persist_terminal_checkpoint(
        self,
        invocation_config: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        node_id: str,
    ) -> None:
        """Persist full state as one root-node update, then clear pending tasks."""

        update_state = getattr(self.compiled_graph, "update_state", None)
        if not callable(update_state):
            raise LangGraphCheckpointError(
                "compiled graph cannot persist a terminal node failure"
            )
        root_node = node_id.split(".", 1)[0]
        try:
            updated_config = update_state(
                invocation_config,
                dict(state),
                as_node=root_node,
            )
            update_state(updated_config, None, as_node=self._api.end)
        except Exception as exc:
            raise LangGraphCheckpointError(
                f"failed to persist terminal state for node {node_id}"
            ) from exc

    def _latest_checkpoint_or(self, fallback: AgentState) -> AgentState:
        try:
            checkpoint = self.checkpoint_state(thread_id=fallback.session_id)
        except Exception:
            return deepcopy(fallback)
        terminal_statuses = {
            AgentStatus.SUCCEEDED,
            AgentStatus.COMPLETED_WITH_NO_RESULT,
            AgentStatus.AWAITING_INPUT,
            AgentStatus.MANUAL_REVIEW,
            AgentStatus.FAILED,
        }

        def rank(state: AgentState) -> tuple[int, int, int, int]:
            return (
                len(state.visited_nodes),
                len(state.tool_calls),
                int(state.status in terminal_statuses),
                len(state.errors),
            )

        return checkpoint if rank(checkpoint) > rank(fallback) else deepcopy(fallback)

    def _terminal_node_id(self, state: AgentState) -> str:
        if isinstance(state.current_node, str) and state.current_node:
            return state.current_node
        outgoing = self.graph.outgoing(self.graph.start)
        if not outgoing:
            raise LangGraphCheckpointError("main graph has no start transition")
        return outgoing[0].target

    def _terminalize_recursion_failure(
        self,
        invocation_config: Mapping[str, Any],
        *,
        fallback: AgentState,
        requested_limit: int,
    ) -> AgentState:
        terminal = self._latest_checkpoint_or(fallback)
        terminal.status = AgentStatus.FAILED
        if not any(
            error.get("code") == "GRAPH_STEP_LIMIT_EXCEEDED"
            for error in terminal.errors
            if isinstance(error, Mapping)
        ):
            terminal.add_error(
                "GRAPH_STEP_LIMIT_EXCEEDED",
                "graph execution exceeded the requested recursion limit",
                retryable=False,
                details={"requested_recursion_limit": requested_limit},
            )
        _copy_state(fallback, terminal)
        self._persist_terminal_checkpoint(
            invocation_config,
            state_to_mapping(terminal),
            node_id=self._terminal_node_id(terminal),
        )
        return terminal

    def _terminalize_checkpoint_uncertainty(
        self,
        invocation_config: Mapping[str, Any],
        *,
        fallback: AgentState,
    ) -> AgentState:
        terminal = self._latest_checkpoint_or(fallback)
        terminal.status = AgentStatus.MANUAL_REVIEW
        terminal.metadata["execution_state_uncertain"] = True
        with self._active_threads_lock:
            self._uncertain_threads.add(terminal.session_id)
        if not any(
            error.get("code") == "CHECKPOINT_DURABILITY_UNCERTAIN"
            for error in terminal.errors
            if isinstance(error, Mapping)
        ):
            terminal.add_error(
                "CHECKPOINT_DURABILITY_UNCERTAIN",
                "checkpoint persistence failed after graph execution may have advanced",
                retryable=False,
            )
        _copy_state(fallback, terminal)
        try:
            self._persist_terminal_checkpoint(
                invocation_config,
                state_to_mapping(terminal),
                node_id=self._terminal_node_id(terminal),
            )
        except LangGraphCheckpointError:
            # The primary state remains explicit and non-retryable even when
            # the same unavailable saver cannot record the uncertainty marker.
            pass
        return terminal

    @staticmethod
    def _require_thread_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("thread_id must be a non-empty string")
        if any(character in value for character in ("\r", "\n", "\x00")):
            raise ValueError("thread_id contains control characters")
        if len(value) >= 255:
            raise ValueError("thread_id must contain fewer than 255 characters")
        return value

    @staticmethod
    def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise GraphExecutionError(f"{location} must be a state mapping")
        return value

    @staticmethod
    def _unwrap_graph_output(value: Any) -> tuple[Any, tuple[Any, ...]]:
        """Normalize LangGraph v2 ``GraphOutput`` without importing its type."""

        if hasattr(value, "value") and hasattr(value, "interrupts"):
            interrupts = getattr(value, "interrupts") or ()
            if not isinstance(interrupts, (list, tuple)):
                raise GraphExecutionError(
                    "LangGraph output interrupts must be a list or tuple"
                )
            return getattr(value, "value"), tuple(interrupts)
        return value, ()

    @staticmethod
    def _interrupts_from_bubble(exc: BaseException) -> tuple[Any, ...]:
        """Read GraphInterrupt payloads without depending on its concrete type."""

        candidate = getattr(exc, "interrupts", _MISSING)
        if candidate is _MISSING:
            args = getattr(exc, "args", ())
            candidate = args[0] if len(args) == 1 else ()
        if candidate is None:
            return ()
        if not isinstance(candidate, (list, tuple)):
            # Other GraphBubbleUp subclasses carry framework control data, not
            # user-facing dynamic interrupt values.
            return ()
        return tuple(candidate)

    @staticmethod
    def _apply_interrupts(state: AgentState, interrupts: tuple[Any, ...]) -> None:
        if not interrupts:
            state.metadata.pop("pending_interrupts", None)
            return
        pending: list[dict[str, Any]] = []
        for index, item in enumerate(interrupts):
            if isinstance(item, Mapping):
                interrupt_id = item.get("id")
                interrupt_value = item.get("value")
            else:
                interrupt_id = getattr(item, "id", None)
                interrupt_value = getattr(item, "value", None)
            if not isinstance(interrupt_id, str) or not interrupt_id:
                raise GraphExecutionError(
                    f"LangGraph interrupt at index {index} has no stable id"
                )
            pending.append(
                {
                    "id": interrupt_id,
                    "value": deepcopy(interrupt_value),
                }
            )
        state.status = AgentStatus.AWAITING_INPUT
        state.metadata["pending_interrupts"] = pending
        # Interrupt payloads become checkpoint/API state and must obey the same
        # raw-bytes, JSON and size constraints as every ordinary node update.
        state_to_mapping(state)

    def _graph_span_context(
        self,
        state: AgentState,
        *,
        resumed: bool,
        streaming: bool = False,
    ) -> Any:
        if not self.tracer:
            return nullcontext(None)
        attributes = {
            "agent.graph_id": self.graph.graph_id,
            "agent.session_id": state.session_id,
            "agent.runtime": "langgraph",
            "agent.resumed": resumed,
        }
        if streaming:
            attributes["agent.streaming"] = True
        trace_id = state.metadata.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip():
            trace_id = None
        try:
            return self.tracer.start_span(
                "agent.graph",
                attributes=attributes,
                trace_id=trace_id,
            )
        except TypeError:
            # Keep compatibility with a minimal third-party TracerLike written
            # before trace continuation was added to this protocol.
            return self.tracer.start_span(
                "agent.graph",
                attributes=attributes,
            )

    @staticmethod
    def _persist_trace_id(state: AgentState, graph_span: Any) -> None:
        trace_id = getattr(graph_span, "trace_id", None)
        if isinstance(trace_id, str) and trace_id.strip():
            state.metadata["trace_id"] = trace_id


def _validate_postgres_dsn(connection_string: str) -> str:
    if not isinstance(connection_string, str) or not connection_string.strip():
        raise ValueError("PostgreSQL connection string must be non-empty")
    parsed = urlsplit(connection_string)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("PostgreSQL connection string must use postgres:// or postgresql://")
    if any(character in connection_string for character in ("\r", "\n", "\x00")):
        raise ValueError("PostgreSQL connection string contains control characters")
    return connection_string


@contextmanager
def open_postgres_checkpointer(
    connection_string: str,
    *,
    setup: bool = True,
) -> Iterator[Any]:
    """Open the official PostgreSQL saver without exposing credentials.

    ``langgraph-checkpoint-postgres`` intentionally remains optional for local
    standard-library tests.  Production startup should enter this context once,
    call ``setup`` during an explicit migration step, compile the runtime, and
    keep the context alive for the runtime lifetime.
    """

    dsn = _validate_postgres_dsn(connection_string)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except (ImportError, ModuleNotFoundError) as exc:
        raise LangGraphDependencyError(
            "PostgreSQL checkpoints require langgraph-checkpoint-postgres and psycopg"
        ) from exc

    resource = PostgresSaver.from_conn_string(dsn)
    manager = resource if hasattr(resource, "__enter__") else nullcontext(resource)
    with manager as saver:
        if setup:
            setup_method = getattr(saver, "setup", None)
            if not callable(setup_method):
                raise LangGraphCheckpointError(
                    "PostgresSaver does not expose the required setup() migration API"
                )
            setup_method()
        yield saver


__all__ = [
    "GraphStreamEvent",
    "LangGraphCheckpointError",
    "LangGraphDependencyError",
    "LangGraphRuntime",
    "langgraph_available",
    "open_postgres_checkpointer",
]
