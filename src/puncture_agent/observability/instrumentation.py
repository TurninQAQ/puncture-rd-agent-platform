"""High-level span helpers for graph, RAG, model, MCP, verifier and checkpoint.

Adapters call these helpers so span names and required attribute keys stay
consistent with ``specs/eval-and-tracing.md``. All attributes are filtered by
:func:`~puncture_agent.observability.attributes.sanitize_attributes` when the
:class:`~puncture_agent.observability.tracing.TraceRecorder` sanitize flag is
enabled (default).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from puncture_agent.observability.attributes import hash_query_text
from puncture_agent.observability.propagation import TraceContext
from puncture_agent.observability.tracing import (
    SPAN_AGENT_GRAPH,
    SPAN_AGENT_NODE,
    SPAN_CHECKPOINT,
    SPAN_MCP_TOOL,
    SPAN_MODEL_GENERATE,
    SPAN_RAG_RERANK,
    SPAN_RAG_RETRIEVE,
    SPAN_RAG_REWRITE,
    SPAN_VERIFIER,
    SpanRecord,
    TraceRecorder,
)


@contextmanager
def graph_span(
    tracer: TraceRecorder,
    *,
    graph_id: str,
    session_id: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    runtime: str | None = None,
    resumed: bool | None = None,
    streaming: bool | None = None,
    trace_id: str | None = None,
    remote_parent: TraceContext | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {
        "agent.graph_id": graph_id,
    }
    if session_id is not None:
        attributes["agent.session_id"] = session_id
    if run_id is not None:
        attributes["run_id"] = run_id
    if request_id is not None:
        attributes["request_id"] = request_id
    if runtime is not None:
        attributes["agent.runtime"] = runtime
    if resumed is not None:
        attributes["agent.resumed"] = resumed
    if streaming is not None:
        attributes["agent.streaming"] = streaming
    if extra:
        attributes.update(extra)
    with tracer.start_span(
        SPAN_AGENT_GRAPH,
        attributes=attributes,
        trace_id=trace_id,
        remote_parent=remote_parent,
    ) as span:
        yield span


@contextmanager
def node_span(
    tracer: TraceRecorder,
    *,
    graph_id: str,
    node_id: str,
    node_kind: str,
    retry_count: int = 0,
    runtime: str | None = None,
    branch_result: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {
        "agent.graph_id": graph_id,
        "agent.node_id": node_id,
        "agent.node_kind": node_kind,
        "agent.retry_count": retry_count,
    }
    if runtime is not None:
        attributes["agent.runtime"] = runtime
    if branch_result is not None:
        attributes["agent.branch_result"] = branch_result
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_AGENT_NODE, attributes=attributes) as span:
        yield span


@contextmanager
def rag_rewrite_span(
    tracer: TraceRecorder,
    *,
    query: str | None = None,
    query_already_hashed: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {"rag.stage": "rewrite"}
    if query_already_hashed is not None:
        attributes["rag.rewritten_query_hash"] = query_already_hashed
    elif query is not None:
        attributes["rag.rewritten_query_hash"] = hash_query_text(query)
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_RAG_REWRITE, attributes=attributes) as span:
        yield span


@contextmanager
def rag_retrieve_span(
    tracer: TraceRecorder,
    *,
    top_k: int | None = None,
    embedding_version: str | None = None,
    index_version: str | None = None,
    document_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    scores: list[float] | None = None,
    current_version_hit: bool | None = None,
    acl_violation_count: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {"rag.stage": "retrieve"}
    if top_k is not None:
        attributes["rag.top_k"] = top_k
    if embedding_version is not None:
        attributes["rag.embedding_version"] = embedding_version
    if index_version is not None:
        attributes["rag.index_version"] = index_version
    if document_ids is not None:
        attributes["rag.retrieved_document_ids"] = list(document_ids)
    if chunk_ids is not None:
        attributes["rag.retrieved_chunk_ids"] = list(chunk_ids)
    if scores is not None:
        attributes["rag.retrieved_scores"] = list(scores)
    if current_version_hit is not None:
        attributes["rag.current_version_hit"] = current_version_hit
    if acl_violation_count is not None:
        attributes["rag.acl_violation_count"] = acl_violation_count
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_RAG_RETRIEVE, attributes=attributes) as span:
        yield span


@contextmanager
def rag_rerank_span(
    tracer: TraceRecorder,
    *,
    reranker_version: str | None = None,
    top_k: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {"rag.stage": "rerank"}
    if reranker_version is not None:
        attributes["rag.reranker_version"] = reranker_version
    if top_k is not None:
        attributes["rag.top_k"] = top_k
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_RAG_RERANK, attributes=attributes) as span:
        yield span


@contextmanager
def model_generate_span(
    tracer: TraceRecorder,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    structured_output_valid: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {}
    if model_name is not None:
        attributes["model.name"] = model_name
    if model_version is not None:
        attributes["model.version"] = model_version
    if temperature is not None:
        attributes["model.temperature"] = temperature
    if max_tokens is not None:
        attributes["model.max_tokens"] = max_tokens
    if input_tokens is not None:
        attributes["model.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["model.output_tokens"] = output_tokens
    if structured_output_valid is not None:
        attributes["model.structured_output_valid"] = structured_output_valid
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_MODEL_GENERATE, attributes=attributes) as span:
        yield span


@contextmanager
def mcp_tool_span(
    tracer: TraceRecorder,
    *,
    tool_name: str,
    tool_version: str | None = None,
    call_id: str | None = None,
    request_id: str | None = None,
    classification: str | None = None,
    attempt: int | None = None,
    schema_version: str | None = None,
    artifact_ids: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {"tool.name": tool_name}
    if tool_version is not None:
        attributes["tool.version"] = tool_version
    if call_id is not None:
        attributes["tool.call_id"] = call_id
    if request_id is not None:
        attributes["tool.request_id"] = request_id
    if classification is not None:
        attributes["tool.classification"] = classification
    if attempt is not None:
        attributes["tool.attempt"] = attempt
    if schema_version is not None:
        attributes["tool.schema_version"] = schema_version
    if artifact_ids is not None:
        attributes["tool.artifact_ids"] = list(artifact_ids)
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_MCP_TOOL, attributes=attributes) as span:
        yield span


@contextmanager
def verifier_span(
    tracer: TraceRecorder,
    *,
    result: str | None = None,
    reason_code: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {}
    if result is not None:
        attributes["verifier.result"] = result
    if reason_code is not None:
        attributes["verifier.reason_code"] = reason_code
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_VERIFIER, attributes=attributes) as span:
        yield span


@contextmanager
def checkpoint_span(
    tracer: TraceRecorder,
    *,
    checkpoint_id: str | None = None,
    thread_id: str | None = None,
    namespace: str | None = None,
    resumed: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[SpanRecord]:
    attributes: dict[str, Any] = {}
    if checkpoint_id is not None:
        attributes["checkpoint.id"] = checkpoint_id
    if thread_id is not None:
        attributes["checkpoint.thread_id"] = thread_id
    if namespace is not None:
        attributes["checkpoint.namespace"] = namespace
    if resumed is not None:
        attributes["checkpoint.resumed"] = resumed
    if extra:
        attributes.update(extra)
    with tracer.start_span(SPAN_CHECKPOINT, attributes=attributes) as span:
        yield span
