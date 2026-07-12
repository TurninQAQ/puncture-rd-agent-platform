"""OpenTelemetry facade, privacy, propagation and concurrency tests (stdlib)."""

from __future__ import annotations

import concurrent.futures
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.observability.attributes import (  # noqa: E402
    REDACTED,
    hash_query_text,
    sanitize_attributes,
)
from puncture_agent.observability.instrumentation import (  # noqa: E402
    checkpoint_span,
    graph_span,
    mcp_tool_span,
    model_generate_span,
    node_span,
    rag_rerank_span,
    rag_retrieve_span,
    rag_rewrite_span,
    verifier_span,
)
from puncture_agent.observability.propagation import (  # noqa: E402
    TRACEPARENT_HEADER,
    TRACE_ID_HEADER,
    extract_trace_context,
    format_traceparent,
    inject_trace_context,
    new_trace_context,
    parse_traceparent,
)
from puncture_agent.observability.tracing import (  # noqa: E402
    CompositeTraceExporter,
    InMemoryOtlpTraceExporter,
    InMemoryTraceExporter,
    SPAN_AGENT_GRAPH,
    SPAN_AGENT_NODE,
    SPAN_CHECKPOINT,
    SPAN_MCP_TOOL,
    SPAN_MODEL_GENERATE,
    SPAN_RAG_RERANK,
    SPAN_RAG_RETRIEVE,
    SPAN_RAG_REWRITE,
    SPAN_VERIFIER,
    TraceRecorder,
)


class AttributePolicyTests(unittest.TestCase):
    def test_allowlist_keeps_operational_fields(self) -> None:
        cleaned = sanitize_attributes(
            {
                "agent.graph_id": "main",
                "agent.prompt_version": "v3",
                "case_id": "Case-1",
                "unknown_debug_blob": "drop-me",
            }
        )
        self.assertEqual("main", cleaned["agent.graph_id"])
        self.assertEqual("v3", cleaned["agent.prompt_version"])
        self.assertEqual("Case-1", cleaned["case_id"])
        self.assertNotIn("unknown_debug_blob", cleaned)

    def test_denylisted_and_sensitive_fields_are_redacted(self) -> None:
        cleaned = sanitize_attributes(
            {
                "authorization": "Bearer secret-token",
                "patient_name": "Alice",
                "full_prompt": "unrestricted medical prompt",
                "api_key": "sk-test",
                "image_bytes": b"\x00\x01",
                "agent.node_id": "retrieve_project_knowledge",
            }
        )
        self.assertEqual(REDACTED, cleaned["authorization"])
        self.assertEqual(REDACTED, cleaned["patient_name"])
        self.assertEqual(REDACTED, cleaned["full_prompt"])
        self.assertEqual(REDACTED, cleaned["api_key"])
        self.assertEqual(REDACTED, cleaned["image_bytes"])
        self.assertEqual("retrieve_project_knowledge", cleaned["agent.node_id"])
        self.assertNotIn("Bearer", str(cleaned.values()))
        self.assertNotIn("Alice", str(cleaned.values()))

    def test_bytes_and_long_strings_are_capped(self) -> None:
        cleaned = sanitize_attributes(
            {
                "rag.query_sanitized": "x" * 2000,
                "tool.artifact_ids": ["a1", "a2"],
            }
        )
        self.assertTrue(cleaned["rag.query_sanitized"].endswith("…[truncated]"))
        self.assertEqual(["a1", "a2"], cleaned["tool.artifact_ids"])

    def test_query_hash_is_stable(self) -> None:
        self.assertEqual(hash_query_text("hello"), hash_query_text("hello"))
        self.assertNotEqual(hash_query_text("hello"), hash_query_text("world"))


class PropagationTests(unittest.TestCase):
    def test_traceparent_round_trip(self) -> None:
        context = new_trace_context()
        encoded = format_traceparent(context)
        parsed = parse_traceparent(encoded)
        self.assertEqual(context.trace_id, parsed.trace_id)
        self.assertEqual(context.span_id, parsed.span_id)
        self.assertEqual("01", parsed.trace_flags)

    def test_invalid_traceparent_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_traceparent("not-a-traceparent")
        with self.assertRaises(ValueError):
            parse_traceparent("ff-" + "a" * 32 + "-" + "b" * 16 + "-01")

    def test_inject_extract_http_headers(self) -> None:
        context = new_trace_context()
        headers: dict[str, str] = {}
        inject_trace_context(context, headers)
        self.assertIn(TRACEPARENT_HEADER, headers)
        extracted = extract_trace_context(headers)
        assert extracted is not None
        self.assertEqual(context.trace_id, extracted.trace_id)
        self.assertEqual(context.span_id, extracted.span_id)

    def test_inject_extract_mcp_metadata(self) -> None:
        context = new_trace_context()
        metadata: dict[str, str] = {}
        inject_trace_context(context, metadata=metadata)
        extracted = extract_trace_context(None, metadata=metadata)
        assert extracted is not None
        self.assertEqual(context.trace_id, extracted.trace_id)

    def test_http_server_propagates_trace_id(self) -> None:
        """Fake HTTP service echoes the inbound trace id back to the client."""

        received: dict[str, str] = {}
        ready = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                # Header lookup is case-insensitive on BaseHTTPRequestHandler.
                received["traceparent"] = self.headers.get(TRACEPARENT_HEADER, "")
                received["trace_id"] = self.headers.get(TRACE_ID_HEADER, "")
                body = received["trace_id"].encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True

        def _serve() -> None:
            ready.set()
            server.serve_forever(poll_interval=0.05)

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(2.0))
        try:
            exporter = InMemoryTraceExporter()
            tracer = TraceRecorder(exporter)
            with tracer.start_span("client.call") as span:
                headers = tracer.inject_headers({})
                # urllib capitalizes header names; inject uses lowercase keys.
                request = Request(
                    f"http://127.0.0.1:{server.server_address[1]}/echo",
                    data=b"{}",
                    headers={
                        "Content-Type": "application/json",
                        **headers,
                    },
                    method="POST",
                )
                # Bypass ambient HTTP proxies so the local fake server is hit.
                opener = build_opener(ProxyHandler({}))
                with opener.open(request, timeout=5) as response:
                    echoed = response.read().decode("utf-8")
            self.assertEqual(span.trace_id, echoed)
            self.assertEqual(span.trace_id, received["trace_id"])
            self.assertIn(span.trace_id, received["traceparent"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_grpc_style_metadata_propagation(self) -> None:
        """Simulate gRPC metadata carrier without a live gRPC stack."""

        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)
        with tracer.start_span("rpc.client") as parent:
            metadata: dict[str, str] = {}
            tracer.inject_headers(metadata=metadata)
            remote = extract_trace_context(None, metadata=metadata)
            assert remote is not None
            with tracer.start_span(
                "rpc.server",
                remote_parent=remote,
            ) as child:
                # Nested under the same process parent stack takes precedence for
                # parent_span_id, but remote_parent is used when there is no local
                # parent. Detach by using a fresh recorder for the server side.
                del child
        server_exporter = InMemoryTraceExporter()
        server_tracer = TraceRecorder(server_exporter)
        remote = extract_trace_context(None, metadata=metadata)
        assert remote is not None
        with server_tracer.start_span("rpc.server", remote_parent=remote) as server_span:
            self.assertEqual(parent.trace_id, server_span.trace_id)
            self.assertEqual(remote.span_id, server_span.parent_span_id)


class FacadeHierarchyTests(unittest.TestCase):
    def test_nested_instrumentation_shares_trace_id(self) -> None:
        exporter = InMemoryTraceExporter()
        otlp = InMemoryOtlpTraceExporter()
        tracer = TraceRecorder(CompositeTraceExporter([exporter, otlp]))

        with graph_span(tracer, graph_id="main_graph", session_id="s1") as root:
            with node_span(
                tracer,
                graph_id="main_graph",
                node_id="retrieve_project_knowledge",
                node_kind="rag",
            ):
                with rag_rewrite_span(tracer, query="路径规划规则"):
                    pass
                with rag_retrieve_span(
                    tracer,
                    top_k=5,
                    document_ids=["doc-1"],
                    current_version_hit=True,
                    acl_violation_count=0,
                ):
                    pass
                with rag_rerank_span(tracer, reranker_version="rr-1", top_k=5):
                    pass
            with node_span(
                tracer,
                graph_id="main_graph",
                node_id="report_generator",
                node_kind="model",
            ):
                with model_generate_span(
                    tracer,
                    model_name="qwen-mock",
                    model_version="0",
                    structured_output_valid=True,
                ):
                    pass
            with node_span(
                tracer,
                graph_id="main_graph",
                node_id="call_tool",
                node_kind="mcp",
            ):
                with mcp_tool_span(
                    tracer,
                    tool_name="generate_candidate_paths",
                    call_id="call-1",
                    attempt=1,
                ):
                    pass
            with verifier_span(tracer, result="PASS"):
                pass
            with checkpoint_span(
                tracer,
                checkpoint_id="cp-1",
                thread_id="thread-1",
                resumed=False,
            ):
                pass

        names = {span.name for span in exporter.spans()}
        expected = {
            SPAN_AGENT_GRAPH,
            SPAN_AGENT_NODE,
            SPAN_RAG_REWRITE,
            SPAN_RAG_RETRIEVE,
            SPAN_RAG_RERANK,
            SPAN_MODEL_GENERATE,
            SPAN_MCP_TOOL,
            SPAN_VERIFIER,
            SPAN_CHECKPOINT,
        }
        self.assertTrue(expected.issubset(names))
        self.assertTrue(all(span.trace_id == root.trace_id for span in exporter.spans()))
        self.assertEqual(len(exporter.spans()), len(otlp.spans()))
        # Sensitive full query never appears; only the hash attribute does.
        rewrite = next(s for s in exporter.spans() if s.name == SPAN_RAG_REWRITE)
        self.assertIn("rag.rewritten_query_hash", rewrite.attributes)
        self.assertNotIn("路径规划规则", str(rewrite.attributes))

    def test_error_span_exported_and_exception_reraise(self) -> None:
        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)
        with self.assertRaises(RuntimeError) as ctx:
            with tracer.start_span("agent.node", attributes={"agent.node_id": "boom"}):
                raise RuntimeError("node failed intentionally")
        self.assertEqual("node failed intentionally", str(ctx.exception))
        spans = exporter.spans()
        self.assertEqual(1, len(spans))
        self.assertEqual("ERROR", spans[0].status)
        self.assertEqual("RuntimeError", spans[0].error["type"])
        self.assertEqual("node failed intentionally", spans[0].error["message"])

    def test_otlp_unavailable_does_not_fail_request(self) -> None:
        otlp = InMemoryOtlpTraceExporter(enqueue_timeout_ms=50.0)
        otlp.block()
        tracer = TraceRecorder(otlp)
        with tracer.start_span("agent.graph"):
            pass
        self.assertEqual(1, tracer.export_failure_count)
        self.assertEqual(1, otlp.enqueue_failure_count)
        self.assertEqual([], otlp.spans())

    def test_recorder_redacts_attributes_before_export(self) -> None:
        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)
        with tracer.start_span(
            "agent.graph",
            attributes={
                "agent.graph_id": "main",
                "authorization": "Bearer abc",
                "mystery_field": "drop",
            },
        ):
            tracer.add_event(
                "debug",
                attributes={"patient_name": "Bob", "tool.name": "x"},
            )
        span = exporter.spans()[0]
        self.assertEqual("main", span.attributes["agent.graph_id"])
        self.assertEqual(REDACTED, span.attributes["authorization"])
        self.assertNotIn("mystery_field", span.attributes)
        self.assertEqual(REDACTED, span.events[0]["attributes"]["patient_name"])
        self.assertEqual("x", span.events[0]["attributes"]["tool.name"])


class ConcurrencyIsolationTests(unittest.TestCase):
    def test_twenty_concurrent_sessions_do_not_leak_context(self) -> None:
        exporter = InMemoryTraceExporter()
        lock = threading.Lock()
        results: list[tuple[str, str, str]] = []

        def worker(index: int) -> None:
            tracer = TraceRecorder(exporter)
            with tracer.start_span(
                "agent.graph",
                attributes={"agent.session_id": f"session-{index}"},
            ) as root:
                with tracer.start_span(
                    "agent.node",
                    attributes={"agent.node_id": f"node-{index}"},
                ) as child:
                    current = tracer.current_span()
                    assert current is not None
                    with lock:
                        results.append(
                            (root.trace_id, child.trace_id, child.parent_span_id or "")
                        )
                    self.assertEqual(root.trace_id, child.trace_id)
                    self.assertEqual(root.span_id, child.parent_span_id)
                    self.assertEqual(root.trace_id, current.trace_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(worker, range(20)))

        self.assertEqual(20, len(results))
        trace_ids = [item[0] for item in results]
        self.assertEqual(20, len(set(trace_ids)))
        for root_id, child_id, parent_id in results:
            self.assertEqual(root_id, child_id)
            self.assertTrue(parent_id)

        # Exported graph spans each have a unique session and unique trace.
        graph_spans = [s for s in exporter.spans() if s.name == "agent.graph"]
        self.assertEqual(20, len(graph_spans))
        self.assertEqual(20, len({s.trace_id for s in graph_spans}))
        sessions = {s.attributes.get("agent.session_id") for s in graph_spans}
        self.assertEqual(20, len(sessions))


class LinkedRetrySpanTests(unittest.TestCase):
    def test_tool_retry_spans_share_trace(self) -> None:
        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)
        with graph_span(tracer, graph_id="main") as root:
            with self.assertRaises(TimeoutError):
                with mcp_tool_span(
                    tracer,
                    tool_name="generate_candidate_paths",
                    call_id="c1",
                    attempt=1,
                ):
                    raise TimeoutError("one-time")
            with mcp_tool_span(
                tracer,
                tool_name="generate_candidate_paths",
                call_id="c1",
                attempt=2,
            ):
                pass
        tool_spans = [s for s in exporter.spans() if s.name == SPAN_MCP_TOOL]
        self.assertEqual(2, len(tool_spans))
        self.assertEqual("ERROR", tool_spans[0].status)
        self.assertEqual("OK", tool_spans[1].status)
        self.assertTrue(all(s.trace_id == root.trace_id for s in tool_spans))
        self.assertEqual(1, tool_spans[0].attributes["tool.attempt"])
        self.assertEqual(2, tool_spans[1].attributes["tool.attempt"])


if __name__ == "__main__":
    unittest.main()
