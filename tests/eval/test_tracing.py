from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import AgentState, GraphRuntime, build_mock_handlers  # noqa: E402
from puncture_agent.observability.tracing import (  # noqa: E402
    InMemoryTraceExporter,
    JsonLinesTraceExporter,
    TraceRecorder,
)


class TracingTests(unittest.TestCase):
    def test_runtime_emits_parented_graph_and_node_spans(self) -> None:
        exporter = InMemoryTraceExporter()
        tracer = TraceRecorder(exporter)
        runtime = GraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(),
            tracer=tracer,
        )
        runtime.run(AgentState(user_query="对 Case-101 做路径规划"))

        spans = exporter.spans()
        graph_spans = [span for span in spans if span.name == "agent.graph"]
        node_spans = [span for span in spans if span.name == "agent.node"]
        self.assertEqual(1, len(graph_spans))
        self.assertGreater(len(node_spans), 5)
        self.assertTrue(all(span.status == "OK" for span in spans))
        self.assertEqual(
            {graph_spans[0].trace_id}, {span.trace_id for span in node_spans}
        )
        known_span_ids = {span.span_id for span in spans}
        self.assertTrue(
            all(span.parent_span_id in known_span_ids for span in node_spans)
        )
        top_level_nodes = [
            span
            for span in node_spans
            if span.attributes["agent.node_id"] == "planning_safety_subgraph"
        ]
        self.assertEqual(graph_spans[0].span_id, top_level_nodes[0].parent_span_id)
        nested_nodes = [
            span
            for span in node_spans
            if span.attributes["agent.node_id"].startswith(
                "planning_safety_subgraph."
            )
        ]
        self.assertTrue(
            all(span.parent_span_id == top_level_nodes[0].span_id for span in nested_nodes)
        )

    def test_json_lines_exporter_writes_replayable_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            tracer = TraceRecorder(JsonLinesTraceExporter(path))
            with tracer.start_span("outer", attributes={"case_id": "Case-1"}):
                tracer.add_event("tool.called", attributes={"tool": "mock"})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(1, len(records))
            self.assertEqual("outer", records[0]["name"])
            self.assertEqual("OK", records[0]["status"])
            self.assertEqual("tool.called", records[0]["events"][0]["name"])

        class FailingExporter:
            def export(self, span):
                del span
                raise OSError("private trace sink detail")

        tracer = TraceRecorder(FailingExporter())
        runtime = GraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(),
            tracer=tracer,
        )
        state = runtime.run(
            AgentState(user_query="对 Case-102 做路径规划")
        )
        self.assertEqual("SUCCEEDED", state.status)
        self.assertGreater(tracer.export_failure_count, 0)


if __name__ == "__main__":
    unittest.main()
