from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from contracts.artifacts import ArtifactRef  # noqa: E402
from contracts.enums import (  # noqa: E402
    ArtifactStatus,
    ArtifactType,
    CoordinateSystem,
)
from contracts.geometry import VolumeGeometry  # noqa: E402
from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    VerificationStatus,
    build_mock_handlers,
)
from puncture_agent.agent.langgraph_runtime import (  # noqa: E402
    LangGraphRuntime,
    langgraph_available,
)
from puncture_agent.agent.tool_bridge import McpToolExecutor  # noqa: E402
from puncture_agent.mcp import (  # noqa: E402
    InMemoryArtifactResolver,
    McpPrincipal,
    McpToolRuntime,
)
from puncture_agent.observability.tracing import (  # noqa: E402
    InMemoryTraceExporter,
    TraceRecorder,
)
from puncture_agent.tooling import build_mock_registry  # noqa: E402


CASE_ID = "Case-901"
EXPECTED_PLANNING_TOOLS = [
    "generate_candidate_paths",
    "evaluate_path_safety",
    "evaluate_intraoperative_risk",
    "verify_skin_penetration",
]
FIXED_NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(128, 128, 96),
        spacing_mm=(1.0, 1.0, 1.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        coordinate_system=CoordinateSystem.LPS,
    )


def _artifact(suffix: str, artifact_type: ArtifactType) -> ArtifactRef:
    artifact_id = f"artifact-{CASE_ID}-{suffix}"
    return ArtifactRef(
        artifact_id=artifact_id,
        case_id=CASE_ID,
        artifact_type=artifact_type,
        uri=f"mock://private/{artifact_id}",
        checksum_sha256="a" * 64,
        status=ArtifactStatus.AVAILABLE,
        geometry=_geometry(),
        producer_name="langgraph-mcp-trace-test",
        producer_version="1",
    )


def _planning_artifacts() -> tuple[ArtifactRef, ...]:
    return (
        _artifact("ct", ArtifactType.CT_VOLUME),
        _artifact("skin-surface", ArtifactType.SKIN_SURFACE_MASK),
        _artifact("target", ArtifactType.TARGET_MASK),
        _artifact("skin", ArtifactType.SEGMENTATION_MASK),
        _artifact("heart", ArtifactType.DANGER_MASK),
        _artifact("bone", ArtifactType.DANGER_MASK),
        _artifact("bronchus", ArtifactType.DANGER_MASK),
        _artifact("vessel", ArtifactType.DANGER_MASK),
        _artifact("lung", ArtifactType.SEGMENTATION_MASK),
    )


class CapturingLocalMcpCaller:
    def __init__(self, delegate: McpToolRuntime) -> None:
        self.delegate = delegate
        self.tool_names = delegate.tool_names
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> Any:
        self.calls.append((name, dict(arguments["context"])))
        return self.delegate.call_tool(name, arguments, principal=principal)


@unittest.skipUnless(
    langgraph_available(),
    "real LangGraph dependency is not installed",
)
class LangGraphMcpTraceIntegrationTests(unittest.TestCase):
    def test_planning_graph_propagates_one_trace_through_nodes_and_mcp(self) -> None:
        resolver = InMemoryArtifactResolver(_planning_artifacts())
        caller = CapturingLocalMcpCaller(
            McpToolRuntime(
                build_mock_registry(),
                resolver,
                server_name="planning-safety",
            )
        )
        executor = McpToolExecutor(
            caller,
            principal=McpPrincipal("agent-runtime", (CASE_ID,)),
            clock=lambda: FIXED_NOW,
        )
        exporter = InMemoryTraceExporter()
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(executor),
            tracer=TraceRecorder(exporter),
        )

        state = runtime.run(
            AgentState(
                user_query=f"对 {CASE_ID} 做路径规划和皮肤穿透安全评估",
                session_id="real-langgraph-local-mcp-trace",
            )
        )

        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(VerificationStatus.PASS, state.verification_status)
        self.assertIn("planning_safety_subgraph", state.visited_nodes)
        self.assertEqual(
            EXPECTED_PLANNING_TOOLS,
            [call["tool_name"] for call in state.tool_calls],
        )
        self.assertEqual(EXPECTED_PLANNING_TOOLS, [name for name, _ in caller.calls])

        trace_id = state.metadata.get("trace_id")
        self.assertIsInstance(trace_id, str)
        self.assertTrue(trace_id)

        spans = exporter.spans()
        graph_spans = [span for span in spans if span.name == "agent.graph"]
        node_spans = [span for span in spans if span.name == "agent.node"]
        self.assertEqual(1, len(graph_spans))
        self.assertTrue(node_spans)
        self.assertEqual({trace_id}, {span.trace_id for span in graph_spans})
        self.assertEqual({trace_id}, {span.trace_id for span in node_spans})
        self.assertEqual(
            {trace_id},
            {context["trace_id"] for _, context in caller.calls},
        )


if __name__ == "__main__":
    unittest.main()
