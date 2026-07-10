from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent.graph_spec import (  # noqa: E402
    EdgeSpec,
    GraphSpecError,
    load_graph_spec,
    validate_graph_spec,
)
from puncture_agent.agent.runtime import evaluate_condition  # noqa: E402
from puncture_agent.agent.state import AgentState  # noqa: E402


class GraphSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph_root = PROJECT_ROOT / "graph"

    def test_all_checked_in_graphs_are_semantically_valid(self) -> None:
        for name in (
            "main_graph.json",
            "data_model_subgraph.json",
            "planning_safety_subgraph.json",
        ):
            with self.subTest(name=name):
                spec = load_graph_spec(self.graph_root / name)
                self.assertGreater(len(spec.nodes), 0)
                self.assertGreater(len(spec.edges), 0)

    def test_main_graph_locks_required_agent_boundaries(self) -> None:
        spec = load_graph_spec(self.graph_root / "main_graph.json")
        required = {
            "parse_request",
            "retrieve_project_knowledge",
            "resolve_case_context",
            "task_router",
            "data_model_subgraph",
            "planning_safety_subgraph",
            "result_verifier",
            "error_recovery",
            "request_missing_data",
            "report_generator",
        }
        self.assertEqual(required, set(spec.node_map))
        self.assertEqual("subgraph", spec.node_map["data_model_subgraph"].kind)
        self.assertEqual("subgraph", spec.node_map["planning_safety_subgraph"].kind)

    def test_unconditional_fallback_must_be_last(self) -> None:
        spec = load_graph_spec(self.graph_root / "planning_safety_subgraph.json")
        bad_edges = list(spec.edges)
        source_indexes = [
            index for index, edge in enumerate(bad_edges) if edge.source == "candidate_router"
        ]
        first, second = source_indexes
        bad_edges[first], bad_edges[second] = bad_edges[second], bad_edges[first]
        invalid = replace(spec, edges=tuple(bad_edges))
        with self.assertRaisesRegex(GraphSpecError, "fallback must be last"):
            validate_graph_spec(invalid, graph_root=self.graph_root)

    def test_unknown_edge_target_is_rejected(self) -> None:
        spec = load_graph_spec(self.graph_root / "data_model_subgraph.json")
        invalid = replace(
            spec,
            edges=spec.edges
            + (
                EdgeSpec(
                    source="finalize_data_model",
                    target="does_not_exist",
                    condition={"operator": "always"},
                ),
            ),
        )
        with self.assertRaisesRegex(GraphSpecError, "unknown target"):
            validate_graph_spec(invalid, graph_root=self.graph_root)

    def test_condition_dsl_supports_nested_and_field_comparison(self) -> None:
        state = AgentState(user_query="test", retry_count=1, max_retries=2)
        state.metadata["ready"] = True
        condition = {
            "all": [
                {"field": "metadata.ready", "operator": "eq", "value": True},
                {"field": "retry_count", "operator": "lte_field", "value": "max_retries"},
            ]
        }
        self.assertTrue(evaluate_condition(condition, state))
        state.retry_count = 3
        self.assertFalse(evaluate_condition(condition, state))


if __name__ == "__main__":
    unittest.main()
