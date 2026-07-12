from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.observability.metrics import (  # noqa: E402
    acl_violation_rate,
    active_version_hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    tool_parameter_validity_rate,
    tool_selection_precision,
    tool_selection_recall,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_uses_unique_relevant_documents(self) -> None:
        self.assertEqual(0.5, recall_at_k(["a", "x", "a"], {"a", "b"}, 3))
        self.assertEqual(1.0, recall_at_k([], set(), 5))

    def test_reciprocal_rank_uses_first_relevant_result(self) -> None:
        self.assertEqual(0.5, reciprocal_rank(["x", "b", "a"], {"a", "b"}))
        self.assertEqual(0.0, reciprocal_rank(["x"], {"a"}))

    def test_ndcg_rewards_correct_order(self) -> None:
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        self.assertAlmostEqual(1.0, ndcg_at_k(["a", "b", "c"], relevance, 3))
        self.assertLess(ndcg_at_k(["c", "b", "a"], relevance, 3), 1.0)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k([], set(), 0)

    def test_duplicate_ids_and_no_hit_edges(self) -> None:
        self.assertEqual(1.0, recall_at_k(["a", "a", "a"], {"a"}, 3))
        self.assertEqual(0.0, reciprocal_rank(["x", "y"], {"a"}))
        self.assertEqual(1.0, ndcg_at_k([], {}, 3))

    def test_negative_relevance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {"a": -1.0}, 1)

    def test_mean_reciprocal_rank_and_precision(self) -> None:
        self.assertAlmostEqual(
            0.75,
            mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}]),
        )
        self.assertEqual(0.5, precision_at_k(["a", "x"], {"a", "b"}, 2))

    def test_active_version_and_acl_rates(self) -> None:
        rate = active_version_hit_rate(
            [
                {"required_version": "v2", "retrieved_versions": ["v1", "v2"]},
                {"required_version": "v3", "retrieved_versions": ["v1"]},
                {"current_version_hit": True},
            ]
        )
        self.assertAlmostEqual(2.0 / 3.0, rate)
        self.assertEqual(0.0, acl_violation_rate(0, 10))
        self.assertEqual(0.1, acl_violation_rate(1, 10))
        self.assertEqual(0.0, acl_violation_rate(0, 0))
        with self.assertRaises(ValueError):
            acl_violation_rate(2, 1)

    def test_tool_selection_and_parameter_predicates(self) -> None:
        self.assertEqual(1.0, tool_selection_recall(["a", "b"], ["a"]))
        self.assertEqual(0.5, tool_selection_precision(["a", "x"], ["a", "b"]))
        calls = [
            {
                "tool_name": "generate_candidate_paths",
                "request": {"max_needle_length_mm": 120.0, "safety_radius_mm": 2.0},
            }
        ]
        predicates = [
            {
                "tool_name": "generate_candidate_paths",
                "field": "max_needle_length_mm",
                "operator": "eq",
                "value": 120.0,
            },
            {
                "tool_name": "generate_candidate_paths",
                "field": "safety_radius_mm",
                "operator": "gt",
                "value": 0,
            },
            {
                "tool_name": "missing_tool",
                "field": "x",
                "operator": "exists",
                "required": False,
            },
        ]
        self.assertEqual(1.0, tool_parameter_validity_rate(calls, predicates))
        bad = tool_parameter_validity_rate(
            calls,
            [
                {
                    "tool_name": "generate_candidate_paths",
                    "field": "max_needle_length_mm",
                    "operator": "eq",
                    "value": 99.0,
                }
            ],
        )
        self.assertEqual(0.0, bad)

    def test_percentile_nearest_rank(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(1.0, percentile(values, 0))
        self.assertEqual(4.0, percentile(values, 100))
        self.assertEqual(4.0, percentile(values, 95))


if __name__ == "__main__":
    unittest.main()
