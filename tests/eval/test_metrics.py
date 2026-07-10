from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.observability.metrics import (  # noqa: E402
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
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


if __name__ == "__main__":
    unittest.main()
