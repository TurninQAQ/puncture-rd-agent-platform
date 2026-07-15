"""Offline contract checks for the opt-in live OpenSearch RAG demo."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
DEMO_FILE = ROOT / "examples" / "live_opensearch_rag_demo.py"
specification = importlib.util.spec_from_file_location("live_opensearch_rag_demo_for_tests", DEMO_FILE)
if specification is None or specification.loader is None:
    raise RuntimeError(f"cannot load live OpenSearch RAG demo: {DEMO_FILE}")
demo = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = demo
specification.loader.exec_module(demo)

from puncture_agent.rag import DeterministicEmbeddingBackend  # noqa: E402


class LiveOpenSearchRagDemoTests(unittest.TestCase):
    def test_seed_documents_match_the_versioned_index_contract(self) -> None:
        embedding = DeterministicEmbeddingBackend(dimension=64)
        documents = demo.build_seed_documents(embedding)

        self.assertEqual(12, len(documents))
        children = [document for document in documents if document["doc_kind"] == "child"]
        parents = [document for document in documents if document["doc_kind"] == "parent"]
        self.assertEqual(6, len(children))
        self.assertEqual(6, len(parents))
        self.assertEqual(6, len({document["chunk_id"] for document in children}))
        self.assertEqual(
            {
                "data_validation",
                "path_planning",
                "planning_safety",
                "safety_evaluation",
                "segmentation",
                "yield_analysis",
            },
            {str(document["module"]) for document in children},
        )
        for document in children:
            self.assertEqual("active", document["status"])
            self.assertEqual(64, len(document["embedding"]))
            self.assertEqual(64, document["embedding_dimension"])
            self.assertEqual("deterministic-hash", document["embedding_model"])
            self.assertRegex(document["checksum_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(document["metadata_terms"])
        for document in parents:
            self.assertNotIn("embedding", document)
            self.assertNotIn("chunk_id", document)
            self.assertIn("parent_id", document)
        json.dumps(documents, allow_nan=False)

    def test_live_write_is_opt_in_before_files_or_network_are_touched(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                socket.socket,
                "connect",
                autospec=True,
                side_effect=AssertionError("live demo attempted a network connection"),
            ):
                with self.assertRaisesRegex(ValueError, "RUN_RAG_INTEGRATION=1"):
                    demo.run_demo()


if __name__ == "__main__":
    unittest.main()
