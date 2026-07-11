"""Executable handoff tests for the deterministic local enterprise RAG demo."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEMO_FILE = PROJECT_ROOT / "examples" / "local_rag_demo.py"
TESTING_DOC = PROJECT_ROOT / "docs" / "testing-rag.md"

specification = importlib.util.spec_from_file_location("local_rag_demo_for_tests", DEMO_FILE)
if specification is None or specification.loader is None:
    raise RuntimeError(f"cannot load local RAG demo: {DEMO_FILE}")
demo = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = demo
specification.loader.exec_module(demo)


class LocalRagDemoTests(unittest.TestCase):
    def test_run_demo_is_deterministic_without_network(self) -> None:
        with mock.patch.object(
            socket.socket,
            "connect",
            autospec=True,
            side_effect=AssertionError("local RAG demo attempted a network connection"),
        ):
            first = demo.run_demo()
            second = demo.run_demo()
        self.assertEqual(first, second)
        self.assertEqual("offline-enterprise-hybrid-rag", first["demo"])
        self.assertEqual(4, first["index"]["document_count"])
        self.assertEqual(7, first["index"]["chunk_count"])
        self.assertEqual(4, first["index"]["generation"])
        self.assertEqual([1, 2, 3, 4], [item["generation"] for item in first["ingestion"]])
        self.assertTrue(all(item["action"] == "indexed" for item in first["ingestion"]))
        json.dumps(first, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def test_authorized_query_uses_hybrid_branches_and_returns_citations(self) -> None:
        result = demo.run_demo()["queries"]["authorized_hybrid"]
        self.assertEqual("hybrid_rrf_rerank_parent", result["retrieval_mode"])
        self.assertEqual([], result["warnings"])
        self.assertEqual(3, len(result["evidence"]))
        self.assertEqual([1, 2, 3], [item["rank"] for item in result["evidence"]])
        self.assertEqual(result["citations"], [item["citation"] for item in result["evidence"]])
        self.assertEqual(
            [
                "eda-timing-signoff-v3",
                "eda-timing-signoff-v3",
                "release-change-control-v2",
            ],
            [item["document_id"] for item in result["evidence"]],
        )
        for item in result["evidence"]:
            self.assertIsInstance(item["branch_ranks"]["lexical"], int)
            self.assertIsInstance(item["branch_ranks"]["dense"], int)
            self.assertGreater(item["branch_ranks"]["rerank"], 0.0)
            self.assertTrue(item["citation"].startswith("["))
            self.assertTrue(item["citation"].endswith("]"))

    def test_acl_negative_query_returns_no_restricted_evidence(self) -> None:
        result = demo.run_demo()["queries"]["acl_negative"]
        self.assertEqual("hybrid_rrf_rerank_parent", result["retrieval_mode"])
        self.assertEqual([], result["evidence"])
        self.assertEqual([], result["citations"])
        self.assertEqual(["NO_RELEVANT_KNOWLEDGE"], result["warnings"])
        observable_evidence = json.dumps(
            {"evidence": result["evidence"], "citations": result["citations"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertNotIn("yield-excursion-restricted-v1", observable_evidence)
        self.assertNotIn("Restricted Yield Excursion Playbook", observable_evidence)

    def test_script_stdout_is_byte_stable_across_python_processes(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
                "PYTHONHASHSEED": "random",
                "PYTHONUNBUFFERED": "1",
            }
        )
        outputs: list[str] = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(DEMO_FILE)],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("", result.stderr)
            json.loads(result.stdout)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_testing_document_lists_exact_commands_and_boundaries(self) -> None:
        text = TESTING_DOC.read_text(encoding="utf-8")
        for expected in (
            "Python `3.10.12`",
            "python3 examples/local_rag_demo.py",
            "python3 -m unittest tests.rag.test_local_demo -v",
            "python3 -m unittest discover -s tests/rag -p 'test_*.py' -v",
            "python3 run_tests.py",
            "test_run_demo_is_deterministic_without_network",
            "test_acl_negative_query_returns_no_restricted_evidence",
            "does **not** prove",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
