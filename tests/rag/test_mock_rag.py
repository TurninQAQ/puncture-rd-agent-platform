"""Behavior, security-filter, and failure tests for MockRagService."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    EnterpriseRagClient,
    EnterpriseRagConfig,
    MockRagService,
    RagServiceError,
    RetrievalRequest,
)


class MockRagServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MockRagService.from_default_fixture()

    def test_health_and_fixture_loading(self) -> None:
        health = self.service.health()
        self.assertEqual(health.status, "UP")
        self.assertGreaterEqual(health.document_count, 5)

    def test_module_scope_and_version_filters_are_enforced_before_ranking(self) -> None:
        request = RetrievalRequest(
            request_id="rag-filter-1",
            query="needle length path planning constraints safety envelope",
            modules=("path_planning",),
            required_version="v3",
            access_scopes=("algorithm_team",),
            top_k=5,
        )
        response = self.service.retrieve(request)
        self.assertEqual(len(response.chunks), 1)
        self.assertEqual(response.chunks[0].document_id, "planning-rules-v3")
        self.assertEqual(response.chunks[0].version, "v3")

    def test_unauthorized_document_is_not_leaked(self) -> None:
        request = RetrievalRequest(
            request_id="rag-acl-1",
            query="needle length path planning constraints safety envelope",
            modules=("path_planning",),
            access_scopes=("public",),
        )
        response = self.service.retrieve(request)
        self.assertEqual(response.chunks, ())
        self.assertIn("NO_RELEVANT_KNOWLEDGE", response.warnings)

    def test_explicit_legacy_version_can_be_selected(self) -> None:
        request = RetrievalRequest(
            request_id="rag-version-1",
            query="deprecated path planning constraints",
            modules=("path_planning",),
            required_version="v2",
            access_scopes=("algorithm_team",),
        )
        response = self.service.retrieve(request)
        self.assertEqual(response.chunks[0].document_id, "planning-rules-v2-deprecated")

    def test_metadata_filter_and_top_k_produce_valid_ranks_and_citations(self) -> None:
        request = RetrievalRequest(
            request_id="rag-metadata-1",
            query="mask model interface planning safety",
            access_scopes=("algorithm_team",),
            metadata_filters={"status": "active"},
            top_k=2,
        )
        response = self.service.retrieve(request)
        self.assertLessEqual(len(response.chunks), 2)
        self.assertEqual([chunk.rank for chunk in response.chunks], list(range(1, len(response.chunks) + 1)))
        self.assertTrue(all(chunk.citation.startswith("[") for chunk in response.chunks))
        self.assertTrue(all(0.0 <= chunk.score <= 1.0 for chunk in response.chunks))

    def test_irrelevant_query_returns_empty_evidence_not_fabricated_content(self) -> None:
        request = RetrievalRequest(
            request_id="rag-empty-1",
            query="zzzxxyyqq completely unrelated token",
            access_scopes=("algorithm_team",),
        )
        response = self.service.retrieve(request)
        self.assertEqual(response.chunks, ())
        self.assertEqual(response.warnings, ("NO_RELEVANT_KNOWLEDGE",))

    def test_timeout_is_normalized_and_retryable(self) -> None:
        service = MockRagService.from_default_fixture(failure_mode="timeout")
        request = RetrievalRequest(request_id="rag-timeout-1", query="spacing")
        with self.assertRaises(RagServiceError) as context:
            service.retrieve(request)
        self.assertEqual(context.exception.code, "RAG_TIMEOUT")
        self.assertTrue(context.exception.retryable)

    def test_production_client_fails_fast_without_explicit_providers(self) -> None:
        with self.assertRaises(RagServiceError) as context:
            EnterpriseRagClient(
                EnterpriseRagConfig(
                    endpoint="http://127.0.0.1:9200",
                    index_name="project-knowledge",
                    embedding_model="embedding-model",
                    reranker_model="reranker-model",
                )
            )
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")
        self.assertFalse(context.exception.retryable)


if __name__ == "__main__":
    unittest.main()
