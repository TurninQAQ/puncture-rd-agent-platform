"""Hybrid retrieval correctness, authorization, and normalized-failure tests."""

from __future__ import annotations

import math
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    BackendProtocolError,
    BackendTimeout,
    DeterministicEmbeddingBackend,
    DeterministicReranker,
    EnterpriseRagClient,
    EnterpriseRagConfig,
    InMemoryHybridIndex,
    InMemoryTraceSink,
    IndexIncompatible,
    RagDependencies,
    RagIngestionService,
    RagRuntimeConfig,
    RagServiceError,
    RerankResult,
    RetrievalRequest,
    SearchHit,
    SourceDocument,
    reciprocal_rank_fusion,
)
from puncture_agent.rag.backends import IndexedChunk  # noqa: E402


def config(**overrides) -> EnterpriseRagConfig:
    values = {
        "endpoint": "memory://",
        "index_name": "enterprise-rag-test",
        "embedding_model": "embedding-test",
        "reranker_model": "reranker-test",
        "timeout_seconds": 0.5,
        "dense_top_k": 30,
        "lexical_top_k": 30,
        "rerank_top_k": 10,
    }
    values.update(overrides)
    return EnterpriseRagConfig(**values)


def document(document_id: str, text: str, **overrides) -> SourceDocument:
    values = {
        "document_id": document_id,
        "title": document_id.replace("-", " ").title(),
        "source_uri": f"internal://knowledge/{document_id}",
        "source_type": "markdown",
        "module": "path_planning",
        "version": "v3",
        "status": "active",
        "owner": "algorithm-team",
        "access_scopes": ("algorithm-team",),
        "content": f"# Rules\n{text}",
        "updated_at": "2026-07-01T00:00:00Z",
        "metadata": {"category": "approved", "language": "en"},
    }
    values.update(overrides)
    return SourceDocument(**values)


class RecordingBackend:
    def __init__(self, delegate):
        self.delegate = delegate
        self.lock = threading.Lock()
        self.lexical_filters = []
        self.dense_filters = []

    def health(self):
        return self.delegate.health()

    def descriptor(self):
        return self.delegate.descriptor()

    def lexical_search(self, query, filters, top_k):
        with self.lock:
            self.lexical_filters.append(filters)
        return self.delegate.lexical_search(query, filters, top_k)

    def dense_search(self, vector, filters, top_k):
        with self.lock:
            self.dense_filters.append(filters)
        return self.delegate.dense_search(vector, filters, top_k)

    def get_parent(self, parent_id, filters):
        return self.delegate.get_parent(parent_id, filters)


class ScriptedReranker:
    model_name = "reranker-test"
    revision = "scripted-v1"

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, chunks):
        return tuple(RerankResult(chunk.chunk_id, self.scores[chunk.document_id]) for chunk in chunks)


class TimeoutReranker:
    model_name = "reranker-test"
    revision = "timeout-v1"

    def rerank(self, query, chunks):
        raise BackendTimeout("scripted timeout")


class SlowReranker:
    model_name = "reranker-test"
    revision = "slow-v1"

    def rerank(self, query, chunks):
        time.sleep(0.12)
        return tuple(RerankResult(chunk.chunk_id, 0.9) for chunk in chunks)


class CloseTrackingBackend(RecordingBackend):
    def __init__(self, delegate):
        super().__init__(delegate)
        self.closed = False

    def close(self):
        self.closed = True


class CloseTrackingEmbedding(DeterministicEmbeddingBackend):
    def __init__(self):
        super().__init__("embedding-test", "rev-1", 64)
        self.closed = False

    def close(self):
        self.closed = True


class CloseTrackingReranker(DeterministicReranker):
    def __init__(self):
        super().__init__("reranker-test", "rev-1")
        self.closed = False

    def close(self):
        self.closed = True


class MalformedReranker:
    model_name = "reranker-test"
    revision = "bad-v1"

    def rerank(self, query, chunks):
        return (RerankResult("unknown-chunk", 2.0),)


class FailingQueryEmbedding(DeterministicEmbeddingBackend):
    def embed_query(self, text):
        raise RuntimeError("scripted embedding failure")


class IncompatibleQueryEmbedding(DeterministicEmbeddingBackend):
    def embed_query(self, text):
        raise IndexIncompatible("scripted model mismatch")


class ProtocolQueryEmbedding(DeterministicEmbeddingBackend):
    def embed_query(self, text):
        raise BackendProtocolError("scripted malformed vector")


class TimeoutSearchBackend(RecordingBackend):
    def lexical_search(self, query, filters, top_k):
        raise BackendTimeout("scripted search timeout")


class FilterViolatingBackend(RecordingBackend):
    def __init__(self, delegate, unauthorized_chunk):
        super().__init__(delegate)
        self.unauthorized_chunk = unauthorized_chunk

    def lexical_search(self, query, filters, top_k):
        return (SearchHit(self.unauthorized_chunk, 999.0),)

    def dense_search(self, vector, filters, top_k):
        return ()


class MissingParentBackend(RecordingBackend):
    def get_parent(self, parent_id, filters):
        return None


class ScriptedScoreBackend(RecordingBackend):
    def __init__(self, delegate, first, second):
        super().__init__(delegate)
        self.first = first
        self.second = second

    def lexical_search(self, query, filters, top_k):
        return (SearchHit(self.first, 0.3), SearchHit(self.second, 5.0))

    def dense_search(self, vector, filters, top_k):
        return ()


class FailingTraceSink:
    def record(self, trace):
        raise RuntimeError("trace backend unavailable")


class MalformedHealthBackend(RecordingBackend):
    def health(self):
        value = self.delegate.health()
        return type(
            "MalformedHealth",
            (),
            {
                "status": value.status,
                "backend": value.backend,
                "document_count": value.document_count,
                "chunk_count": value.chunk_count,
                "details": None,
            },
        )()


class EnterpriseRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = config()
        self.embedding = DeterministicEmbeddingBackend("embedding-test", "rev-1", 64)
        self.backend = InMemoryHybridIndex(self.config.index_name)
        self.ingestion = RagIngestionService(self.backend, self.embedding)
        self.documents = (
            document(
                "exact-error-code",
                "ERR_PATH_42 is emitted when validate_path_v3() detects less than 5 mm clearance.",
            ),
            document(
                "semantic-safety",
                "The path keeps a safe distance from every danger structure and rejects an intersection.",
            ),
            document(
                "legacy-rules",
                "Legacy path constraints for version two.",
                version="v2",
                status="deprecated",
                updated_at="2025-01-01T00:00:00Z",
            ),
            document(
                "secret-perfect-match",
                "ULTRA_SECRET_99 perfect keyword match.",
                access_scopes=("secret-team",),
                owner="secret-team",
            ),
        )
        for item in self.documents:
            self.ingestion.ingest(item)

    def make_client(self, *, backend=None, embedding=None, reranker=None, runtime=None, trace=None):
        return EnterpriseRagClient(
            self.config,
            dependencies=RagDependencies(
                backend or self.backend,
                embedding or self.embedding,
                reranker if reranker is not None else DeterministicReranker("reranker-test", "rev-1"),
                trace,
            ),
            runtime=runtime,
        )

    def request(self, query: str, **overrides) -> RetrievalRequest:
        values = {
            "request_id": f"request-{abs(hash(query))}",
            "query": query,
            "modules": ("path_planning",),
            "access_scopes": ("algorithm-team",),
            "top_k": 5,
        }
        values.update(overrides)
        return RetrievalRequest(**values)

    def test_constructor_fails_fast_without_explicit_dependencies(self) -> None:
        with self.assertRaises(RagServiceError) as context:
            EnterpriseRagClient(self.config)
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")

    def test_retrieve_rejects_wrong_request_type_with_normalized_error(self) -> None:
        with self.assertRaises(RagServiceError) as context:
            self.make_client().retrieve(object())
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")
        self.assertFalse(context.exception.retryable)

    def test_client_context_closes_each_provider(self) -> None:
        backend = CloseTrackingBackend(self.backend)
        embedding = CloseTrackingEmbedding()
        reranker = CloseTrackingReranker()
        client = self.make_client(backend=backend, embedding=embedding, reranker=reranker)
        with client:
            self.assertFalse(backend.closed)
        self.assertTrue(backend.closed)
        self.assertTrue(embedding.closed)
        self.assertTrue(reranker.closed)

    def test_exact_api_and_error_code_are_recalled_lexically(self) -> None:
        response = self.make_client().retrieve(self.request("ERR_PATH_42 validate_path_v3"))
        self.assertEqual(response.chunks[0].document_id, "exact-error-code")
        self.assertEqual(response.chunks[0].metadata["lexical_rank"], 1)
        self.assertIn("ERR_PATH_42", response.chunks[0].text)

    def test_semantic_paraphrase_is_recalled_by_dense_branch(self) -> None:
        response = self.make_client().retrieve(
            self.request("trajectory clearance hazard anatomy")
        )
        self.assertEqual(response.chunks[0].document_id, "semantic-safety")
        self.assertIsNone(response.chunks[0].metadata["lexical_rank"])
        self.assertEqual(response.chunks[0].metadata["dense_rank"], 1)

    def test_lexical_only_ablation_does_not_call_query_embedding(self) -> None:
        failing_embedding = FailingQueryEmbedding("embedding-test", "rev-1", 64)
        response = self.make_client(
            embedding=failing_embedding,
            runtime=RagRuntimeConfig(
                recall_mode="lexical",
                use_reranker=False,
                expand_parent_context=False,
            ),
        ).retrieve(self.request("ERR_PATH_42"))
        self.assertEqual(response.chunks[0].document_id, "exact-error-code")

    def test_both_recall_branches_receive_identical_mandatory_filters(self) -> None:
        recording = RecordingBackend(self.backend)
        response = self.make_client(backend=recording).retrieve(
            self.request("path clearance", required_version="v3", metadata_filters={"status": "active"})
        )
        self.assertTrue(response.chunks)
        self.assertEqual(len(recording.lexical_filters), 1)
        self.assertEqual(len(recording.dense_filters), 1)
        self.assertEqual(recording.lexical_filters[0], recording.dense_filters[0])
        self.assertEqual(recording.lexical_filters[0].required_version, "v3")
        self.assertEqual(recording.lexical_filters[0].access_scopes, ("algorithm-team",))

    def test_unauthorized_perfect_match_is_not_returned_or_named(self) -> None:
        response = self.make_client().retrieve(
            self.request("ULTRA_SECRET_99", access_scopes=("public",))
        )
        self.assertEqual(response.chunks, ())
        self.assertEqual(response.warnings, ("NO_RELEVANT_KNOWLEDGE",))
        self.assertNotIn("secret-perfect-match", repr(response).lower())
        self.assertNotIn("secret perfect match", repr(response).lower())

    def test_filter_violating_backend_fails_closed_before_rrf(self) -> None:
        prepared = self.ingestion.prepare(self.documents[-1])
        backend = FilterViolatingBackend(self.backend, prepared.chunks[0])
        with self.assertRaises(RagServiceError) as context:
            self.make_client(backend=backend).retrieve(
                self.request("ULTRA_SECRET_99", access_scopes=("public",))
            )
        self.assertEqual(context.exception.code, "RAG_PROTOCOL_ERROR")
        self.assertNotIn("secret-perfect-match", context.exception.details.values())

    def test_deprecated_requires_exact_version_and_active_is_default(self) -> None:
        default = self.make_client().retrieve(self.request("Legacy path constraints"))
        self.assertTrue(all(chunk.document_id != "legacy-rules" for chunk in default.chunks))
        legacy = self.make_client().retrieve(
            self.request("Legacy path constraints", required_version="v2")
        )
        self.assertEqual(legacy.chunks[0].document_id, "legacy-rules")
        with self.assertRaises(RagServiceError) as context:
            self.make_client().retrieve(
                self.request("Legacy", metadata_filters={"status": "deprecated"})
            )
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")

    def test_top_k_is_applied_after_reranking_with_contiguous_ranks(self) -> None:
        reranker = ScriptedReranker(
            {"exact-error-code": 0.7, "semantic-safety": 0.9, "legacy-rules": 0.8}
        )
        response = self.make_client(reranker=reranker).retrieve(
            self.request("path", top_k=1)
        )
        self.assertEqual(len(response.chunks), 1)
        self.assertEqual(response.chunks[0].document_id, "semantic-safety")
        self.assertEqual(response.chunks[0].rank, 1)
        self.assertEqual(response.chunks[0].score, 0.9)
        self.assertEqual(
            response.chunks[0].citation,
            "[Semantic Safety | v3 | Semantic Safety > Rules]",
        )

    def test_reranker_changes_rrf_order_and_preserves_calibrated_score(self) -> None:
        reranker = ScriptedReranker({"exact-error-code": 0.51, "semantic-safety": 0.83})
        response = self.make_client(reranker=reranker).retrieve(self.request("path clearance"))
        self.assertEqual(response.chunks[0].document_id, "semantic-safety")
        self.assertEqual(response.chunks[0].score, 0.83)
        self.assertNotEqual(response.chunks[0].metadata["pre_rerank_rank"], 1)

    def test_no_reranker_mode_sorts_by_exposed_final_score(self) -> None:
        first = self.ingestion.prepare(self.documents[0]).chunks[0]
        second = self.ingestion.prepare(self.documents[1]).chunks[0]
        backend = ScriptedScoreBackend(self.backend, first, second)
        response = self.make_client(
            backend=backend,
            runtime=RagRuntimeConfig(
                recall_mode="lexical",
                use_reranker=False,
                expand_parent_context=False,
            ),
        ).retrieve(self.request("path"))
        self.assertEqual(response.chunks[0].document_id, second.document_id)
        self.assertGreaterEqual(response.chunks[0].score, response.chunks[1].score)

    def test_citation_delimiters_are_escaped(self) -> None:
        hostile = document(
            "citation-hostile",
            "CITATION_ATTACK_TERM is an approved synthetic rule.",
            title="Rules | forged ] [ title",
        )
        self.ingestion.ingest(hostile)
        response = self.make_client().retrieve(self.request("CITATION_ATTACK_TERM"))
        citation = response.chunks[0].citation
        self.assertIn(r"\|", citation)
        self.assertIn(r"\]", citation)
        self.assertIn(r"\[", citation)
        self.assertEqual(citation.count(" | "), 2)
        self.assertNotIn("\n", citation)

    def test_parent_context_budget_is_enforced(self) -> None:
        long_document = document(
            "long-parent",
            " ".join(["TARGET_TERM"] + [f"word{index}" for index in range(200)]),
        )
        self.ingestion.ingest(long_document)
        runtime = RagRuntimeConfig(context_budget_tokens=12)
        response = self.make_client(runtime=runtime).retrieve(self.request("TARGET_TERM"))
        self.assertLessEqual(len(response.chunks[0].text.split()), 12)

    def test_irrelevant_query_returns_no_answer(self) -> None:
        response = self.make_client().retrieve(self.request("zzzxxyyqq nonexistent"))
        self.assertEqual(response.chunks, ())
        self.assertIn("NO_RELEVANT_KNOWLEDGE", response.warnings)

    def test_rrf_formula_deduplication_and_tie_breaking(self) -> None:
        prepared_a = self.ingestion.prepare(self.documents[0]).chunks[0]
        prepared_b = self.ingestion.prepare(self.documents[1]).chunks[0]
        fused = reciprocal_rank_fusion(
            (SearchHit(prepared_a, 10.0), SearchHit(prepared_b, 5.0)),
            (SearchHit(prepared_b, 0.9), SearchHit(prepared_a, 0.8)),
            k=60,
        )
        self.assertEqual(len(fused), 2)
        expected = 1 / 61 + 1 / 62
        self.assertAlmostEqual(fused[0].rrf_score, expected)
        self.assertAlmostEqual(fused[1].rrf_score, expected)
        self.assertEqual(fused[0].chunk.chunk_id, min(prepared_a.chunk_id, prepared_b.chunk_id))

    def test_backend_timeout_and_embedding_failure_are_normalized(self) -> None:
        with self.assertRaises(RagServiceError) as timeout:
            self.make_client(backend=TimeoutSearchBackend(self.backend)).retrieve(self.request("path"))
        self.assertEqual(timeout.exception.code, "RAG_TIMEOUT")
        self.assertTrue(timeout.exception.retryable)

        failing_embedding = FailingQueryEmbedding("embedding-test", "rev-1", 64)
        with self.assertRaises(RagServiceError) as embedding:
            self.make_client(embedding=failing_embedding).retrieve(self.request("path"))
        self.assertEqual(embedding.exception.code, "RAG_EMBEDDING_UNAVAILABLE")
        self.assertTrue(embedding.exception.retryable)

    def test_embedding_contract_failures_are_non_retryable(self) -> None:
        incompatible = IncompatibleQueryEmbedding("embedding-test", "rev-1", 64)
        with self.assertRaises(RagServiceError) as mismatch:
            self.make_client(embedding=incompatible).retrieve(self.request("path"))
        self.assertEqual(mismatch.exception.code, "RAG_INDEX_INCOMPATIBLE")
        self.assertFalse(mismatch.exception.retryable)

        malformed = ProtocolQueryEmbedding("embedding-test", "rev-1", 64)
        with self.assertRaises(RagServiceError) as protocol:
            self.make_client(embedding=malformed).retrieve(self.request("path"))
        self.assertEqual(protocol.exception.code, "RAG_PROTOCOL_ERROR")
        self.assertFalse(protocol.exception.retryable)

    def test_reranker_timeout_fails_by_default_or_explicitly_degrades(self) -> None:
        with self.assertRaises(RagServiceError) as timeout:
            self.make_client(reranker=TimeoutReranker()).retrieve(self.request("ERR_PATH_42"))
        self.assertEqual(timeout.exception.code, "RAG_TIMEOUT")

        degraded = self.make_client(
            reranker=TimeoutReranker(),
            runtime=RagRuntimeConfig(reranker_failure_mode="degrade"),
        ).retrieve(self.request("ERR_PATH_42"))
        self.assertIn("RERANKER_DEGRADED", degraded.warnings)
        self.assertLess(degraded.chunks[0].score, 1.0)

    def test_total_deadline_applies_to_a_slow_reranker(self) -> None:
        client = EnterpriseRagClient(
            config(timeout_seconds=0.03),
            dependencies=RagDependencies(self.backend, self.embedding, SlowReranker()),
        )
        started = time.perf_counter()
        with self.assertRaises(RagServiceError) as context:
            client.retrieve(self.request("path clearance"))
        self.assertEqual(context.exception.code, "RAG_TIMEOUT")
        self.assertTrue(context.exception.retryable)
        self.assertLess(time.perf_counter() - started, 0.1)

    def test_malformed_reranker_and_missing_parent_fail_protocol(self) -> None:
        with self.assertRaises(RagServiceError) as malformed:
            self.make_client(reranker=MalformedReranker()).retrieve(self.request("path"))
        self.assertEqual(malformed.exception.code, "RAG_PROTOCOL_ERROR")

        with self.assertRaises(RagServiceError) as missing_parent:
            self.make_client(backend=MissingParentBackend(self.backend)).retrieve(
                self.request("ERR_PATH_42")
            )
        self.assertEqual(missing_parent.exception.code, "RAG_PROTOCOL_ERROR")

    def test_incompatible_embedding_dimension_fails_before_search(self) -> None:
        recording = RecordingBackend(self.backend)
        incompatible = DeterministicEmbeddingBackend("embedding-test", "rev-1", 32)
        with self.assertRaises(RagServiceError) as context:
            self.make_client(backend=recording, embedding=incompatible).retrieve(self.request("path"))
        self.assertEqual(context.exception.code, "RAG_INDEX_INCOMPATIBLE")
        self.assertEqual(recording.lexical_filters, [])
        self.assertEqual(recording.dense_filters, [])

    def test_unapproved_filter_is_rejected_without_interpolation(self) -> None:
        with self.assertRaises(RagServiceError) as context:
            self.make_client().retrieve(
                self.request("path", metadata_filters={"raw_query": "* OR access_scopes:*"})
            )
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")
        self.assertEqual(context.exception.details, {"invalid_filter_count": 1})

    def test_trace_hashes_query_by_default_and_redacts_only_when_explicit(self) -> None:
        sink = InMemoryTraceSink()
        secret_query = "patient-secret-42 ERR_PATH_42"
        self.make_client(trace=sink).retrieve(self.request(secret_query))
        trace = sink.records[-1]
        self.assertIsInstance(trace["original_query"], dict)
        self.assertNotIn(secret_query, repr(trace))

        redacted_sink = InMemoryTraceSink()
        runtime = RagRuntimeConfig(
            trace_query_mode="redacted",
            trace_redaction_patterns=(r"patient-secret-\d+",),
        )
        self.make_client(trace=redacted_sink, runtime=runtime).retrieve(self.request(secret_query))
        redacted = redacted_sink.records[-1]
        self.assertIn("[REDACTED]", redacted["original_query"])
        self.assertNotIn("patient-secret-42", repr(redacted))

    def test_trace_sink_failure_never_masks_success_or_normalized_error(self) -> None:
        response = self.make_client(trace=FailingTraceSink()).retrieve(self.request("ERR_PATH_42"))
        self.assertTrue(response.chunks)
        with self.assertRaises(RagServiceError) as context:
            self.make_client(
                reranker=MalformedReranker(),
                trace=FailingTraceSink(),
            ).retrieve(self.request("path"))
        self.assertEqual(context.exception.code, "RAG_PROTOCOL_ERROR")

    def test_health_reports_index_and_provider_revisions(self) -> None:
        health = self.make_client().health()
        self.assertEqual(health.status, "UP")
        self.assertEqual(health.document_count, 4)
        self.assertEqual(health.details["embedding_revision"], "rev-1")
        self.assertEqual(health.details["reranker_revision"], "rev-1")

    def test_health_rejects_malformed_backend_details_without_raw_exception(self) -> None:
        health = self.make_client(backend=MalformedHealthBackend(self.backend)).health()
        self.assertEqual(health.status, "DOWN")
        self.assertEqual(health.details["error_code"], "RAG_PROTOCOL_ERROR")


if __name__ == "__main__":
    unittest.main()
