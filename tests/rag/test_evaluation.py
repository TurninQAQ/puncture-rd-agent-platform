"""Golden-set metric and real offline ablation tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    DeterministicEmbeddingBackend,
    DeterministicReranker,
    EnterpriseRagClient,
    EnterpriseRagConfig,
    GoldenQuery,
    InMemoryHybridIndex,
    RagDependencies,
    RagIngestionService,
    RagRuntimeConfig,
    RagServiceError,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    SourceDocument,
    evaluate_ablations,
    evaluate_service,
)


def chunk(document_id: str, rank: int, *, version: str = "v1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document_id}#chunk",
        document_id=document_id,
        title=document_id.upper(),
        module="test",
        version=version,
        section="Section",
        text=f"Evidence for {document_id}",
        score=max(0.1, 1.0 - rank * 0.1),
        rank=rank,
        citation=f"[{document_id.upper()} | {version} | Section]",
    )


class ScriptedService:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def health(self):
        raise NotImplementedError

    def retrieve(self, request):
        outcome = self.outcomes[request.request_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class EvaluationTests(unittest.TestCase):
    def request(self, request_id: str, query: str = "query") -> RetrievalRequest:
        return RetrievalRequest(request_id=request_id, query=query, access_scopes=("public",))

    def response(self, request_id: str, chunks=()) -> RetrievalResponse:
        return RetrievalResponse(
            request_id=request_id,
            rewritten_query="query",
            chunks=tuple(chunks),
            retrieval_mode="scripted",
            trace_id=f"trace-{request_id}",
            latency_ms=0.1,
            warnings=() if chunks else ("NO_RELEVANT_KNOWLEDGE",),
        )

    def test_metrics_are_calculated_from_scripted_runs(self) -> None:
        cases = (
            GoldenQuery(
                name="two relevant",
                request=self.request("case-1"),
                relevant_document_ids=("a", "b"),
                expected_version="v2",
                forbidden_document_ids=("x",),
            ),
            GoldenQuery(
                name="missed answer",
                request=self.request("case-2"),
                relevant_document_ids=("c",),
            ),
            GoldenQuery(
                name="correct no answer",
                request=self.request("case-3"),
                expect_no_answer=True,
            ),
            GoldenQuery(
                name="error is not no answer",
                request=self.request("case-4"),
                expect_no_answer=True,
            ),
        )
        service = ScriptedService(
            {
                "case-1": self.response(
                    "case-1",
                    (chunk("b", 1, version="v2"), chunk("x", 2), chunk("a", 3)),
                ),
                "case-2": self.response("case-2"),
                "case-3": self.response("case-3"),
                "case-4": RagServiceError("RAG_TIMEOUT", "timeout", retryable=True),
            }
        )
        report = evaluate_service(service, cases, profile="scripted")

        expected_case_one_ndcg = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
        self.assertEqual(report.query_count, 4)
        self.assertAlmostEqual(report.recall_at_5, 0.5)
        self.assertAlmostEqual(report.recall_at_10, 0.5)
        self.assertAlmostEqual(report.mrr, 0.5)
        self.assertAlmostEqual(report.ndcg_at_10, expected_case_one_ndcg / 2.0)
        self.assertEqual(report.correct_version_hit_rate, 1.0)
        self.assertEqual(report.version_case_count, 1)
        self.assertEqual(report.acl_leak_count, 1)
        self.assertEqual(report.no_answer_accuracy, 0.5)
        self.assertEqual(report.error_count, 1)
        self.assertGreaterEqual(report.p50_latency_ms, 0.0)
        self.assertGreaterEqual(report.p95_latency_ms, report.p50_latency_ms)
        self.assertFalse(report.cases[-1].no_answer_correct)
        self.assertEqual(report.cases[-1].error_code, "RAG_TIMEOUT")

    def test_real_offline_ablation_profiles_run_on_identical_index(self) -> None:
        cfg = EnterpriseRagConfig(
            endpoint="memory://",
            index_name="evaluation-index",
            embedding_model="evaluation-embedding",
            reranker_model="evaluation-reranker",
        )
        embedding = DeterministicEmbeddingBackend("evaluation-embedding", "rev-1", 64)
        reranker = DeterministicReranker("evaluation-reranker", "rev-1")
        backend = InMemoryHybridIndex(cfg.index_name)
        ingestion = RagIngestionService(backend, embedding)
        for document_id, text in (
            ("error-contract", "# API\nERR_MODEL_17 is returned by create_engine_v2()."),
            ("safety-distance", "# Safety\nThe path maintains distance from each danger structure."),
        ):
            ingestion.ingest(
                SourceDocument(
                    document_id=document_id,
                    title=document_id,
                    source_uri=f"internal://{document_id}",
                    source_type="markdown",
                    module="engineering",
                    version="v1",
                    status="active",
                    owner="platform-team",
                    access_scopes=("platform-team",),
                    content=text,
                    updated_at="2026-07-01T00:00:00Z",
                )
            )

        def client(runtime):
            return EnterpriseRagClient(
                cfg,
                dependencies=RagDependencies(backend, embedding, reranker),
                runtime=runtime,
            )

        services = {
            "bm25_only": client(
                RagRuntimeConfig(recall_mode="lexical", use_reranker=False, expand_parent_context=False)
            ),
            "dense_only": client(
                RagRuntimeConfig(recall_mode="dense", use_reranker=False, expand_parent_context=False)
            ),
            "hybrid_rrf": client(
                RagRuntimeConfig(recall_mode="hybrid", use_reranker=False, expand_parent_context=False)
            ),
            "hybrid_reranker": client(
                RagRuntimeConfig(recall_mode="hybrid", use_reranker=True, expand_parent_context=False)
            ),
            "hybrid_reranker_parent": client(RagRuntimeConfig()),
        }
        cases = (
            GoldenQuery(
                name="exact identifier",
                request=RetrievalRequest(
                    request_id="eval-exact",
                    query="ERR_MODEL_17 create_engine_v2",
                    modules=("engineering",),
                    access_scopes=("platform-team",),
                ),
                relevant_document_ids=("error-contract",),
                expected_version="v1",
            ),
            GoldenQuery(
                name="semantic paraphrase",
                request=RetrievalRequest(
                    request_id="eval-semantic",
                    query="trajectory clearance hazard anatomy",
                    modules=("engineering",),
                    access_scopes=("platform-team",),
                ),
                relevant_document_ids=("safety-distance",),
            ),
            GoldenQuery(
                name="unanswerable",
                request=RetrievalRequest(
                    request_id="eval-empty",
                    query="zzzxxyyqq unrelated",
                    modules=("engineering",),
                    access_scopes=("platform-team",),
                ),
                expect_no_answer=True,
            ),
        )
        reports = evaluate_ablations(services, cases)
        self.assertEqual(set(reports), set(services))
        for profile, report in reports.items():
            with self.subTest(profile=profile):
                self.assertEqual(report.query_count, 3)
                self.assertEqual(report.acl_leak_count, 0)
                self.assertGreaterEqual(report.p95_latency_ms, report.p50_latency_ms)
                self.assertIn("profile", report.to_dict())


if __name__ == "__main__":
    unittest.main()
