from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from puncture_agent.rag.backends import (  # noqa: E402
    BackendProtocolError,
    BackendTimeout,
    EmbeddingUnavailable,
    IndexIncompatible,
    IndexedChunk,
    RerankerUnavailable,
)
from puncture_agent.rag.model_http import (  # noqa: E402
    OpenAIEmbeddingBackend,
    VllmRerankerBackend,
)
from puncture_agent.rag.provider_http import (  # noqa: E402
    ProviderEndpoint,
    ProviderHttpResponse,
    ProviderTimeoutError,
)
from tests.rag.provider_fakes import ScriptedTransport, json_response  # noqa: E402


def make_chunk(chunk_id: str, text: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        document_id=f"doc-{chunk_id}",
        title="Industrial Troubleshooting Guide",
        module="diagnostics",
        version="v1",
        status="active",
        section_path=("Guide", "Errors"),
        text=text,
        token_count=4,
        chunk_index=0,
        access_scopes=("engineering",),
        owner="platform-team",
        source_type="markdown",
        updated_at="2026-07-10T00:00:00Z",
        checksum_sha256="a" * 64,
        parser_version="parser-v1",
        chunker_version="chunker-v1",
    )


class EmbeddingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = ProviderEndpoint(
            "https://embedding.internal",
            auth_kind="bearer",
            secret="embedding-secret",
            timeout_seconds=4.0,
        )

    def build(self, transport: ScriptedTransport, **overrides):
        options = {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "revision-123",
            "dimension": 3,
            "query_instruction": "Retrieve internal engineering evidence.",
            "tokenizer_revision": "tokenizer-123",
            "max_input_tokens": 8192,
            "transport": transport,
        }
        options.update(overrides)
        return OpenAIEmbeddingBackend(self.endpoint, **options)

    def test_query_instruction_request_indices_and_vector_normalization(self) -> None:
        transport = ScriptedTransport(
            [
                json_response(
                    {
                        "object": "list",
                        "model": "Qwen/Qwen3-Embedding-0.6B",
                        "data": [{"index": 0, "embedding": [3.0, 4.0, 0.0]}],
                    }
                )
            ],
            endpoint=self.endpoint,
        )
        backend = self.build(transport)
        vector = backend.embed_query("ERR_PLL_LOCK troubleshooting")

        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)
        request = transport.requests[0]
        self.assertEqual(request.path, "/v1/embeddings")
        self.assertEqual(request.timeout_seconds, 4.0)
        self.assertEqual(
            request.json_body["input"],
            [
                "Instruct: Retrieve internal engineering evidence.\n"
                "Query: ERR_PLL_LOCK troubleshooting"
            ],
        )
        self.assertEqual(backend.dimension, 3)
        self.assertTrue(backend.vectors_normalized)
        self.assertEqual(backend.tokenizer_revision, "tokenizer-123")

    def test_document_batch_is_returned_in_input_index_order(self) -> None:
        transport = ScriptedTransport(
            [
                json_response(
                    {
                        "model": "Qwen/Qwen3-Embedding-0.6B",
                        "data": [
                            {"index": 1, "embedding": [0.0, 2.0, 0.0]},
                            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                        ],
                    }
                )
            ],
            endpoint=self.endpoint,
        )
        backend = self.build(
            transport,
            document_instruction="Represent an approved internal document.",
        )
        vectors = backend.embed_documents(("first", "second"))

        self.assertEqual(vectors, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        self.assertEqual(
            transport.requests[0].json_body["input"],
            [
                "Instruct: Represent an approved internal document.\nDocument: first",
                "Instruct: Represent an approved internal document.\nDocument: second",
            ],
        )

    def test_model_dimension_and_index_mismatches_fail_closed(self) -> None:
        cases = (
            {
                "model": "wrong-model",
                "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}],
            },
            {
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
            },
            {
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ],
            },
        )
        expected = (IndexIncompatible, IndexIncompatible, BackendProtocolError)
        inputs = (("query",), ("query",), ("one", "two"))
        for payload, error, values in zip(cases, expected, inputs):
            with self.subTest(payload=payload):
                transport = ScriptedTransport([json_response(payload)], endpoint=self.endpoint)
                backend = self.build(transport)
                with self.assertRaises(error):
                    if len(values) == 1:
                        backend.embed_query(values[0])
                    else:
                        backend.embed_documents(values)

    def test_http_and_transport_failures_are_normalized(self) -> None:
        cases = (
            (ProviderTimeoutError("timeout"), BackendTimeout),
            (json_response({"error": "busy"}, status=503), EmbeddingUnavailable),
            (json_response({"error": "unauthorized"}, status=401), BackendProtocolError),
        )
        for scripted, expected in cases:
            with self.subTest(expected=expected):
                transport = ScriptedTransport([scripted], endpoint=self.endpoint)
                backend = self.build(transport)
                with self.assertRaises(expected):
                    backend.embed_query("query")

    def test_non_json_success_response_is_rejected(self) -> None:
        transport = ScriptedTransport(
            [ProviderHttpResponse(200, {"Content-Type": "text/html"}, b"<html></html>")],
            endpoint=self.endpoint,
        )
        with self.assertRaises(BackendProtocolError):
            self.build(transport).embed_query("query")


class RerankerAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = ProviderEndpoint(
            "https://reranker.internal",
            auth_kind="bearer",
            secret="reranker-secret",
            timeout_seconds=6.0,
        )

    def build(self, transport: ScriptedTransport, **overrides):
        options = {
            "model_name": "Qwen/Qwen3-Reranker-0.6B",
            "revision": "reranker-revision-1",
            "query_instruction": "Rank evidence for an engineering question.",
            "transport": transport,
        }
        options.update(overrides)
        return VllmRerankerBackend(self.endpoint, **options)

    def test_vllm_rerank_payload_and_provider_order_are_strictly_mapped(self) -> None:
        first = make_chunk("chunk-a", "PLL lock error procedure")
        second = make_chunk("chunk-b", "Unrelated packaging note")
        transport = ScriptedTransport(
            [
                json_response(
                    {
                        "model": "Qwen/Qwen3-Reranker-0.6B",
                        "results": [
                            {"index": 1, "relevance_score": 0.2},
                            {
                                "index": 0,
                                "relevance_score": 0.95,
                                "document": {"text": first.text},
                            },
                        ],
                    }
                )
            ],
            endpoint=self.endpoint,
        )
        results = self.build(transport).rerank("ERR_PLL_LOCK", (first, second))

        self.assertEqual([result.chunk_id for result in results], ["chunk-b", "chunk-a"])
        self.assertEqual([result.score for result in results], [0.2, 0.95])
        request = transport.requests[0]
        self.assertEqual(request.path, "/v1/rerank")
        self.assertEqual(request.json_body["documents"], [first.text, second.text])
        self.assertEqual(request.json_body["top_n"], 2)
        self.assertTrue(request.json_body["query"].startswith("Instruct: Rank evidence"))

    def test_reranker_requires_full_unique_coverage_and_normalized_scores(self) -> None:
        chunks = (make_chunk("a", "A"), make_chunk("b", "B"))
        payloads = (
            {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "results": [{"index": 0, "relevance_score": 0.8}],
            },
            {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.7},
                ],
            },
            {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "results": [
                    {"index": 0, "relevance_score": 1.2},
                    {"index": 1, "relevance_score": 0.1},
                ],
            },
            {
                "model": "wrong-model",
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.1},
                ],
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                transport = ScriptedTransport([json_response(payload)], endpoint=self.endpoint)
                with self.assertRaises(BackendProtocolError):
                    self.build(transport).rerank("query", chunks)

    def test_reranker_status_and_timeout_failures_are_normalized(self) -> None:
        chunk = make_chunk("a", "A")
        cases = (
            (ProviderTimeoutError("timeout"), BackendTimeout),
            (json_response({"error": "busy"}, status=429), RerankerUnavailable),
            (json_response({"error": "forbidden"}, status=403), BackendProtocolError),
        )
        for scripted, expected in cases:
            with self.subTest(expected=expected):
                transport = ScriptedTransport([scripted], endpoint=self.endpoint)
                with self.assertRaises(expected):
                    self.build(transport).rerank("query", (chunk,))


if __name__ == "__main__":
    unittest.main()
