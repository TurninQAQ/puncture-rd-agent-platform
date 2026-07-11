from __future__ import annotations

import copy
import json
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
    BackendUnavailable,
    IndexIncompatible,
    RetrievalFilters,
)
from puncture_agent.rag.opensearch import OpenSearchSearchBackend  # noqa: E402
from puncture_agent.rag.provider_http import ProviderEndpoint  # noqa: E402
from tests.rag.provider_fakes import ScriptedTransport, json_response  # noqa: E402


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "embedding-revision-123"


def mapping_payload(*, dimension: int = 3, model: str = MODEL) -> dict:
    return {
        "project-knowledge-v000007": {
            "mappings": {
                "_meta": {
                    "contract": "puncture-rag-chunk-v1",
                    "generation": 7,
                    "embedding_model": model,
                    "embedding_revision": REVISION,
                    "embedding_dimension": dimension,
                    "parser_version": "markdown-normalizer-v1",
                    "chunker_version": "heading-parent-child-v1",
                    "document_count": 3,
                    "chunk_count": 8,
                    "query_instruction": "Retrieve internal engineering evidence.",
                    "document_instruction": "",
                    "vectors_normalized": True,
                    "tokenizer_revision": "tokenizer-revision-123",
                    "max_input_tokens": 8192,
                },
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                    }
                },
            }
        }
    }


def chunk_source(
    *,
    chunk_id: str = "chunk-1",
    scopes: list[str] | None = None,
    status: str = "active",
    version: str = "v3",
) -> dict:
    return {
        "doc_kind": "child",
        "chunk_id": chunk_id,
        "parent_id": "parent-1",
        "document_id": "planning-rules-v3",
        "title": "Engineering Planning Rules",
        "module": "path_planning",
        "version": version,
        "status": status,
        "section_path": ["Planning", "Safety Envelope"],
        "text": "ERR_PATH_COLLISION requires the complete path safety envelope check.",
        "token_count": 9,
        "chunk_index": 0,
        "access_scopes": scopes or ["algorithm_team"],
        "owner": "platform-team",
        "source_type": "markdown",
        "updated_at": "2026-07-10T00:00:00Z",
        "checksum_sha256": "a" * 64,
        "parser_version": "markdown-normalizer-v1",
        "chunker_version": "heading-parent-child-v1",
        "metadata": {"category": "hardware", "language": "en"},
    }


def parent_source() -> dict:
    source = chunk_source()
    return {
        "doc_kind": "parent",
        "parent_id": source["parent_id"],
        "document_id": source["document_id"],
        "title": source["title"],
        "module": source["module"],
        "version": source["version"],
        "status": source["status"],
        "section_path": source["section_path"],
        "text": "Full parent context for the safety envelope rule.",
        "access_scopes": source["access_scopes"],
        "owner": source["owner"],
        "source_type": source["source_type"],
        "updated_at": source["updated_at"],
        "metadata": source["metadata"],
    }


def search_payload(*sources: dict, scores: list[float] | None = None) -> dict:
    values = scores or [1.0] * len(sources)
    return {
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "hits": [
                {"_id": source.get("chunk_id", source.get("parent_id")), "_score": score, "_source": source}
                for source, score in zip(sources, values)
            ]
        },
    }


def filters() -> RetrievalFilters:
    return RetrievalFilters(
        access_scopes=("algorithm_team",),
        modules=("path_planning",),
        required_version="v3",
        allowed_statuses=("active",),
        metadata_filters={"category": "hardware"},
    )


class OpenSearchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = ProviderEndpoint(
            "https://search.internal:9200",
            auth_kind="basic",
            username="rag-reader",
            secret="search-secret",
            timeout_seconds=5.0,
        )

    def build(self, scripts) -> tuple[OpenSearchSearchBackend, ScriptedTransport]:
        transport = ScriptedTransport(list(scripts), endpoint=self.endpoint)
        backend = OpenSearchSearchBackend(
            self.endpoint,
            read_alias="project-knowledge-read",
            transport=transport,
        )
        return backend, transport

    def test_descriptor_and_health_validate_index_manifest(self) -> None:
        backend, transport = self.build(
            [
                json_response({"status": "yellow", "timed_out": False}),
                json_response(mapping_payload()),
                json_response(mapping_payload()),
            ]
        )
        health = backend.health()
        descriptor = backend.descriptor()

        self.assertEqual(health.status, "DEGRADED")
        self.assertEqual(health.backend, "opensearch-rest")
        self.assertEqual(health.document_count, 3)
        self.assertEqual(health.chunk_count, 8)
        self.assertEqual(health.details["concrete_index"], "project-knowledge-v000007")
        self.assertEqual(descriptor.query_instruction, "Retrieve internal engineering evidence.")
        self.assertEqual(descriptor.document_instruction, "")
        self.assertTrue(descriptor.vectors_normalized)
        self.assertEqual(descriptor.tokenizer_revision, "tokenizer-revision-123")
        self.assertEqual(descriptor.max_input_tokens, 8192)
        self.assertEqual(transport.requests[0].path, "/_cluster/health")
        self.assertEqual(transport.requests[1].path, "/project-knowledge-read/_mapping")

    def test_checked_in_deploy_mapping_and_adapter_query_contract_are_compatible(self) -> None:
        template = json.loads(
            (PROJECT_ROOT / "deploy" / "rag-search" / "config" / "index-template.json").read_text(
                encoding="utf-8"
            )
        )
        mappings = copy.deepcopy(template["template"]["mappings"])
        mappings["_meta"].update(
            {
                "contract": "puncture-rag-chunk-v1",
                "generation": 42,
                "embedding_model": MODEL,
                "embedding_revision": REVISION,
                "embedding_dimension": 3,
                "parser_version": "markdown-normalizer-v1",
                "chunker_version": "heading-parent-child-v1",
                "document_count": 1,
                "chunk_count": 1,
                "query_instruction": "Retrieve internal engineering evidence.",
                "document_instruction": "",
                "vectors_normalized": True,
                "tokenizer_revision": "tokenizer-revision-123",
                "max_input_tokens": 8192,
            }
        )
        mappings["properties"]["embedding"]["dimension"] = 3
        rendered_payload = {"project-knowledge-v000042": {"mappings": mappings}}
        source = chunk_source()
        backend, transport = self.build(
            [
                json_response(rendered_payload),
                json_response(search_payload(source)),
                json_response(search_payload(source)),
            ]
        )

        descriptor = backend.descriptor()
        backend.lexical_search("ERR_PATH_COLLISION", filters(), 5)
        backend.dense_search((1.0, 0.0, 0.0), filters(), 5)

        self.assertEqual(descriptor.generation, 42)
        self.assertEqual(descriptor.query_instruction, "Retrieve internal engineering evidence.")
        self.assertEqual(mappings["properties"]["metadata_terms"]["type"], "keyword")
        lexical_filters = transport.requests[1].json_body["query"]["bool"]["filter"]
        dense_filters = transport.requests[2].json_body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        for clauses in (lexical_filters, dense_filters):
            self.assertIn({"term": {"doc_kind": "child"}}, clauses)
            self.assertIn({"term": {"metadata_terms": "category=hardware"}}, clauses)

    def test_lexical_and_dense_requests_apply_identical_mandatory_filters(self) -> None:
        source = chunk_source()
        backend, transport = self.build(
            [
                json_response(search_payload(source, scores=[3.5])),
                json_response(mapping_payload()),
                json_response(search_payload(source, scores=[0.91])),
            ]
        )
        lexical = backend.lexical_search("ERR_PATH_COLLISION", filters(), 5)
        dense = backend.dense_search((1.0, 0.0, 0.0), filters(), 5)

        self.assertEqual(lexical[0].chunk.chunk_id, "chunk-1")
        self.assertEqual(dense[0].score, 0.91)
        lexical_filters = transport.requests[0].json_body["query"]["bool"]["filter"]
        dense_filters = transport.requests[2].json_body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        self.assertEqual(lexical_filters, dense_filters)
        self.assertIn({"terms": {"access_scopes": ["algorithm_team", "public"]}}, lexical_filters)
        self.assertIn({"terms": {"module": ["path_planning"]}}, lexical_filters)
        self.assertIn({"term": {"version": "v3"}}, lexical_filters)
        self.assertIn({"terms": {"status": ["active"]}}, lexical_filters)
        self.assertIn({"term": {"metadata_terms": "category=hardware"}}, lexical_filters)
        self.assertFalse(any("metadata.category" in str(clause) for clause in lexical_filters))
        self.assertNotIn("embedding", transport.requests[0].json_body["_source"])

    def test_backend_filter_mismatch_fails_closed_before_caller_can_rank_it(self) -> None:
        unauthorized = chunk_source(scopes=["secret_team"])
        backend, _ = self.build([json_response(search_payload(unauthorized))])

        with self.assertRaisesRegex(BackendProtocolError, "mandatory filters"):
            backend.lexical_search("perfect keyword match", filters(), 5)

    def test_explicit_status_filter_reuses_the_lifecycle_clause_without_raw_override(self) -> None:
        source = chunk_source()
        explicit = RetrievalFilters(
            access_scopes=("algorithm_team",),
            modules=("path_planning",),
            required_version="v3",
            allowed_statuses=("active",),
            metadata_filters={"status": "active", "category": "hardware"},
        )
        backend, transport = self.build([json_response(search_payload(source))])
        result = backend.lexical_search("query", explicit, 5)

        self.assertEqual(result[0].chunk.status, "active")
        clauses = transport.requests[0].json_body["query"]["bool"]["filter"]
        self.assertEqual(clauses.count({"terms": {"status": ["active"]}}), 1)
        self.assertFalse(any("metadata.status" in str(clause) for clause in clauses))

    def test_canonical_metadata_value_is_used_for_both_query_and_response_validation(self) -> None:
        source = chunk_source()
        canonical = RetrievalFilters(
            access_scopes=("algorithm_team",),
            modules=("path_planning",),
            required_version="v3",
            allowed_statuses=("active",),
            metadata_filters={"category": " hardware "},
        )
        backend, transport = self.build([json_response(search_payload(source))])
        result = backend.lexical_search("query", canonical, 5)

        self.assertEqual(result[0].chunk.metadata["category"], "hardware")
        clauses = transport.requests[0].json_body["query"]["bool"]["filter"]
        self.assertIn({"term": {"metadata_terms": "category=hardware"}}, clauses)

    def test_parent_lookup_is_a_filtered_search_not_an_unprotected_id_get(self) -> None:
        parent = parent_source()
        backend, transport = self.build([json_response(search_payload(parent))])
        result = backend.get_parent("parent-1", filters())

        self.assertEqual(result.parent_id, "parent-1")
        request = transport.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.path, "/project-knowledge-read/_search")
        clauses = request.json_body["query"]["bool"]["filter"]
        self.assertIn({"term": {"doc_kind": "parent"}}, clauses)
        self.assertIn({"term": {"parent_id": "parent-1"}}, clauses)
        self.assertIn({"terms": {"access_scopes": ["algorithm_team", "public"]}}, clauses)

    def test_partial_shard_failures_timeouts_and_duplicate_ids_are_rejected(self) -> None:
        source = chunk_source()
        malformed = (
            {
                "timed_out": True,
                "_shards": {"failed": 0},
                "hits": {"hits": []},
            },
            {
                "timed_out": False,
                "_shards": {"failed": 1},
                "hits": {"hits": []},
            },
            search_payload(source, source),
        )
        expected = (BackendTimeout, BackendProtocolError, BackendProtocolError)
        for payload, error in zip(malformed, expected):
            with self.subTest(payload=payload):
                backend, _ = self.build([json_response(payload)])
                with self.assertRaises(error):
                    backend.lexical_search("query", filters(), 5)

    def test_mapping_dimension_and_alias_fanout_fail_closed(self) -> None:
        wrong_mapping = mapping_payload(dimension=3)
        wrong_mapping["project-knowledge-v000007"]["mappings"]["properties"]["embedding"]["dimension"] = 4
        payloads = (
            wrong_mapping,
            {
                **mapping_payload(),
                "project-knowledge-v000008": mapping_payload()["project-knowledge-v000007"],
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                backend, _ = self.build([json_response(payload)])
                with self.assertRaises(IndexIncompatible):
                    backend.descriptor()

    def test_http_unavailability_is_normalized_without_exposing_response_body(self) -> None:
        backend, _ = self.build(
            [json_response({"error": {"reason": "confidential backend details"}}, status=503)]
        )
        with self.assertRaises(BackendUnavailable) as context:
            backend.lexical_search("query", filters(), 5)
        self.assertNotIn("confidential", str(context.exception))


if __name__ == "__main__":
    unittest.main()
