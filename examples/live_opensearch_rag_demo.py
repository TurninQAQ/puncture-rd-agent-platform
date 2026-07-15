#!/usr/bin/env python3
"""Seed synthetic chunks and run the production RAG client against live OpenSearch."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    DeterministicEmbeddingBackend,
    DeterministicReranker,
    EnterpriseRagClient,
    EnterpriseRagConfig,
    OpenSearchSearchBackend,
    ProviderEndpoint,
    RagDependencies,
    RagRuntimeConfig,
    RetrievalRequest,
)
from puncture_agent.rag.provider_http import (  # noqa: E402
    HttpxProviderTransport,
    ProviderProtocolError,
    decode_json_response,
)


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
_UPDATED_AT = "2026-07-15T00:00:00Z"


def _safe_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or ".." in value:
        raise ValueError(f"{label} is not a safe OpenSearch identifier")
    return value


def _required_regular_file(name: str) -> pathlib.Path:
    raw = os.environ.get(name)
    if not raw:
        raise ValueError(f"{name} must point to a local regular file")
    path = pathlib.Path(raw)
    if not path.is_file():
        raise ValueError(f"{name} must point to a local regular file")
    return path


def endpoint_from_environment() -> ProviderEndpoint:
    password_file = _required_regular_file("OPENSEARCH_PASSWORD_FILE")
    ca_file = _required_regular_file("OPENSEARCH_CA_FILE")
    password = password_file.read_text(encoding="utf-8").strip()
    return ProviderEndpoint(
        base_url=os.environ.get("OPENSEARCH_ENDPOINT", "https://127.0.0.1:9200"),
        auth_kind="basic",
        username=os.environ.get("OPENSEARCH_USERNAME", "admin"),
        secret=password,
        ca_bundle_path=str(ca_file),
        timeout_seconds=float(os.environ.get("RAG_REQUEST_TIMEOUT_SECONDS", "15")),
    )


def build_seed_documents(
    embedding: DeterministicEmbeddingBackend,
) -> tuple[dict[str, Any], ...]:
    records = (
        {
            "document_id": "live-demo-clearance-v1",
            "chunk_id": "live-demo-clearance-v1-chunk-0",
            "parent_id": "live-demo-clearance-v1-parent-0",
            "title": "Route Clearance Diagnostic",
            "module": "planning_safety",
            "access_scopes": ["team-a"],
            "section_path": ["Safety", "Clearance"],
            "text": (
                "ERR_PATH_CLEARANCE_0042 requires the route clearance check to pass "
                "before a candidate path can be approved."
            ),
            "owner": "safety-platform-team",
            "metadata": {"category": "approved", "language": "en"},
        },
        {
            "document_id": "live-demo-path-planning-v1",
            "chunk_id": "live-demo-path-planning-v1-chunk-0",
            "parent_id": "live-demo-path-planning-v1-parent-0",
            "title": "Synthetic Candidate Path Planning Rule",
            "module": "path_planning",
            "access_scopes": ["team-a"],
            "section_path": ["Planning", "Candidate Paths"],
            "text": (
                "Candidate needle paths must preserve configured clearance and length "
                "constraints before deterministic safety evaluation."
            ),
            "owner": "planning-platform-team",
            "metadata": {"category": "approved", "language": "en"},
        },
        {
            "document_id": "live-demo-safety-evaluation-v1",
            "chunk_id": "live-demo-safety-evaluation-v1-chunk-0",
            "parent_id": "live-demo-safety-evaluation-v1-parent-0",
            "title": "Synthetic Path Safety Evaluation Rule",
            "module": "safety_evaluation",
            "access_scopes": ["team-a"],
            "section_path": ["Safety", "Full Path"],
            "text": (
                "Full-path safety evaluation checks every candidate against the "
                "configured danger masks; the language model cannot mark a path safe."
            ),
            "owner": "safety-platform-team",
            "metadata": {"category": "approved", "language": "en"},
        },
        {
            "document_id": "live-demo-data-validation-v1",
            "chunk_id": "live-demo-data-validation-v1-chunk-0",
            "parent_id": "live-demo-data-validation-v1-parent-0",
            "title": "Synthetic MCS and NIfTI Validation Rule",
            "module": "data_validation",
            "access_scopes": ["team-a"],
            "section_path": ["Data", "Geometry"],
            "text": (
                "MCS and NIfTI inputs must preserve spacing, origin, direction, label "
                "schema, and geometry fingerprints before segmentation runs."
            ),
            "owner": "case-data-team",
            "metadata": {"category": "approved", "language": "en"},
        },
        {
            "document_id": "live-demo-segmentation-v1",
            "chunk_id": "live-demo-segmentation-v1-chunk-0",
            "parent_id": "live-demo-segmentation-v1-parent-0",
            "title": "Synthetic Segmentation Quality Rule",
            "module": "segmentation",
            "access_scopes": ["team-a"],
            "section_path": ["Segmentation", "Quality"],
            "text": (
                "Segmentation quality control verifies required labels, output geometry, "
                "and non-empty masks before publishing a versioned artifact."
            ),
            "owner": "segmentation-platform-team",
            "metadata": {"category": "approved", "language": "en"},
        },
        {
            "document_id": "live-demo-restricted-v1",
            "chunk_id": "live-demo-restricted-v1-chunk-0",
            "parent_id": "live-demo-restricted-v1-parent-0",
            "title": "Restricted Yield Exercise",
            "module": "yield_analysis",
            "access_scopes": ["yield-secret"],
            "section_path": ["Restricted", "Containment"],
            "text": (
                "LOT-SECRET-ALPHA is a synthetic restricted exercise identifier "
                "visible only to the yield team."
            ),
            "owner": "yield-analysis-team",
            "metadata": {"category": "restricted", "language": "en"},
        },
    )
    vectors = embedding.embed_documents([str(record["text"]) for record in records])
    documents: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        text = str(record["text"])
        common = {
            **record,
            "canonical_document_id": record["document_id"],
            "supersedes_document_id": "",
            "source_uri": f"internal://synthetic/{record['document_id']}",
            "source_type": "markdown",
            "version": "v1",
            "status": "active",
            "section": str(record["section_path"][-1]),
            "checksum_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parser_version": "1",
            "chunker_version": "1",
            "ingestion_run_id": "live-opensearch-demo-v1",
            "created_at": _UPDATED_AT,
            "updated_at": _UPDATED_AT,
            "indexed_at": _UPDATED_AT,
            "metadata_terms": [
                f"category={record['metadata']['category']}",
                f"language={record['metadata']['language']}",
            ],
        }
        documents.append(
            {
                key: value
                for key, value in {**common, "doc_kind": "parent"}.items()
                if key != "chunk_id"
            }
        )
        documents.append(
            {
                **common,
                "doc_kind": "child",
                "token_count": max(1, len(text.split())),
                "chunk_index": 0,
                "embedding": list(vector),
                "embedding_model": embedding.model_name,
                "embedding_revision": embedding.revision,
                "embedding_dimension": embedding.dimension,
            }
        )
    return tuple(documents)


def _request_json(
    transport: HttpxProviderTransport,
    method: str,
    path: str,
    *,
    payload: Any | None = None,
    allowed_statuses: tuple[int, ...] = (200,),
) -> dict[str, Any]:
    response = transport.request(method, path, json_body=payload)
    if response.status not in allowed_statuses:
        raise ProviderProtocolError(
            f"OpenSearch demo request returned unexpected status {response.status}"
        )
    return decode_json_response(response, max_bytes=transport.endpoint.max_response_bytes)


def seed_documents(
    endpoint: ProviderEndpoint,
    write_alias: str,
    documents: tuple[dict[str, Any], ...],
) -> None:
    alias = quote(_safe_identifier(write_alias, "write alias"), safe="")
    with HttpxProviderTransport(endpoint) as transport:
        for document in documents:
            identity_field = "chunk_id" if document["doc_kind"] == "child" else "parent_id"
            document_id = quote(
                _safe_identifier(str(document[identity_field]), f"{document['doc_kind']} ID"),
                safe="",
            )
            _request_json(
                transport,
                "PUT",
                f"/{alias}/_doc/{document_id}",
                payload=document,
                allowed_statuses=(200, 201),
            )
        _request_json(transport, "POST", f"/{alias}/_refresh")


def _response_view(response: Any) -> dict[str, Any]:
    return {
        "chunk_ids": [chunk.chunk_id for chunk in response.chunks],
        "citations": [chunk.citation for chunk in response.chunks],
        "retrieval_mode": response.retrieval_mode,
        "warnings": list(response.warnings),
    }


def run_demo() -> dict[str, Any]:
    if os.environ.get("RUN_RAG_INTEGRATION") != "1":
        raise ValueError("set RUN_RAG_INTEGRATION=1 to enable live OpenSearch writes")
    endpoint = endpoint_from_environment()
    read_alias = _safe_identifier(
        os.environ.get("RAG_READ_ALIAS", "puncture-rag-read"),
        "read alias",
    )
    write_alias = _safe_identifier(
        os.environ.get("RAG_WRITE_ALIAS", "puncture-rag-write"),
        "write alias",
    )
    search = OpenSearchSearchBackend(endpoint, read_alias=read_alias)
    try:
        descriptor = search.descriptor()
        embedding = DeterministicEmbeddingBackend(
            model_name=descriptor.embedding_model,
            revision=descriptor.embedding_revision,
            dimension=descriptor.embedding_dimension,
        )
        documents = build_seed_documents(embedding)
        seed_documents(endpoint, write_alias, documents)
        reranker = DeterministicReranker(model_name="deterministic-overlap", revision="1")
        config = EnterpriseRagConfig(
            endpoint=endpoint.base_url,
            index_name=read_alias,
            embedding_model=embedding.model_name,
            reranker_model=reranker.model_name,
            timeout_seconds=15,
            dense_top_k=5,
            lexical_top_k=5,
            rerank_top_k=5,
        )
        runtime = RagRuntimeConfig(
            minimum_relevance=0.05,
            recall_mode="hybrid",
            use_reranker=True,
            expand_parent_context=False,
        )
        client = EnterpriseRagClient(
            config,
            dependencies=RagDependencies(search, embedding, reranker),
            runtime=runtime,
        )
        health = client.health()
        authorized = client.retrieve(
            RetrievalRequest(
                request_id="live-opensearch-authorized-001",
                query="ERR_PATH_CLEARANCE_0042 route clearance check",
                modules=("planning_safety",),
                access_scopes=("team-a",),
                required_version="v1",
                top_k=2,
            )
        )
        denied = client.retrieve(
            RetrievalRequest(
                request_id="live-opensearch-denied-001",
                query="LOT-SECRET-ALPHA",
                modules=("yield_analysis",),
                access_scopes=("team-a",),
                required_version="v1",
                top_k=2,
            )
        )
        permitted = client.retrieve(
            RetrievalRequest(
                request_id="live-opensearch-permitted-001",
                query="LOT-SECRET-ALPHA",
                modules=("yield_analysis",),
                access_scopes=("yield-secret",),
                required_version="v1",
                top_k=2,
            )
        )
        restricted_id = "live-demo-restricted-v1-chunk-0"
        if "live-demo-clearance-v1-chunk-0" not in {
            chunk.chunk_id for chunk in authorized.chunks
        }:
            raise RuntimeError("authorized hybrid query did not return the seeded chunk")
        if restricted_id in {chunk.chunk_id for chunk in denied.chunks}:
            raise RuntimeError("ACL-negative query exposed the restricted chunk")
        if restricted_id not in {chunk.chunk_id for chunk in permitted.chunks}:
            raise RuntimeError("authorized restricted query did not return the seeded chunk")
        return {
            "demo": "live-opensearch-hybrid-rag",
            "health": {
                "status": health.status,
                "backend": health.backend,
                "concrete_index": descriptor.index_name,
            },
            "seeded_chunk_ids": [
                str(document["chunk_id"])
                for document in documents
                if document["doc_kind"] == "child"
            ],
            "queries": {
                "authorized_hybrid": _response_view(authorized),
                "acl_negative": _response_view(denied),
                "restricted_authorized": _response_view(permitted),
            },
            "security": {"restricted_visible_without_scope": False},
        }
    finally:
        search.close()


def main() -> int:
    try:
        print(json.dumps(run_demo(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"LIVE_OPENSEARCH_DEMO_FAILED {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
