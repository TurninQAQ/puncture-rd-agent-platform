#!/usr/bin/env python3
"""Run a deterministic, dependency-free enterprise hybrid-RAG demonstration."""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    EnterpriseRagClient,
    EnterpriseRagConfig,
    RagRuntimeConfig,
    RetrievalRequest,
    RetrievalResponse,
    SourceDocument,
)


def synthetic_documents() -> tuple[SourceDocument, ...]:
    """Return four synthetic industrial documents with intentionally different ACLs."""

    return (
        SourceDocument(
            document_id="eda-timing-signoff-v3",
            title="EDA Timing Signoff Rules",
            source_uri="internal://engineering/eda/timing-signoff-v3",
            source_type="markdown",
            module="eda_flow",
            version="v3",
            status="active",
            owner="eda-platform-team",
            access_scopes=("eda-team",),
            content=(
                "# Release Gate\n"
                "ERR_TIMING_0042 blocks tape-out when worst negative slack is below 0 ns. "
                "The approved signoff gate also requires zero unresolved DRC violations.\n\n"
                "# Recovery\n"
                "After a failed gate, restore the last approved constraint set, rerun static "
                "timing analysis, and attach the comparison report to change control."
            ),
            updated_at="2026-06-20T08:00:00Z",
            metadata={"category": "approved", "language": "en", "system": "eda"},
        ),
        SourceDocument(
            document_id="fab-vacuum-maintenance-v2",
            title="Wafer Equipment Vacuum Maintenance",
            source_uri="internal://operations/fab/vacuum-maintenance-v2",
            source_type="markdown",
            module="equipment_ops",
            version="v2",
            status="active",
            owner="fab-equipment-team",
            access_scopes=("fab-ops",),
            content=(
                "# Preventive Maintenance\n"
                "Inspect the dry pump vibration trend every 500 operating hours. "
                "Open a maintenance ticket when the normalized vibration exceeds the approved limit.\n\n"
                "# Return to Service\n"
                "Record the leak-check result, chamber base pressure, technician identity, and approval."
            ),
            updated_at="2026-06-18T09:30:00Z",
            metadata={"category": "approved", "language": "en", "system": "fab"},
        ),
        SourceDocument(
            document_id="release-change-control-v2",
            title="Engineering Release Change Control",
            source_uri="internal://engineering/governance/change-control-v2",
            source_type="markdown",
            module="release_management",
            version="v2",
            status="active",
            owner="engineering-quality-team",
            access_scopes=("eda-team", "fab-ops"),
            content=(
                "# Approval\n"
                "Every production release needs an owner, reviewer, verification evidence, and a "
                "documented rollback plan before approval.\n\n"
                "# Rollback\n"
                "A rollback restores the last approved artifact set. Record the trigger, affected "
                "systems, validation result, and final release decision in the audit trail."
            ),
            updated_at="2026-06-22T10:00:00Z",
            metadata={"category": "approved", "language": "en", "system": "governance"},
        ),
        SourceDocument(
            document_id="yield-excursion-restricted-v1",
            title="Restricted Yield Excursion Playbook",
            source_uri="internal://restricted/yield/excursion-playbook-v1",
            source_type="markdown",
            module="yield_analysis",
            version="v1",
            status="active",
            owner="yield-analysis-team",
            access_scopes=("yield-secret",),
            content=(
                "# Restricted Containment\n"
                "LOT-SECRET-ALPHA is the exact internal exercise identifier. "
                "The containment sequence and affected lot details are restricted to the yield team."
            ),
            updated_at="2026-06-25T11:00:00Z",
            metadata={"category": "restricted", "language": "en", "system": "yield"},
        ),
    )


def build_client() -> EnterpriseRagClient:
    config = EnterpriseRagConfig(
        endpoint="memory://local-enterprise-demo",
        index_name="local-enterprise-rag-demo",
        embedding_model="deterministic-enterprise-embedding",
        reranker_model="deterministic-enterprise-reranker",
        timeout_seconds=5.0,
        dense_top_k=8,
        lexical_top_k=8,
        rerank_top_k=6,
    )
    runtime = RagRuntimeConfig(
        minimum_relevance=0.12,
        context_budget_tokens=1200,
        recall_mode="hybrid",
        use_reranker=True,
        expand_parent_context=True,
    )
    return EnterpriseRagClient.offline(config, runtime=runtime, embedding_dimension=64)


def _response_view(response: RetrievalResponse) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "rewritten_query": response.rewritten_query,
        "retrieval_mode": response.retrieval_mode,
        "trace_id": response.trace_id,
        "warnings": list(response.warnings),
        "citations": [chunk.citation for chunk in response.chunks],
        "evidence": [
            {
                "rank": chunk.rank,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "module": chunk.module,
                "version": chunk.version,
                "score": chunk.score,
                "citation": chunk.citation,
                "branch_ranks": {
                    "lexical": chunk.metadata.get("lexical_rank"),
                    "dense": chunk.metadata.get("dense_rank"),
                    "rerank": chunk.metadata.get("rerank_score"),
                },
            }
            for chunk in response.chunks
        ],
    }


def run_demo() -> dict[str, Any]:
    client = build_client()
    ingestion_reports = [client.ingest(document) for document in synthetic_documents()]

    authorized = client.retrieve(
        RetrievalRequest(
            request_id="local-demo-authorized-001",
            query="ERR_TIMING_0042 timing signoff gate and rollback approval",
            modules=("eda_flow", "release_management"),
            access_scopes=("eda-team",),
            top_k=3,
            metadata_filters={"category": "approved", "language": "en"},
        )
    )
    acl_negative = client.retrieve(
        RetrievalRequest(
            request_id="local-demo-acl-negative-001",
            query="LOT-SECRET-ALPHA yield excursion containment sequence",
            modules=("yield_analysis",),
            access_scopes=("eda-team",),
            top_k=3,
        )
    )
    health = client.health()
    return {
        "demo": "offline-enterprise-hybrid-rag",
        "runtime": {
            "python": "3.10+",
            "external_services": False,
            "third_party_dependencies": False,
        },
        "index": {
            "status": health.status,
            "backend": health.backend,
            "document_count": health.document_count,
            "chunk_count": health.details.get("chunk_count"),
            "generation": health.details.get("generation"),
            "embedding_model": health.details.get("embedding_model"),
            "embedding_revision": health.details.get("embedding_revision"),
        },
        "ingestion": [
            {
                key: value
                for key, value in asdict(report).items()
                if key
                in {
                    "document_id",
                    "version",
                    "action",
                    "generation",
                    "parent_count",
                    "chunk_count",
                }
            }
            for report in ingestion_reports
        ],
        "queries": {
            "authorized_hybrid": _response_view(authorized),
            "acl_negative": _response_view(acl_negative),
        },
    }


def main() -> int:
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
