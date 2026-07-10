"""Deterministic in-memory RAG service for graph and API development."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .client import RagServiceError
from .models import KnowledgeDocument, RagHealth, RetrievalRequest, RetrievalResponse, RetrievedChunk


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


class MockRagService:
    """Lexical test double; it is deliberately not a production RAG algorithm."""

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument],
        *,
        failure_mode: str | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._failure_mode = failure_mode

    @classmethod
    def from_default_fixture(cls, *, failure_mode: str | None = None) -> "MockRagService":
        project_root = Path(__file__).resolve().parents[3]
        return cls.from_json(project_root / "mocks" / "rag" / "documents.json", failure_mode=failure_mode)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        failure_mode: str | None = None,
    ) -> "MockRagService":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw_documents = json.load(handle)
        documents = [
            KnowledgeDocument(
                document_id=item["document_id"],
                title=item["title"],
                module=item["module"],
                version=item["version"],
                section=item["section"],
                text=item["text"],
                access_scopes=tuple(item.get("access_scopes", ["public"])),
                updated_at=item.get("updated_at", ""),
                metadata=item.get("metadata", {}),
            )
            for item in raw_documents
        ]
        return cls(documents, failure_mode=failure_mode)

    def health(self) -> RagHealth:
        if self._failure_mode == "down":
            return RagHealth(status="DOWN", backend="mock-memory", document_count=0)
        return RagHealth(
            status="UP",
            backend="mock-memory",
            document_count=len(self._documents),
            details={"retrieval_mode": "deterministic_lexical"},
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        if self._failure_mode == "timeout":
            raise RagServiceError("RAG_TIMEOUT", "forced mock retrieval timeout", retryable=True)
        if self._failure_mode == "backend_error":
            raise RagServiceError("RAG_BACKEND_ERROR", "forced mock backend error", retryable=True)

        rewritten_query = self._rewrite_query(request.query)
        scored: list[tuple[float, KnowledgeDocument]] = []
        for document in self._documents:
            if request.modules and document.module not in request.modules:
                continue
            if request.required_version and document.version != request.required_version:
                continue
            if not self._is_authorized(document, request.access_scopes):
                continue
            if not self._matches_metadata(document, request.metadata_filters):
                continue
            score = self._lexical_score(rewritten_query, document)
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        selected = scored[: request.top_k]
        chunks = tuple(
            RetrievedChunk(
                chunk_id=f"{document.document_id}#section",
                document_id=document.document_id,
                title=document.title,
                module=document.module,
                version=document.version,
                section=document.section,
                text=document.text,
                score=round(score, 6),
                rank=rank,
                citation=f"[{document.title} | {document.version} | {document.section}]",
                metadata={**document.metadata, "updated_at": document.updated_at},
            )
            for rank, (score, document) in enumerate(selected, start=1)
        )
        warnings = () if chunks else ("NO_RELEVANT_KNOWLEDGE",)
        return RetrievalResponse(
            request_id=request.request_id,
            rewritten_query=rewritten_query,
            chunks=chunks,
            retrieval_mode="mock_lexical",
            trace_id=f"mock-rag-{request.request_id}",
            latency_ms=1.0,
            warnings=warnings,
        )

    @staticmethod
    def _rewrite_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _is_authorized(document: KnowledgeDocument, caller_scopes: tuple[str, ...]) -> bool:
        document_scopes = set(document.access_scopes)
        return "public" in document_scopes or bool(document_scopes.intersection(caller_scopes))

    @staticmethod
    def _matches_metadata(document: KnowledgeDocument, filters: object) -> bool:
        if not isinstance(filters, Mapping):
            return False
        searchable = {
            "document_id": document.document_id,
            "title": document.title,
            "module": document.module,
            "version": document.version,
            "section": document.section,
            **document.metadata,
        }
        return all(searchable.get(key) == value for key, value in filters.items())

    @staticmethod
    def _lexical_score(query: str, document: KnowledgeDocument) -> float:
        query_tokens = set(_TOKEN_PATTERN.findall(query))
        haystack = f"{document.title} {document.module} {document.section} {document.text}".lower()
        document_tokens = set(_TOKEN_PATTERN.findall(haystack))
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(document_tokens)) / len(query_tokens)
        phrase_bonus = 0.2 if len(query) >= 2 and query in haystack else 0.0
        return min(1.0, overlap * 0.8 + phrase_bonus)
