"""RAG retrieval contracts and deterministic in-memory implementation."""

from .backends import (
    BackendHealth,
    BackendProtocolError,
    BackendTimeout,
    BackendUnavailable,
    DeterministicEmbeddingBackend,
    DeterministicReranker,
    EmbeddingUnavailable,
    InMemoryHybridIndex,
    IndexDescriptor,
    IndexIncompatible,
    IndexedChunk,
    ParentRecord,
    RerankResult,
    RerankerUnavailable,
    RetrievalFilters,
    SearchHit,
)
from .client import EnterpriseRagClient, RagDependencies, RagRuntimeConfig, RagService, RagServiceError
from .evaluation import (
    CaseEvaluation,
    EvaluationReport,
    GoldenQuery,
    evaluate_ablations,
    evaluate_service,
)
from .ingestion import (
    HeadingAwareChunker,
    IngestionReport,
    MarkdownSectionParser,
    RagIngestionService,
    SourceDocument,
)
from .mock_service import MockRagService
from .model_http import OpenAIEmbeddingBackend, VllmRerankerBackend
from .models import (
    EnterpriseRagConfig,
    KnowledgeDocument,
    RagHealth,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from .opensearch import OpenSearchSearchBackend
from .provider_http import ProviderEndpoint
from .retrieval import InMemoryTraceSink, normalize_query, reciprocal_rank_fusion

__all__ = [
    "EnterpriseRagClient",
    "EnterpriseRagConfig",
    "EvaluationReport",
    "CaseEvaluation",
    "GoldenQuery",
    "KnowledgeDocument",
    "BackendHealth",
    "BackendProtocolError",
    "BackendTimeout",
    "BackendUnavailable",
    "DeterministicEmbeddingBackend",
    "DeterministicReranker",
    "EmbeddingUnavailable",
    "HeadingAwareChunker",
    "InMemoryHybridIndex",
    "InMemoryTraceSink",
    "IndexDescriptor",
    "IndexIncompatible",
    "IndexedChunk",
    "IngestionReport",
    "MarkdownSectionParser",
    "MockRagService",
    "OpenAIEmbeddingBackend",
    "OpenSearchSearchBackend",
    "ProviderEndpoint",
    "RagHealth",
    "RagDependencies",
    "RagIngestionService",
    "RagRuntimeConfig",
    "RagService",
    "RagServiceError",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedChunk",
    "ParentRecord",
    "RerankResult",
    "RerankerUnavailable",
    "RetrievalFilters",
    "SearchHit",
    "SourceDocument",
    "VllmRerankerBackend",
    "normalize_query",
    "reciprocal_rank_fusion",
    "evaluate_ablations",
    "evaluate_service",
]
