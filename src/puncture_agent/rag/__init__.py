"""RAG retrieval contracts and deterministic in-memory implementation."""

from .client import EnterpriseRagClient, RagService, RagServiceError
from .mock_service import MockRagService
from .models import (
    EnterpriseRagConfig,
    KnowledgeDocument,
    RagHealth,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)

__all__ = [
    "EnterpriseRagClient",
    "EnterpriseRagConfig",
    "KnowledgeDocument",
    "MockRagService",
    "RagHealth",
    "RagService",
    "RagServiceError",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedChunk",
]
