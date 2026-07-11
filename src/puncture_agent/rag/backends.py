"""Provider-neutral RAG backend contracts and deterministic offline adapters.

The in-memory implementation is deliberately complete enough for local development,
contract tests, and deterministic evaluation.  Network search, embedding, and
reranker providers implement the same protocols; the Agent-facing contracts do not
change when a provider is replaced.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|[./:+\-])[A-Za-z0-9_]+)*"
    r"|\d+(?:\.\d+)?(?:mm|cm|ms|mb|gb)?|[\u4e00-\u9fff]",
    re.IGNORECASE,
)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize without losing identifiers such as ``ERR_42`` or ``foo::bar``."""

    normalized = unicodedata.normalize("NFC", text)
    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(normalized))


def estimate_tokens(text: str) -> int:
    """Stable, dependency-free token estimate used only for context budgeting."""

    return max(1, len(tokenize(text)))


class BackendTimeout(TimeoutError):
    """A provider exceeded its configured deadline."""


class BackendUnavailable(RuntimeError):
    """The search provider is temporarily unavailable."""


class EmbeddingUnavailable(RuntimeError):
    """The embedding provider is temporarily unavailable."""


class RerankerUnavailable(RuntimeError):
    """The reranker provider is temporarily unavailable."""


class BackendProtocolError(RuntimeError):
    """A provider returned a malformed or internally inconsistent response."""


class IndexIncompatible(RuntimeError):
    """The active index is incompatible with the configured embedding provider."""


class DocumentConflict(RuntimeError):
    """An immutable document/version identity was reused with changed content."""


@dataclass(frozen=True)
class RetrievalFilters:
    """Mandatory filters supplied independently to both retrieval branches."""

    access_scopes: tuple[str, ...]
    modules: tuple[str, ...] = ()
    required_version: str | None = None
    allowed_statuses: tuple[str, ...] = ("active",)
    metadata_filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "access_scopes", tuple(self.access_scopes))
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "allowed_statuses", tuple(self.allowed_statuses))
        object.__setattr__(self, "metadata_filters", MappingProxyType(dict(self.metadata_filters)))


@dataclass(frozen=True)
class ParentRecord:
    parent_id: str
    document_id: str
    title: str
    module: str
    version: str
    status: str
    section_path: tuple[str, ...]
    text: str
    access_scopes: tuple[str, ...]
    owner: str
    source_type: str
    updated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "access_scopes", tuple(self.access_scopes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    title: str
    module: str
    version: str
    status: str
    section_path: tuple[str, ...]
    text: str
    token_count: int
    chunk_index: int
    access_scopes: tuple[str, ...]
    owner: str
    source_type: str
    updated_at: str
    checksum_sha256: str
    parser_version: str
    chunker_version: str
    embedding: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "access_scopes", tuple(self.access_scopes))
        object.__setattr__(self, "embedding", tuple(float(item) for item in self.embedding))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SearchHit:
    chunk: IndexedChunk
    score: float


@dataclass(frozen=True)
class BackendHealth:
    status: str
    backend: str
    document_count: int
    chunk_count: int
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexDescriptor:
    index_name: str
    generation: int
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    parser_version: str
    chunker_version: str
    document_count: int
    chunk_count: int
    query_instruction: str = ""
    document_instruction: str = ""
    vectors_normalized: bool = True
    tokenizer_revision: str = "unspecified"
    max_input_tokens: int = 0


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    version: str
    checksum_sha256: str
    chunk_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class IndexMutation:
    action: str
    generation: int
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class RerankResult:
    chunk_id: str
    score: float


class EmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def query_instruction(self) -> str: ...

    @property
    def document_instruction(self) -> str: ...

    @property
    def vectors_normalized(self) -> bool: ...

    @property
    def tokenizer_revision(self) -> str: ...

    @property
    def max_input_tokens(self) -> int: ...

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class RerankerBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def rerank(self, query: str, chunks: Sequence[IndexedChunk]) -> Sequence[RerankResult]: ...


class SearchBackend(Protocol):
    def health(self) -> BackendHealth: ...

    def descriptor(self) -> IndexDescriptor: ...

    def lexical_search(
        self,
        query: str,
        filters: RetrievalFilters,
        top_k: int,
    ) -> Sequence[SearchHit]: ...

    def dense_search(
        self,
        vector: Sequence[float],
        filters: RetrievalFilters,
        top_k: int,
    ) -> Sequence[SearchHit]: ...

    def get_parent(self, parent_id: str, filters: RetrievalFilters) -> ParentRecord | None: ...


class WritableSearchBackend(SearchBackend, Protocol):
    def upsert_document(
        self,
        document: StoredDocument,
        parents: Sequence[ParentRecord],
        chunks: Sequence[IndexedChunk],
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str = "",
        document_instruction: str = "",
        vectors_normalized: bool = True,
        tokenizer_revision: str = "unspecified",
        max_input_tokens: int = 0,
        allow_update: bool = False,
        expected_checksum: str | None = None,
    ) -> IndexMutation: ...

    def delete_document(self, document_id: str, version: str | None = None) -> IndexMutation: ...

    def replace_all(
        self,
        documents: Sequence[StoredDocument],
        parents: Sequence[ParentRecord],
        chunks: Sequence[IndexedChunk],
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str = "",
        document_instruction: str = "",
        vectors_normalized: bool = True,
        tokenizer_revision: str = "unspecified",
        max_input_tokens: int = 0,
    ) -> IndexMutation: ...


def _canonical_semantic_token(token: str) -> str:
    aliases = {
        "trajectory": "path",
        "route": "path",
        "track": "path",
        "clearance": "distance",
        "margin": "distance",
        "separation": "distance",
        "hazard": "danger",
        "unsafe": "danger",
        "collision": "intersect",
        "overlap": "intersect",
        "organ": "structure",
        "organs": "structure",
        "anatomy": "structure",
        "nii": "nifti",
        "nii.gz": "nifti",
        "authorization": "acl",
        "permission": "acl",
    }
    return aliases.get(token, token)


class DeterministicEmbeddingBackend:
    """A stable feature-hashing encoder for offline tests and local development."""

    def __init__(self, model_name: str = "deterministic-hash", revision: str = "1", dimension: int = 128) -> None:
        if not model_name.strip() or not revision.strip():
            raise ValueError("embedding model name and revision must be non-empty")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 8:
            raise ValueError("embedding dimension must be an integer of at least 8")
        self._model_name = model_name
        self._revision = revision
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def query_instruction(self) -> str:
        return "Represent the enterprise knowledge query for retrieval."

    @property
    def document_instruction(self) -> str:
        return "Represent the enterprise knowledge passage for retrieval."

    @property
    def vectors_normalized(self) -> bool:
        return True

    @property
    def tokenizer_revision(self) -> str:
        return "rag-regex-tokenizer-v1"

    @property
    def max_input_tokens(self) -> int:
        return 8192

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._embed(text) for text in texts)

    def _embed(self, text: str) -> tuple[float, ...]:
        features = [_canonical_semantic_token(token) for token in tokenize(text)]
        vector = [0.0] * self._dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


class DeterministicReranker:
    """Dependency-free semantic-token overlap reranker with stable tie behavior."""

    def __init__(self, model_name: str = "deterministic-overlap", revision: str = "1") -> None:
        self._model_name = model_name
        self._revision = revision

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    def rerank(self, query: str, chunks: Sequence[IndexedChunk]) -> Sequence[RerankResult]:
        query_tokens = {_canonical_semantic_token(token) for token in tokenize(query)}
        results: list[RerankResult] = []
        for chunk in chunks:
            chunk_tokens = {_canonical_semantic_token(token) for token in tokenize(chunk.text)}
            title_tokens = {_canonical_semantic_token(token) for token in tokenize(chunk.title)}
            section_tokens = {
                _canonical_semantic_token(token)
                for token in tokenize(" ".join(chunk.section_path))
            }
            if not query_tokens:
                score = 0.0
            else:
                body_overlap = len(query_tokens.intersection(chunk_tokens)) / len(query_tokens)
                heading_overlap = len(query_tokens.intersection(title_tokens | section_tokens)) / len(query_tokens)
                score = min(1.0, body_overlap * 0.8 + heading_overlap * 0.2)
            results.append(RerankResult(chunk_id=chunk.chunk_id, score=score))
        return tuple(results)


@dataclass(frozen=True)
class _IndexState:
    descriptor: IndexDescriptor
    documents: Mapping[tuple[str, str], StoredDocument]
    parents: Mapping[str, ParentRecord]
    chunks: Mapping[str, IndexedChunk]


def _authorized(scopes: Sequence[str], effective_scopes: Sequence[str]) -> bool:
    document_scopes = set(scopes)
    return "public" in document_scopes or bool(document_scopes.intersection(effective_scopes))


def chunk_matches_filters(chunk: IndexedChunk, filters: RetrievalFilters) -> bool:
    if not _authorized(chunk.access_scopes, filters.access_scopes):
        return False
    if filters.modules and chunk.module not in filters.modules:
        return False
    if filters.required_version is not None and chunk.version != filters.required_version:
        return False
    if chunk.status not in filters.allowed_statuses:
        return False
    searchable: dict[str, Any] = {
        **dict(chunk.metadata),
        "document_id": chunk.document_id,
        "module": chunk.module,
        "version": chunk.version,
        "status": chunk.status,
        "source_type": chunk.source_type,
        "owner": chunk.owner,
    }
    return all(searchable.get(key) == value for key, value in filters.metadata_filters.items())


def parent_matches_filters(parent: ParentRecord, filters: RetrievalFilters) -> bool:
    if not _authorized(parent.access_scopes, filters.access_scopes):
        return False
    if filters.modules and parent.module not in filters.modules:
        return False
    if filters.required_version is not None and parent.version != filters.required_version:
        return False
    if parent.status not in filters.allowed_statuses:
        return False
    searchable: dict[str, Any] = {
        **dict(parent.metadata),
        "document_id": parent.document_id,
        "module": parent.module,
        "version": parent.version,
        "status": parent.status,
        "source_type": parent.source_type,
        "owner": parent.owner,
    }
    return all(searchable.get(key) == value for key, value in filters.metadata_filters.items())


class InMemoryHybridIndex:
    """Thread-safe versioned index with atomic alias-style state replacement."""

    def __init__(self, index_name: str) -> None:
        if not isinstance(index_name, str) or not index_name.strip():
            raise ValueError("index_name must be non-empty")
        self._lock = threading.RLock()
        self._state = _IndexState(
            descriptor=IndexDescriptor(
                index_name=index_name,
                generation=0,
                embedding_model="uninitialized",
                embedding_revision="uninitialized",
                embedding_dimension=0,
                parser_version="uninitialized",
                chunker_version="uninitialized",
                document_count=0,
                chunk_count=0,
            ),
            documents=MappingProxyType({}),
            parents=MappingProxyType({}),
            chunks=MappingProxyType({}),
        )
        self._history: dict[int, _IndexState] = {}

    def health(self) -> BackendHealth:
        with self._lock:
            state = self._state
        return BackendHealth(
            status="UP",
            backend="deterministic-memory-hybrid",
            document_count=state.descriptor.document_count,
            chunk_count=state.descriptor.chunk_count,
            details={
                "index_name": state.descriptor.index_name,
                "generation": state.descriptor.generation,
                "embedding_model": state.descriptor.embedding_model,
                "embedding_revision": state.descriptor.embedding_revision,
                "embedding_dimension": state.descriptor.embedding_dimension,
            },
        )

    def descriptor(self) -> IndexDescriptor:
        with self._lock:
            return self._state.descriptor

    def generation_history(self) -> tuple[IndexDescriptor, ...]:
        """Return immutable manifests retained for audit/rollback tooling."""

        with self._lock:
            states = [*self._history.values(), self._state]
        return tuple(state.descriptor for state in sorted(states, key=lambda item: item.descriptor.generation))

    def lexical_search(self, query: str, filters: RetrievalFilters, top_k: int) -> Sequence[SearchHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return ()
        with self._lock:
            chunks = tuple(
                chunk for chunk in self._state.chunks.values() if chunk_matches_filters(chunk, filters)
            )
        if not chunks:
            return ()
        tokenized = {
            chunk.chunk_id: tokenize(
                f"{chunk.title} {' '.join(chunk.section_path)} {chunk.text}"
            )
            for chunk in chunks
        }
        document_frequency: Counter[str] = Counter()
        for terms in tokenized.values():
            document_frequency.update(set(terms))
        average_length = sum(len(terms) for terms in tokenized.values()) / len(tokenized)
        query_counts = Counter(query_terms)
        scored: list[SearchHit] = []
        k1 = 1.5
        b = 0.75
        corpus_size = len(chunks)
        for chunk in chunks:
            terms = tokenized[chunk.chunk_id]
            term_counts = Counter(terms)
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                frequency_in_docs = document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (corpus_size - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
                )
                denominator = frequency + k1 * (1.0 - b + b * len(terms) / max(average_length, 1.0))
                score += inverse_frequency * frequency * (k1 + 1.0) / denominator * query_frequency
            if score > 0.0 and math.isfinite(score):
                scored.append(SearchHit(chunk=chunk, score=score))
        scored.sort(key=lambda hit: (-hit.score, _descending_date_key(hit.chunk.updated_at), hit.chunk.chunk_id))
        return tuple(scored[:top_k])

    def dense_search(
        self,
        vector: Sequence[float],
        filters: RetrievalFilters,
        top_k: int,
    ) -> Sequence[SearchHit]:
        query_vector = _validate_vector(vector, "query embedding")
        with self._lock:
            state = self._state
            if state.descriptor.embedding_dimension and len(query_vector) != state.descriptor.embedding_dimension:
                raise IndexIncompatible(
                    f"query dimension {len(query_vector)} does not match index dimension "
                    f"{state.descriptor.embedding_dimension}"
                )
            chunks = tuple(
                chunk for chunk in state.chunks.values() if chunk_matches_filters(chunk, filters)
            )
        scored: list[SearchHit] = []
        for chunk in chunks:
            embedding = _validate_vector(chunk.embedding, f"chunk {chunk.chunk_id} embedding")
            if len(embedding) != len(query_vector):
                raise IndexIncompatible("stored and query embedding dimensions differ")
            similarity = sum(left * right for left, right in zip(query_vector, embedding))
            if math.isfinite(similarity) and similarity > 0.05:
                scored.append(SearchHit(chunk=chunk, score=similarity))
        scored.sort(key=lambda hit: (-hit.score, _descending_date_key(hit.chunk.updated_at), hit.chunk.chunk_id))
        return tuple(scored[:top_k])

    def get_parent(self, parent_id: str, filters: RetrievalFilters) -> ParentRecord | None:
        with self._lock:
            parent = self._state.parents.get(parent_id)
        if parent is None or not parent_matches_filters(parent, filters):
            return None
        return parent

    def upsert_document(
        self,
        document: StoredDocument,
        parents: Sequence[ParentRecord],
        chunks: Sequence[IndexedChunk],
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str = "",
        document_instruction: str = "",
        vectors_normalized: bool = True,
        tokenizer_revision: str = "unspecified",
        max_input_tokens: int = 0,
        allow_update: bool = False,
        expected_checksum: str | None = None,
    ) -> IndexMutation:
        _validate_index_payload(document, parents, chunks, embedding_dimension)
        with self._lock:
            current = self._state
            existing = current.documents.get((document.document_id, document.version))
            if existing is not None and existing.checksum_sha256 == document.checksum_sha256:
                return IndexMutation(
                    action="unchanged",
                    generation=current.descriptor.generation,
                    document_count=current.descriptor.document_count,
                    chunk_count=current.descriptor.chunk_count,
                )
            if existing is not None and not allow_update:
                raise DocumentConflict(
                    "document_id and version are immutable; use an explicit update with checksum precondition"
                )
            if existing is not None and expected_checksum != existing.checksum_sha256:
                raise DocumentConflict("document update checksum precondition failed")
            self._ensure_compatible(
                current,
                embedding_model,
                embedding_revision,
                embedding_dimension,
                parser_version,
                chunker_version,
                query_instruction,
                document_instruction,
                vectors_normalized,
                tokenizer_revision,
                max_input_tokens,
            )
            documents = dict(current.documents)
            parent_map = dict(current.parents)
            chunk_map = dict(current.chunks)
            if existing is not None:
                for parent_id in existing.parent_ids:
                    parent_map.pop(parent_id, None)
                for chunk_id in existing.chunk_ids:
                    chunk_map.pop(chunk_id, None)
            documents[(document.document_id, document.version)] = document
            parent_map.update((parent.parent_id, parent) for parent in parents)
            chunk_map.update((chunk.chunk_id, chunk) for chunk in chunks)
            action = "updated" if existing is not None else "indexed"
            self._state = self._make_state(
                current,
                documents,
                parent_map,
                chunk_map,
                embedding_model=embedding_model,
                embedding_revision=embedding_revision,
                embedding_dimension=embedding_dimension,
                parser_version=parser_version,
                chunker_version=chunker_version,
                query_instruction=query_instruction,
                document_instruction=document_instruction,
                vectors_normalized=vectors_normalized,
                tokenizer_revision=tokenizer_revision,
                max_input_tokens=max_input_tokens,
            )
            self._history[current.descriptor.generation] = current
            return IndexMutation(action, self._state.descriptor.generation, len(documents), len(chunk_map))

    def delete_document(self, document_id: str, version: str | None = None) -> IndexMutation:
        with self._lock:
            current = self._state
            selected = [
                key
                for key in current.documents
                if key[0] == document_id and (version is None or key[1] == version)
            ]
            if not selected:
                return IndexMutation(
                    "unchanged",
                    current.descriptor.generation,
                    current.descriptor.document_count,
                    current.descriptor.chunk_count,
                )
            documents = dict(current.documents)
            parent_map = dict(current.parents)
            chunk_map = dict(current.chunks)
            for key in selected:
                stored = documents.pop(key)
                for parent_id in stored.parent_ids:
                    parent_map.pop(parent_id, None)
                for chunk_id in stored.chunk_ids:
                    chunk_map.pop(chunk_id, None)
            self._state = self._make_state(
                current,
                documents,
                parent_map,
                chunk_map,
                embedding_model=current.descriptor.embedding_model,
                embedding_revision=current.descriptor.embedding_revision,
                embedding_dimension=current.descriptor.embedding_dimension,
                parser_version=current.descriptor.parser_version,
                chunker_version=current.descriptor.chunker_version,
                query_instruction=current.descriptor.query_instruction,
                document_instruction=current.descriptor.document_instruction,
                vectors_normalized=current.descriptor.vectors_normalized,
                tokenizer_revision=current.descriptor.tokenizer_revision,
                max_input_tokens=current.descriptor.max_input_tokens,
            )
            self._history[current.descriptor.generation] = current
            return IndexMutation("deleted", self._state.descriptor.generation, len(documents), len(chunk_map))

    def replace_all(
        self,
        documents: Sequence[StoredDocument],
        parents: Sequence[ParentRecord],
        chunks: Sequence[IndexedChunk],
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str = "",
        document_instruction: str = "",
        vectors_normalized: bool = True,
        tokenizer_revision: str = "unspecified",
        max_input_tokens: int = 0,
    ) -> IndexMutation:
        document_map = {(document.document_id, document.version): document for document in documents}
        if len(document_map) != len(documents):
            raise BackendProtocolError("duplicate document identity in replacement generation")
        parent_map = {parent.parent_id: parent for parent in parents}
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        if len(parent_map) != len(parents) or len(chunk_map) != len(chunks):
            raise BackendProtocolError("duplicate parent or chunk identity in replacement generation")
        for document in documents:
            _validate_index_payload(
                document,
                [parent_map[parent_id] for parent_id in document.parent_ids],
                [chunk_map[chunk_id] for chunk_id in document.chunk_ids],
                embedding_dimension,
            )
        # Build and validate every object before taking the lock.  Any exception leaves
        # the live alias (``self._state``) untouched.
        with self._lock:
            current = self._state
            descriptor = IndexDescriptor(
                index_name=current.descriptor.index_name,
                generation=current.descriptor.generation + 1,
                embedding_model=embedding_model,
                embedding_revision=embedding_revision,
                embedding_dimension=embedding_dimension,
                parser_version=parser_version,
                chunker_version=chunker_version,
                document_count=len(document_map),
                chunk_count=len(chunk_map),
                query_instruction=query_instruction,
                document_instruction=document_instruction,
                vectors_normalized=vectors_normalized,
                tokenizer_revision=tokenizer_revision,
                max_input_tokens=max_input_tokens,
            )
            self._history[current.descriptor.generation] = current
            self._state = _IndexState(
                descriptor=descriptor,
                documents=MappingProxyType(document_map),
                parents=MappingProxyType(parent_map),
                chunks=MappingProxyType(chunk_map),
            )
            return IndexMutation("replaced", descriptor.generation, len(document_map), len(chunk_map))

    def _ensure_compatible(
        self,
        state: _IndexState,
        model: str,
        revision: str,
        dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str,
        document_instruction: str,
        vectors_normalized: bool,
        tokenizer_revision: str,
        max_input_tokens: int,
    ) -> None:
        if not state.chunks:
            return
        descriptor = state.descriptor
        if (
            descriptor.embedding_model != model
            or descriptor.embedding_revision != revision
            or descriptor.embedding_dimension != dimension
            or descriptor.parser_version != parser_version
            or descriptor.chunker_version != chunker_version
            or descriptor.query_instruction != query_instruction
            or descriptor.document_instruction != document_instruction
            or descriptor.vectors_normalized != vectors_normalized
            or descriptor.tokenizer_revision != tokenizer_revision
            or descriptor.max_input_tokens != max_input_tokens
        ):
            raise IndexIncompatible("active index embedding and ingestion manifest is incompatible")

    def _make_state(
        self,
        current: _IndexState,
        documents: Mapping[tuple[str, str], StoredDocument],
        parents: Mapping[str, ParentRecord],
        chunks: Mapping[str, IndexedChunk],
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        parser_version: str,
        chunker_version: str,
        query_instruction: str,
        document_instruction: str,
        vectors_normalized: bool,
        tokenizer_revision: str,
        max_input_tokens: int,
    ) -> _IndexState:
        descriptor = IndexDescriptor(
            index_name=current.descriptor.index_name,
            generation=current.descriptor.generation + 1,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            embedding_dimension=embedding_dimension,
            parser_version=parser_version,
            chunker_version=chunker_version,
            document_count=len(documents),
            chunk_count=len(chunks),
            query_instruction=query_instruction,
            document_instruction=document_instruction,
            vectors_normalized=vectors_normalized,
            tokenizer_revision=tokenizer_revision,
            max_input_tokens=max_input_tokens,
        )
        return _IndexState(
            descriptor=descriptor,
            documents=MappingProxyType(dict(documents)),
            parents=MappingProxyType(dict(parents)),
            chunks=MappingProxyType(dict(chunks)),
        )


class UnavailableSearchBackend:
    """Fail-closed placeholder when a non-memory endpoint has no injected adapter."""

    def __init__(self, index_name: str, endpoint: str) -> None:
        self._index_name = index_name
        self._endpoint = endpoint

    def health(self) -> BackendHealth:
        return BackendHealth(
            status="DOWN",
            backend="unconfigured-provider",
            document_count=0,
            chunk_count=0,
            details={"index_name": self._index_name, "reason": "SEARCH_BACKEND_NOT_INJECTED"},
        )

    def descriptor(self) -> IndexDescriptor:
        raise BackendUnavailable("search backend adapter is not configured")

    def lexical_search(self, query: str, filters: RetrievalFilters, top_k: int) -> Sequence[SearchHit]:
        raise BackendUnavailable("search backend adapter is not configured")

    def dense_search(
        self, vector: Sequence[float], filters: RetrievalFilters, top_k: int
    ) -> Sequence[SearchHit]:
        raise BackendUnavailable("search backend adapter is not configured")

    def get_parent(self, parent_id: str, filters: RetrievalFilters) -> ParentRecord | None:
        raise BackendUnavailable("search backend adapter is not configured")


def _validate_vector(vector: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise BackendProtocolError(f"{label} must be a numeric sequence")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(f"{label} must be a numeric sequence") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise BackendProtocolError(f"{label} must contain finite values")
    return values


def _validate_index_payload(
    document: StoredDocument,
    parents: Sequence[ParentRecord],
    chunks: Sequence[IndexedChunk],
    embedding_dimension: int,
) -> None:
    if not parents or not chunks:
        raise BackendProtocolError("indexed document must contain parents and chunks")
    parent_ids = {parent.parent_id for parent in parents}
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    if tuple(sorted(parent_ids)) != tuple(sorted(document.parent_ids)):
        raise BackendProtocolError("stored document parent IDs do not match payload")
    if tuple(sorted(chunk_ids)) != tuple(sorted(document.chunk_ids)):
        raise BackendProtocolError("stored document chunk IDs do not match payload")
    for parent in parents:
        if parent.document_id != document.document_id or parent.version != document.version:
            raise BackendProtocolError("parent identity does not match stored document")
    for chunk in chunks:
        if chunk.document_id != document.document_id or chunk.version != document.version:
            raise BackendProtocolError("chunk identity does not match stored document")
        if chunk.parent_id not in parent_ids:
            raise BackendProtocolError("chunk refers to an unknown parent")
        vector = _validate_vector(chunk.embedding, f"chunk {chunk.chunk_id} embedding")
        if len(vector) != embedding_dimension:
            raise IndexIncompatible("chunk embedding dimension does not match index mapping")


def _descending_date_key(value: str) -> tuple[int, ...]:
    # ISO-8601 strings are normally lexical.  Invert code points so ascending tuple
    # sorting yields descending update time without platform-dependent locale rules.
    return tuple(-ord(character) for character in value)


__all__ = [
    "BackendHealth",
    "BackendProtocolError",
    "BackendTimeout",
    "BackendUnavailable",
    "DeterministicEmbeddingBackend",
    "DeterministicReranker",
    "DocumentConflict",
    "EmbeddingBackend",
    "EmbeddingUnavailable",
    "InMemoryHybridIndex",
    "IndexDescriptor",
    "IndexIncompatible",
    "IndexMutation",
    "IndexedChunk",
    "ParentRecord",
    "RerankResult",
    "RerankerBackend",
    "RerankerUnavailable",
    "RetrievalFilters",
    "SearchBackend",
    "SearchHit",
    "StoredDocument",
    "UnavailableSearchBackend",
    "WritableSearchBackend",
    "chunk_matches_filters",
    "estimate_tokens",
    "parent_matches_filters",
    "tokenize",
]
