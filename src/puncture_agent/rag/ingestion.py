"""Deterministic source normalization, heading-aware chunking, and indexing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .backends import (
    BackendProtocolError,
    DocumentConflict,
    EmbeddingBackend,
    IndexIncompatible,
    IndexMutation,
    IndexedChunk,
    ParentRecord,
    StoredDocument,
    WritableSearchBackend,
    estimate_tokens,
    tokenize,
)
from .errors import RagServiceError


PARSER_VERSION = "markdown-normalizer-v1"
CHUNKER_VERSION = "heading-parent-child-v1"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RESERVED_METADATA = {
    "document_id",
    "title",
    "module",
    "version",
    "status",
    "owner",
    "source_type",
    "source_uri",
    "access_scopes",
    "checksum_sha256",
    "parser_version",
    "chunker_version",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,255}$")


def _require_text(value: str, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    allowed_controls = "\t\n\r\f" if name == "content" else ""
    if any(ord(character) < 32 and character not in allowed_controls for character in normalized):
        raise ValueError(f"{name} contains unsupported control characters")
    return normalized


@dataclass(frozen=True)
class SourceDocument:
    """Strict ingestion input; every security and lifecycle field is mandatory."""

    document_id: str
    title: str
    source_uri: str
    source_type: str
    module: str
    version: str
    status: str
    owner: str
    access_scopes: tuple[str, ...]
    content: str
    updated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "document_id",
            "title",
            "source_uri",
            "source_type",
            "module",
            "version",
            "owner",
            "content",
            "updated_at",
        ):
            maximum = 2_000_000 if name == "content" else 4096
            object.__setattr__(self, name, _require_text(getattr(self, name), name, maximum=maximum))
        for name in ("document_id", "source_type", "module", "version", "owner"):
            if not _IDENTIFIER.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} contains unsupported identifier characters")
        if self.status not in {"active", "deprecated", "superseded"}:
            raise ValueError("status must be active, deprecated, or superseded")
        if isinstance(self.access_scopes, (str, bytes)):
            raise ValueError("access_scopes must be a sequence of scope names")
        scopes = tuple(self.access_scopes)
        if not scopes or len(scopes) > 64:
            raise ValueError("access_scopes must contain between 1 and 64 entries")
        if len(set(scopes)) != len(scopes):
            raise ValueError("access_scopes must be unique")
        normalized_scopes: list[str] = []
        for scope in scopes:
            normalized_scope = _require_text(scope, "access scope", maximum=128)
            if not _IDENTIFIER.fullmatch(normalized_scope):
                raise ValueError("access scope contains unsupported identifier characters")
            normalized_scopes.append(normalized_scope)
        object.__setattr__(self, "access_scopes", tuple(normalized_scopes))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        metadata = dict(self.metadata)
        if any(key in _RESERVED_METADATA for key in metadata):
            raise ValueError("metadata must not override reserved index fields")
        counter = [0]
        _validate_json_value(metadata, "metadata", depth=0, counter=counter)
        encoded_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded_metadata.encode("utf-8")) > 65_536:
            raise ValueError("metadata exceeds 65536 bytes")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def checksum_sha256(self) -> str:
        payload = {
            "document_id": self.document_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "module": self.module,
            "version": self.version,
            "status": self.status,
            "owner": self.owner,
            "access_scopes": self.access_scopes,
            "content": normalize_source_text(self.content),
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedSection:
    section_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class PreparedDocument:
    stored: StoredDocument
    parents: tuple[ParentRecord, ...]
    chunks: tuple[IndexedChunk, ...]


@dataclass(frozen=True)
class IngestionReport:
    document_id: str
    version: str
    checksum_sha256: str
    action: str
    generation: int
    parent_count: int
    chunk_count: int


def normalize_source_text(text: str) -> str:
    """Normalize Unicode/whitespace and remove repeated per-page header/footer noise."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    pages = normalized.split("\f")
    if len(pages) > 1:
        page_lines = [page.splitlines() for page in pages]
        boundary_counts: dict[str, int] = {}
        for lines in page_lines:
            nonempty = [line.strip() for line in lines if line.strip()]
            for boundary in nonempty[:1] + nonempty[-1:]:
                boundary_counts[boundary] = boundary_counts.get(boundary, 0) + 1
        repeated = {line for line, count in boundary_counts.items() if count >= 2}
        cleaned_pages: list[str] = []
        for lines in page_lines:
            nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
            removable: set[int] = set()
            if nonempty_indexes and lines[nonempty_indexes[0]].strip() in repeated:
                removable.add(nonempty_indexes[0])
            if nonempty_indexes and lines[nonempty_indexes[-1]].strip() in repeated:
                removable.add(nonempty_indexes[-1])
            cleaned_pages.append("\n".join(line for index, line in enumerate(lines) if index not in removable))
        normalized = "\n".join(cleaned_pages)
    normalized_lines = [line.rstrip() for line in normalized.splitlines()]
    output: list[str] = []
    previous_blank = False
    for line in normalized_lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        output.append(line)
        previous_blank = blank
    return "\n".join(output).strip()


class MarkdownSectionParser:
    """Minimal Markdown/normalized-text parser preserving tables and code blocks."""

    version = PARSER_VERSION

    def parse(self, document: SourceDocument) -> tuple[ParsedSection, ...]:
        text = normalize_source_text(document.content)
        hierarchy: list[str] = []
        current_path: tuple[str, ...] = (document.title,)
        body: list[str] = []
        sections: list[ParsedSection] = []

        def flush() -> None:
            section_text = "\n".join(body).strip()
            if section_text:
                sections.append(ParsedSection(section_path=current_path, text=section_text))
            body.clear()

        in_fence = False
        fence_marker = ""
        for line in text.splitlines():
            fence_match = _FENCE.match(line)
            if fence_match:
                marker = fence_match.group(1)
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                body.append(line)
                continue
            heading = None if in_fence else _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                heading_text = _require_text(heading.group(2), "heading", maximum=1024)
                hierarchy[:] = hierarchy[: level - 1]
                while len(hierarchy) < level - 1:
                    hierarchy.append("Untitled")
                hierarchy.append(heading_text)
                current_path = tuple([document.title, *hierarchy])
            else:
                body.append(line)
        flush()
        if not sections:
            raise ValueError("document content contains no indexable text")
        return tuple(sections)


class HeadingAwareChunker:
    """Parent-child chunker that keeps fenced code and Markdown tables coherent."""

    version = CHUNKER_VERSION

    def __init__(self, *, target_tokens: int = 600, overlap_tokens: int = 100) -> None:
        if target_tokens < 16:
            raise ValueError("target_tokens must be at least 16")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def build_records(
        self,
        document: SourceDocument,
        sections: Sequence[ParsedSection],
    ) -> tuple[tuple[ParentRecord, ...], tuple[IndexedChunk, ...]]:
        parents: list[ParentRecord] = []
        chunks: list[IndexedChunk] = []
        checksum = document.checksum_sha256
        for section_ordinal, section in enumerate(sections):
            path_key = "\x1f".join(section.section_path)
            path_digest = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:12]
            parent_id = f"{document.document_id}#{document.version}#p-{path_digest}-{section_ordinal:04d}"
            parent = ParentRecord(
                parent_id=parent_id,
                document_id=document.document_id,
                title=document.title,
                module=document.module,
                version=document.version,
                status=document.status,
                section_path=section.section_path,
                text=section.text,
                access_scopes=document.access_scopes,
                owner=document.owner,
                source_type=document.source_type,
                updated_at=document.updated_at,
                metadata=document.metadata,
            )
            parents.append(parent)
            child_texts = self._chunk_section(section.text)
            for chunk_index, child_text in enumerate(child_texts):
                chunk_id = f"{document.document_id}#{document.version}#{path_digest}#{chunk_index:04d}"
                chunks.append(
                    IndexedChunk(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        document_id=document.document_id,
                        title=document.title,
                        module=document.module,
                        version=document.version,
                        status=document.status,
                        section_path=section.section_path,
                        text=child_text,
                        token_count=estimate_tokens(child_text),
                        chunk_index=chunk_index,
                        access_scopes=document.access_scopes,
                        owner=document.owner,
                        source_type=document.source_type,
                        updated_at=document.updated_at,
                        checksum_sha256=checksum,
                        parser_version=PARSER_VERSION,
                        chunker_version=self.version,
                        metadata=document.metadata,
                    )
                )
        return tuple(parents), tuple(chunks)

    def _chunk_section(self, text: str) -> tuple[str, ...]:
        blocks = _coherent_blocks(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for block in blocks:
            block_tokens = estimate_tokens(block)
            if block_tokens > self.target_tokens and not _is_atomic_block(block):
                if current:
                    chunks.append("\n\n".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_large_text(block))
                continue
            if current and current_tokens + block_tokens > self.target_tokens:
                completed = "\n\n".join(current).strip()
                chunks.append(completed)
                overlap = (
                    ""
                    if _is_atomic_block(completed) or _is_atomic_block(block)
                    else _tail_words(
                        completed,
                        min(
                            self.overlap_tokens,
                            max(0, self.target_tokens - block_tokens),
                        ),
                    )
                )
                current = [overlap] if overlap else []
                current_tokens = estimate_tokens(overlap) if overlap else 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
        return tuple(chunk for chunk in chunks if chunk)

    def _split_large_text(self, text: str) -> list[str]:
        remaining = text.strip()
        if not remaining:
            return []
        chunks: list[str] = []
        while estimate_tokens(remaining) > self.target_tokens:
            cut = _largest_prefix_within_budget(remaining, self.target_tokens)
            candidate = remaining[:cut].rstrip()
            # Prefer a nearby whitespace boundary without sacrificing most of the
            # token budget. Mixed CJK/identifier prose may have no such boundary.
            whitespace_cut = max(
                candidate.rfind(" "),
                candidate.rfind("\n"),
                candidate.rfind("\t"),
            )
            if whitespace_cut >= max(1, len(candidate) // 2):
                cut = whitespace_cut + 1
                candidate = remaining[:cut].rstrip()
            if not candidate:
                candidate = remaining[:cut]
            chunks.append(candidate)
            overlap = _tail_words(candidate, self.overlap_tokens)
            suffix = remaining[cut:].lstrip()
            remaining = f"{overlap} {suffix}".strip() if overlap else suffix
        if remaining:
            chunks.append(remaining)
        return chunks


class RagIngestionService:
    """Prepare and atomically publish versioned documents into a writable backend."""

    def __init__(
        self,
        backend: WritableSearchBackend,
        embedding_backend: EmbeddingBackend,
        *,
        parser: MarkdownSectionParser | None = None,
        chunker: HeadingAwareChunker | None = None,
        max_documents_per_batch: int = 1000,
        max_chunks_per_document: int = 10_000,
        embedding_batch_size: int = 128,
    ) -> None:
        for value, label in (
            (max_documents_per_batch, "max_documents_per_batch"),
            (max_chunks_per_document, "max_chunks_per_document"),
            (embedding_batch_size, "embedding_batch_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.backend = backend
        self.embedding_backend = embedding_backend
        self.parser = parser or MarkdownSectionParser()
        self.chunker = chunker or HeadingAwareChunker()
        self.max_documents_per_batch = max_documents_per_batch
        self.max_chunks_per_document = max_chunks_per_document
        self.embedding_batch_size = embedding_batch_size

    def prepare(self, document: SourceDocument) -> PreparedDocument:
        try:
            sections = self.parser.parse(document)
            parents, chunks = self.chunker.build_records(document, sections)
            if len(chunks) > self.max_chunks_per_document:
                raise ValueError("document exceeds the maximum chunk count")
            max_input_tokens = int(getattr(self.embedding_backend, "max_input_tokens", 0))
            if max_input_tokens > 0 and any(chunk.token_count > max_input_tokens for chunk in chunks):
                raise ValueError("chunk exceeds the embedding provider max_input_tokens")
            embeddings: list[Sequence[float]] = []
            for start in range(0, len(chunks), self.embedding_batch_size):
                batch = chunks[start : start + self.embedding_batch_size]
                batch_embeddings = self.embedding_backend.embed_documents(
                    [chunk.text for chunk in batch]
                )
                if (
                    isinstance(batch_embeddings, (str, bytes))
                    or len(batch_embeddings) != len(batch)
                ):
                    raise BackendProtocolError("embedding backend returned the wrong batch size")
                embeddings.extend(batch_embeddings)
            embedded_chunks: list[IndexedChunk] = []
            for chunk, vector in zip(chunks, embeddings):
                values = _validate_embedding(vector, self.embedding_backend.dimension)
                embedded_chunks.append(replace(chunk, embedding=values))
            stored = StoredDocument(
                document_id=document.document_id,
                version=document.version,
                checksum_sha256=document.checksum_sha256,
                chunk_ids=tuple(chunk.chunk_id for chunk in embedded_chunks),
                parent_ids=tuple(parent.parent_id for parent in parents),
            )
            return PreparedDocument(stored=stored, parents=parents, chunks=tuple(embedded_chunks))
        except RagServiceError:
            raise
        except IndexIncompatible as exc:
            raise RagServiceError("RAG_INDEX_INCOMPATIBLE", str(exc), retryable=False) from exc
        except (TimeoutError,) as exc:
            raise RagServiceError("RAG_TIMEOUT", "embedding timed out", retryable=True, details={"stage": "embedding"}) from exc
        except BackendProtocolError as exc:
            raise RagServiceError("RAG_PROTOCOL_ERROR", str(exc), retryable=False, details={"stage": "ingestion"}) from exc
        except (TypeError, ValueError) as exc:
            raise RagServiceError("RAG_INVALID_REQUEST", str(exc), retryable=False, details={"stage": "ingestion"}) from exc
        except Exception as exc:
            raise RagServiceError(
                "RAG_EMBEDDING_UNAVAILABLE",
                "embedding provider failed during ingestion",
                retryable=True,
                details={"stage": "embedding"},
            ) from exc

    def ingest(self, document: SourceDocument) -> IngestionReport:
        prepared = self.prepare(document)
        return self._publish(document, prepared, allow_update=False, expected_checksum=None)

    def update(self, document: SourceDocument, *, expected_checksum: str) -> IngestionReport:
        if not isinstance(expected_checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_checksum):
            raise RagServiceError(
                "RAG_INVALID_REQUEST", "expected_checksum must be a lowercase SHA-256 digest", retryable=False
            )
        prepared = self.prepare(document)
        return self._publish(
            document,
            prepared,
            allow_update=True,
            expected_checksum=expected_checksum,
        )

    def _publish(
        self,
        document: SourceDocument,
        prepared: PreparedDocument,
        *,
        allow_update: bool,
        expected_checksum: str | None,
    ) -> IngestionReport:
        try:
            mutation = self.backend.upsert_document(
                prepared.stored,
                prepared.parents,
                prepared.chunks,
                embedding_model=self.embedding_backend.model_name,
                embedding_revision=self.embedding_backend.revision,
                embedding_dimension=self.embedding_backend.dimension,
                parser_version=self.parser.version,
                chunker_version=self.chunker.version,
                **_embedding_manifest(self.embedding_backend),
                allow_update=allow_update,
                expected_checksum=expected_checksum,
            )
        except DocumentConflict as exc:
            raise RagServiceError("RAG_INVALID_REQUEST", str(exc), retryable=False) from exc
        except IndexIncompatible as exc:
            raise RagServiceError("RAG_INDEX_INCOMPATIBLE", str(exc), retryable=False) from exc
        except BackendProtocolError as exc:
            raise RagServiceError("RAG_PROTOCOL_ERROR", str(exc), retryable=False) from exc
        except TimeoutError as exc:
            raise RagServiceError("RAG_TIMEOUT", "index publication timed out", retryable=True) from exc
        except Exception as exc:
            raise RagServiceError("RAG_BACKEND_UNAVAILABLE", "index publication failed", retryable=True) from exc
        return _report(document, prepared, mutation)

    def rebuild(self, documents: Iterable[SourceDocument]) -> tuple[IngestionReport, ...]:
        sources = tuple(documents)
        if not sources:
            raise RagServiceError("RAG_INVALID_REQUEST", "rebuild requires at least one document")
        if len(sources) > self.max_documents_per_batch:
            raise RagServiceError("RAG_INVALID_REQUEST", "rebuild exceeds maximum document count")
        identities = {(document.document_id, document.version) for document in sources}
        if len(identities) != len(sources):
            raise RagServiceError("RAG_INVALID_REQUEST", "rebuild contains duplicate document identities")
        # Every parse, chunk, and embedding operation finishes before the backend's
        # atomic alias replacement.  A partial preparation failure cannot become live.
        prepared = tuple(self.prepare(document) for document in sources)
        try:
            mutation = self.backend.replace_all(
                [item.stored for item in prepared],
                [parent for item in prepared for parent in item.parents],
                [chunk for item in prepared for chunk in item.chunks],
                embedding_model=self.embedding_backend.model_name,
                embedding_revision=self.embedding_backend.revision,
                embedding_dimension=self.embedding_backend.dimension,
                parser_version=self.parser.version,
                chunker_version=self.chunker.version,
                **_embedding_manifest(self.embedding_backend),
            )
        except IndexIncompatible as exc:
            raise RagServiceError("RAG_INDEX_INCOMPATIBLE", str(exc), retryable=False) from exc
        except BackendProtocolError as exc:
            raise RagServiceError("RAG_PROTOCOL_ERROR", str(exc), retryable=False) from exc
        except TimeoutError as exc:
            raise RagServiceError("RAG_TIMEOUT", "index replacement timed out", retryable=True) from exc
        except Exception as exc:
            raise RagServiceError("RAG_BACKEND_UNAVAILABLE", "index replacement failed", retryable=True) from exc
        return tuple(
            IngestionReport(
                document_id=document.document_id,
                version=document.version,
                checksum_sha256=document.checksum_sha256,
                action=mutation.action,
                generation=mutation.generation,
                parent_count=len(item.parents),
                chunk_count=len(item.chunks),
            )
            for document, item in zip(sources, prepared)
        )

    def delete(self, document_id: str, version: str | None = None) -> IndexMutation:
        try:
            normalized_id = _require_text(document_id, "document_id")
            normalized_version = None if version is None else _require_text(version, "version")
            return self.backend.delete_document(normalized_id, normalized_version)
        except ValueError as exc:
            raise RagServiceError("RAG_INVALID_REQUEST", str(exc), retryable=False) from exc
        except TimeoutError as exc:
            raise RagServiceError("RAG_TIMEOUT", "index deletion timed out", retryable=True) from exc
        except Exception as exc:
            raise RagServiceError("RAG_BACKEND_UNAVAILABLE", "index deletion failed", retryable=True) from exc


def _coherent_blocks(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if current:
            value = "\n".join(current).strip()
            if value:
                blocks.append(value)
            current.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = fence.group(1)
            current.append(line)
            if in_fence and len(current) > 1 and fence.group(1) == fence_marker:
                in_fence = False
                flush()
            index += 1
            continue
        if in_fence:
            current.append(line)
            index += 1
            continue
        if "|" in line and index + 1 < len(lines) and _looks_like_table_separator(lines[index + 1]):
            flush()
            table = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table.append(lines[index])
                index += 1
            blocks.append("\n".join(table).strip())
            continue
        if not line.strip():
            flush()
        else:
            current.append(line)
        index += 1
    flush()
    return tuple(blocks)


def _looks_like_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _is_atomic_block(block: str) -> bool:
    stripped = block.lstrip()
    lines = block.splitlines()
    return stripped.startswith(("```", "~~~")) or (
        len(lines) >= 2 and "|" in lines[0] and _looks_like_table_separator(lines[1])
    )


def _tail_words(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    value = text.strip()
    if estimate_tokens(value) <= limit:
        return value
    best = ""
    for start in range(len(value) - 1, -1, -1):
        candidate = value[start:].strip()
        if estimate_tokens(candidate) > limit:
            break
        best = candidate
    return best


def _largest_prefix_within_budget(text: str, budget: int) -> int:
    low = 1
    high = len(text)
    best = 1
    while low <= high:
        middle = (low + high) // 2
        if estimate_tokens(text[:middle]) <= budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _validate_embedding(vector: Sequence[float], expected_dimension: int) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise BackendProtocolError("embedding vector must be a numeric sequence")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError("embedding vector must be a numeric sequence") from exc
    if len(values) != expected_dimension:
        raise IndexIncompatible("embedding backend dimension does not match returned vector")
    if any(value != value or value in {float("inf"), float("-inf")} for value in values):
        raise BackendProtocolError("embedding vector contains a non-finite value")
    return values


def _report(document: SourceDocument, prepared: PreparedDocument, mutation: IndexMutation) -> IngestionReport:
    return IngestionReport(
        document_id=document.document_id,
        version=document.version,
        checksum_sha256=document.checksum_sha256,
        action=mutation.action,
        generation=mutation.generation,
        parent_count=len(prepared.parents),
        chunk_count=len(prepared.chunks),
    )


def _validate_json_value(value: Any, path: str, *, depth: int, counter: list[int]) -> None:
    counter[0] += 1
    if counter[0] > 2048:
        raise ValueError("metadata contains too many values")
    if depth > 8:
        raise ValueError("metadata nesting exceeds 8 levels")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError(f"{path} string exceeds 4096 characters")
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            _validate_json_value(child, f"{path}.{key}", depth=depth + 1, counter=counter)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]", depth=depth + 1, counter=counter)
        return
    raise ValueError(f"{path} contains a non-JSON value")


def _embedding_manifest(backend: EmbeddingBackend) -> dict[str, Any]:
    return {
        "query_instruction": str(getattr(backend, "query_instruction", "")),
        "document_instruction": str(getattr(backend, "document_instruction", "")),
        "vectors_normalized": bool(getattr(backend, "vectors_normalized", False)),
        "tokenizer_revision": str(getattr(backend, "tokenizer_revision", "unspecified")),
        "max_input_tokens": int(getattr(backend, "max_input_tokens", 0)),
    }


__all__ = [
    "CHUNKER_VERSION",
    "HeadingAwareChunker",
    "IngestionReport",
    "MarkdownSectionParser",
    "PARSER_VERSION",
    "ParsedSection",
    "PreparedDocument",
    "RagIngestionService",
    "SourceDocument",
    "normalize_source_text",
]
