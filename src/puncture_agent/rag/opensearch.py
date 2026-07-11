"""Strict OpenSearch REST implementation of the provider-neutral SearchBackend."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from .backends import (
    BackendHealth,
    BackendProtocolError,
    BackendTimeout,
    BackendUnavailable,
    IndexDescriptor,
    IndexIncompatible,
    IndexedChunk,
    ParentRecord,
    RetrievalFilters,
    SearchHit,
)
from .provider_http import (
    HttpxProviderTransport,
    ProviderEndpoint,
    ProviderHttpResponse,
    ProviderHttpTransport,
    ProviderProtocolError,
    ProviderSecurityError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    decode_json_response,
)


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
_FIELD_PATH = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,254}$")
_GENERATION_SUFFIX = re.compile(r"-v(\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CHUNK_SOURCE_FIELDS = (
    "doc_kind",
    "chunk_id",
    "parent_id",
    "document_id",
    "title",
    "module",
    "version",
    "status",
    "section_path",
    "text",
    "token_count",
    "chunk_index",
    "access_scopes",
    "owner",
    "source_type",
    "updated_at",
    "checksum_sha256",
    "parser_version",
    "chunker_version",
    "metadata",
)

_PARENT_SOURCE_FIELDS = (
    "doc_kind",
    "parent_id",
    "document_id",
    "title",
    "module",
    "version",
    "status",
    "section_path",
    "text",
    "access_scopes",
    "owner",
    "source_type",
    "updated_at",
    "metadata",
)

_RESERVED_METADATA_FILTERS = frozenset(
    {"document_id", "module", "version", "status", "access_scopes", "parent_id", "chunk_id"}
)


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or ".." in value:
        raise ValueError(f"{label} is not a safe OpenSearch identifier")
    return value


def _require_string(
    value: Any,
    label: str,
    *,
    maximum: int = 1_000_000,
    allow_multiline: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
    allowed_controls = "\t\n\r" if allow_multiline else ""
    if any(ord(character) < 32 and character not in allowed_controls for character in value):
        raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
    return value


def _require_string_tuple(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
    result = tuple(_require_string(item, label, maximum=4096) for item in value)
    if len(set(result)) != len(result):
        raise BackendProtocolError(f"OpenSearch returned duplicate {label} values")
    return result


class OpenSearchSearchBackend:
    """Read-only BM25/vector backend with mandatory filters and strict decoding."""

    def __init__(
        self,
        endpoint: ProviderEndpoint,
        *,
        read_alias: str,
        transport: ProviderHttpTransport | None = None,
        metadata_field_map: Mapping[str, str] | None = None,
    ) -> None:
        self._read_alias = _validate_identifier(read_alias, "read_alias")
        self._transport = transport or HttpxProviderTransport(endpoint)
        self._owns_transport = transport is None
        if self._transport.endpoint != endpoint:
            raise ValueError("OpenSearch transport endpoint does not match the configured endpoint")
        fields = dict(
            metadata_field_map
            or {
                "source_type": "source_type",
                "owner": "owner",
                "category": "metadata_terms",
                "language": "metadata_terms",
            }
        )
        if not fields:
            raise ValueError("metadata_field_map must not be empty")
        for key, path in fields.items():
            if (
                not isinstance(key, str)
                or not key
                or key in _RESERVED_METADATA_FILTERS
                or not isinstance(path, str)
                or not _FIELD_PATH.fullmatch(path)
            ):
                raise ValueError("metadata_field_map contains an unsafe entry")
        self._metadata_field_map = fields
        self._descriptor_cache: IndexDescriptor | None = None

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()

    def health(self) -> BackendHealth:
        cluster = self._request_json("GET", "/_cluster/health")
        if cluster.get("timed_out") is True:
            raise BackendTimeout("OpenSearch cluster health request timed out")
        cluster_status = cluster.get("status")
        if cluster_status not in {"green", "yellow", "red"}:
            raise BackendProtocolError("OpenSearch returned an invalid cluster health status")
        descriptor = self.descriptor()
        status = {"green": "UP", "yellow": "DEGRADED", "red": "DOWN"}[cluster_status]
        return BackendHealth(
            status=status,
            backend="opensearch-rest",
            document_count=descriptor.document_count,
            chunk_count=descriptor.chunk_count,
            details={
                "cluster_status": cluster_status,
                "read_alias": self._read_alias,
                "concrete_index": descriptor.index_name,
                "generation": descriptor.generation,
            },
        )

    def descriptor(self) -> IndexDescriptor:
        payload = self._request_json("GET", f"/{quote(self._read_alias, safe='')}/_mapping")
        if len(payload) != 1:
            raise IndexIncompatible("OpenSearch read alias must resolve to exactly one concrete index")
        concrete_index, index_value = next(iter(payload.items()))
        concrete_index = _validate_identifier(concrete_index, "concrete index")
        if not isinstance(index_value, dict):
            raise BackendProtocolError("OpenSearch mapping response is malformed")
        mappings = index_value.get("mappings")
        if not isinstance(mappings, dict):
            raise BackendProtocolError("OpenSearch mapping response is missing mappings")
        meta = mappings.get("_meta")
        if not isinstance(meta, dict):
            raise IndexIncompatible("OpenSearch mapping is missing the RAG index manifest")
        contract = meta.get("contract")
        if contract not in {"puncture-rag-chunk-v1", "puncture-rag-index"}:
            raise IndexIncompatible("OpenSearch mapping contract is incompatible")
        generation = meta.get("generation")
        if generation is None:
            match = _GENERATION_SUFFIX.search(concrete_index)
            if match is None:
                raise IndexIncompatible("OpenSearch index generation is missing")
            generation = int(match.group(1))
        generation = _require_int(generation, "index generation", minimum=0)
        descriptor = IndexDescriptor(
            index_name=concrete_index,
            generation=generation,
            embedding_model=_require_string(meta.get("embedding_model"), "embedding model", maximum=4096),
            embedding_revision=_require_string(
                meta.get("embedding_revision"),
                "embedding revision",
                maximum=4096,
            ),
            embedding_dimension=_require_int(
                meta.get("embedding_dimension"),
                "embedding dimension",
                minimum=1,
            ),
            parser_version=_require_string(meta.get("parser_version"), "parser version", maximum=4096),
            chunker_version=_require_string(meta.get("chunker_version"), "chunker version", maximum=4096),
            document_count=_require_int(meta.get("document_count", 0), "document count", minimum=0),
            chunk_count=_require_int(meta.get("chunk_count", 0), "chunk count", minimum=0),
            query_instruction=self._optional_string(meta.get("query_instruction"), "query instruction"),
            document_instruction=self._optional_string(
                meta.get("document_instruction"),
                "document instruction",
            ),
            vectors_normalized=_require_bool(
                meta.get("vectors_normalized"),
                "vectors_normalized",
            ),
            tokenizer_revision=_require_string(
                meta.get("tokenizer_revision"),
                "tokenizer revision",
                maximum=4096,
            ),
            max_input_tokens=_require_int(
                meta.get("max_input_tokens"),
                "max input tokens",
                minimum=1,
            ),
        )
        properties = mappings.get("properties")
        if not isinstance(properties, dict):
            raise IndexIncompatible("OpenSearch mapping is missing field properties")
        embedding = properties.get("embedding")
        if not isinstance(embedding, dict) or embedding.get("type") != "knn_vector":
            raise IndexIncompatible("OpenSearch mapping is missing the k-NN embedding field")
        if embedding.get("dimension") != descriptor.embedding_dimension:
            raise IndexIncompatible("OpenSearch vector mapping and index manifest dimensions differ")
        self._descriptor_cache = descriptor
        return descriptor

    def lexical_search(
        self,
        query: str,
        filters: RetrievalFilters,
        top_k: int,
    ) -> Sequence[SearchHit]:
        query_value = _require_string(query, "lexical query", maximum=131_072)
        limit = self._validate_top_k(top_k)
        body = {
            "size": limit,
            "track_total_hits": False,
            "_source": list(_CHUNK_SOURCE_FIELDS),
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_value,
                                "fields": [
                                    "title^3",
                                    "title.code^5",
                                    "section^2",
                                    "text",
                                    "text.code^4",
                                ],
                                "type": "best_fields",
                                "operator": "or",
                            }
                        }
                    ],
                    "filter": self._mandatory_filter_clauses(filters, doc_kind="child"),
                }
            },
        }
        payload = self._request_json(
            "POST",
            f"/{quote(self._read_alias, safe='')}/_search",
            json_body=body,
        )
        return self._parse_chunk_hits(payload, filters, limit)

    def dense_search(
        self,
        vector: Sequence[float],
        filters: RetrievalFilters,
        top_k: int,
    ) -> Sequence[SearchHit]:
        values = self._validate_vector(vector)
        descriptor = self._descriptor_cache or self.descriptor()
        if len(values) != descriptor.embedding_dimension:
            raise IndexIncompatible("query embedding dimension does not match the OpenSearch index")
        limit = self._validate_top_k(top_k)
        body = {
            "size": limit,
            "track_total_hits": False,
            "_source": list(_CHUNK_SOURCE_FIELDS),
            "query": {
                "knn": {
                    "embedding": {
                        "vector": list(values),
                        "k": limit,
                        "filter": {
                            "bool": {
                                "filter": self._mandatory_filter_clauses(
                                    filters,
                                    doc_kind="child",
                                )
                            }
                        },
                    }
                }
            },
        }
        payload = self._request_json(
            "POST",
            f"/{quote(self._read_alias, safe='')}/_search",
            json_body=body,
        )
        return self._parse_chunk_hits(payload, filters, limit)

    def get_parent(self, parent_id: str, filters: RetrievalFilters) -> ParentRecord | None:
        parent_value = _require_string(parent_id, "parent ID", maximum=4096)
        clauses = self._mandatory_filter_clauses(filters, doc_kind="parent")
        clauses.append({"term": {"parent_id": parent_value}})
        payload = self._request_json(
            "POST",
            f"/{quote(self._read_alias, safe='')}/_search",
            json_body={
                "size": 2,
                "track_total_hits": False,
                "_source": list(_PARENT_SOURCE_FIELDS),
                "query": {"bool": {"filter": clauses}},
            },
        )
        raw_hits = self._extract_hits(payload)
        if not raw_hits:
            return None
        if len(raw_hits) != 1:
            raise BackendProtocolError("OpenSearch returned duplicate parent records")
        hit = raw_hits[0]
        if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
            raise BackendProtocolError("OpenSearch returned a malformed parent hit")
        parent = self._parse_parent(hit["_source"])
        if hit.get("_id") != parent.parent_id or parent.parent_id != parent_value:
            raise BackendProtocolError("OpenSearch parent identity is inconsistent")
        if not self._parent_matches(parent, filters):
            raise BackendProtocolError("OpenSearch parent violated mandatory retrieval filters")
        return parent

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._transport.request(
                method,
                path,
                json_body=json_body,
                timeout_seconds=self._transport.endpoint.timeout_seconds,
            )
        except ProviderTimeoutError as exc:
            raise BackendTimeout("OpenSearch request timed out") from exc
        except ProviderUnavailableError as exc:
            raise BackendUnavailable("OpenSearch endpoint is unavailable") from exc
        except (ProviderSecurityError, ProviderProtocolError) as exc:
            raise BackendProtocolError("OpenSearch transport validation failed") from exc
        self._validate_status(response)
        try:
            payload = decode_json_response(
                response,
                max_bytes=self._transport.endpoint.max_response_bytes,
            )
        except ProviderProtocolError as exc:
            raise BackendProtocolError("OpenSearch returned invalid JSON") from exc
        return payload

    @staticmethod
    def _validate_status(response: ProviderHttpResponse) -> None:
        status = response.status
        if 200 <= status < 300:
            return
        if status in {408, 504}:
            raise BackendTimeout("OpenSearch request timed out")
        if status == 429 or 500 <= status <= 599:
            raise BackendUnavailable("OpenSearch endpoint is unavailable")
        if status in {401, 403}:
            raise BackendProtocolError("OpenSearch authentication was rejected")
        if status == 404:
            raise BackendUnavailable("OpenSearch read alias or index is unavailable")
        raise BackendProtocolError("OpenSearch returned an unexpected HTTP status")

    def _mandatory_filter_clauses(
        self,
        filters: RetrievalFilters,
        *,
        doc_kind: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(filters, RetrievalFilters):
            raise BackendProtocolError("retrieval filters are malformed")
        scopes = self._safe_filter_strings(filters.access_scopes, "access scopes")
        effective_scopes = sorted(set(scopes) | {"public"})
        statuses = self._safe_filter_strings(filters.allowed_statuses, "allowed statuses")
        clauses: list[dict[str, Any]] = [
            {"term": {"doc_kind": doc_kind}},
            {"terms": {"access_scopes": effective_scopes}},
            {"terms": {"status": list(statuses)}},
        ]
        if filters.modules:
            clauses.append({"terms": {"module": list(self._safe_filter_strings(filters.modules, "modules"))}})
        if filters.required_version is not None:
            clauses.append(
                {
                    "term": {
                        "version": _require_string(
                            filters.required_version,
                            "required version",
                            maximum=4096,
                        )
                    }
                }
            )
        for key, value in filters.metadata_filters.items():
            if key == "status":
                if value not in filters.allowed_statuses:
                    raise BackendProtocolError("status metadata filter conflicts with lifecycle policy")
                continue
            if key in _RESERVED_METADATA_FILTERS or key not in self._metadata_field_map:
                raise BackendProtocolError("metadata filter is not approved for OpenSearch")
            if value is None or isinstance(value, (Mapping, list, tuple, set)):
                raise BackendProtocolError("metadata filter value must be scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise BackendProtocolError("metadata filter value must be finite")
            field_path = self._metadata_field_map[key]
            filter_value = (
                self._canonical_metadata_term(key, value)
                if field_path == "metadata_terms"
                else value
            )
            clauses.append({"term": {field_path: filter_value}})
        return clauses

    def _parse_chunk_hits(
        self,
        payload: Mapping[str, Any],
        filters: RetrievalFilters,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        raw_hits = self._extract_hits(payload)
        if len(raw_hits) > limit:
            raise BackendProtocolError("OpenSearch exceeded the requested hit limit")
        output: list[SearchHit] = []
        seen: set[str] = set()
        for hit in raw_hits:
            if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
                raise BackendProtocolError("OpenSearch returned a malformed search hit")
            score_value = hit.get("_score")
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise BackendProtocolError("OpenSearch returned a non-numeric search score")
            score = float(score_value)
            if not math.isfinite(score):
                raise BackendProtocolError("OpenSearch returned a non-finite search score")
            chunk = self._parse_chunk(hit["_source"])
            if hit.get("_id") != chunk.chunk_id:
                raise BackendProtocolError("OpenSearch hit identity is inconsistent")
            if chunk.chunk_id in seen:
                raise BackendProtocolError("OpenSearch returned a duplicate chunk ID")
            if not self._chunk_matches(chunk, filters):
                raise BackendProtocolError("OpenSearch returned a hit outside mandatory filters")
            seen.add(chunk.chunk_id)
            output.append(SearchHit(chunk=chunk, score=score))
        return tuple(output)

    @staticmethod
    def _extract_hits(payload: Mapping[str, Any]) -> list[Any]:
        timed_out = payload.get("timed_out")
        if timed_out is True:
            raise BackendTimeout("OpenSearch search timed out")
        if not isinstance(timed_out, bool):
            raise BackendProtocolError("OpenSearch returned an invalid timed_out flag")
        shards = payload.get("_shards")
        if not isinstance(shards, dict):
            raise BackendProtocolError("OpenSearch response is missing shard status")
        failed = shards.get("failed")
        if isinstance(failed, bool) or not isinstance(failed, int) or failed != 0:
            raise BackendProtocolError("OpenSearch search had failed shards")
        hits = payload.get("hits")
        if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
            raise BackendProtocolError("OpenSearch response is missing hits")
        return hits["hits"]

    @staticmethod
    def _parse_chunk(source: Mapping[str, Any]) -> IndexedChunk:
        if source.get("doc_kind") != "child":
            raise BackendProtocolError("OpenSearch returned a non-chunk search result")
        metadata = source.get("metadata", {})
        if not isinstance(metadata, dict):
            raise BackendProtocolError("OpenSearch chunk metadata is malformed")
        checksum = _require_string(source.get("checksum_sha256"), "chunk checksum", maximum=64)
        if not _SHA256.fullmatch(checksum):
            raise BackendProtocolError("OpenSearch returned an invalid chunk checksum")
        return IndexedChunk(
            chunk_id=_require_string(source.get("chunk_id"), "chunk ID", maximum=4096),
            parent_id=_require_string(source.get("parent_id"), "parent ID", maximum=4096),
            document_id=_require_string(source.get("document_id"), "document ID", maximum=4096),
            title=_require_string(source.get("title"), "title", maximum=65_536),
            module=_require_string(source.get("module"), "module", maximum=4096),
            version=_require_string(source.get("version"), "version", maximum=4096),
            status=_require_string(source.get("status"), "status", maximum=128),
            section_path=_require_string_tuple(source.get("section_path"), "section path"),
            text=_require_string(source.get("text"), "chunk text", allow_multiline=True),
            token_count=_require_int(source.get("token_count"), "token count", minimum=1),
            chunk_index=_require_int(source.get("chunk_index"), "chunk index", minimum=0),
            access_scopes=_require_string_tuple(source.get("access_scopes"), "access scopes"),
            owner=_require_string(source.get("owner"), "owner", maximum=4096),
            source_type=_require_string(source.get("source_type"), "source type", maximum=4096),
            updated_at=_require_string(source.get("updated_at"), "updated_at", maximum=4096),
            checksum_sha256=checksum,
            parser_version=_require_string(source.get("parser_version"), "parser version", maximum=4096),
            chunker_version=_require_string(source.get("chunker_version"), "chunker version", maximum=4096),
            metadata=metadata,
        )

    @staticmethod
    def _parse_parent(source: Mapping[str, Any]) -> ParentRecord:
        if source.get("doc_kind") != "parent":
            raise BackendProtocolError("OpenSearch returned a non-parent context result")
        metadata = source.get("metadata", {})
        if not isinstance(metadata, dict):
            raise BackendProtocolError("OpenSearch parent metadata is malformed")
        return ParentRecord(
            parent_id=_require_string(source.get("parent_id"), "parent ID", maximum=4096),
            document_id=_require_string(source.get("document_id"), "document ID", maximum=4096),
            title=_require_string(source.get("title"), "title", maximum=65_536),
            module=_require_string(source.get("module"), "module", maximum=4096),
            version=_require_string(source.get("version"), "version", maximum=4096),
            status=_require_string(source.get("status"), "status", maximum=128),
            section_path=_require_string_tuple(source.get("section_path"), "section path"),
            text=_require_string(source.get("text"), "parent text", allow_multiline=True),
            access_scopes=_require_string_tuple(source.get("access_scopes"), "access scopes"),
            owner=_require_string(source.get("owner"), "owner", maximum=4096),
            source_type=_require_string(source.get("source_type"), "source type", maximum=4096),
            updated_at=_require_string(source.get("updated_at"), "updated_at", maximum=4096),
            metadata=metadata,
        )

    def _chunk_matches(self, chunk: IndexedChunk, filters: RetrievalFilters) -> bool:
        if not self._authorized(chunk.access_scopes, filters.access_scopes):
            return False
        if filters.modules and chunk.module not in filters.modules:
            return False
        if filters.required_version is not None and chunk.version != filters.required_version:
            return False
        if chunk.status not in filters.allowed_statuses:
            return False
        return self._metadata_matches(
            chunk.status,
            chunk.source_type,
            chunk.owner,
            chunk.metadata,
            filters,
        )

    def _parent_matches(self, parent: ParentRecord, filters: RetrievalFilters) -> bool:
        if not self._authorized(parent.access_scopes, filters.access_scopes):
            return False
        if filters.modules and parent.module not in filters.modules:
            return False
        if filters.required_version is not None and parent.version != filters.required_version:
            return False
        if parent.status not in filters.allowed_statuses:
            return False
        return self._metadata_matches(
            parent.status,
            parent.source_type,
            parent.owner,
            parent.metadata,
            filters,
        )

    def _metadata_matches(
        self,
        status: str,
        source_type: str,
        owner: str,
        metadata: Mapping[str, Any],
        filters: RetrievalFilters,
    ) -> bool:
        values = {
            **dict(metadata),
            "status": status,
            "source_type": source_type,
            "owner": owner,
        }
        for key, expected in filters.metadata_filters.items():
            if key not in values:
                return False
            actual = values[key]
            if self._metadata_field_map.get(key) == "metadata_terms":
                if self._canonical_metadata_term(key, actual) != self._canonical_metadata_term(
                    key,
                    expected,
                ):
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _authorized(document_scopes: Sequence[str], caller_scopes: Sequence[str]) -> bool:
        scopes = set(document_scopes)
        return "public" in scopes or bool(scopes.intersection(caller_scopes))

    @staticmethod
    def _safe_filter_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)) or not values:
            raise BackendProtocolError(f"{label} are malformed")
        result = tuple(_require_string(value, label, maximum=4096) for value in values)
        if len(set(result)) != len(result):
            raise BackendProtocolError(f"{label} contain duplicates")
        return result

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes)) or not vector:
            raise BackendProtocolError("query embedding is malformed")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BackendProtocolError("query embedding is malformed")
            number = float(value)
            if not math.isfinite(number):
                raise BackendProtocolError("query embedding is malformed")
            values.append(number)
        return tuple(values)

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10_000:
            raise BackendProtocolError("OpenSearch top_k is invalid")
        return top_k

    @staticmethod
    def _canonical_metadata_term(key: str, value: Any) -> str:
        if not isinstance(key, str) or not key or any(ord(character) < 33 for character in key):
            raise BackendProtocolError("metadata filter key is invalid")
        if isinstance(value, str):
            normalized = unicodedata.normalize("NFC", value).strip()
            if not normalized or any(ord(character) < 32 or ord(character) == 127 for character in normalized):
                raise BackendProtocolError("metadata filter string is invalid")
            canonical_value = normalized
        elif isinstance(value, bool):
            canonical_value = "true" if value else "false"
        elif isinstance(value, int):
            canonical_value = str(value)
        elif isinstance(value, float) and math.isfinite(value):
            canonical_value = repr(value)
        else:
            raise BackendProtocolError("metadata filter value is not canonicalizable")
        term = f"{key}={canonical_value}"
        if len(term) > 2048:
            raise BackendProtocolError("canonical metadata filter exceeds the index limit")
        return term

    @staticmethod
    def _optional_string(value: Any, label: str) -> str:
        if not isinstance(value, str) or len(value) > 65_536:
            raise BackendProtocolError(f"OpenSearch returned an invalid {label}")
        return value


__all__ = ["OpenSearchSearchBackend"]
