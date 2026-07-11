#!/usr/bin/env python3
"""Render and validate the versioned OpenSearch RAG index contract."""

from __future__ import annotations

import copy
import json
import pathlib
from dataclasses import dataclass
from typing import Any

from http_utils import ServiceError, validate_identifier


MAX_EMBEDDING_DIMENSION = 16_000


@dataclass(frozen=True)
class IndexContract:
    schema_version: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    query_instruction: str
    document_instruction: str
    vectors_normalized: bool
    tokenizer_revision: str
    max_input_tokens: int
    parser_version: str
    chunker_version: str

    def validate(self) -> None:
        for label, value in (
            ("schema version", self.schema_version),
            ("embedding model", self.embedding_model),
            ("embedding revision", self.embedding_revision),
            ("tokenizer revision", self.tokenizer_revision),
            ("parser version", self.parser_version),
            ("chunker version", self.chunker_version),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{label} must be a string")
            normalized = value.strip()
            placeholder = normalized.upper()
            if (
                not normalized
                or placeholder in {"SET_ME", "SET_DURING_BOOTSTRAP", "LATEST", "MAIN", "UNSPECIFIED"}
                or placeholder.startswith("SET_")
            ):
                raise ValueError(f"{label} must be an explicit immutable release value")
            if len(normalized) > 4096 or any(ord(character) < 32 for character in normalized):
                raise ValueError(f"{label} contains unsupported characters")
        for label, value in (
            ("query instruction", self.query_instruction),
            ("document instruction", self.document_instruction),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{label} must be a string")
            if len(value) > 8192 or any(
                ord(character) < 32 and character not in "\t\n\r" for character in value
            ):
                raise ValueError(f"{label} contains unsupported characters")
        if not isinstance(self.vectors_normalized, bool):
            raise ValueError("vectors_normalized must be a boolean")
        if (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or not 1 <= self.embedding_dimension <= MAX_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                f"embedding dimension must be between 1 and {MAX_EMBEDDING_DIMENSION}"
            )
        if (
            isinstance(self.max_input_tokens, bool)
            or not isinstance(self.max_input_tokens, int)
            or self.max_input_tokens <= 0
        ):
            raise ValueError("max_input_tokens must be a positive integer")


def load_template(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("index template is not readable valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("index template root must be an object")
    return value


def render_template(template: dict[str, Any], contract: IndexContract) -> dict[str, Any]:
    contract.validate()
    rendered = copy.deepcopy(template)
    try:
        template_meta = rendered["_meta"]
        mappings = rendered["template"]["mappings"]
        mapping_meta = mappings["_meta"]
        embedding = mappings["properties"]["embedding"]
    except (KeyError, TypeError) as exc:
        raise ValueError("index template is missing required contract fields") from exc
    values: dict[str, Any] = {
        "schema_version": contract.schema_version,
        "embedding_model": contract.embedding_model,
        "embedding_revision": contract.embedding_revision,
        "embedding_dimension": contract.embedding_dimension,
        "query_instruction": contract.query_instruction,
        "document_instruction": contract.document_instruction,
        "vectors_normalized": contract.vectors_normalized,
        "tokenizer_revision": contract.tokenizer_revision,
        "max_input_tokens": contract.max_input_tokens,
        "parser_version": contract.parser_version,
        "chunker_version": contract.chunker_version,
    }
    template_meta.update(values)
    mapping_meta.update(values)
    embedding["dimension"] = contract.embedding_dimension
    return rendered


def mapping_for_index(response: dict[str, Any], index_name: str) -> dict[str, Any]:
    value = response.get(index_name)
    if not isinstance(value, dict):
        raise ServiceError("mapping response does not contain the expected concrete index")
    mappings = value.get("mappings")
    if not isinstance(mappings, dict):
        raise ServiceError("mapping response is missing mappings")
    return mappings


def validate_mapping(mappings: dict[str, Any], contract: IndexContract) -> None:
    contract.validate()
    metadata = mappings.get("_meta")
    properties = mappings.get("properties")
    if not isinstance(metadata, dict) or not isinstance(properties, dict):
        raise ServiceError("active index mapping is missing metadata or properties")
    expected: dict[str, Any] = {
        "schema_version": contract.schema_version,
        "embedding_model": contract.embedding_model,
        "embedding_revision": contract.embedding_revision,
        "embedding_dimension": contract.embedding_dimension,
        "query_instruction": contract.query_instruction,
        "document_instruction": contract.document_instruction,
        "vectors_normalized": contract.vectors_normalized,
        "tokenizer_revision": contract.tokenizer_revision,
        "max_input_tokens": contract.max_input_tokens,
        "parser_version": contract.parser_version,
        "chunker_version": contract.chunker_version,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ServiceError(f"active index mapping metadata mismatch: {field}")
    embedding = properties.get("embedding")
    if not isinstance(embedding, dict):
        raise ServiceError("active index mapping has no embedding field")
    if embedding.get("type") != "knn_vector" or embedding.get("dimension") != contract.embedding_dimension:
        raise ServiceError("active index embedding mapping is incompatible")
    method = embedding.get("method")
    if not isinstance(method, dict):
        raise ServiceError("active index embedding mapping has no method")
    if (
        method.get("name") != "hnsw"
        or method.get("engine") != "lucene"
        or method.get("space_type") != "cosinesimil"
    ):
        raise ServiceError("active index vector method is incompatible")
    required_keywords = {
        "doc_kind",
        "document_id",
        "chunk_id",
        "owner",
        "module",
        "version",
        "status",
        "access_scopes",
        "embedding_model",
        "embedding_revision",
        "parser_version",
        "chunker_version",
        "metadata_terms",
    }
    missing = sorted(
        field
        for field in required_keywords
        if not isinstance(properties.get(field), dict)
        or properties[field].get("type") != "keyword"
    )
    if missing:
        raise ServiceError(f"active index mapping is missing required keyword fields: {missing}")
    if not isinstance(properties.get("text"), dict) or properties["text"].get("type") != "text":
        raise ServiceError("active index mapping is missing the BM25 text field")


def alias_indexes(response: dict[str, Any], alias: str) -> tuple[str, ...]:
    validate_identifier(alias, label="alias")
    indexes: list[str] = []
    for index_name, body in response.items():
        if not isinstance(index_name, str) or not isinstance(body, dict):
            raise ServiceError("alias response is malformed")
        aliases = body.get("aliases")
        if not isinstance(aliases, dict) or alias not in aliases:
            raise ServiceError("alias response does not contain the requested alias")
        validate_identifier(index_name, label="concrete index", versioned_index=True)
        indexes.append(index_name)
    return tuple(sorted(indexes))


def mandatory_filters(access_scope: str = "__rag_health_probe__") -> list[dict[str, Any]]:
    if not access_scope or any(character in access_scope for character in "\r\n"):
        raise ValueError("access scope is invalid")
    return [
        {"term": {"doc_kind": "child"}},
        {"term": {"status": "active"}},
        {"terms": {"access_scopes": [access_scope]}},
    ]
