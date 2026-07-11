"""HTTP adapters for Qwen embeddings and vLLM-compatible reranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .backends import (
    BackendProtocolError,
    BackendTimeout,
    EmbeddingUnavailable,
    IndexIncompatible,
    IndexedChunk,
    RerankResult,
    RerankerUnavailable,
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
    validate_request_path,
)


def _require_text(value: str, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    if any(ord(character) < 32 and character not in "\t\n" for character in normalized):
        raise ValueError(f"{label} contains unsupported control characters")
    return normalized


def _validate_positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _format_input(text: str, instruction: str, label: str) -> str:
    value = _require_text(text, label, maximum=1_000_000)
    if not instruction:
        return value
    prefix = "Query" if label == "query" else "Document"
    return f"Instruct: {instruction}\n{prefix}: {value}"


def _validate_http_status(
    response: ProviderHttpResponse,
    *,
    unavailable_type: type[EmbeddingUnavailable] | type[RerankerUnavailable],
    provider_label: str,
) -> None:
    status = response.status
    if 200 <= status < 300:
        return
    if status in {408, 504}:
        raise BackendTimeout(f"{provider_label} request timed out")
    if status == 429 or 500 <= status <= 599:
        raise unavailable_type(f"{provider_label} endpoint is unavailable")
    if status in {401, 403}:
        raise BackendProtocolError(f"{provider_label} authentication was rejected")
    raise BackendProtocolError(f"{provider_label} returned an unexpected HTTP status")


def _translate_transport_error(
    exc: BaseException,
    *,
    unavailable_type: type[EmbeddingUnavailable] | type[RerankerUnavailable],
    provider_label: str,
) -> BaseException:
    if isinstance(exc, ProviderTimeoutError):
        return BackendTimeout(f"{provider_label} request timed out")
    if isinstance(exc, ProviderUnavailableError):
        return unavailable_type(f"{provider_label} endpoint is unavailable")
    if isinstance(exc, (ProviderSecurityError, ProviderProtocolError)):
        return BackendProtocolError(f"{provider_label} transport or protocol validation failed")
    return exc


class OpenAIEmbeddingBackend:
    """OpenAI-compatible ``/v1/embeddings`` adapter for a private Qwen service."""

    def __init__(
        self,
        endpoint: ProviderEndpoint,
        *,
        model_name: str,
        revision: str,
        dimension: int,
        query_instruction: str,
        document_instruction: str = "",
        tokenizer_revision: str = "unspecified",
        max_input_tokens: int = 8192,
        normalize_vectors: bool = True,
        api_path: str = "/v1/embeddings",
        max_batch_size: int = 128,
        max_input_characters: int = 131_072,
        transport: ProviderHttpTransport | None = None,
    ) -> None:
        self._model_name = _require_text(model_name, "embedding model_name")
        self._revision = _require_text(revision, "embedding revision")
        self._dimension = _validate_positive_int(dimension, "embedding dimension")
        self._query_instruction = _require_text(
            query_instruction,
            "query_instruction",
            maximum=4096,
        )
        if document_instruction:
            document_instruction = _require_text(
                document_instruction,
                "document_instruction",
                maximum=4096,
            )
        self._document_instruction = document_instruction
        self._tokenizer_revision = _require_text(tokenizer_revision, "tokenizer_revision")
        self._max_input_tokens = _validate_positive_int(max_input_tokens, "max_input_tokens")
        if not isinstance(normalize_vectors, bool):
            raise ValueError("normalize_vectors must be a boolean")
        self._normalize_vectors = normalize_vectors
        self._api_path = validate_request_path(api_path)
        self._max_batch_size = _validate_positive_int(max_batch_size, "max_batch_size")
        self._max_input_characters = _validate_positive_int(
            max_input_characters,
            "max_input_characters",
        )
        self._transport = transport or HttpxProviderTransport(endpoint)
        self._owns_transport = transport is None
        if self._transport.endpoint != endpoint:
            raise ValueError("embedding transport endpoint does not match the configured endpoint")

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
        return self._query_instruction

    @property
    def document_instruction(self) -> str:
        return self._document_instruction

    @property
    def vectors_normalized(self) -> bool:
        return self._normalize_vectors

    @property
    def tokenizer_revision(self) -> str:
        return self._tokenizer_revision

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()

    def embed_query(self, text: str) -> Sequence[float]:
        formatted = _format_input(text, self._query_instruction, "query")
        return self._embed((formatted,))[0]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if isinstance(texts, (str, bytes)):
            raise ValueError("embedding documents must be a sequence of strings")
        values = tuple(texts)
        if not values:
            return ()
        if len(values) > self._max_batch_size:
            raise ValueError("embedding document batch exceeds the configured limit")
        formatted = tuple(
            _format_input(text, self._document_instruction, "document")
            for text in values
        )
        return self._embed(formatted)

    def _embed(self, inputs: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if any(len(value) > self._max_input_characters for value in inputs):
            raise ValueError("embedding input exceeds the configured character limit")
        try:
            response = self._transport.request(
                "POST",
                self._api_path,
                json_body={
                    "model": self._model_name,
                    "input": list(inputs),
                    "encoding_format": "float",
                },
                timeout_seconds=self._transport.endpoint.timeout_seconds,
            )
        except Exception as exc:
            translated = _translate_transport_error(
                exc,
                unavailable_type=EmbeddingUnavailable,
                provider_label="embedding provider",
            )
            if translated is exc:
                raise
            raise translated from exc
        _validate_http_status(
            response,
            unavailable_type=EmbeddingUnavailable,
            provider_label="embedding provider",
        )
        try:
            payload = decode_json_response(
                response,
                max_bytes=self._transport.endpoint.max_response_bytes,
            )
        except ProviderProtocolError as exc:
            raise BackendProtocolError("embedding provider returned invalid JSON") from exc
        if payload.get("model") != self._model_name:
            raise IndexIncompatible("embedding provider returned a different model identity")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise BackendProtocolError("embedding provider returned the wrong result count")
        by_index: dict[int, tuple[float, ...]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise BackendProtocolError("embedding provider returned a malformed result")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(inputs):
                raise BackendProtocolError("embedding provider returned an invalid result index")
            if index in by_index:
                raise BackendProtocolError("embedding provider returned a duplicate result index")
            by_index[index] = self._validate_vector(item.get("embedding"))
        if set(by_index) != set(range(len(inputs))):
            raise BackendProtocolError("embedding provider omitted a result index")
        return tuple(by_index[index] for index in range(len(inputs)))

    def _validate_vector(self, value: Any) -> tuple[float, ...]:
        if not isinstance(value, list) or len(value) != self._dimension:
            raise IndexIncompatible("embedding provider dimension does not match configuration")
        values: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise BackendProtocolError("embedding provider returned a non-numeric vector")
            number = float(item)
            if not math.isfinite(number):
                raise BackendProtocolError("embedding provider returned a non-finite vector")
            values.append(number)
        if not self._normalize_vectors:
            return tuple(values)
        norm = math.sqrt(sum(number * number for number in values))
        if not math.isfinite(norm) or norm <= 0.0:
            raise BackendProtocolError("embedding provider returned a zero or invalid vector")
        return tuple(number / norm for number in values)


class VllmRerankerBackend:
    """Strict adapter for vLLM's Cohere/Jina-compatible ``/v1/rerank`` API."""

    def __init__(
        self,
        endpoint: ProviderEndpoint,
        *,
        model_name: str,
        revision: str,
        query_instruction: str = "",
        api_path: str = "/v1/rerank",
        max_candidates: int = 128,
        max_query_characters: int = 131_072,
        max_document_characters: int = 262_144,
        transport: ProviderHttpTransport | None = None,
    ) -> None:
        self._model_name = _require_text(model_name, "reranker model_name")
        self._revision = _require_text(revision, "reranker revision")
        if query_instruction:
            query_instruction = _require_text(query_instruction, "reranker query_instruction")
        self._query_instruction = query_instruction
        self._api_path = validate_request_path(api_path)
        self._max_candidates = _validate_positive_int(max_candidates, "max_candidates")
        self._max_query_characters = _validate_positive_int(
            max_query_characters,
            "max_query_characters",
        )
        self._max_document_characters = _validate_positive_int(
            max_document_characters,
            "max_document_characters",
        )
        self._transport = transport or HttpxProviderTransport(endpoint)
        self._owns_transport = transport is None
        if self._transport.endpoint != endpoint:
            raise ValueError("reranker transport endpoint does not match the configured endpoint")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()

    def rerank(self, query: str, chunks: Sequence[IndexedChunk]) -> Sequence[RerankResult]:
        if isinstance(chunks, (str, bytes)):
            raise ValueError("reranker chunks must be a sequence")
        candidates = tuple(chunks)
        if not candidates:
            return ()
        if len(candidates) > self._max_candidates:
            raise ValueError("reranker candidate count exceeds the configured limit")
        if any(not isinstance(chunk, IndexedChunk) for chunk in candidates):
            raise ValueError("reranker candidates must be IndexedChunk values")
        chunk_ids = [chunk.chunk_id for chunk in candidates]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("reranker candidate chunk IDs must be unique")
        formatted_query = _format_input(query, self._query_instruction, "query")
        if len(formatted_query) > self._max_query_characters:
            raise ValueError("reranker query exceeds the configured limit")
        documents: list[str] = []
        for chunk in candidates:
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ValueError("reranker candidate text must be non-empty")
            if len(chunk.text) > self._max_document_characters:
                raise ValueError("reranker document exceeds the configured limit")
            documents.append(chunk.text)
        try:
            response = self._transport.request(
                "POST",
                self._api_path,
                json_body={
                    "model": self._model_name,
                    "query": formatted_query,
                    "documents": documents,
                    "top_n": len(documents),
                },
                timeout_seconds=self._transport.endpoint.timeout_seconds,
            )
        except Exception as exc:
            translated = _translate_transport_error(
                exc,
                unavailable_type=RerankerUnavailable,
                provider_label="reranker provider",
            )
            if translated is exc:
                raise
            raise translated from exc
        _validate_http_status(
            response,
            unavailable_type=RerankerUnavailable,
            provider_label="reranker provider",
        )
        try:
            payload = decode_json_response(
                response,
                max_bytes=self._transport.endpoint.max_response_bytes,
            )
        except ProviderProtocolError as exc:
            raise BackendProtocolError("reranker provider returned invalid JSON") from exc
        if payload.get("model") != self._model_name:
            raise BackendProtocolError("reranker provider returned a different model identity")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(candidates):
            raise BackendProtocolError("reranker provider did not return every candidate")
        seen: set[int] = set()
        output: list[RerankResult] = []
        for item in results:
            if not isinstance(item, dict):
                raise BackendProtocolError("reranker provider returned a malformed result")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidates):
                raise BackendProtocolError("reranker provider returned an invalid candidate index")
            if index in seen:
                raise BackendProtocolError("reranker provider returned a duplicate candidate index")
            seen.add(index)
            score_value = item.get("relevance_score")
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise BackendProtocolError("reranker provider returned a non-numeric score")
            score = float(score_value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise BackendProtocolError("reranker provider score is outside [0, 1]")
            returned_document = item.get("document")
            if returned_document is not None:
                if (
                    not isinstance(returned_document, dict)
                    or returned_document.get("text") != documents[index]
                ):
                    raise BackendProtocolError("reranker provider returned mismatched document content")
            output.append(RerankResult(chunk_id=candidates[index].chunk_id, score=score))
        if seen != set(range(len(candidates))):
            raise BackendProtocolError("reranker provider omitted a candidate index")
        return tuple(output)


__all__ = ["OpenAIEmbeddingBackend", "VllmRerankerBackend"]
