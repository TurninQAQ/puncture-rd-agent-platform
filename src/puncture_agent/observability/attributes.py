"""Allowlist-based trace attribute sanitization for privacy-safe export.

Production spans must never carry PHI, credentials, unrestricted prompts, or
raw medical payloads. Callers may pass convenience keys; unknown or denied keys
are dropped before export. Explicit denylist matches are replaced with a
stable redaction marker so tests can prove the field was seen and stripped.
"""

from __future__ import annotations

from typing import Any, Mapping

REDACTED = "[REDACTED]"

# Exact keys permitted on exported spans/events. Keep low-cardinality and
# operationally useful. Extend only when a new instrumented boundary needs it.
ATTRIBUTE_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Request / run correlation
        "request_id",
        "run_id",
        "case_id",
        "tenant_id",
        "project_id",
        "session_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "user_id_hash",
        # Graph / agent
        "agent.graph_id",
        "agent.session_id",
        "agent.node_id",
        "agent.node_kind",
        "agent.retry_count",
        "agent.runtime",
        "agent.resumed",
        "agent.streaming",
        "agent.final_status",
        "agent.verification_status",
        "agent.prompt_version",
        "agent.branch_result",
        "agent.step_count",
        "agent.token_usage_total",
        # RAG stages
        "rag.stage",
        "rag.top_k",
        "rag.rewritten_query_hash",
        "rag.query_sanitized",
        "rag.embedding_version",
        "rag.index_version",
        "rag.reranker_version",
        "rag.retrieved_document_ids",
        "rag.retrieved_chunk_ids",
        "rag.retrieved_scores",
        "rag.current_version_hit",
        "rag.acl_violation_count",
        "rag.latency_ms",
        "rag.filter_count",
        # Model
        "model.name",
        "model.version",
        "model.deployment",
        "model.temperature",
        "model.max_tokens",
        "model.input_tokens",
        "model.output_tokens",
        "model.ttft_ms",
        "model.latency_ms",
        "model.structured_output_valid",
        # MCP / tools
        "tool.name",
        "tool.version",
        "tool.call_id",
        "tool.request_id",
        "tool.classification",
        "tool.schema_version",
        "tool.response_status",
        "tool.error_code",
        "tool.latency_ms",
        "tool.attempt",
        "tool.artifact_ids",
        "tool.artifact_checksums",
        "mcp.server",
        "mcp.method",
        # Verifier / checkpoint
        "verifier.result",
        "verifier.reason_code",
        "checkpoint.id",
        "checkpoint.thread_id",
        "checkpoint.namespace",
        "checkpoint.resumed",
        # Generic operational
        "error.type",
        "error.code",
        "error.retryable",
        "http.method",
        "http.route",
        "http.status_code",
        "rpc.system",
        "rpc.service",
        "rpc.method",
        "rpc.grpc.status_code",
        "component",
        "outcome",
        "duration_ms",
        "status",
        "schema_version",
        "event.name",
    }
)

# Exact sensitive keys that should be redacted (not merely dropped) when seen.
DENYLIST_EXACT: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "bearer",
        "cookie",
        "set-cookie",
        "patient_name",
        "patient_id",
        "phi",
        "ssn",
        "dicom_tags",
        "voxel_data",
        "raw_image",
        "image_bytes",
        "pixel_data",
        "prompt",
        "system_prompt",
        "full_prompt",
        "messages",
        "internal_uri",
        "storage_uri",
        "credential",
        "jwt",
        "authorization_header",
    }
)

# Substring markers that force redaction for non-allowlisted keys only.
_DENY_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "patient",
    "phi",
    "ssn",
    "dicom",
    "voxel",
    "raw_image",
    "image_bytes",
    "pixel_data",
    "full_prompt",
    "system_prompt",
    "internal_uri",
    "storage_uri",
    "credential",
    "refresh_token",
)


def is_denied_key(key: str) -> bool:
    """Return True when a key must not leave the process unredacted.

    Exact allowlisted operational keys are never denied (for example
    ``agent.prompt_version``). Unknown keys that look sensitive are denied so
    they are redacted rather than silently dropped without evidence.
    """

    if not isinstance(key, str) or not key:
        return False
    if key in ATTRIBUTE_ALLOWLIST:
        return False
    lowered = key.lower().replace("-", "_")
    if lowered in DENYLIST_EXACT or key in DENYLIST_EXACT:
        return True
    return any(part in lowered for part in _DENY_SUBSTRINGS)


def is_allowed_key(key: str) -> bool:
    if not isinstance(key, str) or not key:
        return False
    if is_denied_key(key):
        return False
    return key in ATTRIBUTE_ALLOWLIST


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # Hard cap: keep attribute payloads small and free of bulk blobs.
        if len(value) > 512:
            return value[:512] + "…[truncated]"
        return value
    if isinstance(value, bytes):
        return f"[bytes:{len(value)}]"
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        # Nested maps under an allowed parent are filtered recursively.
        return sanitize_attributes(value, depth=depth + 1)
    return str(value)[:256]


def sanitize_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    depth: int = 0,
) -> dict[str, Any]:
    """Return a privacy-safe attribute dict.

    - Denylisted keys are retained as ``REDACTED`` so exporters/tests can prove
      the sensitive field was observed and stripped.
    - Unknown keys are dropped (strict allowlist at top level and nested).
    - Nested dicts under allowed keys are filtered recursively.
    """

    if not attributes:
        return {}
    if depth > 4:
        return {}

    cleaned: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue
        if is_denied_key(key):
            cleaned[key] = REDACTED
            continue
        if key not in ATTRIBUTE_ALLOWLIST:
            continue
        cleaned[key] = _sanitize_value(value, depth=depth)
    return cleaned


def hash_query_text(text: str, *, salt: str = "puncture-rag-query") -> str:
    """Stable, non-reversible fingerprint for rewritten queries in traces."""

    import hashlib

    payload = f"{salt}:{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:32]
