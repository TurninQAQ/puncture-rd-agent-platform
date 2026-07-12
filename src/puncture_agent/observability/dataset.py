"""Versioned offline evaluation dataset loader and validator.

Supported formats:

* JSON object: ``{"dataset_version": "...", "schema_version": "...", "cases": [...]}``
* JSON array of cases (schema/dataset version taken from loader kwargs/defaults)
* JSONL: one case object per line; optional leading ``# meta: {...}`` comment

Schema version ``eval-case-v1`` matches the production ``EvalCase`` fields plus
optional RAG/tool/recovery extensions documented in ``specs/eval-and-tracing.md``.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from puncture_agent.observability.eval_harness import EvalCase

DATASET_SCHEMA_VERSION = "eval-case-v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({DATASET_SCHEMA_VERSION})

_REQUIRED_CASE_KEYS = frozenset(
    {
        "case_id",
        "query",
        "expected_task_type",
        "expected_terminal_status",
    }
)

_KNOWN_CASE_KEYS = frozenset(
    {
        "case_id",
        "query",
        "expected_task_type",
        "expected_terminal_status",
        "required_nodes",
        "forbidden_nodes",
        "expected_tools",
        "forbidden_tools",
        "minimum_citations",
        "agent_case_id",
        "metadata",
        "planning_constraints",
        # Optional production extensions
        "expected_relevant_document_ids",
        "expected_relevant_chunk_ids",
        "expected_document_version",
        "expected_tool_argument_predicates",
        "expected_error_code",
        "max_steps",
        "max_retries",
        "security_policy",
        "is_recovery_case",
        "expected_retry_count_min",
        "expected_retry_count_max",
        "expect_no_answer",
        "tags",
        "notes",
    }
)


class DatasetValidationError(ValueError):
    """Raised when a dataset file fails schema or version checks."""


def _as_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise DatasetValidationError(f"{field_name} must be a list, not a string")
    if not isinstance(value, Sequence):
        raise DatasetValidationError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetValidationError(f"{field_name} items must be non-empty strings")
        result.append(item)
    return tuple(result)


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{field_name} must be an object")
    return dict(value)


def validate_case_payload(payload: Mapping[str, Any], *, index: int | None = None) -> dict[str, Any]:
    """Validate and normalize a raw case mapping."""

    prefix = f"case[{index}]" if index is not None else "case"
    if not isinstance(payload, Mapping):
        raise DatasetValidationError(f"{prefix} must be an object")

    unknown = set(payload) - _KNOWN_CASE_KEYS
    if unknown:
        raise DatasetValidationError(
            f"{prefix} contains unknown fields: {sorted(unknown)}"
        )

    missing = _REQUIRED_CASE_KEYS - set(payload)
    if missing:
        raise DatasetValidationError(
            f"{prefix} missing required fields: {sorted(missing)}"
        )

    case_id = payload["case_id"]
    query = payload["query"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetValidationError(f"{prefix}.case_id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise DatasetValidationError(f"{prefix}.query must be a non-empty string")

    for key in ("expected_task_type", "expected_terminal_status"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"{prefix}.{key} must be a non-empty string")

    minimum_citations = payload.get("minimum_citations", 0)
    if not isinstance(minimum_citations, int) or isinstance(minimum_citations, bool):
        raise DatasetValidationError(f"{prefix}.minimum_citations must be an int")
    if minimum_citations < 0:
        raise DatasetValidationError(f"{prefix}.minimum_citations must be >= 0")

    agent_case_id = payload.get("agent_case_id")
    if agent_case_id is not None and (
        not isinstance(agent_case_id, str) or not agent_case_id.strip()
    ):
        raise DatasetValidationError(
            f"{prefix}.agent_case_id must be a non-empty string when set"
        )

    predicates = payload.get("expected_tool_argument_predicates", [])
    if predicates is None:
        predicates = []
    if not isinstance(predicates, list):
        raise DatasetValidationError(
            f"{prefix}.expected_tool_argument_predicates must be a list"
        )
    for pred_index, predicate in enumerate(predicates):
        if not isinstance(predicate, Mapping):
            raise DatasetValidationError(
                f"{prefix}.expected_tool_argument_predicates[{pred_index}] must be an object"
            )
        if "tool_name" not in predicate:
            raise DatasetValidationError(
                f"{prefix}.expected_tool_argument_predicates[{pred_index}].tool_name is required"
            )

    normalized = {
        "case_id": case_id.strip(),
        "query": query,
        "expected_task_type": payload["expected_task_type"],
        "expected_terminal_status": payload["expected_terminal_status"],
        "required_nodes": _as_tuple(
            payload.get("required_nodes"), field_name=f"{prefix}.required_nodes"
        ),
        "forbidden_nodes": _as_tuple(
            payload.get("forbidden_nodes"), field_name=f"{prefix}.forbidden_nodes"
        ),
        "expected_tools": _as_tuple(
            payload.get("expected_tools"), field_name=f"{prefix}.expected_tools"
        ),
        "forbidden_tools": _as_tuple(
            payload.get("forbidden_tools"), field_name=f"{prefix}.forbidden_tools"
        ),
        "minimum_citations": minimum_citations,
        "agent_case_id": agent_case_id.strip() if isinstance(agent_case_id, str) else None,
        "metadata": _as_dict(payload.get("metadata"), field_name=f"{prefix}.metadata"),
        "planning_constraints": _as_dict(
            payload.get("planning_constraints"),
            field_name=f"{prefix}.planning_constraints",
        ),
        "expected_relevant_document_ids": _as_tuple(
            payload.get("expected_relevant_document_ids"),
            field_name=f"{prefix}.expected_relevant_document_ids",
        ),
        "expected_relevant_chunk_ids": _as_tuple(
            payload.get("expected_relevant_chunk_ids"),
            field_name=f"{prefix}.expected_relevant_chunk_ids",
        ),
        "expected_document_version": payload.get("expected_document_version"),
        "expected_tool_argument_predicates": [dict(item) for item in predicates],
        "expected_error_code": payload.get("expected_error_code"),
        "max_steps": payload.get("max_steps"),
        "max_retries": payload.get("max_retries"),
        "security_policy": payload.get("security_policy"),
        "is_recovery_case": bool(payload.get("is_recovery_case", False)),
        "expected_retry_count_min": payload.get("expected_retry_count_min"),
        "expected_retry_count_max": payload.get("expected_retry_count_max"),
        "expect_no_answer": bool(payload.get("expect_no_answer", False)),
        "tags": _as_tuple(payload.get("tags"), field_name=f"{prefix}.tags"),
        "notes": payload.get("notes"),
    }

    version = normalized["expected_document_version"]
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise DatasetValidationError(
            f"{prefix}.expected_document_version must be a non-empty string when set"
        )
    return normalized


def case_from_payload(payload: Mapping[str, Any]) -> EvalCase:
    """Build an :class:`EvalCase` from a validated payload."""

    normalized = (
        payload
        if "case_id" in payload and "required_nodes" in payload and isinstance(payload.get("required_nodes"), tuple)
        else validate_case_payload(payload)
    )
    # Only pass constructor fields known to EvalCase; extensions ride in metadata.
    extension_keys = (
        "forbidden_tools",
        "expected_relevant_document_ids",
        "expected_relevant_chunk_ids",
        "expected_document_version",
        "expected_tool_argument_predicates",
        "expected_error_code",
        "max_steps",
        "max_retries",
        "security_policy",
        "is_recovery_case",
        "expected_retry_count_min",
        "expected_retry_count_max",
        "expect_no_answer",
        "tags",
        "notes",
    )
    metadata = dict(normalized.get("metadata") or {})
    extensions = {
        key: normalized[key]
        for key in extension_keys
        if key in normalized and normalized[key] not in (None, (), [], False)
    }
    if extensions:
        # Nested under a reserved key so runtime mock flags stay flat.
        metadata.setdefault("eval_extensions", {}).update(extensions)
        # Promote recovery flags used by the harness without colliding with
        # tool-fault injection keys.
        if "is_recovery_case" in extensions:
            metadata["is_recovery_case"] = extensions["is_recovery_case"]
        if "expected_tool_argument_predicates" in extensions:
            metadata["expected_tool_argument_predicates"] = extensions[
                "expected_tool_argument_predicates"
            ]
        if "forbidden_tools" in extensions:
            metadata["forbidden_tools"] = list(extensions["forbidden_tools"])
        if "expected_relevant_document_ids" in extensions:
            metadata["expected_relevant_document_ids"] = list(
                extensions["expected_relevant_document_ids"]
            )
        if "expected_document_version" in extensions:
            metadata["expected_document_version"] = extensions[
                "expected_document_version"
            ]
        if "expected_error_code" in extensions:
            metadata["expected_error_code"] = extensions["expected_error_code"]
        if "expected_retry_count_min" in extensions:
            metadata["expected_retry_count_min"] = extensions["expected_retry_count_min"]
        if "expected_retry_count_max" in extensions:
            metadata["expected_retry_count_max"] = extensions["expected_retry_count_max"]

    return EvalCase(
        case_id=str(normalized["case_id"]),
        query=str(normalized["query"]),
        expected_task_type=str(normalized["expected_task_type"]),
        expected_terminal_status=str(normalized["expected_terminal_status"]),
        required_nodes=tuple(normalized.get("required_nodes") or ()),
        forbidden_nodes=tuple(normalized.get("forbidden_nodes") or ()),
        expected_tools=tuple(normalized.get("expected_tools") or ()),
        minimum_citations=int(normalized.get("minimum_citations") or 0),
        agent_case_id=normalized.get("agent_case_id"),
        metadata=metadata,
        planning_constraints=dict(normalized.get("planning_constraints") or {}),
    )


def _load_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"invalid JSON: {exc}") from exc


def _parse_jsonl(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.lower().startswith("meta:"):
                meta_payload = body[5:].strip()
                try:
                    parsed_meta = json.loads(meta_payload)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(
                        f"invalid JSONL meta on line {line_no}: {exc}"
                    ) from exc
                if not isinstance(parsed_meta, Mapping):
                    raise DatasetValidationError(
                        f"JSONL meta on line {line_no} must be an object"
                    )
                meta.update(dict(parsed_meta))
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"invalid JSONL on line {line_no}: {exc}"
            ) from exc
        if not isinstance(item, Mapping):
            raise DatasetValidationError(
                f"JSONL line {line_no} must be a case object"
            )
        cases.append(dict(item))
    return meta, cases


def load_eval_dataset(
    path: str | Path,
    *,
    schema_version: str | None = None,
    dataset_version: str | None = None,
) -> tuple[str, str, list[EvalCase]]:
    """Load and validate a versioned evaluation dataset.

    Returns ``(schema_version, dataset_version, cases)``.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetValidationError(f"dataset file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise DatasetValidationError("dataset file is empty")

    suffix = file_path.suffix.lower()
    meta: dict[str, Any] = {}
    raw_cases: list[dict[str, Any]]

    if suffix == ".jsonl":
        meta, raw_cases = _parse_jsonl(text)
    else:
        payload = _load_json_text(text)
        if isinstance(payload, list):
            raw_cases = []
            for index, item in enumerate(payload):
                if not isinstance(item, Mapping):
                    raise DatasetValidationError(
                        f"cases[{index}] must be an object"
                    )
                raw_cases.append(dict(item))
        elif isinstance(payload, Mapping):
            meta = {
                key: payload[key]
                for key in ("schema_version", "dataset_version", "name", "description")
                if key in payload
            }
            cases_value = payload.get("cases")
            if cases_value is None:
                raise DatasetValidationError("dataset object requires a 'cases' array")
            if not isinstance(cases_value, list):
                raise DatasetValidationError("'cases' must be an array")
            raw_cases = []
            for index, item in enumerate(cases_value):
                if not isinstance(item, Mapping):
                    raise DatasetValidationError(
                        f"cases[{index}] must be an object"
                    )
                raw_cases.append(dict(item))
        else:
            raise DatasetValidationError(
                "dataset root must be an object or array of cases"
            )

    resolved_schema = (
        schema_version
        or meta.get("schema_version")
        or DATASET_SCHEMA_VERSION
    )
    if resolved_schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise DatasetValidationError(
            f"unsupported schema_version {resolved_schema!r}; "
            f"supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    resolved_dataset = (
        dataset_version
        or meta.get("dataset_version")
        or file_path.stem
    )
    if not isinstance(resolved_dataset, str) or not resolved_dataset.strip():
        raise DatasetValidationError("dataset_version must be a non-empty string")

    if not raw_cases:
        raise DatasetValidationError("dataset contains no cases")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        validated = validate_case_payload(raw, index=index)
        case_id = validated["case_id"]
        if case_id in seen_ids:
            raise DatasetValidationError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        cases.append(case_from_payload(validated))
    return str(resolved_schema), str(resolved_dataset), cases


def dump_eval_dataset(
    cases: Sequence[EvalCase],
    path: str | Path,
    *,
    schema_version: str = DATASET_SCHEMA_VERSION,
    dataset_version: str = "unversioned",
    jsonl: bool = False,
) -> None:
    """Write cases to JSON or JSONL for fixtures and regression pins."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    for case in cases:
        item = {
            "case_id": case.case_id,
            "query": case.query,
            "expected_task_type": case.expected_task_type,
            "expected_terminal_status": case.expected_terminal_status,
            "required_nodes": list(case.required_nodes),
            "forbidden_nodes": list(case.forbidden_nodes),
            "expected_tools": list(case.expected_tools),
            "minimum_citations": case.minimum_citations,
            "agent_case_id": case.agent_case_id,
            "metadata": dict(case.metadata),
            "planning_constraints": dict(case.planning_constraints),
        }
        payloads.append(item)

    if jsonl:
        lines = [
            f"# meta: {json.dumps({'schema_version': schema_version, 'dataset_version': dataset_version}, ensure_ascii=False)}"
        ]
        for item in payloads:
            lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    document = {
        "schema_version": schema_version,
        "dataset_version": dataset_version,
        "cases": payloads,
    }
    file_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
