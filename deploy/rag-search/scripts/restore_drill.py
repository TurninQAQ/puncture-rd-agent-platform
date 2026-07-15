#!/usr/bin/env python3
"""Restore a snapshot into an isolated index, verify it, and delete the restore."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
from typing import Any

from http_utils import (
    ServiceError,
    connection_from_environment,
    request_json_with,
    validate_identifier,
)
from index_contract import alias_indexes


RESTORE_INDEX_PREFIX = "puncture-rag-restore-"


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def validate_restore_index(value: str) -> str:
    value = validate_identifier(value, label="restore index")
    if not value.startswith(RESTORE_INDEX_PREFIX) or len(value) <= len(RESTORE_INDEX_PREFIX):
        raise ValueError(
            f"restore index must use the isolated prefix {RESTORE_INDEX_PREFIX!r}"
        )
    return value


def mapping_for(response: dict[str, Any], index: str) -> dict[str, Any]:
    index_payload = response.get(index)
    mapping = index_payload.get("mappings") if isinstance(index_payload, dict) else None
    if not isinstance(mapping, dict):
        raise ServiceError("mapping response is malformed")
    return mapping


def count_for(response: dict[str, Any]) -> int:
    count = response.get("count")
    shards = response.get("_shards")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or not isinstance(shards, dict)
        or shards.get("failed") != 0
    ):
        raise ServiceError("count response is malformed or has shard failures")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--restore-index", required=True)
    parser.add_argument(
        "--read-alias", default=os.getenv("RAG_READ_ALIAS", "puncture-rag-read")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    restore_index = ""
    config = None
    cleanup_required = False
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    cleanup_failure: Exception | None = None
    try:
        repository = validate_identifier(args.repository, label="snapshot repository")
        snapshot = validate_identifier(args.snapshot, label="snapshot")
        restore_index = validate_restore_index(args.restore_index)
        read_alias = validate_identifier(args.read_alias, label="read alias")
        config = connection_from_environment()

        _, alias_response = request_json_with(
            config, "GET", f"/_alias/{encoded(read_alias)}", allowed_statuses=(200,)
        )
        source_indexes = alias_indexes(alias_response, read_alias)
        if len(source_indexes) != 1:
            raise ServiceError("live read alias must resolve to one concrete index")
        source_index = source_indexes[0]

        status, _ = request_json_with(
            config, "HEAD", f"/{encoded(restore_index)}", allowed_statuses=(200, 404)
        )
        if status == 200:
            raise ServiceError("isolated restore index already exists")

        _, restore_response = request_json_with(
            config,
            "POST",
            f"/_snapshot/{encoded(repository)}/{encoded(snapshot)}/_restore",
            query={"wait_for_completion": "true"},
            payload={
                "indices": source_index,
                "ignore_unavailable": False,
                "include_global_state": False,
                "include_aliases": False,
                "rename_pattern": f"^{re.escape(source_index)}$",
                "rename_replacement": restore_index,
            },
            allowed_statuses=(200,),
        )
        # Only a successful restore response proves that this invocation owns
        # the target. A failed/ambiguous request must never delete an index that
        # could have been created concurrently by another operator.
        cleanup_required = True
        snapshot_result = restore_response.get("snapshot")
        shards = snapshot_result.get("shards") if isinstance(snapshot_result, dict) else None
        if not isinstance(shards, dict):
            raise ServiceError("restore response is missing shard results")
        total = shards.get("total")
        successful = shards.get("successful")
        failed = shards.get("failed")
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or successful != total
            or failed != 0
        ):
            raise ServiceError("snapshot restore completed with shard failures")

        _, source_mapping_response = request_json_with(
            config, "GET", f"/{encoded(source_index)}/_mapping", allowed_statuses=(200,)
        )
        _, restored_mapping_response = request_json_with(
            config, "GET", f"/{encoded(restore_index)}/_mapping", allowed_statuses=(200,)
        )
        if mapping_for(source_mapping_response, source_index) != mapping_for(
            restored_mapping_response, restore_index
        ):
            raise ServiceError("restored mapping differs from the live source mapping")

        _, source_count_response = request_json_with(
            config, "GET", f"/{encoded(source_index)}/_count", allowed_statuses=(200,)
        )
        _, restored_count_response = request_json_with(
            config, "GET", f"/{encoded(restore_index)}/_count", allowed_statuses=(200,)
        )
        source_count = count_for(source_count_response)
        restored_count = count_for(restored_count_response)
        if source_count != restored_count:
            raise ServiceError("restored document count differs from the live source count")

        result = {
            "status": "RESTORE_DRILL_PASS",
            "repository": repository,
            "snapshot": snapshot,
            "source_index": source_index,
            "restore_index": restore_index,
            "document_count": restored_count,
            "checks": ["shards", "mapping", "document_count", "isolated_cleanup"],
        }
    except (ServiceError, ValueError, OSError) as exc:
        failure = exc
    finally:
        if cleanup_required and config is not None and restore_index:
            try:
                request_json_with(
                    config,
                    "DELETE",
                    f"/{encoded(restore_index)}",
                    allowed_statuses=(200, 404),
                )
            except (ServiceError, ValueError, OSError) as exc:
                cleanup_failure = exc

    if cleanup_failure is not None:
        print(f"RESTORE_DRILL_FAILED cleanup: {cleanup_failure}", file=sys.stderr)
        return 1
    if failure is not None:
        print(f"RESTORE_DRILL_FAILED {failure}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
