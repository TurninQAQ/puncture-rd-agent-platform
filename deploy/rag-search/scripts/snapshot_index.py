#!/usr/bin/env python3
"""Create and verify a non-partial snapshot of the concrete live RAG index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse

from http_utils import ServiceError, connection_from_environment, request_json_with, validate_identifier
from index_contract import alias_indexes


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--read-alias", default=os.getenv("RAG_READ_ALIAS", "puncture-rag-read"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        repository = validate_identifier(args.repository, label="snapshot repository")
        snapshot = validate_identifier(args.snapshot, label="snapshot")
        read_alias = validate_identifier(args.read_alias, label="read alias")
        config = connection_from_environment()
        _, alias_response = request_json_with(
            config, "GET", f"/_alias/{encoded(read_alias)}", allowed_statuses=(200,)
        )
        indexes = alias_indexes(alias_response, read_alias)
        if len(indexes) != 1:
            raise ServiceError("live read alias must resolve to one concrete index")
        concrete_index = indexes[0]
        _, response = request_json_with(
            config,
            "PUT",
            f"/_snapshot/{encoded(repository)}/{encoded(snapshot)}",
            query={"wait_for_completion": "true"},
            payload={
                "indices": concrete_index,
                "ignore_unavailable": False,
                "include_global_state": False,
                "partial": False,
            },
            allowed_statuses=(200,),
        )
        snapshot_result = response.get("snapshot")
        if not isinstance(snapshot_result, dict) or snapshot_result.get("state") != "SUCCESS":
            raise ServiceError("snapshot did not complete successfully")
        failures = snapshot_result.get("failures", [])
        if failures not in ([], None):
            raise ServiceError("snapshot completed with shard failures")
        print(
            json.dumps(
                {
                    "status": "SNAPSHOT_CREATED",
                    "repository": repository,
                    "snapshot": snapshot,
                    "index": concrete_index,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ServiceError, ValueError, OSError) as exc:
        print(f"SNAPSHOT_FAILED {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
