#!/usr/bin/env python3
"""Atomically promote or roll back a fully validated versioned RAG index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse

from bootstrap_index import contract_from_environment
from http_utils import ServiceError, connection_from_environment, request_json_with, validate_identifier
from index_contract import alias_indexes, mapping_for_index, validate_mapping


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def resolve_alias(config: object, alias: str) -> tuple[str, ...]:
    status, response = request_json_with(  # type: ignore[arg-type]
        config,
        "GET",
        f"/_alias/{encoded(alias)}",
        allowed_statuses=(200, 404),
    )
    return () if status == 404 else alias_indexes(response, alias)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-index", required=True)
    parser.add_argument("--expected-current", required=True)
    parser.add_argument("--read-alias", default=os.getenv("RAG_READ_ALIAS", "puncture-rag-read"))
    parser.add_argument("--write-alias", default=os.getenv("RAG_WRITE_ALIAS", "puncture-rag-write"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        new_index = validate_identifier(args.new_index, label="new index", versioned_index=True)
        expected = validate_identifier(
            args.expected_current, label="expected current index", versioned_index=True
        )
        read_alias = validate_identifier(args.read_alias, label="read alias")
        write_alias = validate_identifier(args.write_alias, label="write alias")
        if new_index == expected:
            raise ValueError("new index must differ from expected current index")
        config = connection_from_environment()
        contract = contract_from_environment()

        _, mapping_response = request_json_with(
            config, "GET", f"/{encoded(new_index)}/_mapping", allowed_statuses=(200,)
        )
        validate_mapping(mapping_for_index(mapping_response, new_index), contract)

        read_indexes = resolve_alias(config, read_alias)
        write_indexes = resolve_alias(config, write_alias)
        if read_indexes != (expected,) or write_indexes != (expected,):
            raise ServiceError("live aliases changed; expected-current check failed")

        request_json_with(
            config,
            "POST",
            "/_aliases",
            payload={
                "actions": [
                    {"remove": {"index": expected, "alias": read_alias}},
                    {"remove": {"index": expected, "alias": write_alias}},
                    {"add": {"index": new_index, "alias": read_alias}},
                    {
                        "add": {
                            "index": new_index,
                            "alias": write_alias,
                            "is_write_index": True,
                        }
                    },
                ]
            },
            allowed_statuses=(200,),
        )
        print(
            json.dumps(
                {
                    "status": "PROMOTED",
                    "previous_index": expected,
                    "current_index": new_index,
                    "rollback_command": (
                        "python3 scripts/promote_index.py "
                        f"--new-index {expected} --expected-current {new_index}"
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ServiceError, ValueError, OSError) as exc:
        print(f"PROMOTION_FAILED {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
