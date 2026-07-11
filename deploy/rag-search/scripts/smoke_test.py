#!/usr/bin/env python3
"""Run read-only mapping, alias, BM25, dense-filter, and ACL smoke checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from typing import Any

from bootstrap_index import contract_from_environment
from health_check import probe
from http_utils import ServiceError, connection_from_environment, request_json_with, validate_identifier
from index_contract import alias_indexes, mandatory_filters, mapping_for_index, validate_mapping


RESERVED_PROBE_SCOPE = "__rag_health_probe__"


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def validate_search_response(response: dict[str, Any], *, label: str) -> int:
    hits = response.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        raise ServiceError(f"{label} response is missing hits")
    if hits["hits"]:
        raise ServiceError(f"{label} returned a document for the reserved ACL probe scope")
    return len(hits["hits"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-alias", default=os.getenv("RAG_READ_ALIAS", "puncture-rag-read"))
    parser.add_argument("--write-alias", default=os.getenv("RAG_WRITE_ALIAS", "puncture-rag-write"))
    parser.add_argument(
        "--wait-seconds", type=float, default=float(os.getenv("RAG_HEALTH_WAIT_SECONDS", "0"))
    )
    parser.add_argument("--interval", type=float, default=5.0)
    return parser


def run_checks(config: object, read_alias: str, write_alias: str) -> dict[str, Any]:
    health = probe(config)
    contract = contract_from_environment()
    _, read_response = request_json_with(  # type: ignore[arg-type]
        config, "GET", f"/_alias/{encoded(read_alias)}", allowed_statuses=(200,)
    )
    _, write_response = request_json_with(  # type: ignore[arg-type]
        config, "GET", f"/_alias/{encoded(write_alias)}", allowed_statuses=(200,)
    )
    read_indexes = alias_indexes(read_response, read_alias)
    write_indexes = alias_indexes(write_response, write_alias)
    if len(read_indexes) != 1 or read_indexes != write_indexes:
        raise ServiceError("read/write aliases must resolve to the same single concrete index")
    concrete_index = read_indexes[0]

    _, mapping_response = request_json_with(  # type: ignore[arg-type]
        config, "GET", f"/{encoded(concrete_index)}/_mapping", allowed_statuses=(200,)
    )
    validate_mapping(mapping_for_index(mapping_response, concrete_index), contract)

    filters = mandatory_filters(RESERVED_PROBE_SCOPE)
    lexical_payload = {
        "size": 1,
        "_source": False,
        "track_total_hits": False,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "__rag_health_probe_no_match__",
                            "fields": ["title^2", "title.code", "text", "text.code"],
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }
    _, lexical_response = request_json_with(  # type: ignore[arg-type]
        config,
        "POST",
        f"/{encoded(read_alias)}/_search",
        payload=lexical_payload,
        allowed_statuses=(200,),
    )
    validate_search_response(lexical_response, label="BM25 probe")

    vector = [0.0] * contract.embedding_dimension
    vector[0] = 1.0
    dense_payload = {
        "size": 1,
        "_source": False,
        "track_total_hits": False,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": 1,
                    "filter": {"bool": {"filter": filters}},
                }
            }
        },
    }
    _, dense_response = request_json_with(  # type: ignore[arg-type]
        config,
        "POST",
        f"/{encoded(read_alias)}/_search",
        payload=dense_payload,
        allowed_statuses=(200,),
    )
    validate_search_response(dense_response, label="dense ACL probe")

    _, count_response = request_json_with(  # type: ignore[arg-type]
        config, "GET", f"/{encoded(read_alias)}/_count", allowed_statuses=(200,)
    )
    count = count_response.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ServiceError("OpenSearch count response is malformed")
    return {
        "status": "PASS",
        "health": health,
        "concrete_index": concrete_index,
        "document_count": count,
        "embedding_dimension": contract.embedding_dimension,
        "checks": ["alias", "mapping", "bm25_acl_filter", "dense_acl_filter"],
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        read_alias = validate_identifier(args.read_alias, label="read alias")
        write_alias = validate_identifier(args.write_alias, label="write alias")
        config = connection_from_environment()
    except (ValueError, OSError) as exc:
        print(f"SMOKE_TEST_FAILED {exc}", file=sys.stderr)
        return 1
    deadline = time.monotonic() + max(0.0, args.wait_seconds)
    while True:
        try:
            print(json.dumps(run_checks(config, read_alias, write_alias), sort_keys=True))
            return 0
        except (ServiceError, ValueError, OSError) as exc:
            if time.monotonic() >= deadline:
                print(f"SMOKE_TEST_FAILED {exc}", file=sys.stderr)
                return 1
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
