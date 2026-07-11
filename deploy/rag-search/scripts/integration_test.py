#!/usr/bin/env python3
"""Opt-in live BM25/vector/ACL test using only an isolated disposable index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import uuid
from typing import Any

from http_utils import (
    ServiceError,
    connection_from_environment,
    parse_bool,
    request_json_with,
    validate_identifier,
)


TEST_INDEX_PREFIX = "puncture-rag-test-"
PRODUCTION_INDEX_PREFIX = "puncture-rag-v"
PRODUCTION_ALIASES = {"puncture-rag-read", "puncture-rag-write"}


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def validate_test_index(value: str) -> str:
    value = validate_identifier(value, label="test index")
    if (
        not value.startswith(TEST_INDEX_PREFIX)
        or value.startswith(PRODUCTION_INDEX_PREFIX)
        or value in PRODUCTION_ALIASES
        or len(value) <= len(TEST_INDEX_PREFIX)
    ):
        raise ValueError(
            f"test index must use the reserved disposable prefix {TEST_INDEX_PREFIX!r}"
        )
    return value


def filters(scope: str) -> list[dict[str, Any]]:
    return [
        {"term": {"doc_kind": "child"}},
        {"term": {"status": "active"}},
        {"term": {"version": "v1"}},
        {"terms": {"access_scopes": [scope]}},
    ]


def hit_ids(response: dict[str, Any]) -> list[str]:
    hits = response.get("hits")
    values = hits.get("hits") if isinstance(hits, dict) else None
    if not isinstance(values, list):
        raise ServiceError("integration search response is missing hits")
    result: list[str] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("_id"), str):
            raise ServiceError("integration search hit is malformed")
        result.append(item["_id"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-index",
        default=os.getenv("RAG_TEST_INDEX", f"{TEST_INDEX_PREFIX}{uuid.uuid4().hex[:12]}"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        test_index = validate_test_index(args.test_index)
        if not parse_bool(os.getenv("RUN_RAG_INTEGRATION", "false"), label="RUN_RAG_INTEGRATION"):
            print("SKIPPED set RUN_RAG_INTEGRATION=1 to create an isolated live test index")
            return 0
        config = connection_from_environment()
    except (ValueError, OSError) as exc:
        print(f"INTEGRATION_TEST_FAILED {exc}", file=sys.stderr)
        return 1

    created = False
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    cleanup_failure: Exception | None = None
    try:
        request_json_with(
            config,
            "PUT",
            f"/{encoded(test_index)}",
            payload={
                "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "doc_kind": {"type": "keyword"},
                        "text": {"type": "text"},
                        "version": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "access_scopes": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 4,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                    },
                },
            },
            allowed_statuses=(200,),
        )
        created = True
        documents = {
            "authorized-exact": {
                "doc_kind": "child",
                "text": "ERR_PATH_CLEARANCE_0042 approved diagnostic",
                "version": "v1",
                "status": "active",
                "access_scopes": ["team-a"],
                "embedding": [1.0, 0.0, 0.0, 0.0],
            },
            "authorized-semantic": {
                "doc_kind": "child",
                "text": "troubleshoot minimum route separation",
                "version": "v1",
                "status": "active",
                "access_scopes": ["team-a"],
                "embedding": [0.99, 0.01, 0.0, 0.0],
            },
            "unauthorized-perfect": {
                "doc_kind": "child",
                "text": "TOP_SECRET_PERFECT_MATCH",
                "version": "v1",
                "status": "active",
                "access_scopes": ["restricted-team"],
                "embedding": [1.0, 0.0, 0.0, 0.0],
            },
        }
        for document_id, document in documents.items():
            request_json_with(
                config,
                "PUT",
                f"/{encoded(test_index)}/_doc/{encoded(document_id)}",
                query={"refresh": "wait_for"},
                payload=document,
                allowed_statuses=(200, 201),
            )

        _, lexical = request_json_with(
            config,
            "POST",
            f"/{encoded(test_index)}/_search",
            payload={
                "size": 3,
                "_source": False,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": "ERR_PATH_CLEARANCE_0042"}}],
                        "filter": filters("team-a"),
                    }
                },
            },
            allowed_statuses=(200,),
        )
        if hit_ids(lexical) != ["authorized-exact"]:
            raise ServiceError("BM25 exact-term integration assertion failed")

        _, dense = request_json_with(
            config,
            "POST",
            f"/{encoded(test_index)}/_search",
            payload={
                "size": 2,
                "_source": False,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": [1.0, 0.0, 0.0, 0.0],
                            "k": 2,
                            "filter": {"bool": {"filter": filters("team-a")}},
                        }
                    }
                },
            },
            allowed_statuses=(200,),
        )
        dense_ids = hit_ids(dense)
        if "unauthorized-perfect" in dense_ids or not {
            "authorized-exact",
            "authorized-semantic",
        }.issubset(dense_ids):
            raise ServiceError("dense ACL integration assertion failed")

        _, negative = request_json_with(
            config,
            "POST",
            f"/{encoded(test_index)}/_search",
            payload={
                "size": 3,
                "_source": False,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": "TOP_SECRET_PERFECT_MATCH"}}],
                        "filter": filters("team-a"),
                    }
                },
            },
            allowed_statuses=(200,),
        )
        if hit_ids(negative):
            raise ServiceError("ACL-negative integration assertion failed")
        result = {
            "status": "PASS",
            "test_index": test_index,
            "checks": ["bm25_exact", "dense_semantic", "acl_negative"],
        }
    except (ServiceError, ValueError, OSError) as exc:
        failure = exc
    finally:
        if created:
            try:
                request_json_with(
                    config,
                    "DELETE",
                    f"/{encoded(test_index)}",
                    allowed_statuses=(200, 404),
                )
            except (ServiceError, ValueError, OSError) as exc:
                cleanup_failure = exc
    if cleanup_failure is not None:
        print(f"INTEGRATION_TEST_CLEANUP_FAILED {cleanup_failure}", file=sys.stderr)
        return 1
    if failure is not None:
        print(f"INTEGRATION_TEST_FAILED {failure}", file=sys.stderr)
        return 1
    if result is None:
        print("INTEGRATION_TEST_FAILED no result", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
