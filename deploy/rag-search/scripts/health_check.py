#!/usr/bin/env python3
"""Wait for a secured OpenSearch cluster to reach yellow or green health."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from http_utils import ServiceError, connection_from_environment, request_json_with


def probe(config: object, expected_cluster: str | None = None) -> dict[str, object]:
    _, response = request_json_with(
        config,  # type: ignore[arg-type]
        "GET",
        "/_cluster/health",
        query={"wait_for_status": "yellow", "timeout": "5s"},
    )
    status = response.get("status")
    cluster_name = response.get("cluster_name")
    if status not in {"yellow", "green"}:
        raise ServiceError("OpenSearch cluster is not yellow or green")
    if not isinstance(cluster_name, str) or not cluster_name:
        raise ServiceError("OpenSearch health response has no cluster name")
    if expected_cluster and cluster_name != expected_cluster:
        raise ServiceError("OpenSearch health response has an unexpected cluster name")
    if response.get("timed_out") is True:
        raise ServiceError("OpenSearch health request timed out")
    return {
        "cluster_name": cluster_name,
        "status": status,
        "number_of_nodes": response.get("number_of_nodes"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-seconds", type=float, default=float(os.getenv("RAG_HEALTH_WAIT_SECONDS", "0"))
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--cluster-name", default=os.getenv("OPENSEARCH_CLUSTER_NAME", ""))
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deadline = time.monotonic() + (0 if args.once else max(0.0, args.wait_seconds))
    try:
        config = connection_from_environment()
    except (ValueError, OSError) as exc:
        print(f"NOT_READY {exc}", file=sys.stderr)
        return 1
    while True:
        try:
            result = probe(config, args.cluster_name or None)
            print(json.dumps({"status": "READY", **result}, sort_keys=True))
            return 0
        except (ServiceError, ValueError, OSError) as exc:
            if args.once or time.monotonic() >= deadline:
                print(f"NOT_READY {exc}", file=sys.stderr)
                return 1
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
