#!/usr/bin/env python3
"""Check vLLM liveness and confirm that the configured served model is ready."""

from __future__ import annotations

import argparse
import os
import sys
import time

from http_utils import ServiceError, load_api_key, normalize_base_url, request_bytes, request_json, server_root


def probe(base_url: str, expected_model: str, api_key: str, timeout: float) -> None:
    base_url = normalize_base_url(base_url)
    request_bytes("GET", f"{server_root(base_url)}/health", api_key=api_key, timeout=timeout)
    response = request_json("GET", f"{base_url}/models", api_key=api_key, timeout=timeout)
    models = response.get("data")
    if not isinstance(models, list):
        raise ServiceError("/v1/models response is missing the data array")
    served_ids = {
        item.get("id")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if expected_model not in served_ids:
        raise ServiceError(
            f"configured model is not ready; expected={expected_model!r}, served={sorted(served_ids)!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    container_port = os.getenv("VLLM_CONTAINER_PORT", "8000")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{container_port}/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", os.getenv("VLLM_SERVED_MODEL_NAME", "qwen-enterprise-agent")),
    )
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Perform one probe for a container healthcheck")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deadline = time.monotonic() + (0.0 if args.once else max(0.0, args.wait_seconds))
    api_key = load_api_key()
    while True:
        try:
            probe(args.base_url, args.model, api_key, args.request_timeout)
            print(f"READY model={args.model} base_url={normalize_base_url(args.base_url)}")
            return 0
        except (ServiceError, ValueError, OSError) as exc:
            if args.once or time.monotonic() >= deadline:
                print(f"NOT_READY {exc}", file=sys.stderr)
                return 1
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

