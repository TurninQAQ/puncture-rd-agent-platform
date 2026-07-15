#!/usr/bin/env python3
"""Fail-closed, secret-safe readiness checks for the local full-stack demo."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable
import urllib.parse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples.live_api_demo import validate_base_url  # noqa: E402
from examples.live_api_server import (  # noqa: E402
    _loopback_host,
    _port,
    load_or_create_demo_token,
    require_demo_opt_in,
)


def _down(error_code: str) -> dict[str, Any]:
    return {"status": "DOWN", "error_code": error_code}


def _guard(check: Callable[[], dict[str, Any]], error_code: str) -> dict[str, Any]:
    try:
        result = check()
    except Exception:
        return _down(error_code)
    if result.get("status") not in {"UP", "DEGRADED", "DOWN"}:
        return _down(error_code)
    return result


def _check_configuration() -> dict[str, Any]:
    require_demo_opt_in()
    host = _loopback_host(os.environ.get("PUNCTURE_DEMO_HOST", "127.0.0.1"))
    port = _port(os.environ.get("PUNCTURE_DEMO_PORT", "8010"))
    base_url = validate_base_url(
        os.environ.get("PUNCTURE_DEMO_BASE_URL", "http://127.0.0.1:8010")
    )
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.hostname is None:
        raise ValueError("local demo base URL is missing a host")
    client_port = parsed.port if parsed.port is not None else 80
    if parsed.hostname.lower() != host.lower() or client_port != port:
        raise ValueError("local demo server and client addresses do not match")
    load_or_create_demo_token()
    return {
        "status": "UP",
        "api_binding": "loopback",
        "port": port,
        "explicit_opt_in": True,
        "private_token_ready": True,
    }


def _check_python_dependencies() -> dict[str, Any]:
    distributions = (
        "fastapi",
        "pydantic",
        "uvicorn",
        "psycopg",
        "httpx",
        "langgraph",
        "langgraph-checkpoint-postgres",
        "mcp",
        "opentelemetry-api",
    )
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        return {
            "status": "DOWN",
            "error_code": "PYTHON_DEPENDENCY_MISSING",
            "missing": missing,
        }
    return {
        "status": "UP",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "packages": versions,
    }


def _check_postgres() -> dict[str, Any]:
    import psycopg

    from puncture_agent.api.postgres_app import PostgresApiSettings

    settings = PostgresApiSettings.from_env(os.environ)
    timeout = max(1, int(math.ceil(min(settings.connect_timeout_seconds, 5.0))))
    with psycopg.connect(
        settings.connection_string,
        connect_timeout=timeout,
        autocommit=True,
    ) as connection:
        row = connection.execute(
            "SELECT current_setting('server_version'), "
            "current_setting('server_version_num')::integer"
        ).fetchone()
    if row is None or len(row) != 2:
        raise RuntimeError("PostgreSQL version query returned no row")
    version = str(row[0])
    version_number = int(row[1])
    if version_number // 10_000 != 16:
        return {
            "status": "DEGRADED",
            "error_code": "POSTGRES_VERSION_UNEXPECTED",
            "server_version": version,
        }
    return {"status": "UP", "server_version": version}


def _check_qwen() -> dict[str, Any]:
    from puncture_agent.model_gateway import VllmGatewayConfig, VllmModelGateway

    gateway = VllmModelGateway(
        VllmGatewayConfig(
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8008/v1"),
            model=os.environ.get("VLLM_MODEL", "qwen-enterprise-agent"),
            timeout_seconds=min(
                float(os.environ.get("VLLM_TIMEOUT_SECONDS", "120")),
                5.0,
            ),
            max_retries=0,
        )
    )
    try:
        health = gateway.health()
    finally:
        gateway.close()
    result = {
        "status": health.status,
        "provider": health.provider,
        "model": health.model,
    }
    error_code = health.details.get("error_code")
    if isinstance(error_code, str):
        result["error_code"] = error_code
    return result


def _check_opensearch() -> dict[str, Any]:
    from examples.live_opensearch_rag_demo import endpoint_from_environment
    from puncture_agent.rag import OpenSearchSearchBackend

    backend = OpenSearchSearchBackend(
        endpoint_from_environment(),
        read_alias=os.environ.get("RAG_READ_ALIAS", "puncture-rag-read"),
    )
    try:
        health = backend.health()
    finally:
        backend.close()
    result = {
        "status": health.status,
        "backend": health.backend,
        "document_count": health.document_count,
        "chunk_count": health.chunk_count,
    }
    cluster_status = health.details.get("cluster_status")
    if isinstance(cluster_status, str):
        result["cluster_status"] = cluster_status
    return result


def run_checks() -> dict[str, Any]:
    configuration = _guard(_check_configuration, "LOCAL_CONFIGURATION_INVALID")
    components: dict[str, dict[str, Any]] = {"configuration": configuration}
    if configuration["status"] != "UP":
        return {
            "check": "local-full-stack-readiness",
            "ready": False,
            "components": components,
            "boundaries": {
                "database_or_index_mutation": False,
                "secret_values_printed": False,
            },
        }

    components.update(
        {
            "python_dependencies": _guard(
                _check_python_dependencies,
                "PYTHON_DEPENDENCY_CHECK_FAILED",
            ),
            "postgresql": _guard(_check_postgres, "POSTGRES_UNAVAILABLE"),
            "qwen_vllm": _guard(_check_qwen, "QWEN_VLLM_UNAVAILABLE"),
            "opensearch": _guard(_check_opensearch, "OPENSEARCH_UNAVAILABLE"),
        }
    )
    ready = (
        components["python_dependencies"]["status"] == "UP"
        and components["postgresql"]["status"] == "UP"
        and components["qwen_vllm"]["status"] == "UP"
        and components["opensearch"]["status"] in {"UP", "DEGRADED"}
    )
    return {
        "check": "local-full-stack-readiness",
        "ready": ready,
        "components": components,
        "boundaries": {
            "database_or_index_mutation": False,
            "secret_values_printed": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only when a readiness check fails",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_checks()
    if not args.quiet or not report["ready"]:
        stream = sys.stdout if report["ready"] else sys.stderr
        print(json.dumps(report, indent=2, sort_keys=True), file=stream)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
