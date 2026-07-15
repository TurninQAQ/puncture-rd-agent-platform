#!/usr/bin/env python3
"""Verify the loopback full-stack API through real HTTP and SSE requests."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import ssl
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.live_api_server import load_or_create_demo_token, require_demo_opt_in  # noqa: E402


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class DemoHttpError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        del request, code, message, headers, new_url
        fp.close()
        raise DemoHttpError("local demo redirect rejected")


def validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("PUNCTURE_DEMO_BASE_URL must be loopback HTTP")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("PUNCTURE_DEMO_BASE_URL must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("PUNCTURE_DEMO_BASE_URL must not contain path syntax")
    host = parsed.hostname
    if host.lower() != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("PUNCTURE_DEMO_BASE_URL must use a loopback host")
        except ValueError as exc:
            raise ValueError("PUNCTURE_DEMO_BASE_URL must use a loopback host") from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("PUNCTURE_DEMO_BASE_URL has an invalid port") from exc
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit(("http", netloc, "", "", ""))


class DemoClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = validate_base_url(base_url)
        self.token = token
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        accept: str = "application/json",
        authenticated: bool = True,
        timeout: float = 30.0,
    ) -> tuple[int, str, bytes]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("demo request path must be an absolute path without query syntax")
        body = None
        headers = {"Accept": accept}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise DemoHttpError("local demo response exceeds the size limit")
                return int(response.status), response.headers.get("Content-Type", ""), raw
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            exc.close()
            raise DemoHttpError(
                f"local demo API returned HTTP {status}", status=status
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DemoHttpError("local demo API connection failed") from exc

    def json_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        status, content_type, raw = self.request(method, path, **kwargs)
        if status != 200 or "application/json" not in content_type.lower():
            raise DemoHttpError("local demo API returned an unexpected response")
        try:
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DemoHttpError("local demo API returned invalid JSON") from exc
        if not isinstance(value, (dict, list)):
            raise DemoHttpError("local demo API returned an invalid JSON shape")
        return value

    def wait_ready(self, timeout_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                value = self.json_request("GET", "/health", authenticated=False, timeout=2.0)
                if isinstance(value, dict) and value.get("status") in {"UP", "DEGRADED"}:
                    return value
            except DemoHttpError:
                pass
            time.sleep(0.2)
        raise DemoHttpError("local demo API did not become ready")


def run_demo(*, wait_seconds: float = 120.0) -> dict[str, Any]:
    require_demo_opt_in()
    token = load_or_create_demo_token()
    base_url = os.environ.get("PUNCTURE_DEMO_BASE_URL", "http://127.0.0.1:8010")
    client = DemoClient(base_url, token)
    health = client.wait_ready(wait_seconds)

    unauthorized = DemoClient(base_url, f"{token}x")
    try:
        unauthorized.json_request("GET", "/api/v1/runs/run-auth-negative")
    except DemoHttpError as exc:
        if exc.status != 403:
            raise DemoHttpError("invalid bearer token did not return HTTP 403") from exc
    else:
        raise DemoHttpError("invalid bearer token was accepted")

    def await_terminal(created: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(created.get("run_id"), str):
            raise DemoHttpError("run creation response is malformed")
        current = created
        deadline = time.monotonic() + wait_seconds
        while current.get("status") == "RUNNING" and time.monotonic() < deadline:
            time.sleep(0.2)
            value = client.json_request("GET", f"/api/v1/runs/{created['run_id']}")
            if not isinstance(value, dict):
                raise DemoHttpError("run read response is malformed")
            current = value
        if current.get("status") != "SUCCEEDED":
            raise DemoHttpError("local full-stack Run did not succeed")
        return current

    data_body = {
        "case_id": "Case-203",
        "user_query": "检查 Case-203 的 MCS 标签、几何和分割质量，并生成可追踪报告",
        "task_type": "DATA_MODEL_VALIDATION",
        "idempotency_key": f"live-full-stack-{secrets.token_hex(12)}",
        "artifact_ids": [],
        "metadata": {},
    }
    data_created = client.json_request("POST", "/api/v1/runs", payload=data_body)
    if not isinstance(data_created, dict):
        raise DemoHttpError("run creation response is malformed")
    data_terminal = await_terminal(data_created)
    data_run_id = data_terminal["run_id"]

    replay = client.json_request("POST", "/api/v1/runs", payload=data_body)
    if not isinstance(replay, dict) or replay.get("run_id") != data_run_id:
        raise DemoHttpError("idempotent Run replay returned a different identity")
    data_events = client.json_request("GET", f"/api/v1/runs/{data_run_id}/events")
    if not isinstance(data_events, list) or not data_events:
        raise DemoHttpError("event replay is empty or malformed")
    event_types = [
        event.get("event_type") for event in data_events if isinstance(event, dict)
    ]
    node_names = [
        event.get("node_name")
        for event in data_events
        if isinstance(event, dict) and isinstance(event.get("node_name"), str)
    ]
    required_nodes = {"model_gateway.plan", "rag.retrieve", "agent.graph"}
    if "RUN_COMPLETED" not in event_types or not required_nodes.issubset(node_names):
        raise DemoHttpError("event replay omitted full-stack execution evidence")

    status, content_type, sse = client.request(
        "GET",
        f"/api/v1/runs/{data_run_id}/events",
        accept="text/event-stream",
        timeout=30.0,
    )
    if (
        status != 200
        or "text/event-stream" not in content_type.lower()
        or b"event: RUN_COMPLETED\n" not in sse
    ):
        raise DemoHttpError("terminal SSE replay did not complete")

    data_report = data_terminal.get("final_report")
    data_evidence = (
        data_report.get("runtime_evidence") if isinstance(data_report, dict) else None
    )
    if (
        not isinstance(data_evidence, dict)
        or data_evidence.get("model") != os.environ.get(
            "VLLM_MODEL", "qwen-enterprise-agent"
        )
        or not isinstance(data_evidence.get("rag_chunk_count"), int)
        or data_evidence["rag_chunk_count"] < 2
        or not isinstance(data_evidence.get("tool_call_count"), int)
        or data_evidence["tool_call_count"] < 1
    ):
        raise DemoHttpError("terminal report omitted real model/RAG or tool evidence")

    planning_body = {
        "case_id": "Case-102",
        "user_query": "请对 Case-102 执行候选路径规划、全路径安全和皮肤穿透评估",
        "task_type": "PLANNING_SAFETY",
        "idempotency_key": f"live-full-stack-{secrets.token_hex(12)}",
        "artifact_ids": [],
        "metadata": {},
    }
    planning_created = client.json_request(
        "POST", "/api/v1/runs", payload=planning_body
    )
    if not isinstance(planning_created, dict):
        raise DemoHttpError("planning Run creation response is malformed")
    planning_terminal = await_terminal(planning_created)
    planning_report = planning_terminal.get("final_report")
    planning_evidence = (
        planning_report.get("runtime_evidence")
        if isinstance(planning_report, dict)
        else None
    )
    if (
        not isinstance(planning_evidence, dict)
        or planning_evidence.get("rag_chunk_count", 0) < 2
        or planning_evidence.get("tool_call_count", 0) < 4
        or planning_report.get("verification_status") != "PASS"
    ):
        raise DemoHttpError("planning Run omitted model/RAG/tool/verifier evidence")

    return {
        "demo": "live-fastapi-postgres-qwen-opensearch-agent",
        "health": health,
        "runs": {
            "data_validation": {
                "run_id": data_run_id,
                "trace_id": data_terminal.get("trace_id"),
                "status": data_terminal.get("status"),
                "idempotent_replay": True,
                "rag_chunk_count": data_evidence["rag_chunk_count"],
                "tool_call_count": data_evidence["tool_call_count"],
                "visited_node_count": data_evidence.get("visited_node_count"),
                "event_count": len(data_events),
                "sse_terminal_replay": True,
            },
            "planning_safety": {
                "run_id": planning_terminal.get("run_id"),
                "trace_id": planning_terminal.get("trace_id"),
                "status": planning_terminal.get("status"),
                "rag_chunk_count": planning_evidence["rag_chunk_count"],
                "tool_call_count": planning_evidence["tool_call_count"],
                "visited_node_count": planning_evidence.get("visited_node_count"),
                "verification_status": planning_report.get("verification_status"),
            },
        },
        "boundaries": {
            "algorithm_backend": "deterministic-synthetic",
            "invalid_bearer_rejected": True,
            "company_or_patient_data": False,
            "loopback_only": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.wait_seconds <= 0:
            raise ValueError("--wait-seconds must be positive")
        print(json.dumps(run_demo(wait_seconds=args.wait_seconds), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"LIVE_API_DEMO_FAILED {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
