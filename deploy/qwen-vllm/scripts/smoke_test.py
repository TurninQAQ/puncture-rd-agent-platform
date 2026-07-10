#!/usr/bin/env python3
"""Run plain-chat, tool-call, and structured-output smoke checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from health_check import probe
from http_utils import ServiceError, iter_sse_json, load_api_key, normalize_base_url, request_json


def first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ServiceError("chat response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ServiceError("chat response has no assistant message")
    return message


def run_plain_chat(base_url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    response = request_json(
        "POST",
        f"{base_url}/chat/completions",
        api_key=api_key,
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an enterprise R&D assistant."},
                {"role": "user", "content": "Reply with the single word READY."},
            ],
            "temperature": 0,
            "max_tokens": 32,
        },
    )
    content = first_message(response).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ServiceError("plain chat produced empty content")
    return {"content_preview": content.strip()[:80], "response_id": response.get("id")}


def run_streaming_chat(base_url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    content_parts: list[str] = []
    terminal_reason: str | None = None
    event_count = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word STREAM_READY."}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for event in iter_sse_json(
        f"{base_url}/chat/completions", payload, api_key=api_key, timeout=timeout
    ):
        event_count += 1
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                content_parts.append(content)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                terminal_reason = finish_reason
    if not content_parts:
        raise ServiceError("streaming chat produced no content delta")
    if terminal_reason not in {"stop", "length", "tool_calls", "content_filter"}:
        raise ServiceError("streaming chat produced no supported terminal finish reason")
    return {
        "content_preview": "".join(content_parts).strip()[:80],
        "finish_reason": terminal_reason,
        "event_count": event_count,
    }


def run_tool_call(base_url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    tool_name = "lookup_document"
    response = request_json(
        "POST",
        f"{base_url}/chat/completions",
        api_key=api_key,
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use lookup_document to find the current equipment acceptance procedure.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Look up a versioned internal engineering document.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": 0,
            "max_tokens": 128,
        },
    )
    tool_calls = first_message(response).get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ServiceError("forced tool-call response contains no tool call")
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != tool_name:
        raise ServiceError("forced tool-call response used the wrong tool")
    raw_arguments = function.get("arguments")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError as exc:
        raise ServiceError("tool arguments are not valid JSON") from exc
    if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str):
        raise ServiceError("tool arguments do not satisfy the required schema")
    return {"tool": tool_name, "arguments": arguments}


def run_structured_output(base_url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["search", "explain"]},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    }
    response = request_json(
        "POST",
        f"{base_url}/chat/completions",
        api_key=api_key,
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": "user", "content": "Classify this request: search for the latest test report."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "intent_result", "schema": schema, "strict": True},
            },
            "temperature": 0,
            "max_tokens": 128,
        },
    )
    content = first_message(response).get("content")
    if not isinstance(content, str):
        raise ServiceError("structured-output response content is not text")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ServiceError("structured-output response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"intent", "confidence"}:
        raise ServiceError("structured-output response has unexpected fields")
    if value["intent"] not in {"search", "explain"}:
        raise ServiceError("structured-output intent is outside the enum")
    if not isinstance(value["confidence"], (int, float)) or isinstance(value["confidence"], bool):
        raise ServiceError("structured-output confidence is not numeric")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "qwen-enterprise-agent"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("VERIFY_TIMEOUT_SECONDS", "120")))
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-structured", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    try:
        base_url = normalize_base_url(args.base_url)
        api_key = load_api_key()
        probe(base_url, args.model, api_key, min(args.timeout, 10.0))
        results: dict[str, Any] = {
            "model": args.model,
            "plain_chat": run_plain_chat(base_url, args.model, api_key, args.timeout),
            "streaming_chat": run_streaming_chat(base_url, args.model, api_key, args.timeout),
        }
        if not args.skip_tools:
            results["tool_call"] = run_tool_call(base_url, args.model, api_key, args.timeout)
        if not args.skip_structured:
            results["structured_output"] = run_structured_output(
                base_url, args.model, api_key, args.timeout
            )
        results["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
        results["status"] = "PASS"
        print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ServiceError, ValueError, OSError) as exc:
        print(f"SMOKE_TEST_FAILED {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
