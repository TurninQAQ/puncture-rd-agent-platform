#!/usr/bin/env python3
"""Run a small reproducible OpenAI-compatible chat benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from http_utils import ServiceError, iter_sse_json, load_api_key, normalize_base_url, request_json


@dataclass
class Sample:
    index: int
    ok: bool
    latency_ms: float
    ttft_ms: float | None
    completion_tokens: int | None
    output_characters: int
    error: str | None = None


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def extract_usage(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and value >= 0 else None


def make_payload(model: str, max_tokens: int, stream: bool, index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "In two short sentences, explain why deterministic interface contracts "
                    f"help enterprise agents. Request number: {index}."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def run_sample(
    index: int,
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
) -> Sample:
    started = time.monotonic()
    first_token_at: float | None = None
    output_parts: list[str] = []
    completion_tokens: int | None = None
    try:
        payload = make_payload(model, max_tokens, stream, index)
        if stream:
            for event in iter_sse_json(
                f"{base_url}/chat/completions", payload, api_key=api_key, timeout=timeout
            ):
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        delta = choice.get("delta") if isinstance(choice, dict) else None
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            output_parts.append(content)
                usage_value = extract_usage(event)
                if usage_value is not None:
                    completion_tokens = usage_value
        else:
            response = request_json(
                "POST",
                f"{base_url}/chat/completions",
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ServiceError("benchmark response has no choice")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content:
                raise ServiceError("benchmark response has empty content")
            output_parts.append(content)
            completion_tokens = extract_usage(response)
        finished = time.monotonic()
        if not output_parts:
            raise ServiceError("benchmark response produced no text")
        return Sample(
            index=index,
            ok=True,
            latency_ms=(finished - started) * 1000,
            ttft_ms=None if first_token_at is None else (first_token_at - started) * 1000,
            completion_tokens=completion_tokens,
            output_characters=len("".join(output_parts)),
        )
    except Exception as exc:  # each failed sample must be captured without provider data
        if isinstance(exc, ServiceError):
            safe_error = (
                f"ServiceError(status={exc.status})"
                if exc.status is not None
                else "ServiceError"
            )
        else:
            safe_error = type(exc).__name__
        return Sample(
            index=index,
            ok=False,
            latency_ms=(time.monotonic() - started) * 1000,
            ttft_ms=None,
            completion_tokens=None,
            output_characters=0,
            error=safe_error,
        )


def build_report(samples: list[Sample], wall_seconds: float, settings: dict[str, Any]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.ok]
    latencies = [sample.latency_ms for sample in successful]
    ttfts = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    known_tokens = [
        sample.completion_tokens
        for sample in successful
        if sample.completion_tokens is not None
    ]
    return {
        "status": "PASS" if len(successful) == len(samples) else "FAIL",
        "settings": settings,
        "summary": {
            "requests": len(samples),
            "successful": len(successful),
            "failed": len(samples) - len(successful),
            "success_rate": round(len(successful) / len(samples), 6) if samples else 0,
            "wall_seconds": round(wall_seconds, 6),
            "request_throughput_per_second": round(len(successful) / wall_seconds, 6)
            if wall_seconds > 0
            else None,
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "ttft_ms_p50": percentile(ttfts, 50),
            "ttft_ms_p95": percentile(ttfts, 95),
            "known_completion_tokens": sum(known_tokens) if known_tokens else None,
            "completion_tokens_per_second": round(sum(known_tokens) / wall_seconds, 6)
            if known_tokens and wall_seconds > 0
            else None,
            "output_characters": sum(sample.output_characters for sample in successful),
        },
        "samples": [asdict(sample) for sample in samples],
        "notes": [
            "TTFT is available only in streaming mode.",
            "Token throughput is omitted when the server does not return usage.",
            "Record GPU memory and vLLM metrics separately in the sizing worksheet.",
        ],
    }


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "qwen-enterprise-agent"))
    parser.add_argument("--requests", type=int, default=int(os.getenv("BENCHMARK_REQUESTS", "20")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("BENCHMARK_CONCURRENCY", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("BENCHMARK_MAX_TOKENS", "128")))
    parser.add_argument("--stream", type=parse_bool, default=parse_bool(os.getenv("BENCHMARK_STREAM", "true")))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) < 1:
        print("requests, concurrency, and max-tokens must be positive", file=sys.stderr)
        return 2
    base_url = normalize_base_url(args.base_url)
    api_key = load_api_key()
    settings = {
        "base_url": base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_sample,
                index,
                base_url=base_url,
                model=args.model,
                api_key=api_key,
                max_tokens=args.max_tokens,
                stream=args.stream,
                timeout=args.timeout,
            )
            for index in range(args.requests)
        ]
        samples = [future.result() for future in futures]
    report = build_report(samples, time.monotonic() - started, settings)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        destination = pathlib.Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"benchmark report written to {destination}")
    else:
        print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
