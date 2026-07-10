#!/usr/bin/env python3
"""Estimate a conservative memory starting point before live GPU measurement."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any


GIB = 1024**3


def estimate_memory(
    *,
    parameters_billions: float,
    weight_bits: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_tokens: int,
    concurrent_sequences: int,
    kv_bytes: float,
    tensor_parallel: int,
    runtime_overhead_percent: float,
    activation_reserve_gib: float,
    gpu_memory_gib: float,
    gpu_count: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    positive_values = {
        "parameters_billions": parameters_billions,
        "weight_bits": weight_bits,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "context_tokens": context_tokens,
        "concurrent_sequences": concurrent_sequences,
        "kv_bytes": kv_bytes,
        "tensor_parallel": tensor_parallel,
        "gpu_memory_gib": gpu_memory_gib,
        "gpu_count": gpu_count,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise ValueError("all model, workload, parallelism, and GPU inputs must be positive")
    if runtime_overhead_percent < 0 or activation_reserve_gib < 0:
        raise ValueError("overhead and activation reserve must not be negative")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if tensor_parallel > gpu_count:
        raise ValueError("tensor_parallel must not exceed gpu_count")

    raw_weights_gib = parameters_billions * 1_000_000_000 * weight_bits / 8 / GIB
    weights_with_overhead_gib = raw_weights_gib * (1 + runtime_overhead_percent / 100)
    kv_cache_gib = (
        2
        * layers
        * kv_heads
        * head_dim
        * context_tokens
        * concurrent_sequences
        * kv_bytes
        / GIB
    )
    estimated_per_gpu_gib = (
        weights_with_overhead_gib + kv_cache_gib
    ) / tensor_parallel + activation_reserve_gib
    usable_per_gpu_gib = gpu_memory_gib * gpu_memory_utilization
    headroom_gib = usable_per_gpu_gib - estimated_per_gpu_gib
    return {
        "inputs": {
            key: value
            for key, value in {
                **positive_values,
                "runtime_overhead_percent": runtime_overhead_percent,
                "activation_reserve_gib": activation_reserve_gib,
                "gpu_memory_utilization": gpu_memory_utilization,
            }.items()
        },
        "estimate": {
            "raw_weights_gib": round(raw_weights_gib, 3),
            "weights_with_runtime_overhead_gib": round(weights_with_overhead_gib, 3),
            "kv_cache_total_gib": round(kv_cache_gib, 3),
            "estimated_per_gpu_gib": round(estimated_per_gpu_gib, 3),
            "usable_per_gpu_gib": round(usable_per_gpu_gib, 3),
            "headroom_per_gpu_gib": round(headroom_gib, 3),
            "fits_estimate": headroom_gib >= 0,
            "minimum_gpus_by_memory_only": math.ceil(
                (weights_with_overhead_gib + kv_cache_gib)
                / max(0.001, usable_per_gpu_gib - activation_reserve_gib)
            )
            if usable_per_gpu_gib > activation_reserve_gib
            else None,
        },
        "warning": (
            "This is a planning estimate, not a compatibility or capacity result. "
            "Measure startup peak, steady-state peak, TTFT, TPOT, and OOM behavior on the exact stack."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters-billions", type=float, required=True)
    parser.add_argument("--weight-bits", type=float, default=16)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--concurrent-sequences", type=int, default=1)
    parser.add_argument("--kv-bytes", type=float, default=2)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--runtime-overhead-percent", type=float, default=15)
    parser.add_argument("--activation-reserve-gib", type=float, default=2)
    parser.add_argument("--gpu-memory-gib", type=float, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = estimate_memory(**vars(args))
    except ValueError as exc:
        raise SystemExit(f"invalid sizing input: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

