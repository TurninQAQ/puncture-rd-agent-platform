"""Reproducible metadata/idempotency micro-benchmark for Module 0."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from contracts.enums import ArtifactType  # noqa: E402
from puncture_agent.artifacts import (  # noqa: E402
    InMemoryArtifactRegistry,
    SQLiteArtifactRegistry,
)


CHECKSUM = "a" * 64


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def create_registry(backend: str, temporary_directory: str):
    if backend == "memory":
        return InMemoryArtifactRegistry()
    return SQLiteArtifactRegistry(Path(temporary_directory) / "benchmark.sqlite3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--records", type=int, default=10_000)
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be positive")

    with tempfile.TemporaryDirectory() as temporary_directory:
        registry = create_registry(args.backend, temporary_directory)
        setup_started = time.perf_counter()
        for index in range(args.records):
            artifact_id = f"artifact-{index:06d}"
            key = f"key-{index:06d}"
            registry.begin_registration(
                artifact_id=artifact_id,
                case_id="benchmark-case",
                artifact_type=ArtifactType.CT_VOLUME,
                internal_uri=f"benchmark:{artifact_id}",
                created_by="benchmark",
                idempotency_key=key,
                producer_name="benchmark",
                producer_version="1.0.0",
            )
            registry.finalize(artifact_id, CHECKSUM, index)
        setup_seconds = time.perf_counter() - setup_started

        metadata_us: list[float] = []
        idempotency_us: list[float] = []
        for index in range(args.records):
            started = time.perf_counter_ns()
            registry.get_metadata(f"artifact-{index:06d}")
            metadata_us.append((time.perf_counter_ns() - started) / 1_000)

            started = time.perf_counter_ns()
            registry.find_available_by_idempotency_key(
                f"key-{index:06d}",
                case_id="benchmark-case",
            )
            idempotency_us.append((time.perf_counter_ns() - started) / 1_000)

        result = {
            "schema_version": "1",
            "backend": args.backend,
            "records": args.records,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "setup_seconds": round(setup_seconds, 6),
            "metadata_lookup_us": {
                "p50": round(statistics.median(metadata_us), 3),
                "p95": round(percentile(metadata_us, 0.95), 3),
            },
            "idempotency_lookup_us": {
                "p50": round(statistics.median(idempotency_us), 3),
                "p95": round(percentile(idempotency_us, 0.95), 3),
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        close = getattr(registry, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
