# API runtime implementation status

## Current completed boundary

Task 07 is not complete. The first production-facing boundary now provides:

- Pydantic v2 request, approval, snapshot, event and structured-error adapters
  without changing the locked framework-neutral dataclasses;
- authenticated tenant/principal and access-scope injection outside the body;
- recursive rejection of caller-supplied authority fields, raw image/voxel
  payloads, binary/non-JSON values, non-finite numbers, credentials, JWTs and
  URI-bearing values, plus explicit rejection of top-level test-only controls;
- bounded normalized JSON depth, node count and 1 MiB serialized size;
- detached public snapshot/event views with recursive secret, patient-identity,
  prompt, internal-location and binary redaction;
- an allowlisted HTTP error code/status mapping. Unknown internal errors use a
  fixed `500 INTERNAL_ERROR`; retryable unknown dependencies use a fixed
  `503 SERVICE_UNAVAILABLE` and never expose the source exception;
- explicit Pydantic `2.13.4` CI installation plus a dedicated test gate that
  fails if any Pydantic contract test is skipped.

The normalized 1 MiB model limit is not an HTTP parsing limit. The future
FastAPI transport must reject an oversized raw request body before JSON parsing.

## Verification

Local Python 3.10 results on 2026-07-11:

```text
python3 run_tests.py
Ran 573 tests in 9.850s
OK (skipped=29)

PYTHONPATH=/tmp/lgpg:src:. python3 -m unittest discover -s tests -p 'test_*.py' -q
Ran 573 tests in 25.715s
OK (skipped=12)

PYTHONPATH=/tmp/lgpg:src:. python3 -m unittest -v tests.api.test_http_contracts
Ran 9 tests
OK
```

The dependency-free run executes the two framework-neutral privacy tests and
skips seven Pydantic tests explicitly. The implementation-dependency run and CI
execute all nine. The Pydantic suite pins JSON-schema required fields, task
enum, artifact count, extra-field rejection and JSON round-trip behavior.

Remote evidence on 2026-07-11:

- commit `3c4c6fcfc6d0bd54bcf52089e45707267f1f04c2` completed
  [GitHub Actions run 29148721224](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29148721224)
  successfully;
- the Python 3.10, 3.11 and 3.12 jobs each installed Pydantic `2.13.4`, imported
  the adapters and passed the dedicated `Pydantic API contracts (skips
  forbidden)` step with zero skips;
- the same workflow also passed all regular suites, PostgreSQL checkpoint and
  lease gates, service restart, checkpoint benchmark and process-kill recovery.

## Still not implemented

The following remain `NOT_RUN`/not implemented and must not be inferred from the
contract adapter:

- all nine FastAPI HTTP endpoints and generated OpenAPI evidence;
- SSE event replay, `Last-Event-ID`, heartbeat, disconnect and backpressure;
- raw HTTP request-size enforcement before parsing;
- a repository abstraction and atomic cancel/execution state transitions;
- PostgreSQL Run/event/checkpoint/idempotency persistence and migrations;
- process/SIGTERM restart recovery at the API Run layer;
- project/case/artifact authorization and an OIDC/JWT verifier;
- artifact metadata, health and low-cardinality metrics endpoints;
- HTTP concurrency, 10,000-event replay and production performance baselines.
