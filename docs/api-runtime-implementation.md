# API runtime implementation status

## Completed production boundaries

Task 07 is not complete. The first production-facing boundary provides:

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

The second production-facing boundary now provides:

- a framework-neutral `RunRepository` protocol and thread-safe
  `InMemoryRunRepository` reference backend;
- one atomic create operation that claims `(tenant_id, idempotency_key)` and
  persists `RUN_CREATED` plus `RUN_STARTED` before execution begins;
- a private monotonically increasing execution version. Normal node/tool events
  require the exact running version, while state changes use compare-and-swap;
- atomic state/event coupling for approval, success, failure and cancellation.
  Executors cannot append lifecycle events through the ordinary event stream;
- detached snapshot/event reads and tenant-mismatch `NOT_FOUND` behavior;
- bounded durable-JSON normalization at every repository write path, including
  request metadata, snapshots, executor outcomes and events. Binary values,
  cycles, non-finite numbers, non-string keys and oversized values fail closed;
- service and repository state invariants for approval IDs, errors and final
  reports, plus repository-level rejection of unrecoverable failure resume;
- cancellation/version fencing before and after executor callbacks, including
  buffered approval requests, so a superseded execution cannot publish a late
  event or overwrite a newer terminal state;
- atomic rollback when event construction, clock access, JSON normalization or
  compare-and-swap validation fails.

The execution version is deliberately internal and does not change the locked
public `RunSnapshot` or event contracts.

## Verification

Local Python 3.10 results on 2026-07-11:

```text
python3 run_tests.py
Ran 593 tests in 10.135s
OK (skipped=29)

PYTHONPATH=/tmp/lgpg:src:. python3 run_tests.py
Ran 593 tests in 26.354s
OK (skipped=12)

PYTHONPATH=src:. python3 -m unittest -q \
  tests.api.test_run_repository \
  tests.api.test_run_service \
  tests.integration.test_mock_workflow
Ran 32 tests in 0.258s
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
- commit `189040b0e38bcd12e6f5fec4bda20115a06e332e` completed
  [GitHub Actions run 29149762976](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29149762976)
  successfully;
- the Python 3.10, 3.11 and 3.12 jobs passed the complete suite and dedicated
  no-skip Pydantic/PostgreSQL gates; the independent PostgreSQL service-restart,
  checkpoint benchmark and real process-kill recovery jobs also passed;
- the new 20-test repository/concurrency suite covers 100-way idempotent create,
  100 concurrent ordered events, cancel/complete races, concurrent approve and
  resume, approval-event fencing, rollback, stale versions, durable JSON and
  forged lifecycle/outcome contract failures.

## Still not implemented

The following remain `NOT_RUN`/not implemented and must not be inferred from the
completed boundaries:

- all nine FastAPI HTTP endpoints and generated OpenAPI evidence;
- SSE event replay, `Last-Event-ID`, heartbeat, disconnect and backpressure;
- raw HTTP request-size enforcement before parsing;
- PostgreSQL Run/event/checkpoint/idempotency persistence and migrations (the
  current repository backend is intentionally in-memory only);
- process/SIGTERM restart recovery at the API Run layer;
- project/case/artifact authorization and an OIDC/JWT verifier;
- artifact metadata, health and low-cardinality metrics endpoints;
- HTTP concurrency, 10,000-event replay and production performance baselines.
