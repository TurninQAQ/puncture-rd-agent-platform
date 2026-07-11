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

The repository boundary also carries a private stream-event `event_key` that is
not exposed through `RunEvent`, HTTP or SSE. The service namespaces explicit
executor keys by execution version and provides an ordinal fallback for the
deterministic Mock executors. An exact key/content replay returns the original
sequence; changed content is a fixed conflict. The service rechecks the version
fence after reconciliation so an old executor cannot continue after cancel.
Canonical SHA-256 fingerprints preserve JSON type differences such as `true`
versus `1`. Production graph executors must use checkpoint task IDs and tool
call/idempotency IDs instead of relying on the ordinal fallback during mid-graph
recovery.

The third production-facing boundary now provides:

- a lazy-psycopg `PostgresRunRepository` with explicit, advisory-lock-protected,
  checksum-verified deployment migration; request paths never auto-migrate;
- tenant-scoped idempotent create, row-locked monotonic event sequences and
  version-fenced compare-and-swap transitions across repository instances;
- canonical text sidecars beside JSONB for requests, snapshot dynamic fields
  and event payloads. Reads re-parse the canonical text and recompute request,
  snapshot and event fingerprints, preserving values such as `-0.0` and
  exponent-form floats that JSONB may normalize;
- one strict shared `YYYY-MM-DDTHH:MM:SS.mmmZ` boundary for memory and database
  snapshots/events, with PostgreSQL millisecond-precision constraints;
- a durable mutation journal written in the same transaction as snapshot and
  lifecycle-event changes. Exact CAS replay therefore returns the original
  target version even after a later approval/resume has advanced the current
  Run;
- explicit COMMIT-unknown reconciliation for create, keyed stream append and
  lifecycle CAS. Tests wrap a real psycopg connection so the underlying
  `commit()` succeeds before an acknowledgement-loss exception is raised;
- sanitized configuration, availability and integrity errors without DSN or
  backend exception leakage.

The execution version is deliberately internal and does not change the locked
public `RunSnapshot` or event contracts.

The fourth production-facing boundary now provides:

- an isolated FastAPI application factory with all nine fixed REST/OpenAPI
  paths, stable operation IDs, fixed structured errors, and no endpoint-local
  Agent or medical-algorithm logic;
- Bearer verification before body buffering/JSON parsing, plus explicit
  `PrincipalAuthenticator`, tenant/project/case `ResourceAuthorizer`, and atomic
  `ArtifactAccessGateway` ports. There is no permissive default principal;
- a server-owned project binding persisted in Run request metadata, redacted in
  public snapshots, and compared with current authorization on every Run read
  or state-changing operation;
- pure-ASGI raw body admission that rejects ambiguous/forged lengths,
  compression, wrong JSON media types, chunked overflow and single oversized
  chunks before copying them into the bounded parser buffer;
- a seven-field artifact metadata response with requested-ID consistency,
  public-value validation and no URI/checksum/private metadata;
- low-cardinality Prometheus HTTP metrics, non-cacheable operational responses,
  repository readiness that verifies migration checksum plus all core tables,
  and `UP`/`DEGRADED`/`DOWN` health semantics;
- `PostgresApiSettings` and `create_postgres_app`, which compose an injected
  executor with `InMemoryRunService` and `PostgresRunRepository`. Migration is
  an explicit lifespan option and never occurs in a request path.

The class name `InMemoryRunService` is historical; the service accepts the
durable PostgreSQL repository. Company algorithms, identity verification data,
case/project indexes and artifact storage implementations remain injected
interfaces, as required.

## Verification

Local Python 3.10 results on 2026-07-11:

```text
python3 run_tests.py
Ran 608 tests in 10.067s
OK (skipped=37)

PYTHONPATH=/tmp/lgpg:src:. python3 run_tests.py
Ran 608 tests in 26.214s
OK (skipped=20)

PYTHONPATH=src:. python3 -m unittest -q \
  tests.api.test_run_repository \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryContractTests
Ran 27 tests in 0.350s
OK

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=/tmp/lgpg:src:. python3 -m unittest \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests -v
Ran 8 tests in 3.117s
OK

python3 run_tests.py (dependency-free final regression)
Ran 623 tests in 10.116s
OK (skipped=48)

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 run_tests.py
Ran 623 tests in 34.129s
OK (skipped=6)

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 -m unittest \
  tests.api.test_fastapi_app.FastApiPostgresIntegrationTests \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests -v
Ran 9 tests
OK
```

The dependency-free run executes the pure-ASGI body-admission tests, the two
framework-neutral privacy tests and repository contract tests while explicitly
skipping implementation-backed transport/database cases. The dependency run
executes all seven Pydantic tests and all thirteen FastAPI transport/body tests;
the PostgreSQL environment adds the one HTTP composition test and eight Run
repository tests. The Pydantic suite pins JSON-schema required fields, task
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
- commit `15af386033d7881748861a1333556b68c675a602` completed
  [GitHub Actions run 29150409932](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29150409932)
  successfully across the complete Python/PostgreSQL matrix;
- the added event-identity tests cover exact replay, changed-content conflict,
  JSON type-sensitive fingerprints, invalid raw keys and cancel-after-replay
  fencing without changing the public event contract.
- commit `93391f4743f4a0592671eba231fc3f2684d0d0f4` completed
  [GitHub Actions run 29152000095](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29152000095)
  successfully;
- each Python 3.10, 3.11 and 3.12 matrix job executed exactly eight PostgreSQL
  16 Run Repository tests with skips forbidden. The tests cover persistence and
  migration checksum enforcement, 20-way idempotent create, tenant isolation,
  100 concurrent ordered events, CAS competition/history, transaction rollback,
  backend termination, and create/append/CAS acknowledgement loss after a real
  successful commit;
- the same workflow kept the independent PostgreSQL service-restart,
  checkpoint benchmark and real process-kill recovery jobs green.
- commit `b8da00f3e2e51c967308fe52ab7fbb8b7e37cf6d` completed
  [GitHub Actions run 29155728747](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29155728747)
  successfully;
- each Python 3.10, 3.11 and 3.12 matrix job passed the full suite plus exact,
  zero-skip gates for seven Pydantic contracts, thirteen FastAPI/body-admission
  tests, eight PostgreSQL Run repository tests and one FastAPI/PostgreSQL
  composition test;
- the same workflow kept PostgreSQL restart recovery, real process-kill replay
  protection and the hosted checkpoint benchmark green.

## Still not implemented

The following remain `NOT_RUN`/not implemented and must not be inferred from the
completed boundaries:

- SSE event replay, `Last-Event-ID`, heartbeat, disconnect and backpressure;
- an execution reclaim/heartbeat protocol for workers that die while a Run
  remains `RUNNING`;
- process/SIGTERM restart recovery at the API Run layer;
- a concrete OIDC/JWT verifier, company case/project authorizer, company
  artifact gateway and company algorithm executor (their ports are complete);
- HTTP concurrency, 10,000-event replay and production performance baselines.
