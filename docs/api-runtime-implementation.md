# API runtime implementation status

## Completed production boundaries

Task 07's repository/API runtime scope is complete through six
production-facing boundaries. The first boundary provides:

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

The fifth production-facing boundary now provides:

- representation negotiation on the existing event endpoint: bounded JSON by
  default and explicit `text/event-stream` for SSE, with `406` for unsupported
  media types and `Vary: Accept` on both successful representations;
- strict exclusive reconnect cursors through `after_sequence` and
  `Last-Event-ID`. Duplicate, conflicting, non-canonical, negative and oversized
  values fail before stream startup;
- a separate bounded `RunEventPage` repository/service port. The PostgreSQL
  implementation captures a high-water sequence, then performs a lock-free
  keyset query bounded by that high water so concurrent appends do not cause
  gaps, duplicates or unbounded reads;
- fixed `id`/`event`/compact-JSON `data` frames, comment-only heartbeats,
  Bearer re-authentication plus resource re-authorization on every poll,
  terminal high-water draining and normal EOF;
- preflight authentication, authorization, first page read and serialization.
  Once the 200 response starts, authorization, dependency and contract failures
  close silently instead of exposing exception text in the stream;
- idempotent cleanup across normal completion, cancellation, authorization
  revocation and ASGI send failure, plus per-process global and tenant
  connection limits;
- low-cardinality SSE metrics for fixed cursor source, close outcome,
  `EventType` and heartbeat count. No tenant, Run, case, principal, trace or
  event ID becomes a label.

The connection limiter is intentionally process-local, and polling is used
instead of PostgreSQL notification. Those deployment tradeoffs are documented
and do not implement a company identity or algorithm backend. The stream
deadline wins over terminal draining; a client interrupted mid-tail resumes
from its last event ID. Injected synchronous authentication/authorization ports
must also enforce their own backend deadlines because Python cannot kill a
company callback thread that ignores cancellation.

The sixth production-facing boundary now provides:

- a private durable execution intent per Run version. `CREATE`, full
  `APPROVAL` decisions and `RESUME` are committed with their lifecycle
  transition instead of relying on process memory;
- PostgreSQL migration v2 for `run_execution_jobs`, while preserving the v1
  migration SQL/checksum byte-for-byte. Claims use a generation, unique owner
  token, worker ID, database-clock heartbeat and lease;
- `FOR UPDATE SKIP LOCKED` bounded claiming across API/worker instances, plus
  commit-unknown reconciliation for claim, heartbeat, claimed append and
  claimed lifecycle CAS;
- same-transaction claim fencing on every execution event and terminal write.
  Reclaim increments generation, while stable version-scoped event keys remain
  generation-independent for exact replay;
- a bounded `RunWorker` supervisor with independent heartbeat threads,
  configurable concurrency/poll/lease/grace, low-cardinality metrics, visible
  supervisor failure, duplicate-owner defense and conservative TTL takeover;
- FastAPI lifespan composition that migrates before worker startup, reports a
  stopped supervisor as unhealthy, exports worker metrics, stops new claims on
  shutdown, signals active execution and keeps heartbeating only through the
  grace period;
- process-level SIGTERM evidence: process A commits one injected-port result,
  times out shutdown without releasing the active lease, and exits; process B
  reclaims the same Run at generation 2, reuses the stable `call_id`, reaches
  `SUCCEEDED`, preserves contiguous events and emits exactly one terminal
  event;
- an explicit `RecoverableRunExecutor` port and `RunExecutionContext`. Company
  algorithms remain unimplemented; the test executor only proves connectivity
  to an injected, PostgreSQL-idempotent port.

Every historical Run receives completed version-1 `CREATE` history so
idempotent replay remains valid. A version-1 `RUNNING` row keeps that CREATE job
claimable; a later-version `RUNNING` row also receives a conservative current
`RESUME` job. A decision that existed only in a pre-v2 crashed approval process
cannot be reconstructed and requires deployment review. New approval jobs
store the complete decision atomically.

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
Ran 653 tests
OK (skipped=61)

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 run_tests.py
Ran 653 tests
OK (skipped=6)

PYTHONPATH=<FastAPI dependencies>:src:. python3 -m unittest \
  tests.api.test_fastapi_app.SseCoreTests \
  tests.api.test_fastapi_app.FastApiSseTests -v
Ran 8 tests
OK

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 -m unittest \
  tests.api.test_fastapi_app.FastApiPostgresIntegrationTests \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests -v
Ran 9 tests
OK

PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 -m unittest \
  tests.api.test_http_contracts.PydanticHttpContractTests \
  tests.api.test_fastapi_app.RawBodyAdmissionTests \
  tests.api.test_fastapi_app.FastApiTransportTests \
  tests.api.test_fastapi_app.SseCoreTests \
  tests.api.test_fastapi_app.FastApiSseTests \
  tests.api.test_run_worker.RunWorkerTests -v
Ran 38 tests
OK (zero skips)

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. python3 -m unittest \
  tests.api.test_postgres_run_repository.PostgresRunExecutionRepositoryTests -v
Ran 4 tests
OK (zero skips)

PUNCTURE_TEST_POSTGRES_DSN=<private-test-dsn> \
PUNCTURE_API_SIGTERM_EVIDENCE_DIR=<private-output-directory> \
PYTHONPATH=<FastAPI/LangGraph dependencies>:src:. \
python3 -m tests.api.postgres_api_sigterm_probe orchestrate
generation 1 -> 2; status SUCCEEDED; side_effect_count 1; event_count 5
```

The dependency-free run executes the pure-ASGI body-admission tests, the two
framework-neutral privacy tests and repository contract tests while explicitly
skipping implementation-backed transport/database cases. The dependency run
executes all seven Pydantic tests, all fourteen FastAPI transport/body tests,
the nine worker tests and the dedicated eight-test SSE gate; the PostgreSQL
environment adds the one HTTP composition test, eight Run repository tests and
four execution-job tests. The Pydantic suite pins
JSON-schema required fields, task enum, artifact count, extra-field rejection
and JSON round-trip behavior.

### API-001 through API-016

| Case | Result | Evidence |
|---|---|---|
| API-001 valid create | PASS | deferred PostgreSQL/FastAPI composition reaches the Mock terminal state |
| API-002 idempotent create | PASS | tenant-scoped unique request returns one Run and one execution |
| API-003 empty case | PASS | Pydantic/body admission rejects before Run creation |
| API-004 unknown task type | PASS | locked task enum returns structured invalid request |
| API-005 unknown Run | PASS | tenant-safe fixed `NOT_FOUND` mapping |
| API-006 ordered replay | PASS | contiguous repository sequence and SSE/JSON replay tests |
| API-007 reconnect cursor | PASS | strict exclusive cursor and no acknowledged-event replay |
| API-008 approval resume | PASS | complete approval intent is atomically enqueued and executed |
| API-009 wrong approval | PASS | fixed conflict with unchanged checkpoint |
| API-010 repeated approval | PASS | no second job/node execution |
| API-011 cancel running | PASS | version/claim fence rejects later executor writes |
| API-012 terminal cancel/resume | PASS | lifecycle conflict leaves terminal state unchanged |
| API-013 unauthorized resources | PASS | current tenant/case/artifact authorization fails closed |
| API-014 dependency timeout | PASS | recoverable `FAILED` checkpoint and fixed error |
| API-015 process recovery | PASS | API SIGTERM generation reclaim plus existing durable tool replay evidence |
| API-016 redaction | PASS | public response/event/privacy and metric-label tests |

The local injected failure probe used a 0.60-second lease and 0.15-second
shutdown grace. Process B reached the recovered terminal state in about 0.42
seconds after it started polling. This is fault-injection evidence, not a
production latency target. No new production HTTP/PostgreSQL throughput claim
is made.

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
- commit `99587564d50b71f82fa762d6c64fcf20308fc85a` completed
  [GitHub Actions run 29157341457](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29157341457)
  successfully;
- each Python 3.10, 3.11 and 3.12 job passed the full suite plus the dedicated,
  exact eight-test SSE replay/streaming gate with zero skips, while the existing
  Pydantic, FastAPI, PostgreSQL Run repository and PostgreSQL composition gates
  remained green;
- the SSE tests cover canonical reconnect cursors, Accept specificity,
  heartbeat/approval completion, 10,000-event bounded paging, token revocation,
  capacity rejection, preflight failures, ASGI start/body-send cleanup and
  low-cardinality privacy metrics;
- PostgreSQL restart recovery, real process-kill replay protection and the
  hosted checkpoint benchmark also remained successful.
- commit `d30603850b18ccff46709de799a1fca6eae08883` completed
  [GitHub Actions run 29159768707](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29159768707)
  successfully;
- each Python 3.10, 3.11 and 3.12 matrix job passed the complete 653-test suite
  plus exact zero-skip gates for fourteen FastAPI/body tests, nine worker
  tests, eight PostgreSQL Run Repository tests, four PostgreSQL execution-job
  tests and one FastAPI/PostgreSQL composition test;
- the independent PostgreSQL 16 API SIGTERM job exported evidence for the same
  Run moving from generation 1 to 2, five contiguous events, one terminal
  `RUN_COMPLETED`, one stable-call side effect and stopped workers in both
  processes;
- PostgreSQL restart, graph process-kill replay protection and checkpoint
  benchmark jobs also remained green.

## Still not implemented

The following remain `NOT_RUN`/not implemented and must not be inferred from the
completed boundaries:

- a concrete OIDC/JWT verifier, company case/project authorizer, company
  artifact gateway and company algorithm executor (their ports are complete);
- production exactly-once semantics for arbitrary company/GPU/external side
  effects. The recovery probe supplies a test-only PostgreSQL stable-call
  ledger; each production adapter still needs a durable idempotency/outbox
  boundary;
- a production HTTP/PostgreSQL concurrency and performance baseline. The
  in-memory bounded 10,000-event replay correctness test is complete, but it is
  not a production throughput claim;
- reverse-proxy slow-consumer/backpressure evidence and cluster-wide SSE
  capacity control;
- host failure, network partition, PostgreSQL SIGKILL/WAL crash recovery and
  forcible cancellation of a synchronous GPU/company callback that ignores the
  cooperative stop signal.
