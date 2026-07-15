# FastAPI Run gateway

This module exposes the fixed Run contracts over HTTP while keeping company
algorithms, identity data, case/project ownership, and private artifact storage
behind injected ports.

## Boundary

Implemented in this node:

- all nine REST/OpenAPI paths from `specs/fastapi-gateway.md`;
- strict Bearer-header admission and an injected token verifier;
- action-specific server-side project/case authorization;
- an atomic artifact authorization/metadata port that returns only
  `ArtifactPublicView`;
- raw request-size, `Content-Length`, `Content-Encoding`, and JSON media-type
  checks before FastAPI parses the body;
- bounded JSON event pages plus ordered SSE replay with canonical reconnect
  cursors, heartbeat comments, terminal-tail draining and connection limits;
- fixed public error responses, health, low-cardinality Prometheus metrics, and
  PostgreSQL Run/event repository composition;
- durable execution intents, background worker claims, independent heartbeat,
  lease expiry/reclaim, fenced writes and FastAPI lifespan shutdown/recovery.

The company executor is intentionally not implemented. With the production
worker enabled, `create_postgres_app` requires an injected
`RecoverableRunExecutor` with `recovery_safe = True`; its claimed entry point
receives the Run/trace/version/generation identity, active-claim assertion and
cooperative stop callback. The real implementation can therefore be connected
later without changing an endpoint. Artifact and authorization follow the same
injected-port rule.

## Endpoints

| Method | Path | Normal response |
|---|---|---|
| POST | `/api/v1/runs` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/runs/{run_id}` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/runs/{run_id}/events` | bounded `RunEventResponse[]` or ordered SSE, 200 |
| POST | `/api/v1/runs/{run_id}/approvals/{approval_id}` | `RunSnapshotResponse`, 200 |
| POST | `/api/v1/runs/{run_id}/cancel` | `RunSnapshotResponse`, 200 |
| POST | `/api/v1/runs/{run_id}/resume` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/artifacts/{artifact_id}/metadata` | `ArtifactMetadataResponse`, 200 |
| GET | `/health` | `UP`, `DEGRADED`, or 503 `DOWN` |
| GET | `/metrics` | Prometheus text |

Create returns 200 for both the first request and an exact idempotent replay.
The current service contract does not expose whether the repository created or
reused the Run, so returning 201 would be unreliable.

## Event replay and SSE

`GET /api/v1/runs/{run_id}/events` returns a bounded JSON page by default.
`after_sequence` is an exclusive cursor and `limit` is 1–512 with a default of
512. An explicit `Accept: text/event-stream` selects SSE; unsupported media
types return a structured `406 INVALID_ARGUMENT`.

SSE reconnect accepts `Last-Event-ID` and/or `after_sequence`. Values use the
canonical non-negative decimal form (`0` or a number without leading zeroes),
and duplicate or conflicting cursors fail before a 200 response. Every event is
one fixed frame:

```text
id: 3
event: NODE_STARTED
data: {"run_id":"run-1","sequence":3,...}

```

The `data` value is the compact public `RunEventResponse`; heartbeat frames are
comments and carry no event ID. The stream reads committed events in bounded
pages, re-authenticates the Bearer token and re-authorizes the resource on every
poll, and drains the committed high-water tail before closing a terminal Run.
Authentication, authorization, cursor parsing, the first repository page and
first-page serialization all complete before the SSE 200 starts. Later failures
close the stream without emitting exception text or a private synthetic event.
Only the `q` Accept parameter is supported; parameterized media types are
rejected rather than silently changing the UTF-8 stream contract.

The configured maximum lifetime takes precedence over terminal-tail draining.
If a deadline or disconnect interrupts a committed tail, the client reconnects
with the last event ID and resumes from the next sequence without duplication.

Connection limits are per API process: one global limit and one per-tenant
limit. They are not a cluster-wide quota. `Cache-Control` disables storage and
proxy transformation, `X-Accel-Buffering: no` disables common reverse-proxy
buffering, and both JSON and SSE responses use `Vary: Accept`.

Browser `EventSource` cannot attach the required Bearer header. Browser clients
must use streaming `fetch` (or an equivalent authenticated streaming client)
and reconnect with the last committed event ID.

## Required injected ports

`puncture_agent.api.fastapi_app.create_app` requires:

- `PrincipalAuthenticator`: verifies the Bearer token and returns an
  `AuthenticatedPrincipal`;
- `ResourceAuthorizer`: resolves authoritative tenant/project/case ownership
  and checks every Run action independently;
- `RunService`: owns lifecycle and repository behavior;
- optionally `ArtifactAccessGateway`: atomically authorizes artifact use/read
  and returns public metadata only.

There is no permissive default identity. Missing, repeated, malformed, empty,
or oversized Authorization headers fail closed. Caller-provided tenant,
principal, role, scope, project, or case authority is never trusted.

The project ID is resolved by `ResourceAuthorizer`; it is not a request field.
An implementation can therefore connect the company's case-to-project index
without changing the frozen Run contract. The resolved ID is persisted as a
server-owned `request.metadata.project_id`, redacted in public snapshots, and
must still match the current authorizer result on every later Run operation.

Artifact-bearing create requests require the gateway to return the exact input
IDs in order, all in the authorized case and all `AVAILABLE`. The HTTP endpoint
never receives an internal URI, checksum, private metadata, or raw volume.

## PostgreSQL composition

`puncture_agent.api.postgres_app.create_postgres_app` composes:

```text
FastAPI transport
        |
InMemoryRunService (deferred mode; historical class name)
        | durable intent + wakeup
PostgresRunRepository <---- RunWorker heartbeat/claim/reclaim
        |
injected RecoverableRunExecutor
```

Despite its historical class name, `InMemoryRunService` accepts a durable
`RunRepository`; only its default repository is in memory.

Configuration uses:

| Variable | Default | Meaning |
|---|---:|---|
| `PUNCTURE_API_POSTGRES_DSN` | required | private PostgreSQL connection string |
| `PUNCTURE_API_POSTGRES_SCHEMA` | `puncture_runtime` | Run/event schema |
| `PUNCTURE_API_POSTGRES_CONNECT_TIMEOUT_SECONDS` | `5` | connection timeout |
| `PUNCTURE_API_POSTGRES_STATEMENT_TIMEOUT_MS` | `5000` | statement timeout |
| `PUNCTURE_API_POSTGRES_LOCK_TIMEOUT_MS` | `1000` | lock timeout |
| `PUNCTURE_API_MAX_REQUEST_BODY_BYTES` | `1048576` | raw body limit |
| `PUNCTURE_API_MIGRATE_ON_STARTUP` | `false` | explicit lifespan migration hook |
| `PUNCTURE_API_SSE_PAGE_SIZE` | `128` | committed events read per SSE page, 1–512 |
| `PUNCTURE_API_SSE_POLL_INTERVAL_SECONDS` | `1` | idle polling interval, 0.01–10 s |
| `PUNCTURE_API_SSE_HEARTBEAT_SECONDS` | `15` | idle heartbeat interval, 0.01–60 s and not shorter than polling |
| `PUNCTURE_API_SSE_MAX_CONNECTION_SECONDS` | `600` | maximum stream lifetime, 0.05–3600 s |
| `PUNCTURE_API_SSE_MAX_CONNECTIONS` | `200` | per-process global stream limit |
| `PUNCTURE_API_SSE_MAX_CONNECTIONS_PER_TENANT` | `20` | per-process tenant stream limit |
| `PUNCTURE_API_WORKER_ENABLED` | `true` | enable durable background execution |
| `PUNCTURE_API_WORKER_ID` | generated | bounded process/worker identity |
| `PUNCTURE_API_WORKER_CONCURRENCY` | `1` | maximum active claims, 1–256 |
| `PUNCTURE_API_WORKER_POLL_INTERVAL_SECONDS` | `0.5` | idle durable-job polling interval |
| `PUNCTURE_API_WORKER_HEARTBEAT_INTERVAL_SECONDS` | `5` | independent claim heartbeat interval |
| `PUNCTURE_API_WORKER_LEASE_SECONDS` | `30` | database-clock lease; at least 3 heartbeats |
| `PUNCTURE_API_WORKER_SHUTDOWN_GRACE_SECONDS` | `30` | cooperative execution shutdown grace |

Prefer a one-shot deployment migration and leave
`PUNCTURE_API_MIGRATE_ON_STARTUP=false` on API replicas. If startup migration is
enabled, it runs once in the FastAPI lifespan before requests are accepted; it
never runs in a request path. `/health` verifies both PostgreSQL connectivity
and the exact stored migration checksum without returning the DSN, host, schema,
or backend exception.

The v1-to-v2 deployment is not a mixed-version rolling-write protocol. Quiesce
v1 API writers, apply v2, then start worker-enabled replicas; an old process
cannot persist the new approval intent after v2 is installed.

Worker startup follows migration in the lifespan. Shutdown first stops new
claims, signals active executors, and continues heartbeat during the grace
period. If an executor still has not returned, heartbeat stops and the claim is
left untouched until its PostgreSQL lease expires; it is not immediately
released into concurrent execution. `/health` is `DOWN` when the configured
worker supervisor is no longer running. `/metrics` adds fixed-dimension worker
claim, heartbeat, outcome, claim-loss, shutdown-timeout and supervisor-failure
counters without Run, tenant, case, worker or trace labels.

Setting `PUNCTURE_API_WORKER_ENABLED=false` retains the old synchronous mode for
local compatibility. It does not provide process-restart recovery and should
not be presented as the production topology.

## Durable execution and recovery

Migration v2 adds private `run_execution_jobs` rows keyed by
`(tenant_id, run_id, execution_version)`. Each row stores the canonical and
fingerprinted execution intent (`CREATE`, full `APPROVAL` decision, or
`RESUME`), generation, owner token, worker ID, claimed/heartbeat/lease times,
and release/completion times. The public Run and event schemas do not change.

Claims use PostgreSQL's clock and `FOR UPDATE SKIP LOCKED`. Every ordinary
event append and terminal transition validates generation, owner, intent,
unexpired lease, current Run version and `RUNNING` status in the same
transaction as the write. Reclaim increments generation. Generation is not
part of the version-scoped event key, so an exact recovery replay returns the
original sequence while changed content conflicts. Approval decisions are
persisted atomically with `WAITING_APPROVAL -> RUNNING`, rather than being held
only in API-process memory.

The v2 migration backfills a completed version-1 `CREATE` history for every
legacy Run so idempotent create replay remains valid. A version-1 Run that is
still `RUNNING` keeps that CREATE job claimable; a later-version `RUNNING` Run
also receives a conservative current `RESUME` job. A pre-v2 process crash that
lost an in-memory approval decision cannot reconstruct that decision; the
deployer must review such legacy Runs. New approval jobs preserve the complete
decision.

The process-level probe starts the real FastAPI/PostgreSQL composition, commits
one injected-port result under a stable PostgreSQL `call_id`, sends SIGTERM,
and starts a second API process. The second worker reclaims the same Run at a
higher generation, replays the stable event identity, observes the existing
port result and completes with one terminal event. This verifies the recovery
plumbing and injected idempotency contract. It does not claim that an arbitrary
company algorithm or external service is exactly-once: production adapters
must supply stable tool/task IDs plus their own durable idempotency/outbox
boundary for side effects.

Example composition (the named implementations are supplied by the deployer):

```python
from puncture_agent.api.postgres_app import (
    PostgresApiSettings,
    create_postgres_app,
)

settings = PostgresApiSettings.from_env()
app = create_postgres_app(
    settings,
    executor=company_executor,
    authenticator=oidc_authenticator,
    authorizer=company_case_authorizer,
    artifact_gateway=company_artifact_gateway,
)
```

The example names are integration placeholders, not implementations shipped by
this repository.

## Runnable local full-stack composition

`deploy/local-demo` is the explicit workstation composition for learning and
integration verification. After PostgreSQL, Qwen/vLLM, and OpenSearch are up,
copy its `.env.example`, keep the resulting file at mode 0600, review the
synthetic boundary, and run:

```bash
./deploy/local-demo/run_demo.sh
```

The temporary loopback API uses the real PostgreSQL repository, vLLM gateway,
OpenSearch adapter, JSON event API, and SSE replay. The integrated executor's
model and RAG ports are injected with those live clients; its algorithm tools
remain deterministic synthetic implementations. The script validates both
fixed workflows, bad-token rejection, idempotent creation, event evidence, and
clean SIGTERM shutdown. Exact setup and results are in
`deploy/local-demo/README.md` and
`deploy/local-demo/evidence/local-full-stack-validation.md`.

This profile deliberately forces synchronous execution and has no Artifact
Access Gateway, so `/health` is `DEGRADED`. It is not a substitute for the
recovery-safe worker topology, OIDC/IAM, API TLS, company algorithms, or a
production performance run.

## Privacy and operational behavior

- API error handlers never return Pydantic input, exception text, tokens,
  database locations, or private provider messages.
- Persisted creator access scopes are redacted from public snapshots. They are
  not a substitute for current authorization; every read, approval, cancel, and
  resume is authorized again.
- Production execution still needs a current-scope provider before persisted
  creation-time scopes can be removed from the graph integration.
- HTTP metrics use only method, fixed route template, and status class. SSE
  metrics use only fixed cursor source, close outcome, and `EventType` labels.
  They never use tenant, project, case, Run, artifact, principal, trace,
  approval, or event identifiers as labels.
- `/api/**` responses use `Cache-Control: no-store` and
  `X-Content-Type-Options: nosniff`.

## Verification

Dependency-backed transport tests:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[implementation]'
python -m unittest tests.api.test_fastapi_app -v
```

Dedicated SSE replay/streaming tests:

```bash
python -m unittest \
  tests.api.test_fastapi_app.SseCoreTests \
  tests.api.test_fastapi_app.FastApiSseTests -v
```

PostgreSQL wiring test:

```bash
PUNCTURE_TEST_POSTGRES_DSN='<private-test-dsn>' \
python -m unittest \
  tests.api.test_fastapi_app.FastApiPostgresIntegrationTests -v
```

Worker and process-recovery tests:

```bash
python -m unittest tests.api.test_run_worker -v

PUNCTURE_TEST_POSTGRES_DSN='<private-test-dsn>' \
PUNCTURE_API_SIGTERM_EVIDENCE_DIR='<private-output-directory>' \
python -m tests.api.postgres_api_sigterm_probe orchestrate
```

CI pins FastAPI `0.115.12`, HTTPX `0.28.1`, and Pydantic `2.13.4`, and has
dedicated no-skip gates for fourteen transport/body tests, nine worker tests,
eight SSE tests, eight PostgreSQL Run repository tests, four PostgreSQL
execution-job tests, PostgreSQL composition, and an independent PostgreSQL 16
SIGTERM/reclaim job with uploaded evidence.

The 10,000-event test proves bounded, gap-free in-memory paging and frame order;
it is not a production latency or throughput baseline. Real reverse-proxy slow
consumer behavior, cluster-wide capacity control, database fault timing during
an already-open stream, and production HTTP/PostgreSQL performance still need
deployment-environment evidence.

The recovery probe does not cover host failure, network partition, PostgreSQL
WAL crash recovery, GPU-kernel cancellation, or atomicity between a company
system's internal side effect and its idempotency record. An in-flight
synchronous callback that ignores the stop signal can remain alive until the
process exits; generation/lease fencing prevents its late repository writes.

Injected authentication/authorization implementations must enforce their own
network and backend deadlines. The SSE wrapper stops awaiting them at the stream
deadline, but Python cannot forcibly terminate a synchronous company callback
that ignores cancellation; PostgreSQL calls are bounded separately by connect
and statement timeouts.
