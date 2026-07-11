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
- fixed public error responses, health, low-cardinality Prometheus metrics, and
  PostgreSQL Run/event repository composition.

The company executor is intentionally not implemented. `create_postgres_app`
requires a `RunExecutor` instance, so the real implementation can be connected
later without changing an endpoint. The artifact and authorization ports follow
the same rule.

This node returns committed events as JSON from
`GET /api/v1/runs/{run_id}/events?after_sequence=N`. SSE reconnect, heartbeat,
disconnect/backpressure handling, asynchronous worker dispatch, execution
heartbeat/reclaim, and SIGTERM recovery are later Task 07 nodes.

## Endpoints

| Method | Path | Normal response |
|---|---|---|
| POST | `/api/v1/runs` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/runs/{run_id}` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/runs/{run_id}/events` | ordered `RunEventResponse[]`, 200 |
| POST | `/api/v1/runs/{run_id}/approvals/{approval_id}` | `RunSnapshotResponse`, 200 |
| POST | `/api/v1/runs/{run_id}/cancel` | `RunSnapshotResponse`, 200 |
| POST | `/api/v1/runs/{run_id}/resume` | `RunSnapshotResponse`, 200 |
| GET | `/api/v1/artifacts/{artifact_id}/metadata` | `ArtifactMetadataResponse`, 200 |
| GET | `/health` | `UP`, `DEGRADED`, or 503 `DOWN` |
| GET | `/metrics` | Prometheus text |

Create returns 200 for both the first request and an exact idempotent replay.
The current service contract does not expose whether the repository created or
reused the Run, so returning 201 would be unreliable.

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
injected RunExecutor
        |
InMemoryRunService (service name retained for compatibility)
        |
PostgresRunRepository
        |
FastAPI transport
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

Prefer a one-shot deployment migration and leave
`PUNCTURE_API_MIGRATE_ON_STARTUP=false` on API replicas. If startup migration is
enabled, it runs once in the FastAPI lifespan before requests are accepted; it
never runs in a request path. `/health` verifies both PostgreSQL connectivity
and the exact stored migration checksum without returning the DSN, host, schema,
or backend exception.

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

## Privacy and operational behavior

- API error handlers never return Pydantic input, exception text, tokens,
  database locations, or private provider messages.
- Persisted creator access scopes are redacted from public snapshots. They are
  not a substitute for current authorization; every read, approval, cancel, and
  resume is authorized again.
- Production execution still needs a current-scope provider before persisted
  creation-time scopes can be removed from the graph integration.
- Metrics use only method, fixed route template, and status class. They never
  use tenant, project, case, Run, artifact, principal, trace, approval, or event
  identifiers as labels.
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

PostgreSQL wiring test:

```bash
PUNCTURE_TEST_POSTGRES_DSN='<private-test-dsn>' \
python -m unittest \
  tests.api.test_fastapi_app.FastApiPostgresIntegrationTests -v
```

CI pins FastAPI `0.115.12`, HTTPX `0.28.1`, and Pydantic `2.13.4`, and has
dedicated no-skip gates for the transport suite and PostgreSQL composition.
