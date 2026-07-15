# Local full-stack API demo

This directory turns the repository's implemented adapters into a runnable
workstation demonstration:

```text
HTTP/SSE client
    -> FastAPI security and Run API
    -> PostgreSQL Run/event repository
    -> integrated Agent executor
         -> real Qwen/vLLM model gateway
         -> real secured OpenSearch hybrid retrieval
         -> checked-in graph and deterministic synthetic tools/verifier
```

It is intentionally not a production deployment. The algorithm tools do not
read medical volumes or call company MCS, TensorRT, planning, or safety systems.
The API binds only to loopback, uses a generated private local bearer token, and
requires an explicit opt-in.

## Prerequisites

- the repository `.venv` with the pinned implementation dependencies;
- a reachable PostgreSQL 16 database;
- the Qwen/vLLM service from `deploy/qwen-vllm`;
- the secured OpenSearch service and bootstrapped aliases from
  `deploy/rag-search`.

The demo seeds only synthetic RAG records with stable IDs. Repeated startup
updates those records idempotently.

## Configure once

```bash
cd /home/turnin/puncture-rd-agent-platform
cp deploy/local-demo/.env.example deploy/local-demo/.env
chmod 600 deploy/local-demo/.env
```

Edit the private `.env` and replace the PostgreSQL DSN, OpenSearch password-file
path, and CA-file path. Then set `RUN_FULL_STACK_DEMO=1`. Values are parsed
literally by an allowlist parser; the file is never sourced or evaluated, so do
not add shell quotes or expansions. The file and generated token are ignored by
Git.

Check all prerequisites without changing PostgreSQL or OpenSearch:

```bash
./deploy/local-demo/doctor.sh
```

The JSON names the failing component with a fixed error code and never prints
the PostgreSQL DSN, OpenSearch password, bearer token, or raw exception text.
Single-node OpenSearch `DEGRADED/yellow` is accepted for this local profile.

## One-command round trip

```bash
./deploy/local-demo/run_demo.sh
```

The command starts a temporary Uvicorn process, waits for health, rejects a bad
bearer token, executes both fixed workflows, verifies PostgreSQL idempotency and
JSON event replay, verifies terminal SSE replay, and then sends SIGTERM and
waits for clean shutdown. It runs the same readiness doctor first. Qwen,
OpenSearch, and PostgreSQL remain running.

The expected JSON reports:

- `data_validation` and `planning_safety` as `SUCCEEDED`;
- the served Qwen model and at least two RAG chunks per workflow;
- deterministic tool-call and visited-node counts;
- `verification_status=PASS` for planning safety;
- `invalid_bearer_rejected=true`, idempotent replay, and terminal SSE replay.

## Keep the API running

Terminal 1:

```bash
./deploy/local-demo/serve.sh
```

Terminal 2:

```bash
./deploy/local-demo/verify.sh
```

Both scripts safely load the same private `.env`. For a short verification,
prefer `run_demo.sh`, which also owns cleanup. The API intentionally disables Swagger/Redoc.
Inspect the fixed routes in `docs/fastapi-runtime.md` or the OpenAPI contract
tests rather than exposing interactive docs.

## Why health is DEGRADED

`DEGRADED` is expected for this workstation profile. It has no production
Artifact Access Gateway, and the one-node OpenSearch bootstrap retains an
unassigned replica. PostgreSQL, Qwen, and OpenSearch must still be reachable;
otherwise startup or the round trip fails.

The local API uses synchronous execution (`worker_enabled=false`) because the
integrated synthetic executor is not the company's recovery-safe side-effect
adapter. Durable worker claim/reclaim is tested elsewhere with an injected
recovery-safe port; this demo must not be presented as process-crash recovery
evidence.

## Local files and cleanup

- `deploy/local-demo/.env`: private configuration, mode 0600, ignored;
- `var/local-demo/bearer-token`: generated URL-safe token, mode 0600, ignored;
- `var/local-demo/api.log`: local Uvicorn log, mode 0600, ignored;
- PostgreSQL schema: `puncture_local_demo` by default.

Delete only the local schema and ignored files when you deliberately want a
fresh demonstration. Never point the demo at a shared production schema.
