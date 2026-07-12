# Changelog

All notable changes are recorded here. Dates use UTC and the project follows
Semantic Versioning while it is below 1.0.

## [Unreleased]

### Added

- No additional unreleased feature work in the demo gate.

## [0.8.0] - 2026-07-12

### Added

- OpenTelemetry-compatible tracing facade with graph/node/RAG/model/MCP/
  verifier/checkpoint spans and privacy-safe attribute allowlist.
- W3C `traceparent` plus HTTP/MCP/gRPC-style metadata propagation and
  concurrent-session isolation tests.
- Versioned Eval dataset loader (`eval-case-v1`), deterministic RAG/Agent
  metrics (`metrics-v1`), per-case diagnostics and baseline regression reports.
- Offline CLI `python -m puncture_agent.observability.eval_runner` with
  `--traced` / compare / release-block gates.
- Tracing overhead benchmark and CI zero-skip eval/tracing gates (46 tests).
- Release record `docs/releases/v0.8.0.md` and implementation notes under
  `docs/eval-and-tracing-implementation.md`.

### Security

- Denylist redaction of credentials, PHI-like fields, unrestricted prompts and
  raw image attributes before export; unknown keys are dropped.

### Verification

- Local Python 3.10: 690 tests run, 629 passed, 61 gated skips.
- Task 08 suite: 46/46 zero-skip.
- Mock reference CLI: `passed=3/3`, `release_blocked=False`.
- Live OTLP/Langfuse/Phoenix and internal Golden Set quality targets remain
  `NOT_RUN` / field configuration.

## [0.7.0] - 2026-07-11

### Added

- Pydantic v2 secure HTTP contract adapters with injected authority only.
- Atomic in-memory and PostgreSQL Run/event repositories with version fence,
  tenant-scoped idempotency, private event keys and COMMIT-unknown reconciliation.
- FastAPI Run Gateway (9 REST/OpenAPI paths), pre-parse Bearer/body admission,
  privacy-safe public views, health and low-cardinality HTTP metrics.
- Bounded SSE event replay with strict cursors, heartbeat, token revalidation
  and per-process quotas.
- Durable execution jobs with worker heartbeat/reclaim and API SIGTERM recovery.
- Release record `docs/releases/v0.7.0.md` and runtime docs under
  `docs/api-runtime-implementation.md` / `docs/fastapi-runtime.md`.

### Security

- Reject body-forged tenant/principal/role/scope; recursive denial of raw image
  fields, credentials, arbitrary URIs and JWT-like secrets.
- Deep redaction of public snapshot/event/error views; fixed 500/503 messages.

### Verification

- CI matrices for Python 3.10/3.11/3.12 with PostgreSQL 16 no-skip repository,
  FastAPI, SSE and durable recovery jobs (see README evidence links).
- Company OIDC binding, cluster SSE quotas and GPU cancel remain external.

## [0.6.0] - 2026-07-11

### Added

- Optional LangGraph 1.2 production runtime compiling the locked JSON main graph
  and both child graphs while retaining the dependency-free reference runner.
- Exact AgentState/TypedDict checkpoint conversion with raw-byte, JSON and 1 MiB
  state guards, synchronous durability, isolated thread IDs and event streams.
- Dynamic interrupt/resume, same-thread missing-input restart, and conservative
  event buffering until sync streams drain.
- Explicit Qwen structured-request and enterprise RAG node adapters.
- Ten-tool Agent-to-MCP contract bridge with opaque Artifact handles, principal
  propagation, versioned replay identity and frozen policy defaults.
- SQLite MCP replay ledger with full-sync commits, response integrity hashes,
  restart-safe terminal replay, authorization rechecks and explicit
  `PENDING`/`COMPLETED`/`UNCERTAIN` state transitions.
- PostgreSQL checkpointer, cross-runtime advisory locks, trusted Artifact
  Registry validation, process-kill recovery window and real-LangGraph matrices.
- Release record `docs/releases/v0.6.0.md` and
  `docs/langgraph-runtime-implementation.md`.

### Changed

- Production node normalization accepts frozen MCP result fields and reports
  only candidates accepted by deterministic safety evaluation.
- Remote MCP responses are validated recursively against frozen result types;
  storage URIs fail closed. RAG must cover every task-required active module
  before tools may run.

### Verification

- Real LangGraph 1.2.9 success/failure matrix and 20-way concurrent isolation.
- PostgreSQL 16 CI persistence/restart/process-kill evidence linked from README.
- Company algorithms and host/WAL crash classes remain external / `NOT_RUN`.

## [0.5.0] - 2026-07-11

### Added

- Three logical MCP servers exposing all ten frozen algorithm tools through deterministic `tools/list` and `tools/call` behavior.
- Dependency-free MCP 2025-11-25 JSON-RPC/stdio demo with generated input/output JSON Schema, structured content and text fallback.
- Strict dataclass request codec, opaque artifact-handle resolution and model-visible artifact public projections with no storage URI/checksum leakage.
- Principal/case/tool authorization, bounded deadlines, stable error envelopes, trace identity validation and exact handler-catalog binding.
- Injectable case-data, segmentation and planning/safety ports plus deterministic manifest backends that preserve company-algorithm boundaries.
- Optional stable official Python SDK v1 adapter and explicit `mcp>=1.27,<2` dependency pin.
- Local ten-tool demonstration, subprocess stdio handshake test, implementation guide and module verification runbook.

### Security

- Reject full `ArtifactRef` injection at the MCP boundary; only registered opaque artifact IDs may resolve to internal URIs/checksums.
- Enforce authenticated caller equality and case/tool allowlists before a handler executes.
- Redact internal artifact locations from structured content, text compatibility blocks, trace summaries and adapter errors.
- Fail closed on geometry, permission, missing required masks, unsafe backend downgrades and response identity/version drift.

### Verification

- Local Python 3.10 suite: 428 tests run, 421 passed and 7 explicitly gated model-service tests skipped.
- Ten adapter calls across three local MCP servers pass; stdio initialize, initialized notification and tool discovery pass in a subprocess.
- Contract, adapter, negative/failure, idempotency, compileall and whitespace checks pass.
- Official SDK transport installation, Streamable HTTP/OAuth, company algorithms, real MCS/NIfTI, TensorRT/GPU and target-hardware benchmarks remain `NOT_RUN`.

## [0.4.0] - 2026-07-11

### Added

- Local, dependency-free enterprise Hybrid RAG demo with heading-aware ingestion, deterministic BM25/dense recall, RRF, reranking, parent context and stable citations.
- ACL/module/version/lifecycle enforcement in both recall branches, parent lookup and final output, including explicit no-evidence behavior.
- Versioned in-memory index generations, checksum-guarded updates, bounded embedding batches and embedding/parser/chunker manifest compatibility.
- Offline Golden Set and ablation metrics for Recall@5/10, MRR, NDCG@10, correct-version rate, ACL leaks, no-answer accuracy and latency percentiles.
- Secure optional adapters for OpenSearch, OpenAI-compatible Qwen embeddings and vLLM reranking, plus a versioned OpenSearch deployment profile.
- Deterministic industrial/EDA local demo and detailed local/deployment test runbooks.

### Security

- Added fail-closed backend-filter verification, citation escaping, strict public RAG type validation, bounded total retrieval deadlines, hashed query traces and non-fatal trace export.
- Added strict provider URL/auth/TLS/JSON handling, disabled ambient proxies/redirects, bounded response bodies and secret-redacted configuration.

### Verification

- Local Python 3.10 suite: 346 tests run, 339 passed and 7 explicitly gated integration tests skipped.
- Local RAG demo, compileall, shell syntax, JSON validation and whitespace checks pass.
- Live OpenSearch, Qwen embedding/reranker services, GPU inference and production-corpus quality/latency evaluation remain `NOT_RUN`.

## [0.3.0] - 2026-07-10

### Added

- Production `VllmModelGateway` for private Qwen/vLLM OpenAI-compatible chat, tool calling, structured output and SSE.
- Pooled `httpx` transport with explicit lifecycle, TLS verification, disabled ambient proxies/redirects and bounded payload handling.
- Contract v2 assistant tool-call replay, explicit unknown-usage sentinel and strongly structured terminal stream errors.
- Qwen3/vLLM Compose deployment profile, safe entrypoint, health/smoke/benchmark scripts, GPU sizing worksheet and rollout/rollback runbook.
- Environment-gated live gateway suite and deterministic deployment-asset tests.

### Security

- Added strict JSON duplicate/non-finite/complexity rejection, exact served-model and choice validation, header-control protection and secret-redacted errors.
- Added bounded retry/deadline handling, safe `Retry-After`, fail-closed tool/schema validation and no stream retry after visible output.

### Verification

- Local dependency-free run: 241 passed and 7 gated skips out of 248 tests; CI installs pinned `httpx` to execute 2 transport-integration tests, while 5 live vLLM tests remain `NOT_RUN` without a private endpoint.
- A source-pinned `httpx==0.28.1` compatibility run executes the real pooled-client tests: 243 passed and only the 5 live vLLM tests skipped.
- Compile, shell syntax, whitespace and repository secret-pattern checks pass.
- No live GPU latency, throughput or memory claim is made by this source release.

## [0.2.0] - 2026-07-10

### Added

- Persistent SQLite Artifact Registry with restart recovery and transaction-safe lifecycle transitions.
- Secure local artifact store with private staging, service-side checksums and immutable atomic publication.
- Artifact publication coordinator, scoped idempotency, authorized reads and URI-free access audit.
- Canonical identity helpers, CI workflow, Module 0 release evidence and reproducible 10k benchmarks.

### Security

- Scoped idempotency keys by case to prevent cross-case reuse.
- Rejected storage/database symlinks, traversal, unsafe mapping keys and permissive database files.
- Added process-wide publication/commit coordination and retry-safe PENDING recovery.

### Verification

- 172 standard-library tests pass.
- Module 0 acceptance coverage maps AR-001 through AR-014 to automated tests.

## [0.1.0] - 2026-07-10

### Added

- Contract-first enterprise Agent scaffold.
- Ten strongly typed algorithm tool contracts with deterministic mocks.
- Qwen/vLLM and enterprise RAG gateway contracts and mocks.
- JSON Agent graphs, deterministic verifier, tracing and Eval harness.
- API/run lifecycle and integrated no-dependency Mock workflow.
- Module task cards and acceptance specifications.
- 101 passing standard-library tests.
