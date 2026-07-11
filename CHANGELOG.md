# Changelog

All notable changes are recorded here. Dates use UTC and the project follows
Semantic Versioning while it is below 1.0.

## [Unreleased]

### Added

- Optional LangGraph 1.2 production runtime that compiles the locked JSON main
  graph and both child graphs while retaining the dependency-free reference runner.
- Exact AgentState/TypedDict checkpoint conversion with raw-byte, JSON and 1 MiB
  state guards, synchronous durability, isolated thread IDs and an API event stream.
- Dynamic interrupt/resume, same-thread missing-input restart, trace continuation
  across runtime instances, and conservative event buffering until sync streams drain.
- Explicit Qwen structured-request and enterprise RAG node adapters.
- Ten-tool Agent-to-MCP contract bridge with opaque Artifact handles, principal
  propagation, versioned replay identity and frozen policy defaults.

### Changed

- Production node normalization now accepts frozen MCP result fields and reports
  only candidates accepted by deterministic safety evaluation.
- Remote MCP responses are validated recursively against frozen result types;
  normalized contract values are retained, result Artifact identities are bound
  to request/envelope identities, and storage URIs fail closed instead of being
  accepted or silently rewritten. RAG must cover every task-required active
  module before tools may run.
- Retryable transport timeouts/dependency failures remain distinct from contract
  violations, and safety candidate summary/assessment partitions fail closed.

### Verification

- Local Python 3.10 standard environment: 498 tests run, 482 passed and 16
  explicitly gated dependency/private-service tests skipped.
- Isolated LangGraph 1.2.9 run: 498 tests run, 490 passed and 8 gated tests
  skipped. Eight tests execute the real `StateGraph`; the larger deterministic
  branch matrix remains on the Fake API. Real coverage includes local MCP trace
  propagation, dynamic interrupt/resume, durable state/interrupt boundary
  failures and cross-runtime in-memory recovery.
- `langgraph-checkpoint-postgres` 3.1.0 and psycopg import locally; PostgreSQL 16
  service wiring is present in CI but has not been observed running from this
  worktree. Local PostgreSQL restart execution, distributed same-thread locking,
  trusted output-Artifact registry validation and the post-tool/pre-checkpoint
  process-kill window remain `NOT_RUN` without the required service/harness.
- The real-LangGraph <=100 ms P95 threshold is retained but is not a normal CI
  hard gate because repeated shared-host runs showed threshold-crossing jitter;
  controlled runners can enforce it with `PUNCTURE_ENFORCE_PERFORMANCE_GATES=1`.

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
