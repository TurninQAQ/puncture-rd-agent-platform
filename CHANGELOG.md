# Changelog

All notable changes are recorded here. Dates use UTC and the project follows
Semantic Versioning while it is below 1.0.

## [Unreleased]

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
