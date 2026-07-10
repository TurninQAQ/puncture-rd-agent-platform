# Changelog

All notable changes are recorded here. Dates use UTC and the project follows
Semantic Versioning while it is below 1.0.

## [Unreleased]

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
