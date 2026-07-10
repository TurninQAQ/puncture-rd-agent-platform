# Versioning and Release Records

The repository uses semantic versions and one release per completed module.

## Release sequence

| Version | Scope |
|---|---|
| `v0.1.0` | Contract-first scaffold baseline |
| `v0.2.0` | Persistent Artifact Registry and object-store boundary |
| `v0.3.0` | Qwen/vLLM model gateway and deployment profile |
| `v0.4.0` | Enterprise hybrid RAG |
| `v0.5.0` | MCP tool servers and production adapters |
| `v0.6.0` | LangGraph runtime, checkpoint and human approval |
| `v0.7.0` | FastAPI/SSE runtime and durable run/event store |
| `v0.8.0` | OpenTelemetry and production Eval pipeline |

Detailed release evidence is stored under `docs/releases/`, for example
`docs/releases/v0.2.0.md` and `docs/releases/v0.3.0.md`.

Source release and operational deployment evidence are separate. A module may be
released when its code, offline protocol tests and deployable assets pass the
completion gate, while a hardware-specific live deployment remains explicitly
`NOT_RUN`. Such a release must not publish latency, throughput, GPU-memory or
model-validity claims until the live evidence record is complete.

## Completion gate

A module is released only after:

1. its contract, unit, failure and integration tests pass;
2. full `python3 run_tests.py` regression passes;
3. `python3 -m compileall -q contracts src tests` passes;
4. `CHANGELOG.md` and `pyproject.toml` are updated;
5. the work is committed as one module-level commit;
6. an annotated `vX.Y.Z` tag is created;
7. commit and tag are pushed to `origin`;
8. the pushed commit/tag are verified against the remote.

A commit whose exact subject is `release: vX.Y.Z` delegates steps 6–7 to CI.
The release-tag job runs only after the Python 3.10/3.11/3.12 test matrix passes,
validates that `pyproject.toml` and `docs/releases/vX.Y.Z.md` agree, and then
publishes the annotated tag. The tag's presence therefore records a green remote
release gate rather than a local-only assertion.

Small fixes within an unfinished module may be pushed as ordinary commits, but
the version tag is created only when the module acceptance gate is complete.
