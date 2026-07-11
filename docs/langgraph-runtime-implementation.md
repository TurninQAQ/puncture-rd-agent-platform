# LangGraph Runtime Implementation Note

## Status

Task 06 now has an unreleased production-runtime implementation while the
dependency-free `GraphRuntime` remains the behavioral reference.

Implemented and locally verified:

- `AgentState` <-> `TypedDict` checkpoint conversion with deep-copy isolation;
- recursive rejection of raw bytes, non-finite numbers, non-string object keys,
  non-JSON values and checkpoints larger than 1 MiB;
- compilation of the three locked JSON graphs into a LangGraph-compatible main
  graph and two directly attached compiled subgraphs;
- deterministic routing in checked-in edge order, bounded main/subgraph steps,
  `configurable.thread_id` isolation and synchronous checkpoint durability;
- in-memory test checkpointer, PostgreSQL saver lifecycle factory and a
  framework-neutral event stream;
- dynamic interrupt/resume, same-thread missing-input restart and cross-runtime
  continuation from a safe child-graph checkpoint;
- cross-runtime `run`/`resume`/`stream` single-flight with a SQLite TTL/CAS test
  double and a production PostgreSQL session advisory-lock manager;
- bounded model structured-output parsing and compact ACL-aware RAG evidence;
- fail-closed RAG module coverage for both task families;
- all ten legacy Agent node request shapes mapped to the frozen MCP wire schemas;
- opaque Artifact handles, principal propagation, stable idempotency identities
  and URI/checksum/raw-byte rejection;
- recursive frozen MCP response/result validation, normalized frozen values,
  request/envelope Artifact identity binding and versioned idempotency identities
  without changing shared contracts;
- atomic trusted-Registry snapshots for every supplied Artifact and every remote
  output, including current case/ACL/status/type, full geometry, producer/version
  and exact direct-lineage policies before any output can mutate Agent state;
- accepted-candidate-only reporting and deterministic manual-review handling.

The standard-library environment intentionally has no optional LangGraph or
PostgreSQL packages, so dependency-gated tests remain explicit there. An isolated
dependency path was used to run LangGraph `1.2.9` and the real `StateGraph`
tests locally. `langgraph-checkpoint-postgres` `3.1.0` and `psycopg` `3.3.4`
import successfully, while actual PostgreSQL execution remains CI-gated because
this host has no disposable database/DSN.

## Node-to-callable mapping

### Main graph

| JSON node | Production callable |
|---|---|
| `parse_request` | `production_nodes._model_parse_node` + `GatewayRequestPlanner` |
| `retrieve_project_knowledge` | `production_nodes._rag_node` + `RagKnowledgeRetriever` |
| `resolve_case_context` | `production_nodes._resolve_production_case_context` |
| `task_router` | `nodes._noop`; outgoing JSON conditions decide the branch |
| `data_model_subgraph` | compiled `graph/data_model_subgraph.json` |
| `planning_safety_subgraph` | compiled `graph/planning_safety_subgraph.json` |
| `result_verifier` | `nodes._result_verifier` -> deterministic `verify_agent_state` |
| `error_recovery` | `nodes._error_recovery` |
| `request_missing_data` | `nodes._request_missing_data` |
| `report_generator` | `nodes._report_generator` |

### Data/model subgraph

| JSON node | Callable |
|---|---|
| `inspect_case_metadata` | `nodes._inspect_case_metadata` -> `McpToolExecutor` |
| `validate_geometry` | `nodes._validate_geometry` |
| `conversion_router` | `nodes._noop` + JSON conditions |
| `convert_mcs_to_nifti` | `nodes._convert_mcs_to_nifti` -> `McpToolExecutor` |
| `validate_label_schema` | `nodes._validate_label_schema` -> `McpToolExecutor` |
| `segmentation_router` | `nodes._noop` + JSON conditions |
| `run_segmentation` | `nodes._run_segmentation` -> `McpToolExecutor` |
| `validate_segmentation_result` | `nodes._validate_segmentation_result` -> `McpToolExecutor` |
| `skin_processing_router` | `nodes._noop` + JSON conditions |
| `extract_skin_surface` | `nodes._extract_skin_surface` -> `McpToolExecutor` |
| `finalize_data_model` | `nodes._finalize_data_model` |

### Planning/safety subgraph

| JSON node | Callable |
|---|---|
| `ensure_required_artifacts` | `nodes._ensure_required_artifacts` |
| `artifact_router` | `nodes._noop` + JSON conditions |
| `resolve_planning_constraints` | `nodes._resolve_planning_constraints` |
| `generate_candidate_paths` | `nodes._generate_candidate_paths` -> `McpToolExecutor` |
| `candidate_router` | `nodes._noop` + JSON conditions |
| `evaluate_path_safety` | `nodes._evaluate_path_safety` -> `McpToolExecutor` |
| `evaluate_intraoperative_risk` | `nodes._evaluate_intraoperative_risk` -> `McpToolExecutor` |
| `verify_skin_penetration` | `nodes._verify_skin_penetration` -> `McpToolExecutor` |
| `finalize_planning` | `nodes._finalize_planning` |

## Construction

Production startup must inject all external dependencies explicitly:

```python
from puncture_agent.agent import (
    LangGraphRuntime,
    McpToolExecutor,
    PostgresAdvisoryThreadExecutionLeaseManager,
    RegistryToolArtifactValidator,
    build_production_handlers,
    open_postgres_checkpointer,
)

artifact_validator = RegistryToolArtifactValidator(artifact_registry)
tool_executor = McpToolExecutor(
    mcp_runtimes_or_clients,
    principal=principal_provider,
    artifact_validator=artifact_validator,
)
handlers = build_production_handlers(
    tool_executor=tool_executor,
    model_gateway=qwen_gateway,
    rag_service=rag_client,
    access_scope_provider=authenticated_scope_provider,
)
lease_manager = PostgresAdvisoryThreadExecutionLeaseManager(postgres_dsn)

with open_postgres_checkpointer(postgres_dsn, setup=False) as saver:
    runtime = LangGraphRuntime(
        "graph/main_graph.json",
        handlers,
        checkpointer=saver,
        execution_lease_manager=lease_manager,
    )
    final_state = runtime.run(initial_state)
```

`setup=True` performs the official saver migration and belongs in an explicit
deployment migration/startup step, not on every request. Do not use the default
in-memory saver for a production process that must survive restart. The lease
manager opens a separate autocommit connection per execution and must never
reuse the saver connection.

`artifact_registry` must be the deployment's independent authority, not data
copied from the remote MCP response. Its `get_validation_record()` operation
returns one atomic snapshot containing only the public Artifact view, full
geometry and direct parent IDs; it deliberately omits URI, checksum and private
metadata. The checked-in in-memory and SQLite registries implement this port.
A multi-host deployment must provide the same atomic contract from its shared
Registry service/database.

## Failure routing and checkpoint boundaries

- Missing or ambiguous model input, incomplete task-required RAG evidence, or
  missing Artifact IDs routes to `AWAITING_INPUT` before an algorithm tool runs.
- `TIMEOUT` and temporary `DEPENDENCY_FAILED` transport failures are retryable.
  The child graph finalizes `ERROR`, the verifier returns `NEED_RETRY`, and
  `error_recovery` re-enters the task subgraph only while the bounded budget
  remains. The platform cap is three retries; reference state defaults to one.
- Contract/schema/permission/geometry failures are non-retryable and fail closed
  to manual review. A valid `NO_CANDIDATE_PATH`/`NO_FEASIBLE_PATH` response is a
  no-result business terminal and does not run downstream safety tools.
- Trusted Artifact inputs are checked before transport. Successful/partial remote
  outputs are checked again after frozen response normalization and sanitization,
  but before the node can update `state.artifacts` or invoke a dependent tool.
  A Registry outage becomes retryable `DEPENDENCY_FAILED`; an unregistered,
  cross-case, unavailable, mistyped or forged Artifact becomes non-retryable
  `CONTRACT_VIOLATION` without exposing Registry internals.
- Artifact-producing tools have explicit output policies. Conversion,
  segmentation and skin-surface outputs must match the Registry public view,
  producer/version and exact parent set; geometry is exact except for the
  approved conversion-compatibility tolerance. Candidate generation may return
  zero or multiple `PATH_MASK` outputs, whose envelope set, CT geometry and full
  request lineage must match exactly. Tools without an output policy cannot
  smuggle an Artifact envelope.
- LangGraph writes with `durability="sync"`. API events are buffered until the
  underlying stream drains, so a visible transition is not acknowledged while
  the synchronous graph stream is still checkpointing.
- Dynamic interrupts resume with the same `configurable.thread_id` and trace ID.
  A completed child node before an interrupt is restored without replaying that
  node. A terminal missing-input workflow can merge explicit updates and start a
  new pass from `START` on the same thread.
- This boundary protects completed graph nodes. It cannot by itself make an
  external side effect exactly-once if a process dies after the tool returns but
  before the following graph checkpoint; the tool service needs a persistent
  idempotency ledger for that crash window. `SQLiteToolReplayLedger` now records
  a terminal MCP response before returning it, and the checked-in bridge restart
  test proves that losing subsequent AgentState mutations does not call the
  handler twice on one shared host.
- Dynamic interrupt values cross the same JSON/raw-byte/size boundary before
  LangGraph can persist them. Invalid values become a durable terminal
  `STATE_BOUNDARY_ERROR` instead of leaving an unreadable interrupted thread.
- The local active-thread guard remains a fast path, while the injected lease
  manager serializes the same thread across runtime/worker instances. A busy
  lease leaves the rejected state and checkpoint untouched. Backend
  unavailability fails closed before a handler runs.
- Ownership is renewed/verified before each node, checked immediately after a
  handler returns or raises, and checked before explicit terminal checkpoint
  writes and before an API result/event becomes visible. Lease loss stops the
  stale worker, marks the in-memory state `MANUAL_REVIEW` with
  `EXECUTION_LEASE_LOST`, and deliberately does not write that marker through
  the checkpoint saver after ownership is lost.

## Verification evidence

Local Python 3.10.12 results on 2026-07-11:

```text
python3 run_tests.py
Ran 554 tests in 9.729s
OK (skipped=17)

PYTHONPATH=/tmp/lginstall3:src:. python3 run_tests.py
Ran 554 tests in 21.759s
OK (skipped=9)
```

- 537 tests pass in the dependency-free environment;
- 545 tests pass with real LangGraph 1.2.9 available;
- the graph suite with real dependencies available runs 125 tests: 121 pass and
  three PostgreSQL tests plus the inverse missing-dependency guard are skipped;
- eight tests execute the actual `StateGraph` (seven graph integration/smoke/
  fault tests and one Eval test); the broader failure/concurrency matrix uses the
  deterministic Fake API to isolate branch semantics;
- the Eval suite runs 10 tests with 100% reference contract success, 100% report
  schema validity and zero forbidden-node violations;
- 20 concurrent isolated sessions complete without cross-session state leakage
  on the deterministic Fake API; the equivalent real-LangGraph matrix is `NOT_RUN`;
- all ten legacy node request shapes decode as their frozen request dataclasses;
- 15 durable replay tests prove one handler execution across bridge/runtime/
  ledger reconstruction, plus fail-closed authorization and uncertainty paths;
- 14 execution-lease tests prove backend contention/expiry/loss behavior and
  cross-runtime `run/run`, `resume/resume`, and `stream/run` exclusion while
  allowing different thread IDs to enter handlers concurrently;
- 18 trusted Artifact validator tests, five bridge/state integration tests and
  three atomic Registry-snapshot tests prove forged/unregistered outputs,
  incorrect case/status/type/producer/version/geometry/lineage, ACL denial and
  Registry outages all fail closed without storage-secret leakage;
- real LangGraph child nodes and four local MCP planning calls share one trace ID;
- a real dynamic interrupt resumes across a new runtime from the child checkpoint
  without replaying the already completed candidate-generation node;
- compileall and whitespace checks are separate final gates.

Isolated graph-only benchmark on Linux 5.15, Intel Xeon Gold 6230, Python
3.10.12 and LangGraph 1.2.9, with five warm-ups, three rounds of 50
immediate-fake samples and nearest-rank percentiles:

```text
PYTHONPATH=/tmp/lginstall3:src:. python3 -m unittest \
  tests.graph.test_langgraph_runtime.RealLangGraphSmokeTests.test_real_stategraph_graph_only_p95_records_engineering_gate -v

round P95 63.173 / 72.295 / 76.292 ms
median round P95 72.295 ms
aggregate P50 40.702 ms
aggregate P95 71.088 ms
max 123.432 ms
serialized reference checkpoint 12,979 bytes
```

The checked-in test prints its measured values so future runs can be compared.
The <=100 ms engineering threshold was not consistently met across repeated
manual runs: another isolated run contained a 102.745 ms round, and a loaded
graph-suite run measured a 101.397 ms median round P95. Those two observations
are manual records rather than checked-in fixtures. The threshold was not raised.
Normal CI records a sanity baseline;
`PUNCTURE_ENFORCE_PERFORMANCE_GATES=1` enables the original hard <=100 ms gate
for a controlled benchmark host.

## Not yet verified

The following remain `NOT_RUN`, not implicitly complete:

- disposable PostgreSQL setup and restart/resume execution on this host (the
  automated CI-gated tests and PostgreSQL 16 service wiring are present, but no
  successful remote workflow result was available during this implementation);
- forced process termination after a side-effecting tool returns but before the
  graph checkpoint. Deterministic state-loss/restart tests now prove SQLite
  replay count remains one, but an actual process-kill harness is still `NOT_RUN`;
- a shared PostgreSQL/dedicated replay ledger for multi-host MCP servers and
  atomic coordination between the tool's internal side effect and ledger commit;
- actual PostgreSQL execution of the gated same-thread advisory-lock test and a
  true multi-process/network-partition fault run; deterministic dual-runtime
  coverage is complete, but this host has no PostgreSQL test DSN;
- durable fencing or an incident record for the narrow case where the dedicated
  advisory-lock connection is lost while a stale handler's external side effect
  is still completing; the MCP replay ledger and manual reconciliation remain
  required because a session lock alone cannot fence arbitrary external systems;
- blocking real-saver proof that no API node event is acknowledged before a sync
  checkpoint finishes; deterministic stream-buffer tests cover the adapter logic;
- the complete failure matrix and 20-session concurrency test on real LangGraph,
  rather than only the focused real smoke/integration cases;
- live multi-host execution against the deployment's shared Artifact Registry
  service, including Registry failover/load evidence. The authority protocol,
  in-memory/SQLite implementations and fail-closed bridge integration are locally
  verified, but this host has no production shared-Registry endpoint;
- using retrieved rule contents to derive numeric planning constraints and label
  policy, rather than using RAG as a required versioned-evidence gate while the
  frozen `ToolBridgePolicy` remains authoritative;
- an end-to-end retrieved-document prompt-injection case proving model output
  cannot expand the fixed graph/tool authorization set;
- one trace containing model, RAG, graph/node and remote tool spans; current real
  integration proves graph/node/state/local-MCP trace identity only;
- child-subgraph container spans; child node spans currently remain directly
  parented to the graph span;
- API streaming capacity/heartbeat behavior: safe acknowledgement currently
  buffers all node states until the graph stream drains, with a worst-case memory
  shape of roughly `max_steps * 1 MiB`;
- live model/RAG/MCP network clients, OAuth/OIDC, OpenTelemetry and API wiring;
- PostgreSQL checkpoint save/resume P50/P95 evidence.

Checkpointing does not make an external side effect exactly-once by itself.
Stable bridge idempotency keys and a persistent tool-side idempotency ledger are
required for crash-safe replay. Chained output Artifact IDs also require a live
registry-backed MCP Artifact resolver; the static demo resolver is not a
production registry.
