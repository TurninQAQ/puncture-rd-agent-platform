# Task 06: Replace Mock Runtime with Production LangGraph Runtime

## Task objective

Implement the real LangGraph orchestration while preserving the checked-in
graph, state, tool, verification, tracing, and evaluation contracts. This task
does not implement medical algorithms, RAG internals, or Qwen serving.

## Read before coding

Provide the implementing model these files:

1. `specs/langgraph-runtime.md` (primary specification)
2. `graph/main_graph.json`
3. `graph/data_model_subgraph.json`
4. `graph/planning_safety_subgraph.json`
5. `src/puncture_agent/agent/state.py`
6. `src/puncture_agent/agent/graph_spec.py`
7. `src/puncture_agent/agent/verifier.py`
8. shared `contracts/*.py`
9. Tool Registry/MCP client public interface
10. `tests/graph/*` and relevant `tests/eval/*`

Do not provide company source code, data, credentials, or proprietary algorithm
implementation. Use Mock tools for all unit tests.

## Files allowed to change

- `src/puncture_agent/agent/**`
- production dependency/config files explicitly assigned by the maintainer
- new runtime-specific tests under `tests/graph/**`

Do not silently modify `graph/*.json`, shared tool schemas, error codes, or Eval
expectations to make an implementation pass. Propose such changes separately.

## Required deliverables

1. Production `StateGraph` for the main graph.
2. Compiled data/model and planning/safety subgraphs.
3. Deterministic conditional routing matching JSON edge order.
4. Adapter from shared Tool Registry/MCP client to node partial-state updates.
5. Adapter from RAG client and Qwen model gateway to their nodes.
6. PostgreSQL checkpoint configuration plus in-memory test checkpointer.
7. Bounded retry/idempotency logic.
8. Stream/event interface for API consumption.
9. Unit, branch, restart, concurrency, and fault-injection tests.
10. Short implementation note mapping every JSON node to its callable.

## Interfaces that are locked

- Input state/checkpoint shape: `AgentState.to_dict()`.
- Runtime logical call: `run(state, configurable thread/session ID) -> final state`.
- Node IDs/handler names: exactly those in `graph/*.json`.
- Terminal statuses and verifier semantics from the runtime specification.
- Tools: exactly the shared tool request/response contracts.
- Large artifacts are IDs; never place voxels in state/model context.

If LangGraph requires a `TypedDict`, create an internal compatible type and
explicit conversion functions. Conversion round trips must be tested.

## Implementation sequence

1. Run all existing tests and save a baseline.
2. Validate JSON specs at startup/CI.
3. Implement state conversion and one trivial node; prove checkpoint round trip.
4. Implement the two child graphs without real tools, using Tool Registry fakes.
5. Implement the main graph and deterministic routing.
6. Add structured error mapping, bounded retry, and idempotency keys.
7. Add PostgreSQL persistence/restart tests.
8. Attach TraceRecorder/OpenTelemetry hooks.
9. Run the full Eval harness and compare node paths with the Mock baseline.

## Required tests

Execute at minimum:

```bash
python3 -m unittest discover -s tests/graph -p 'test_*.py' -v
python3 -m unittest discover -s tests/eval -p 'test_*.py' -v
```

Add production-runtime tests for:

- planning success;
- MCS data/model success;
- missing case stops before tools;
- geometry/label failure does not continue;
- no feasible path is not a system exception;
- one-time retryable error recovers exactly once;
- non-retryable error is not retried;
- retry exhaustion becomes manual review/failure as specified;
- malformed model structured output fails safely;
- PostgreSQL restart resumes without duplicate tool execution;
- two simultaneous session IDs cannot observe each other's state;
- trace context propagates through child graph and fake tool;
- raw byte payload rejection.

Tests must assert exact node/tool call counts, not merely final text.

### Exact existing test IDs

Run these while developing a replacement; each is a locked behavioral example:

```bash
python3 -m unittest -v \
  tests.graph.test_graph_specs.GraphSpecTests.test_all_checked_in_graphs_are_semantically_valid \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_planning_and_safety_flow_reaches_verified_report \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_mcs_data_flow_runs_conversion_and_segmentation \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_missing_case_id_stops_before_any_algorithm_tool \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_missing_planning_artifact_stops_before_planner \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_no_feasible_path_is_a_valid_terminal_outcome \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_one_time_tool_timeout_is_retried_from_subgraph_boundary \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_persistent_timeout_exhausts_retry_budget \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_non_retryable_tool_error_is_not_retried \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_geometry_failure_requires_manual_review_without_segmentation \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_checkpoint_round_trip_preserves_state \
  tests.graph.test_verifier.VerifierTests
```

Create equivalent `ProductionLangGraphRuntimeTests` rather than deleting or
weakening the Mock reference tests.

### Failure-injection matrix

| Metadata/test fixture | Expected behavior |
|---|---|
| `fail_tool_once=[name]` | one failed call, one recovery cycle, then success |
| `fail_tool_always=[name]` | calls stop at retry budget; `MANUAL_REVIEW` |
| `fail_tool_non_retryable=[name]` | exactly one call; no retry; `MANUAL_REVIEW` |
| `force_geometry_mismatch=true` | stop before label/model tools |
| `force_label_schema_error=true` | stop before segmentation |
| `missing_required_artifacts=[target]` | no planning tool; `AWAITING_INPUT` |
| `force_no_feasible_path=true` | no downstream safety tool; valid no-result status |
| malformed model JSON fixture | bounded repair or safe failure; no tool execution |
| checkpoint process kill after tool | resume without duplicate execution |

Translate these flags into fake dependency behavior in production tests; they
need not remain production configuration options.

## Performance and evaluation thresholds

Measure with fake model/RAG/tools so the runtime itself is isolated. Record OS,
Python, LangGraph/PostgreSQL versions, CPU, sample count, and warm-up. Initial
engineering gates (not claims about the finished system) are:

- graph-only P95 orchestration latency <= 100 ms for a successful flow when all
  fake dependencies return immediately;
- checkpoint save P95 <= 50 ms and checkpoint resume P95 <= 150 ms on the
  documented local/PostgreSQL test environment;
- at least 20 concurrent isolated sessions in the integration test with zero
  cross-session state leakage;
- serialized checkpoint state <= 1 MiB for reference cases and contains no raw
  image bytes;
- graph/report schema validity 100%; forbidden-node/tool violations 0%;
- reference Eval task success 100%; designated one-time failure recovery 100%;
- persistent/non-retryable failure routing accuracy 100%;
- step counts never exceed the checked-in `max_steps`.

If infrastructure cannot meet a latency threshold, report the measured baseline
and bottleneck rather than changing the threshold or fabricating a result.

## Copy-paste implementation instruction

```text
Implement Task 06 using specs/langgraph-runtime.md as the source of truth.
Preserve AgentState serialization, all graph/*.json node IDs/edges, shared tool
schemas, terminal statuses, and verifier semantics. Replace only the execution
layer with real LangGraph, child subgraphs, bounded retry, idempotent MCP tool
calls, and PostgreSQL checkpointing. First run the exact test IDs listed in this
task. Then add ProductionLangGraphRuntimeTests covering every failure-injection
row, checkpoint restart, concurrency isolation, malformed model JSON, trace
propagation, and performance gates. Do not implement medical algorithms and do
not edit expected tests to make code pass. Return changed files, node-to-callable
mapping, complete test output, benchmark environment/results, mismatches, and
remaining risks.
```

## Definition of done

- All pre-existing and new tests pass.
- JSON topology and node IDs remain unchanged.
- Checkpoint restart and idempotent resume are demonstrated automatically.
- Every tool input is schema validated before execution.
- Verifier remains deterministic and has no model dependency.
- All loops have retry/step limits.
- Eval reference suite remains at 100% contract success and 0 forbidden-node
  violations.
- The implementation note explains failure routing and checkpoint boundaries.

## Expected response from the implementing model

Return:

1. files changed;
2. mapping of JSON nodes to implementation callables;
3. test commands and complete results;
4. any interface mismatch found (do not work around it silently);
5. remaining production risks.
