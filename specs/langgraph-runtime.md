# LangGraph Agent Runtime Specification

## 1. Objective and boundary

Implement a durable orchestration layer around existing data, segmentation,
planning, and safety algorithms. The language model may understand a request,
select tools, and summarize results. It must not compute image geometry,
collision, safety distance, warning/stop boundaries, or skin penetration.

The checked-in standard-library runtime proves the workflow without requiring
LangGraph. A production implementation must replace the executor, persistence,
and model-backed nodes while preserving:

- `AgentState` field names and meanings;
- graph/node IDs in `graph/*.json`;
- tool names and tool input/output contracts;
- verification and terminal status semantics;
- tests that do not explicitly test Mock-only values.

Large CT, Mask, MCS, NIfTI, and path-mask payloads never enter an LLM message or
LangGraph state. State contains `artifact_id` values and compact metadata only.

## 2. Current implementation map

| File | Responsibility | May be replaced? |
|---|---|---|
| `graph/*.json` | Locked topology and branch conditions | Only through a reviewed graph change |
| `agent/state.py` | Serializable state/checkpoint contract | Extend only; do not rename/remove fields |
| `agent/graph_spec.py` | JSON loader and semantic validator | Keep as graph CI validator |
| `agent/runtime.py` | Dependency-free executor | Replace execution with LangGraph |
| `agent/nodes.py` | Mock nodes and tool adapter | Replace internals, preserve handler names |
| `agent/verifier.py` | Deterministic cross-module checks | Extend with real invariants |
| `tests/graph/*` | Topology, branch, retry, checkpoint tests | Must remain green |

## 3. State contract

`AgentState` is the only data exchanged between nodes. Production code may use
`TypedDict`, a Pydantic model, or LangGraph reducers internally, but checkpoints
must serialize to the shape produced by `AgentState.to_dict()`.

### Identity and routing

| Field | Type | Producer | Consumer |
|---|---|---|---|
| `session_id` | string | API/runtime | Checkpointer, trace |
| `user_query` | string | API | request parser, RAG |
| `task_type` | enum string | `parse_request` | `task_router`, verifier |
| `case_id` | string/null | API or parser | all tools |
| `status` | enum string | runtime/final nodes | API/evaluation |

Allowed task values are `UNKNOWN`, `DATA_MODEL_VALIDATION`, and
`PLANNING_SAFETY`. Adding a task type requires a graph branch and tests.

### Artifact and algorithm data

| Field | Type | Rules |
|---|---|---|
| `artifacts` | object | IDs/metadata only; never raw voxels |
| `planning_constraints` | object | Physical units must be explicit, normally mm/degree |
| `candidate_paths` | array | Tool-produced structured candidates |
| `safety_result` | object | Accepted/rejected IDs and evidence |
| `risk_flags` | object | Deterministic warning/stop flags |
| `skin_penetration_result` | object | 3D ray-tracing result, not an LLM inference |

### Agent and reliability data

| Field | Type | Rules |
|---|---|---|
| `retrieved_documents` | array | Compact retrieval result metadata |
| `citations` | array | Document ID, version, section |
| `tool_plan` | array | Planned tool names; advisory, not authorization |
| `tool_calls` | array | Sanitized request audit trail |
| `tool_results` | array | Structured response summary; no huge artifact bytes |
| `subgraph_result` | object | `SUCCESS`, `ERROR`, `MISSING_INPUT`, `VALIDATION_FAILED`, or `NO_FEASIBLE_PATH` |
| `verification_status` | string | Deterministic verifier output |
| `retry_count` / `max_retries` | integer | Bounded retry; never an unbounded ReAct loop |
| `errors` | array | Stable code, message, node, retryable flag, details |
| `visited_nodes` | array | Exact execution path used by Eval |
| `metadata` | object | Optional feature/fault flags and small intermediate values |
| `final_report` | object | Stable API-facing report |

Checkpoint compatibility test: serialize with `to_dict()`, deserialize with
`from_dict()`, and require equality. A production migration needs an explicit
schema version and migration function before altering stored fields.

## 4. Main graph

The source of truth is `graph/main_graph.json`.

| Node | Input required | Output written | Correctness rule |
|---|---|---|---|
| `parse_request` | `user_query`, optional `case_id` | `task_type`, `case_id`, `tool_plan` | Output only supported task enum; never invent missing case data |
| `retrieve_project_knowledge` | query, task type | documents, citations | Apply ACL/version filters; return source evidence or an empty result |
| `resolve_case_context` | case/artifact IDs | `case_context_ready`, resolved artifact refs | IDs must all belong to the same case and geometry lineage |
| `task_router` | task type/context status | no business data | Deterministic branch; missing case wins before task branch |
| `*_subgraph` | resolved context | algorithm/tool results | Execute checked-in child graph |
| `result_verifier` | all subgraph outputs | verification status/reasons/evidence | No LLM call; fail closed on missing/contradictory evidence |
| `error_recovery` | structured error/retry budget | incremented retry count, cleaned transient state | Retry only retryable failures and only within budget |
| `request_missing_data` | known missing fields | `AWAITING_INPUT` report | Must not call algorithm tools |
| `report_generator` | verified data | final report/status | May explain but cannot change tool/verifier facts |

### Terminal mapping

| Verification status | Agent status |
|---|---|
| `PASS` | `SUCCEEDED` |
| `NO_FEASIBLE_PATH` | `COMPLETED_WITH_NO_RESULT` |
| `MISSING_INPUT` | `AWAITING_INPUT` |
| `MANUAL_REVIEW` | `MANUAL_REVIEW` |
| unexpected/unhandled | `FAILED` |

`NO_FEASIBLE_PATH` is a valid business outcome, not a system exception.

## 5. Data/model validation subgraph

The source of truth is `graph/data_model_subgraph.json`.

1. `inspect_case_metadata` calls the metadata tool and obtains shape, spacing,
   direction/origin or a geometry fingerprint, coordinate convention, and
   input format.
2. `validate_geometry` deterministically compares CT/label geometry. A mismatch
   stops the algorithm chain and produces `VALIDATION_FAILED`.
3. `conversion_router` invokes `convert_mcs_to_nifti` only for MCS input.
4. `validate_label_schema` checks label IDs/names against the retrieved active
   schema version. Invalid labels cannot enter segmentation/planning.
5. `segmentation_router` respects the explicit `run_segmentation` flag.
6. `run_segmentation` invokes the existing C++/TensorRT-backed service.
7. `validate_segmentation_result` rejects empty, wrong-geometry, or missing
   required Masks.
8. `skin_processing_router` invokes `extract_skin_surface` when requested.
9. `finalize_data_model` emits exactly one subgraph status and reasons.

All tool inputs/outputs come from `contracts/tool_inputs.py` and
`contracts/tool_outputs.py`. If a Tool Registry expects Pydantic objects, add a
thin adapter at the node boundary; do not pass unvalidated dictionaries deeper
into a real tool server.

## 6. Planning/safety subgraph

The source of truth is `graph/planning_safety_subgraph.json`.

1. `ensure_required_artifacts` verifies CT, skin surface, target, and required
   danger Masks and confirms same-case/same-geometry lineage.
2. `resolve_planning_constraints` combines user constraints with the active
   versioned RAG rule. User values must be unit-normalized and range-validated.
3. `generate_candidate_paths` returns candidates satisfying basic geometric
   needle-length and insertion-angle constraints.
4. Empty candidates route directly to `NO_FEASIBLE_PATH`.
5. `evaluate_path_safety` checks the entire dilated path envelope against danger
   Masks, not just the needle tip.
6. `evaluate_intraoperative_risk` returns warning/stop flags from deterministic
   Mask logic.
7. `verify_skin_penetration` uses 3D ray sampling (nominal step 0.5 voxel or a
   documented physical equivalent). It must not infer penetration from tip
   label alone.
8. `finalize_planning` emits one stable subgraph status.

The report may include only candidate IDs accepted by the safety tool. A
contradictory risk result must become `MANUAL_REVIEW`.

## 7. Structured error and retry rules

Tools return a response envelope with status, result, metrics, and error. Map
the shared contract error codes into:

- retryable: `TIMEOUT`, temporary dependency failure, rate limit, transient GPU
  resource failure;
- non-retryable: invalid argument, geometry mismatch, schema error, permission
  denied, model-version mismatch, missing artifact;
- valid outcome: no candidate/no feasible path.

Retry requirements:

1. Use `request_id`/idempotency key for every tool call.
2. Never retry a write-side effect without idempotency support.
3. Record the failed call and retry as separate trace/tool-call entries.
4. Increment `retry_count` once per recovery cycle.
5. Stop at `max_retries`; route to `MANUAL_REVIEW` or `FAILED` as specified.
6. Resume from the subgraph boundary or a safe checkpoint, never silently skip
   a failed validation.

## 8. Real LangGraph implementation

### Required dependencies

- `langgraph` and its checkpoint package;
- PostgreSQL checkpointer in production, in-memory saver in unit tests;
- Pydantic v2/TypedDict for validated state updates;
- model gateway and Tool Registry adapters defined by their contracts.

### Construction sequence

1. Load and validate all JSON graph specs in CI.
2. Create one Python callable per JSON handler name. Its signature is logically
   `node(state, runtime_context) -> partial_state_update`.
3. Build child `StateGraph` objects first and compile them.
4. Add child graphs as main-graph nodes without changing their JSON node IDs.
5. Convert every JSON condition into a deterministic route function. Do not use
   `eval` or ask the LLM to choose an edge that is defined deterministically.
6. Compile with a checkpointer.
7. Invoke with `configurable.thread_id = session_id` and a request/run ID.
8. Add recursion/step limits matching `max_steps`.
9. Stream node events to the API, but checkpoint state before acknowledging a
   durable transition.

Illustrative shape (not copy-paste production code):

```python
builder = StateGraph(ProductionAgentState)
builder.add_node("parse_request", parse_request)
builder.add_node("retrieve_project_knowledge", retrieve_project_knowledge)
builder.add_node("data_model_subgraph", compiled_data_graph)
builder.add_node("planning_safety_subgraph", compiled_planning_graph)
builder.add_conditional_edges("task_router", route_task, {...})
compiled = builder.compile(checkpointer=postgres_checkpointer)
```

### Checkpoint acceptance

- State resumes after process restart using the same `session_id`/thread ID.
- A completed tool call is not executed twice after resume.
- A waiting-for-input run resumes from the expected node after input merge.
- Checkpoint payload contains no raw volume bytes and no unrestricted secrets.

## 9. Correctness test plan

### A. Graph/static contract tests

Run:

```bash
python3 -m unittest discover -s tests/graph -p 'test_graph_specs.py' -v
```

Pass criteria:

- all three JSON files parse and validate;
- required node IDs are unchanged;
- every edge target exists and `END` is reachable;
- unconditional fallback is last;
- subgraph paths stay inside `graph/`;
- condition DSL rejects unsupported/executable expressions.

### B. End-to-end branch tests

Run:

```bash
python3 -m unittest discover -s tests/graph -p 'test_mock_runtime.py' -v
```

Required scenarios:

| Scenario | Required result |
|---|---|
| Planning success | all four planning/safety tools called; verified `PASS` |
| MCS data flow | conversion, schema, segmentation, validation, skin tools called |
| Missing case ID | `AWAITING_INPUT`; zero algorithm tool calls |
| No feasible path | valid `COMPLETED_WITH_NO_RESULT`; no safety tool after empty candidates |
| One-time timeout | failed call recorded, one retry, eventual success |
| Checkpoint round trip | serialized/restored state equals original |

When moving to LangGraph, duplicate these scenarios against the compiled graph.
Mock numeric values may change; branch path and status assertions may not.

### C. Node/tool contract tests

For each node that calls a tool, test:

1. valid contract input reaches the expected tool exactly once;
2. returned contract object is mapped into the documented state field;
3. invalid schema fails before the tool implementation runs;
4. retryable and non-retryable errors route differently;
5. a response for another case/artifact lineage is rejected;
6. no raw imaging bytes are added to state or trace.

Use fake Tool Registry implementations, not real algorithms, for unit tests.

### D. Verifier tests

Run:

```bash
python3 -m unittest discover -s tests/graph -p 'test_verifier.py' -v
```

Add tests for every new invariant. The verifier must remain deterministic and
must not call Qwen/RAG.

### E. Integration tests for production runtime

Use disposable PostgreSQL and fake model/tool servers. Verify:

- checkpoint restart and idempotent resume;
- concurrent sessions never share state;
- timeout cancellation closes pending calls;
- trace IDs propagate model -> RAG -> MCP tool;
- model returns malformed JSON: parser repairs once or rejects safely;
- prompt injection in a retrieved document cannot bypass the graph/tool ACL.

## 10. Acceptance gate

A real LangGraph runtime is complete only when:

- all existing graph/eval unit tests pass;
- topology matches JSON source of truth;
- successful, missing-input, no-path, validation-failure, transient-error, and
  retry-exhaustion cases have automated tests;
- state resumes correctly from PostgreSQL checkpoint;
- no tool call bypasses schema validation or permission checks;
- no LLM output can overwrite deterministic safety results;
- trace contains graph, node, RAG/model, and tool spans with one shared trace ID;
- evaluation report shows 100% schema validity and no forbidden-node execution
  on the contract test set.
