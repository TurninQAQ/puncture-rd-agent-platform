# Agent Evaluation and Tracing Specification

## 1. Goals

The evaluation layer answers four different questions independently:

1. Did RAG retrieve the correct active internal evidence?
2. Did the Agent choose the correct graph branch and tools?
3. Did tools receive valid parameters and recover safely from failures?
4. Can an engineer replay why the final result was produced?

Do not collapse these into one LLM-as-judge score. Deterministic metrics and
contract checks are primary; an LLM judge is optional for explanation quality.

## 2. Current scaffold

| File | Purpose |
|---|---|
| `observability/tracing.py` | nested spans, in-memory exporter, JSONL exporter |
| `observability/metrics.py` | Recall@K, reciprocal rank, NDCG@K |
| `observability/eval_harness.py` | EvalCase schema, execution, diagnostics, aggregate report |
| `tests/eval/test_tracing.py` | parent/trace linkage and replayable JSONL |
| `tests/eval/test_metrics.py` | exact metric behavior and edge cases |
| `tests/eval/test_eval_harness.py` | passing suite and actionable failing diagnostics |

These modules use only the standard library. A real implementation may export
to OpenTelemetry/Langfuse/Phoenix but must preserve the testable semantics.

## 3. Trace model

Every completed span contains:

| Field | Meaning |
|---|---|
| `trace_id` | one ID shared by the full user request |
| `span_id` | unique operation ID |
| `parent_span_id` | caller span or null for root |
| `name` | stable low-cardinality operation name |
| start/end/duration | timing data |
| `status` | `OK` or `ERROR` |
| `attributes` | sanitized indexed metadata |
| `events` | timestamped state/tool events |
| `error` | exception type/message when failed |

### Required span hierarchy

```text
agent.graph
├── agent.node (parse_request)
├── agent.node (retrieve_project_knowledge)
│   ├── rag.rewrite
│   ├── rag.retrieve
│   └── rag.rerank
├── agent.node (planning_safety_subgraph)
│   ├── agent.node (generate_candidate_paths)
│   │   └── mcp.tool
│   └── ...
└── agent.node (report_generator)
    └── model.generate
```

The standard-library runtime currently emits `agent.graph` and `agent.node`.
Real RAG, model, and MCP adapters add the child spans.

### Required attributes

Root graph span:

- `agent.graph_id`, `agent.session_id`, request/run ID;
- model name/version, prompt version (when known);
- final status, verification status, retry count;
- aggregate token usage and latency.

Node span:

- `agent.node_id`, `agent.node_kind`, retry count;
- input/output schema version;
- branch result for router nodes.

RAG span:

- rewritten query hash or sanitized text;
- embedding/index/reranker versions;
- top-k values, filters, retrieved document/chunk IDs and scores;
- current-version hit and retrieval latency.

Tool span:

- tool name/version, call/request ID, read/write classification;
- request schema version, response status/error code;
- latency, retry attempt, artifact IDs/checksums;
- never raw CT/Mask data.

Model span:

- model/deployment version, temperature, max tokens;
- input/output token counts, TTFT, total latency;
- structured-output validation result;
- do not persist unrestricted full prompts in production.

### Sensitive-data rules

- Use internal `case_id`, artifact ID, checksum, and user ID hash; do not record
  patient names, raw DICOM tags, access tokens, or voxel bytes.
- Apply allowlist-based attribute serialization.
- Store prompt/tool payload samples only in an approved restricted store with a
  retention policy.
- Trace viewer access must follow the same internal ACL as source documents.

## 4. Production export

Instrument with OpenTelemetry. Export traces/metrics to one selected backend,
for example Langfuse or Phoenix. Do not couple business code to a vendor SDK:
adapt the existing `TraceRecorder` interface or use an observability facade.

Propagation requirements:

1. API creates or accepts one trace context.
2. LangGraph nodes inherit it.
3. RAG/model requests inject it into headers/metadata.
4. MCP/gRPC/REST tool clients propagate it.
5. Tool-side C++/TensorRT services return/log the same trace ID.

Sampling: keep 100% of errors, retries, permission denials, manual-review runs,
and evaluation traffic. Normal successful traffic may be sampled according to
capacity, but aggregate metrics remain complete.

## 5. Evaluation-case schema

The minimal executable case is represented by `EvalCase`:

```json
{
  "case_id": "planning-timeout-recovery-001",
  "query": "对 Case-405 做路径规划和风险判断",
  "expected_task_type": "PLANNING_SAFETY",
  "expected_terminal_status": "SUCCEEDED",
  "required_nodes": [
    "generate_candidate_paths",
    "error_recovery",
    "evaluate_path_safety"
  ],
  "forbidden_nodes": ["convert_mcs_to_nifti"],
  "expected_tools": [
    "generate_candidate_paths",
    "evaluate_path_safety"
  ],
  "minimum_citations": 2,
  "agent_case_id": null,
  "metadata": {
    "fail_tool_once": ["generate_candidate_paths"]
  },
  "planning_constraints": {
    "max_needle_length_mm": 120.0
  }
}
```

Production datasets should add expected relevant document IDs, expected tool
argument constraints, expected error code, max steps/retries, security policy,
and optional human-reference answer. Keep fixture content synthetic or approved
internal data.

## 6. Dataset composition

Initial internal suite target: approximately 150 cases.

| Group | Suggested count | Examples |
|---|---:|---|
| Data/format/label | 30 | MCS conversion, spacing mismatch, wrong label value |
| Segmentation/model | 25 | correct inference, empty Mask, model version mismatch |
| Planning | 30 | success, length/angle constraint, no candidate |
| Safety/risk | 25 | envelope collision, warning/stop, large vessel, skin penetration |
| RAG/version/ACL | 20 | active vs obsolete rule, table lookup, permission denial |
| Reliability/security | 20 | timeout, malformed JSON, prompt injection, retry exhaustion |

Every bug fixed in production adds one immutable regression case.

## 7. Metric definitions

### RAG retrieval

- `Recall@K`: relevant unique documents retrieved in top K / all relevant
  documents. Empty relevance sets score 1 only for explicit no-answer cases.
- `MRR`: mean inverse rank of the first relevant document.
- `NDCG@K`: graded relevance with logarithmic rank discount.
- active-version hit rate: cases retrieving the required current version.
- ACL violation rate: unauthorized retrieved chunks / all retrieved chunks;
  required value is exactly zero.

Test these metrics separately from answer generation. The reference functions
are in `observability/metrics.py`.

### RAG answer

- grounded claim rate: supported factual claims / factual claims;
- citation precision: citations that support their attached claims / citations;
- citation recall: expected supporting sources cited / expected sources;
- correct refusal rate on unanswerable/unauthorized cases;
- obsolete-version answer rate, required to trend toward zero.

Claims/citations require a labeled evaluator or reviewed judge. Store judge
prompt/version and perform regular human calibration if an LLM judge is used.

### Agent/runtime

- task success rate: cases satisfying every hard expectation / all cases;
- routing accuracy: exact expected task branch;
- terminal status accuracy;
- required-node coverage;
- forbidden-node violation rate;
- tool selection precision and recall;
- tool parameter/schema validity rate;
- retry recovery rate and retry-exhaustion correctness;
- hallucinated tool-call rate;
- average/P95 graph steps, latency, tokens, and cost.

The current harness computes the first group of graph-level metrics. Extend it
using the same `EvalCaseResult` diagnostic style, never a silent scalar only.

### Safety-oriented gates

These are contract/algorithm integration gates, not LLM judgment:

- missing critical artifact must never reach planning tools;
- no path rejected by `evaluate_path_safety` may appear as accepted in report;
- contradictory risk flags must result in `MANUAL_REVIEW`;
- an algorithm/tool failure must not become a successful report;
- an unauthorized/write tool must not execute without policy/approval;
- raw medical image data must not appear in model context or trace.

## 8. Fault injection

The Mock runtime supports deterministic `metadata` flags such as
`fail_tool_once`, `fail_tool_always`, `fail_tool_non_retryable`,
`force_no_feasible_path`, `force_geometry_mismatch`,
`force_label_schema_error`, and `missing_required_artifacts`.

Production fault tests should additionally inject:

- model timeout/malformed tool JSON;
- RAG empty result, wrong version, index timeout;
- MCP timeout, unavailable service, permission denied;
- PostgreSQL checkpoint restart;
- duplicate request/idempotency replay;
- GPU out of memory and TensorRT service restart;
- prompt injection inside a retrieved document;
- artifact case/geometry mismatch.

Each injection defines exact expected node path, status, error code, retry
count, tool-call count, and trace event.

## 9. Correctness test plan by module

### Trace recorder/exporters

Run:

```bash
python3 -m unittest discover -s tests/eval -p 'test_tracing.py' -v
```

Verify:

- root and nested spans share a trace ID;
- parent IDs reproduce graph/subgraph nesting;
- successful/error status and duration are populated;
- JSONL is one valid JSON object per completed span;
- events are ordered and timestamped;
- exceptions are re-raised after an error span is exported;
- concurrent contexts do not leak parent spans across requests.

Add exporter integration tests when OpenTelemetry is introduced, using an
in-memory collector rather than a live external backend.

### Retrieval metrics

Run:

```bash
python3 -m unittest discover -s tests/eval -p 'test_metrics.py' -v
```

Verify exact hand-calculated examples, no-hit cases, empty relevance policy,
duplicate retrieved IDs, graded relevance, and invalid K/negative relevance.

### Agent evaluation harness

Run:

```bash
python3 -m unittest discover -s tests/eval -p 'test_eval_harness.py' -v
```

Verify:

- reference planning/data cases pass;
- failures identify the exact wrong route/status/node/tool/citation condition;
- aggregate metrics equal per-case calculations;
- cases are isolated with a fresh runtime/checkpoint namespace;
- empty suites fail rather than report misleading 100%;
- reports serialize to JSON.

### RAG evaluator extension

For each labeled question:

1. freeze query, authorized user scope, document/index versions;
2. execute retrieval without answer generation;
3. compare ranked IDs with relevant IDs;
4. assert ACL/version constraints;
5. then execute generation and evaluate claim/citation grounding separately.

### Tool evaluator extension

Record expected tool name and field predicates, for example
`max_needle_length_mm == 120`, `safety_radius_mm > 0`, artifact case IDs match.
Validate the generated request against shared schemas before comparing values.

## 10. CI and regression process

Suggested stages:

1. `contract`: graph/static/schema and unit tests, required on every change.
2. `mock-e2e`: full deterministic graph and 20-50 fast Eval cases.
3. `rag-offline`: retrieval dataset against a pinned index snapshot.
4. `integration`: disposable PostgreSQL/model/tool stubs.
5. `nightly`: real Qwen/RAG/tool services, latency and quality trend report.

Compare candidate and baseline using the exact same model, prompt, index,
dataset, temperature, and seed where supported. Report both aggregate change
and the list of improved/regressed cases. Block release on safety/security
regressions even if average success improves.

## 11. Initial acceptance thresholds

Thresholds should later be calibrated on internal baselines. The first contract
gate is:

- graph/report schema validity: 100%;
- forbidden-node/tool violation: 0%;
- ACL violation: 0%;
- missing-input unsafe continuation: 0 cases;
- transient one-time failure recovery: 100% of designated recovery cases;
- retry exhaustion routed correctly: 100%;
- reference Mock suite task success: 100%;
- every failed Eval case includes non-empty actionable diagnostics;
- all error/retry runs have a complete trace.

Do not claim medical performance from this Agent evaluation. Dice, collision
accuracy, safety-distance accuracy, and ray-tracing correctness belong to the
underlying algorithm modules and their own test specifications.
