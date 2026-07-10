# Task 08: Implement Production Agent Eval and Trace

## Task objective

Replace/extend the standard-library observability scaffold with production
OpenTelemetry export and a reproducible offline evaluation pipeline. Preserve
the existing EvalCase, metric, and diagnostic semantics.

This task evaluates and observes the Agent. It does not rewrite prompts to make
the current dataset pass, implement underlying medical algorithms, or change
expected results.

## Read before coding

Provide the implementing model:

1. `specs/eval-and-tracing.md` (primary specification)
2. `specs/langgraph-runtime.md`
3. `src/puncture_agent/observability/**`
4. `src/puncture_agent/agent/state.py`
5. all `graph/*.json`
6. `tests/eval/**` and `tests/graph/**`
7. RAG response contracts and model/tool trace metadata contracts

Use synthetic fixtures unless approved internal evaluation data is available in
the execution environment.

## Files allowed to change

- `src/puncture_agent/observability/**`
- `tests/eval/**`
- evaluation configuration/dataset paths explicitly assigned by the maintainer
- observability deployment configuration explicitly assigned by the maintainer

Do not change graph routes, tool schemas, or expected case labels to improve
scores.

## Required deliverables

1. OpenTelemetry instrumentation facade compatible with current TraceRecorder.
2. Spans for graph, node, model, RAG stages, MCP/tool, verifier, and checkpoint.
3. Trace propagation across HTTP/gRPC/MCP boundaries.
4. Sensitive-attribute allowlist/redaction.
5. Configurable exporter to Langfuse, Phoenix, or an OTLP collector.
6. Versioned Eval dataset loader/validator.
7. RAG retrieval metrics: Recall@K, MRR, NDCG@K, active-version and ACL rates.
8. Agent metrics: routing, node/tool selection, parameter validity, recovery,
   status, schema, latency/token/step distributions.
9. Per-case diagnostics and baseline-vs-candidate regression report.
10. CI entry points for fast contract, offline RAG, integration, and nightly runs.

## Locked interfaces and behavior

- One request has one trace ID across every nested operation.
- `EvalCase` hard expectations remain deterministic.
- Empty evaluation suites fail.
- Metric edge-case policy matches `observability/metrics.py` or is versioned and
  migrated explicitly.
- Every failed case includes actionable reason(s), observed path/tools/status,
  and relevant trace ID.
- No PHI, credentials, raw image bytes, or unrestricted prompts in trace attrs.
- Safety/security failures block release regardless of mean quality improvement.

## Implementation sequence

1. Run and retain results of all current Eval tests.
2. Implement a generic observability facade and in-memory OTLP test exporter.
3. Instrument graph/node spans, then RAG/model/tool/checkpoint spans.
4. Add context propagation tests before configuring a live backend.
5. Add versioned JSONL/JSON Eval dataset loader and schema validation.
6. Implement deterministic metrics and per-case diagnostics.
7. Add baseline comparison with lists of regressions/improvements.
8. Add fault-injection cases and CI commands.
9. Only then configure the selected production dashboard/export backend.

## Required tests

Run:

```bash
python3 -m unittest discover -s tests/eval -p 'test_*.py' -v
python3 -m unittest discover -s tests/graph -p 'test_*.py' -v
```

Add tests for:

- nested trace parenting and shared trace ID;
- error span exported while exception is re-raised;
- concurrent sessions do not leak context;
- HTTP/gRPC/MCP trace propagation with fake servers;
- redaction of denylisted/unknown sensitive fields;
- metric hand calculations, duplicate IDs, no-hit and empty-relevance cases;
- dataset schema/version rejection;
- correct current-version and ACL metrics;
- exact tool parameter predicate evaluation;
- baseline regression classification;
- malformed trace/eval input handled with clear error;
- JSON/JSONL report serialization;
- error/retry cases retain complete trace linkage.

Do not require a live Langfuse/Phoenix service in unit tests. Use an in-memory
collector/exporter and reserve live export for an opt-in integration test.

### Exact existing test IDs

```bash
python3 -m unittest -v \
  tests.eval.test_tracing.TracingTests.test_runtime_emits_parented_graph_and_node_spans \
  tests.eval.test_tracing.TracingTests.test_json_lines_exporter_writes_replayable_records \
  tests.eval.test_metrics.RetrievalMetricTests \
  tests.eval.test_eval_harness.EvalHarnessTests.test_reference_cases_pass_all_contract_metrics \
  tests.eval.test_eval_harness.EvalHarnessTests.test_failed_expectation_contains_actionable_diagnostics \
  tests.eval.test_eval_harness.EvalHarnessTests.test_empty_eval_suite_is_rejected
```

Add production tests without removing these dependency-free references.

### Failure-injection/evaluation rows

| Injection | Required trace/eval assertion |
|---|---|
| node raises exception | error span exported, same exception re-raised |
| model timeout/malformed JSON | model span error; no unauthorized tool call |
| RAG timeout/empty retrieval | retrieval span/error; correct retry or refusal |
| wrong/obsolete document version | active-version metric fails that case |
| unauthorized chunk | ACL violation recorded and release gate fails |
| tool one-time timeout | two linked tool spans, recovery metric passes |
| tool persistent timeout | retry spans stop at budget, terminal status correct |
| checkpoint restart | pre/post-restart spans share run correlation identifiers |
| concurrent requests | no parent/trace ID leakage between sessions |
| sensitive payload attribute | redacted/dropped before export |

## Performance and evaluation thresholds

Benchmark tracing both disabled and enabled with a local in-memory/OTLP test
collector. Document hardware, exporter batch settings, warm-up, and sample size.
Initial gates are:

- batched tracing adds <= 5% P95 latency to the graph-only benchmark;
- application requests never block on an unavailable remote collector beyond a
  configured <= 50 ms export enqueue timeout;
- 100% trace coverage for error, retry, manual-review, and Eval traffic in the
  controlled integration suite;
- zero cross-request parent/trace leakage under at least 20 concurrent sessions;
- report/metric JSON serialization success 100%;
- forbidden-node/tool rate and ACL violation rate exactly 0%;
- contract Eval task success, routing accuracy, terminal-status accuracy, tool
  recall, citation pass, and report schema validity all 100% on the pinned Mock
  reference suite;
- every failed case has at least one diagnostic plus observed route/tools/status
  and trace/run ID;
- deterministic metric reruns on a pinned dataset are byte-identical except
  explicitly excluded timestamp/duration fields.

Retrieval/answer quality targets must be established from the real pinned
internal dataset. Report Recall@5/10, MRR, NDCG@10, groundedness, and citation
accuracy, but do not invent passing numbers before a baseline exists.

## Copy-paste implementation instruction

```text
Implement Task 08 using specs/eval-and-tracing.md as the source of truth.
Preserve TraceRecorder/EvalCase semantics and all exact reference tests. Add an
OpenTelemetry facade with graph, node, RAG, model, MCP tool, verifier, and
checkpoint spans; propagate one trace context across process boundaries; enforce
an allowlist/redaction policy; and add versioned offline Eval dataset loading,
deterministic RAG/Agent metrics, per-case diagnostics, and baseline regression
reports. Implement every failure-injection row and performance/evaluation gate
listed here using in-memory fake collectors/services in tests. Do not require a
live vendor backend and do not alter labels to improve scores. Return changed
files, span hierarchy/attribute allowlist, dataset and metric versions, complete
test/benchmark output, one passing aggregate report, one failing diagnostic,
and remaining privacy/instrumentation risks.
```

## Initial release gates

- 100% report and tool-argument schema validity on the contract dataset;
- 0 forbidden-node/tool and ACL violations;
- 100% correct missing-input, retry-recovery, and retry-exhaustion behavior;
- 100% trace coverage for failure and retry cases;
- no sensitive-field leak in redaction tests;
- all failed cases have non-empty diagnostics and trace ID;
- no unexplained regression against the pinned baseline.

Quality thresholds such as Recall@10 or task success must be reported from an
actual pinned internal dataset; do not invent target achievements.

## Expected response from the implementing model

Return:

1. files/configuration changed;
2. span hierarchy and attribute allowlist;
3. dataset/metric schema versions;
4. exact test commands and complete results;
5. sample aggregate report plus one failing-case diagnostic;
6. remaining instrumentation gaps and privacy risks.
