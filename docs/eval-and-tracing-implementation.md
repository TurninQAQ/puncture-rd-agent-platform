# Task 08: Evaluation and Tracing Implementation

## Status

Production-facing observability and offline evaluation contracts are implemented
on top of the standard-library scaffold. Vendor backends (Langfuse/Phoenix/OTLP
collector) remain optional adapters; unit tests never require a live service.

## Span hierarchy

```text
eval.case                    # optional eval runner parent
└── agent.graph
    ├── agent.node (...)
    │   ├── rag.rewrite
    │   ├── rag.retrieve
    │   └── rag.rerank
    ├── agent.node (...)
    │   └── mcp.tool
    ├── agent.node (...)
    │   └── model.generate
    ├── agent.verifier
    └── agent.checkpoint
```

Reference graph runtime already emits `agent.graph` / `agent.node`. Adapters use
helpers in `puncture_agent.observability.instrumentation` for RAG/model/MCP/
verifier/checkpoint children without importing a vendor SDK.

## Attribute allowlist

Strict allowlist + denylist redaction lives in
`src/puncture_agent/observability/attributes.py`.

- Allowlisted keys keep operational metadata (`agent.*`, `rag.*`, `model.*`,
  `tool.*`, `checkpoint.*`, correlation IDs).
- Denylisted or sensitive keys (`authorization`, `patient_*`, `full_prompt`,
  `image_bytes`, …) export as `[REDACTED]`.
- Unknown keys are dropped.
- Query text is hashed (`rag.rewritten_query_hash`); unrestricted prompts are
  never stored on spans.

## Propagation

`propagation.py` implements W3C `traceparent` plus `x-puncture-trace-id` /
`x-puncture-span-id` carriers for HTTP headers and MCP/gRPC-style metadata.
`TraceRecorder.inject_headers` / `attach_remote_parent` are the public ports.

## Exporters

| Exporter | Role |
|---|---|
| `InMemoryTraceExporter` | unit tests / demos |
| `JsonLinesTraceExporter` | local replay |
| `InMemoryOtlpTraceExporter` | OTLP-shaped in-memory collector |
| `CompositeTraceExporter` | fan-out |
| `OpenTelemetryTraceExporter` | optional real OTEL SDK bridge |

Export failures increment `export_failure_count` and never fail the request.
Blocked remote collectors surface as enqueue failures without blocking beyond
the configured timeout policy (simulated immediately in tests).

## Dataset and metrics versions

| Artifact | Version |
|---|---|
| Eval case schema | `eval-case-v1` |
| Metric edge policy | `metrics-v1` |
| Aggregate report | `eval-report-v1` |
| Regression report | `eval-regression-v1` |
| Pinned mock suite | `mock-reference-v1` |

Loader: `load_eval_dataset` accepts JSON objects, JSON arrays, and JSONL with an
optional `# meta: {...}` header. Unsupported schema versions, unknown fields,
duplicate `case_id`, and empty suites fail closed.

## Metrics and gates

Deterministic metrics include Recall@K, MRR, NDCG@K, active-version hit rate,
ACL violation rate, routing/status accuracy, required-node coverage,
forbidden-node/tool rates, tool precision/recall, tool-parameter predicates,
retry recovery, and step/latency percentiles.

Release gates block on:

- forbidden node/tool rate > 0
- ACL violation rate > 0
- report schema validity < 1
- (regression) worsened safety metrics or newly failing cases

Safety/security regressions block even when mean quality improves.

## CLI

```bash
PYTHONPATH=src:. python -m puncture_agent.observability.eval_runner run \
  --dataset tests/eval/fixtures/mock_reference_v1.json \
  --traced --fail-on-release-block \
  --output /tmp/mock-eval-report.json

PYTHONPATH=src:. python -m puncture_agent.observability.eval_runner compare \
  --baseline /tmp/baseline.json --candidate /tmp/candidate.json \
  --fail-on-regression
```

## Benchmark

```bash
PYTHONPATH=src:. python benchmarks/tracing_overhead_benchmark.py \
  --warmups 5 --samples 50 \
  --environment-label controlled-host \
  --output /tmp/tracing-overhead.json
```

Default mode records only. Relative P95 overhead <= 5% is an engineering gate
for controlled hosts (`--enforce` / `PUNCTURE_ENFORCE_PERFORMANCE_GATES=1`). On
sub-millisecond mock graph paths the relative ratio is dominated by absolute
export cost (~1 ms) and is not a production SLA.

## Privacy and remaining risks

- Company OIDC principal hashing and restricted prompt stores are not wired.
- Live OTLP/Langfuse/Phoenix dashboards are opt-in; configure
  `OpenTelemetryTraceExporter` only after collector capacity review.
- Sampling policy (100% errors/retries/eval, partial success traffic) is
  documented but not yet a remote sampler integration.
- Medical algorithm quality (Dice, collision distance, etc.) is out of scope.
- Production adapters must still implement their own durable idempotency; tracing
  does not claim exactly-once tool side effects.

## Sample outputs

Passing aggregate report (mock-reference-v1, traced):

- `case_count=3`, `passed_case_count=3`
- `task_success_rate=1.0`, `forbidden_node_violation_rate=0.0`
- `tool_parameter_validity_rate=1.0`, `retry_recovery_rate=1.0`
- every case has non-empty `trace_id`

Failing diagnostic example (intentional wrong expectations):

- failures include exact task_type mismatch, missing required nodes, missing
  tools, and citation shortfall
- `observed` carries visited nodes, called tools, status, and optional trace id
