# Testing Qwen / vLLM Gateway Module

## 1. Purpose and evidence labels

This document is the executable verification contract for Module 1. It separates
offline protocol correctness from live Qwen/vLLM deployment evidence.

Use only these status values in release notes:

- `PASS`: the named command/check ran against the recorded revision and passed;
- `FAIL`: it ran and failed; retain the sanitized failure evidence;
- `NOT_RUN`: it did not run, including because Docker, a model, or an idle GPU was
  unavailable;
- `SKIP`: an individual test was intentionally gated; the corresponding live
  evidence remains `NOT_RUN`.

Fake-transport success proves the adapter, not the GPU service. Static Compose or
shell checks prove asset syntax, not that a container can load the checkpoint.

## 2. Clean offline test sequence

Run from the repository root. These commands must not contact the external network, start
Docker, download a model, or reserve a GPU.

### 2.1 Public contract-v2 tests

```bash
python3 tests/contract/test_model_rag_contracts.py -v
```

Pass criteria:

- process exits `0`;
- contract version is `2`;
- assistant tool-call history, unknown usage, structured stream errors, and
  tools/schema exclusivity assertions pass.

### 2.2 Model-gateway unit and failure tests

```bash
python3 -m unittest discover -s tests/model_gateway -p 'test*.py' -v
```

Pass criteria:

- process exits `0` with no unexpected skips;
- mock tests and production-adapter fake-transport tests pass;
- every negative test asserts an exact normalized code or terminal error event;
- no test performs external network access.

### 2.3 Deployment asset tests

Run both the colocated asset suite and its repository-discovery wrapper:

```bash
python3 -m unittest discover -s deploy/qwen-vllm/tests -p 'test*.py' -v
python3 -m unittest discover -s tests/deployment -p 'test*.py' -v
```

Always run the dependency-free syntax/help checks:

```bash
bash -n deploy/qwen-vllm/entrypoint.sh
python3 deploy/qwen-vllm/scripts/health_check.py --help
python3 deploy/qwen-vllm/scripts/smoke_test.py --help
python3 deploy/qwen-vllm/scripts/benchmark.py --help
python3 deploy/qwen-vllm/scripts/sizing_estimator.py --help
```

Pass criteria:

- every command exits `0`;
- shell parsing does not execute or print a secret;
- help commands do not make a provider request;
- both deployment suites discover at least one test; a zero-test run is `FAIL`.

### 2.4 Repository regression

```bash
python3 run_tests.py
```

Pass criteria: process exits `0`; changes to Module 1 do not regress Artifact
Registry, RAG contracts, mocks, or other existing suites.

### 2.5 Import and syntax compilation

```bash
python3 -m compileall -q contracts src tests deploy/qwen-vllm/scripts
```

Pass criteria: process exits `0` and does not create a tracked change.

### 2.6 Secret and generated-file review

The following are review aids, not a substitute for a secret scanner:

```bash
git status --short
git diff --check
git grep -n -E 'VLLM_API_KEY=.+|hf_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{12,}' -- . ':!docs/testing-qwen-vllm.md'
```

Pass criteria:

- `git diff --check` is empty;
- no real credential is found;
- `.env`, model cache, benchmark output containing prompts, and secret files are
  untracked/ignored;
- expected source changes are reviewed rather than requiring a globally clean
  worktree during active development.

## 3. Offline test matrix

The implementation is not acceptable unless every row has a deterministic test.

| Area | Fixture/action | Required result |
|---|---|---|
| Method mode | call `generate` with `stream=True` | non-retryable `MODEL_REQUEST_REJECTED`, attempts 0; zero transport calls |
| Method mode | call `stream` with `stream=False` | one terminal `MODEL_REQUEST_REJECTED` error event; zero transport calls |
| Assistant replay | assistant has two prior calls followed by matching tool messages | provider body preserves IDs/names/strict argument JSON |
| Assistant replay | duplicate call IDs or calls on a user message | construction fails |
| Output mode | tools and `response_schema` together | construction fails before network |
| Choice | request sends `n=1`; response has one choice at integer index 0 | accepted |
| Choice | zero/multiple/missing-index/index 1/non-object non-stream choice | `MODEL_PROTOCOL_ERROR` |
| Stream choice | empty choices with usage | accepted as usage-only frame |
| Stream choice | empty choices without usage or multiple/index 1 | terminal `MODEL_PROTOCOL_ERROR` event |
| Model ID | exact configured ID in health and response | accepted |
| Model ID | absent or different response ID | protocol failure; no silent alias |
| Usage | valid counts present | exact counts; `usage_known=True` |
| Usage | absent | zero sentinel; `usage_known=False` |
| Usage | negative/inconsistent/non-object | `MODEL_PROTOCOL_ERROR` |
| Tool | offered name and valid object arguments | validated `ToolCall` returned/emitted |
| Tool | unknown name | `UNKNOWN_TOOL`; no tool event |
| Tool | malformed/non-object/duplicate-key/non-finite JSON | `TOOL_ARGUMENT_PARSE_ERROR`; no tool event |
| Tool | missing/extra/wrong-type field | `TOOL_ARGUMENT_SCHEMA_ERROR`; no tool event |
| Tool | duplicate provider call ID | protocol failure; no tool event |
| Structured | valid strict JSON object | local validation succeeds; text is not duplicated |
| Structured | missing/malformed/duplicate-key/non-finite/schema mismatch | `STRUCTURED_OUTPUT_SCHEMA_ERROR` |
| Body JSON | invalid UTF-8/top-level/list/duplicate key/non-finite | `MODEL_PROTOCOL_ERROR` |
| Empty response | no text/tool/structured value | `EMPTY_MODEL_RESPONSE` |
| Retry | timeout or 408 then success | exact two calls and one bounded delay |
| Retry | 429 then success | honors bounded numeric/date `Retry-After` |
| Retry | 503 through exhaustion | `max_retries + 1` calls and retry-exhausted details |
| No retry | TLS/security/401/403/400/404/413/422/schema errors | exactly one call |
| SSE | byte/UTF-8/CRLF/multi-line fragmentation | content reconstructed independent of chunks |
| SSE tool | fragmented ID/name/argument JSON | no event until complete validation |
| SSE success | text/tool/usage plus `[DONE]` | contiguous sequence; exactly one completed terminal event |
| SSE failure | initial transient exhaustion | exactly one structured error terminal event |
| SSE failure | transient disconnect after delta | no retry; terminal error has `retryable=false`, `output_visible=true`, `upstream_retryable=true`; no completed |
| Error payload | construct/emit incomplete, mistyped, or mixed error payload | rejected; requires non-empty code/message, boolean retryable, object details, and no other event payload |
| Limits | oversized body/event/total stream | non-retryable security/protocol terminal failure |
| Redirect/proxy | cross-origin redirect or ambient proxy variable | blocked; authorization not forwarded |
| Header | request ID contains CR/LF | rejected before transport |
| Redaction | secret appears in fake provider header/body/exception | absent from error string, repr, details, and captured logs |

When the implementation chooses a more specific local invalid-request code, add
it to the public error-code documentation and assert it consistently. Do not map a
local caller error to a retryable provider failure.

## 4. Optional live environment contract

Live verification is disabled by default. An automated live test must check
`RUN_VLLM_INTEGRATION == "1"`; otherwise it skips without opening a socket.

| Variable | Required | Meaning |
|---|---:|---|
| `RUN_VLLM_INTEGRATION` | yes | exact value `1` opts in |
| `VLLM_BASE_URL` | yes | private endpoint ending in `/v1` |
| `VLLM_MODEL` | yes | exact served model ID, not a display alias |
| `VLLM_API_KEY` | no | bearer secret consumed by the gateway live suite; never commit it |
| `VLLM_CA_BUNDLE_PATH` | no | approved enterprise CA bundle passed explicitly to the production transport |
| `VLLM_TIMEOUT_SECONDS` | no | gateway attempt timeout, default `60` |
| `VLLM_MAX_RETRIES` | no | gateway retries after first attempt, default `1` |
| `VLLM_API_KEY_FILE` | no | supported by deployment scripts; load into the process environment before the gateway live suite |
| `VERIFY_TIMEOUT_SECONDS` | no | direct smoke-script timeout, default `120` |
| `BENCHMARK_REQUESTS` | no | sample count |
| `BENCHMARK_CONCURRENCY` | no | workers for one benchmark profile |
| `BENCHMARK_MAX_TOKENS` | no | requested output bound |
| `BENCHMARK_STREAM` | no | true/false |

Do not put any of these values in shell history, source, a committed `.env`, test
output, screenshots, or trace exports. Only the base URL's sanitized host class
(for example `private-loopback`) belongs in public release evidence.

## 5. Live verification sequence

Run this section only when the endpoint, checkpoint revision, and GPU allocation
are approved. Manual execution is itself explicit opt-in; the `RUN_...` variable
is mandatory for an automated test suite.

### 5.1 Health and capability smoke test

```bash
export RUN_VLLM_INTEGRATION=1
export VLLM_BASE_URL='http://127.0.0.1:8000/v1'
export VLLM_MODEL='qwen-enterprise-agent'
python3 deploy/qwen-vllm/scripts/health_check.py \
  --base-url "$VLLM_BASE_URL" \
  --model "$VLLM_MODEL" \
  --wait-seconds 900
python3 deploy/qwen-vllm/scripts/smoke_test.py \
  --base-url "$VLLM_BASE_URL" \
  --model "$VLLM_MODEL"
```

Do not use `--skip-tools` or `--skip-structured` for release acceptance. Those
flags are fault-isolation aids only.

Pass criteria:

1. exact model appears in `/models`;
2. plain response is non-empty;
3. forced tool name and argument object are valid;
4. structured response contains exactly the schema fields;
5. output is sanitized and no prompt/provider body is retained as evidence.

### 5.2 Gateway-level live test

Run the committed environment-gated gateway suite:

```bash
RUN_VLLM_INTEGRATION=1 python3 tests/model_gateway/test_live_vllm.py -v
```

It must cover `VllmModelGateway`, not only direct HTTP scripts:

1. health;
2. `generate()` plain chat;
3. one validated tool call;
4. one structured response;
5. `stream()` with contiguous events and one completed event.

Assistant replay is covered offline. Add a sanitized two-turn live replay case
before claiming that a specific model/chat-template pair has live multi-turn tool
compatibility. If the variable is unset, the suite's skips mean live status is
`NOT_RUN`. Do not claim gateway live compatibility based only on a direct
cURL/script request.

### 5.3 Context and concurrency

Use only an approved sanitized corpus. Verify one request near the project context
limit without exceeding the admission envelope. Then benchmark concurrency 1, 4,
the intended production load, and one overload case separately.

Example profile:

```bash
mkdir -p benchmark-results
python3 deploy/qwen-vllm/scripts/benchmark.py \
  --base-url "$VLLM_BASE_URL" \
  --model "$VLLM_MODEL" \
  --requests 40 \
  --concurrency 4 \
  --max-tokens 128 \
  --stream true \
  --output benchmark-results/concurrency-4.json
```

`benchmark-results/` is working evidence and must remain ignored until a sanitized
summary is deliberately added to release documentation.

## 6. Benchmark evidence template

Copy this section into the private evidence record or release note. Use
`NOT_RUN`/`NOT_MEASURED` rather than estimates.

### 6.1 Identity and configuration

| Field | Recorded value |
|---|---|
| Evidence status | `NOT_RUN` |
| UTC date/time | `NOT_RUN` |
| Gateway commit/tag | `NOT_RUN` |
| vLLM image tag and digest | `NOT_RUN` |
| vLLM version | `NOT_RUN` |
| Qwen repository/model | `NOT_RUN` |
| Immutable model revision | `NOT_RUN` |
| Weight format/quantization | `NOT_RUN` |
| Chat template source/checksum | `NOT_RUN` |
| Tool-call parser/plugin | `NOT_RUN` |
| Reasoning parser/mode | `NOT_RUN` |
| Structured-output backend/flags | `NOT_RUN` |
| Driver/CUDA | `NOT_RUN` |
| GPU type/count | `NOT_RUN` |
| Tensor-parallel size | `NOT_RUN` |
| GPU-memory-utilization target | `NOT_RUN` |
| Maximum model length | `NOT_RUN` |
| Maximum sequences/batch tokens | `NOT_RUN` |
| Prefix caching/chunked prefill | `NOT_RUN` |

### 6.2 Workload

| Field | Recorded value |
|---|---|
| Sanitized dataset/version/hash | `NOT_RUN` |
| Warm-up requests | `NOT_RUN` |
| Measured requests | `NOT_RUN` |
| Prompt tokens P50/P95/max | `NOT_RUN` |
| Requested output tokens | `NOT_RUN` |
| Actual output tokens P50/P95 | `NOT_RUN` |
| Temperature | `NOT_RUN` |
| Streaming mode | `NOT_RUN` |
| Concurrency | `NOT_RUN` |
| Timeout/admission limits | `NOT_RUN` |

### 6.3 Results

| Metric | Value |
|---|---|
| Successful / total requests | `NOT_RUN` |
| Success rate | `NOT_RUN` |
| End-to-end latency P50/P95 | `NOT_RUN` |
| TTFT P50/P95 | `NOT_RUN` |
| TPOT P50/P95 | `NOT_MEASURED` |
| Request throughput | `NOT_RUN` |
| Completion-token throughput | `NOT_RUN` |
| Peak GPU memory per device | `NOT_MEASURED` |
| Queue depth/scheduler wait P95 | `NOT_MEASURED` |
| Tool-call parse-valid rate | `NOT_RUN` |
| Tool-call schema-valid rate | `NOT_RUN` |
| Structured-output parse-valid rate | `NOT_RUN` |
| Structured-output schema-valid rate | `NOT_RUN` |
| Timeout/429/5xx/OOM count | `NOT_RUN` |

### 6.4 Decision

```text
Release decision: NOT_RUN
Approved concurrency/context/output envelope: NOT_RUN
Known failures and sanitized references: NONE RECORDED
Rollback image/model/gateway revision: NOT_RUN
Reviewer and UTC timestamp: NOT_RUN
```

## 7. Release acceptance checklist

Module 1 passes its offline acceptance gate only when all checked items have saved
command output or CI evidence for the exact commit:

- [ ] contract-v2 direct suite passes;
- [ ] all model-gateway mock/adapter suites pass;
- [ ] both deployment asset suites and syntax checks pass;
- [ ] full `python3 run_tests.py` regression passes;
- [ ] compile and diff checks pass;
- [ ] method, choice, model-ID, usage sentinel, tool, schema, strict JSON, retry,
      streaming terminal, size, proxy, redirect, TLS, header, and redaction rows
      have deterministic coverage;
- [ ] no invalid tool call can be returned or emitted;
- [ ] no retry occurs after visible streamed output;
- [ ] pre-visible transient stream disconnects retry within the same total attempt/deadline budget;
- [ ] every stream ends in exactly one structured terminal event;
- [ ] live status is stated as `PASS`, `FAIL`, or `NOT_RUN`;
- [ ] no latency, throughput, memory, or validity-rate number is published unless
      the benchmark identity/workload/result fields are populated.

Live deployment acceptance additionally requires every capability smoke check,
gateway-level live test, context test, intended-load benchmark, telemetry snapshot,
and rollback record to be `PASS`. An offline-only commit can be useful and pushed,
but it is not proof of a running Qwen private deployment.
