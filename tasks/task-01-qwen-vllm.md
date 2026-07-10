# Task 01 — Complete the Contract-v2 Qwen / vLLM Gateway

## Copyable assignment

Implement and verify the production protocol adapter that connects this
contract-first Agent repository to a private vLLM OpenAI-compatible endpoint
serving one exact Qwen model ID. The work must remain usable without a GPU through
fake-transport tests, while live deployment and benchmark proof stay explicitly
gated.

Do not implement medical imaging algorithms, MCP execution, RAG, LangGraph, or a
cloud-model fallback in this task.

## Read completely before editing

1. `specs/qwen-vllm-service.md`
2. `docs/testing-qwen-vllm.md`
3. `src/puncture_agent/model_gateway/models.py`
4. `src/puncture_agent/model_gateway/client.py`
5. `src/puncture_agent/model_gateway/http_transport.py`
6. `src/puncture_agent/model_gateway/mock_qwen.py`
7. `tests/contract/test_model_rag_contracts.py`
8. `tests/model_gateway/test_mock_qwen.py`
9. `tests/model_gateway/test_vllm_gateway.py`
10. `docs/qwen-deployment-runbook.md`

Repository contracts take precedence over framework tutorials. If implementation
and specification disagree, report the exact disagreement before broadening the
public contract.

## Goal and callable interface

Complete these methods:

```python
health() -> GatewayHealth
generate(ModelRequest) -> ModelResponse
stream(ModelRequest) -> Iterator[ModelStreamEvent]
```

The public contract version is `MODEL_GATEWAY_CONTRACT_VERSION = "2"`.
Provider client/response types must not escape `model_gateway/`.

## Allowed implementation area

Primary code and tests:

```text
src/puncture_agent/model_gateway/
tests/contract/test_model_rag_contracts.py
tests/model_gateway/
```

Deployment-specific work belongs under:

```text
deploy/qwen-vllm/
docs/qwen-deployment-runbook.md
```

Do not change RAG, graph, MCP, or medical-tool implementations. Do not delete or
weaken existing assertions. A production dependency may be added only when the
standard-library import path and offline tests remain usable.

## Fixed contract-v2 decisions

These decisions are complete. Do not reopen them inside the implementation.

### 1. Assistant tool-call replay

`ChatMessage.tool_calls` is a tuple of repository `ToolCall` objects and is legal
only for the assistant role. Serialize it to OpenAI assistant `tool_calls`, with
arguments as a strict JSON string. Preserve each call ID so following tool messages
can reference it. Reject duplicate assistant call IDs locally.

### 2. Tool mode versus structured-result mode

`ModelRequest.tools` and `ModelRequest.response_schema` are mutually exclusive.
Construction must fail before network I/O when both are supplied.

### 3. Generate and stream methods

- `generate()` accepts only `request.stream is False` and sends `stream=false`.
- `stream()` accepts only `request.stream is True` and sends `stream=true`.
- A mismatch raises local non-retryable
  `ModelGatewayError(code="MODEL_REQUEST_REJECTED")` with `details.attempts=0`
  and makes zero transport calls for `generate()`. For `stream()`, iteration must
  yield that failure as the only structured terminal `error` event.

### 4. One-choice policy

The repository explicitly sends `n=1`. Non-stream responses must contain exactly
one choice at integer index 0. Streaming frames may contain zero choices only for a usage
frame; all other data frames contain exactly one choice at index 0. Multiple
choices, missing/alternative indices, and malformed choices fail with
`MODEL_PROTOCOL_ERROR`. Never concatenate or silently ignore alternatives.

### 5. Exact model-ID policy

`VllmGatewayConfig.model` is the exact served ID. Health requires that exact value
in `/models`. Every completion/SSE response must report the same non-empty model
ID. Missing or different IDs fail with `MODEL_PROTOCOL_ERROR`. Do not accept an
alias or dynamically switch model.

### 6. Unknown usage

If usage exists, validate it and return `usage_known=True`. If it is absent, return
zero counts with `usage_known=False`. Never represent unknown usage as known zero.

### 7. Structured terminal stream errors

`stream()` must return exactly one terminal event:

- success: `ModelStreamEvent(event_type="completed", response=...)`;
- failure: `ModelStreamEvent(event_type="error", error=...)`.

The error object requires these fields: non-empty string `code`,
non-empty safe string `message`, boolean `retryable`, and object `details`. The
event has no delta, tool-call, or response payload.
Do not expose `ModelGatewayError` to the iterator consumer. Initial transient
failures may retry before any visible event. Once a text/tool event has been
yielded, do not retry and do not emit `completed` after an error.

## Required implementation behavior

### A. Configuration and lifecycle

1. Validate `base_url` as HTTP(S), with host and without embedded credentials,
   query, or fragment.
2. Use `model`, `api_key`, optional `ca_bundle_path`, `timeout_seconds`, and
   `max_retries` exactly. `timeout_seconds` is one total gateway-operation
   deadline, including retries and the complete SSE lifetime.
3. Reuse one bounded connection pool for the gateway lifetime.
4. Provide deterministic close/context-manager behavior.
5. Keep the API key out of source, logs, details, exception messages, fixtures,
   and provider-body diagnostics.

### B. Transport security

1. Verify TLS certificates for HTTPS.
2. Disable environment proxy inheritance.
3. Reject cross-origin redirects; do not leak `Authorization` on redirects.
4. Bound response bodies, SSE totals, and individual SSE events while reading.
5. Normalize TLS failure as non-retryable `MODEL_TLS_ERROR` and URL/redirect/body
   policy failure as non-retryable `MODEL_SECURITY_ERROR`.
6. Reject CR or LF in `request_id` before constructing `X-Request-ID`.
7. Normalize timeouts separately from connection failures.

### C. Request serialization

1. Serialize every role, optional name, tool-result call ID, and assistant history
   call without data loss.
2. Convert each `ToolDefinition` to OpenAI function-tool format.
3. Forward only model, messages, temperature, max tokens, one-completion policy,
   stream settings, tools, and response format.
4. Never forward arbitrary `ModelRequest.metadata`.
5. Reject non-JSON/NaN schema values before network I/O.
6. Use OpenAI-style strict JSON Schema `response_format` for structured mode.

### D. Strict decoding

Use a single strict JSON decoder for completion bodies, SSE payloads, tool
arguments, and structured content. It must reject:

- duplicate object keys;
- `NaN`, `Infinity`, and `-Infinity`;
- invalid UTF-8;
- a non-object where an object is required.

Do not use a later dictionary conversion that has already discarded duplicate-key
evidence.

### E. Non-stream response parsing

1. Enforce the exact one-choice/model rules.
2. Parse ordinary text, including provider content-parts only when their shape is
   explicitly supported.
3. Parse one or multiple tool calls.
4. Require unique, non-empty tool call IDs.
5. Reject unoffered tools with `UNKNOWN_TOOL`.
6. Parse arguments as strict JSON objects and validate against the exact offered
   schema.
7. Parse and locally validate structured output.
8. Normalize finish reason and provider response ID.
9. Return explicit known/unknown usage.
10. Measure latency with a monotonic clock.
11. Fail with `EMPTY_MODEL_RESPONSE` when no usable payload exists.

### F. Streaming parsing

1. Parse arbitrary UTF-8 and SSE byte fragmentation.
2. Support comments, CRLF/LF, multi-line `data`, `[DONE]`, and usage-only frames.
3. Enforce the exact choice/model policy on every relevant frame.
4. Buffer tool call ID/name/argument fragments by non-negative tool index.
5. Emit a tool event only after the complete call passes name, JSON, duplicate-ID,
   and schema validation.
6. Emit contiguous sequence numbers beginning at zero.
7. Preserve text/tool calls/usage/provider ID in the final `ModelResponse`.
8. Emit exactly one completed or structured error terminal event.
9. After output visibility, normalize disconnect/timeout/protocol/schema failures
   to an error event without retry.

### G. Retries and normalized failures

Implement the matrix in `specs/qwen-vllm-service.md`, including HTTP 408, 429,
500/502/503/504, TLS, security failures, and all non-retryable parse/schema codes.

Rules:

- `max_retries` means retries after the first attempt;
- exponential backoff and jitter are bounded;
- a valid delta-seconds or HTTP-date `Retry-After` is honored and bounded;
- attempts and safe status are added to details;
- retry exhaustion is explicit;
- no model switch or hidden “repair JSON” generation occurs;
- no retry occurs after a visible stream event.

### H. Health and safe observability

`health()` calls `GET /models` with no generation, no retry, and a timeout no
greater than five seconds.

- `UP`: exact configured model present;
- `DEGRADED`: endpoint reachable but model-list shape/secondary checks invalid;
- `DOWN`: unavailable, unauthorized/security failure, or exact model absent.

Expose only safe details. Record request ID, exact model, provider response ID,
attempt count, normalized code, latency/TTFT, terminal state, and usage-known
status. Do not record prompts, RAG text, tool arguments, or provider bodies.

## Required offline tests

Tests use an injectable fake transport or local in-process fake. They require no
network, model download, Docker, or GPU.

### Contract/input tests

1. contract version is `2`;
2. assistant replay accepts valid calls and rejects wrong role/duplicate IDs;
3. tools and response schema are mutually exclusive;
4. unknown usage sentinel is valid;
5. error stream event requires exactly one structured payload;
6. method/stream mismatch returns non-retryable `MODEL_REQUEST_REJECTED` with
   attempts zero and causes zero transport calls;
7. CR/LF request ID is blocked before HTTP.

### Serialization success tests

1. every message role and optional field;
2. assistant tool-call replay JSON shape;
3. multiple tool definitions;
4. structured `response_format`;
5. metadata and secrets not forwarded;
6. exact configured model and one-completion request policy.

### Non-stream success tests

1. plain text;
2. one and multiple tool calls;
3. structured output;
4. provider response ID and latency;
5. usage present (`usage_known=True`);
6. usage absent (`usage_known=False`).

### Fail-closed parser tests

1. zero, multiple, non-object, and index-not-zero choices;
2. missing/mismatched response model;
3. malformed JSON and invalid UTF-8;
4. duplicate JSON keys and non-finite values;
5. unknown tool;
6. malformed/non-object tool arguments;
7. duplicate tool call IDs;
8. missing/extra/wrong-type tool fields;
9. structured output missing, malformed, or schema-invalid;
10. inconsistent/negative usage;
11. empty provider payload;
12. oversized body.

### Retry and security tests

1. timeout and HTTP 408 then success;
2. HTTP 429 then success for numeric and date `Retry-After`;
3. repeated HTTP 503 until exact retry exhaustion;
4. 401/403 without retry;
5. 400/404/413/422 without retry;
6. TLS/security failures without retry;
7. environment proxy disabled;
8. cross-origin redirect rejected without credential leak;
9. error string/details/repr do not contain a configured secret or provider body.

### Streaming tests

1. UTF-8/SSE/text fragmentation;
2. fragmented tool ID/name/arguments;
3. usage-only frame;
4. contiguous sequences and one completed event;
5. one structured error event for pre-output exhaustion;
6. mid-stream timeout/disconnect after a delta, with zero retry;
7. malformed tool/schema/model/choice after a delta, with error and no completed;
8. duplicate streamed call IDs;
9. stream/event size limits;
10. no executable tool event before complete validation.

For failure tests assert the exact code, retryability, attempt count, transport call
count, terminal event type, and absence of an executable invalid tool.

## Offline commands

Run from the repository root exactly as documented in
`docs/testing-qwen-vllm.md`. At minimum:

```bash
python3 tests/contract/test_model_rag_contracts.py -v
python3 -m unittest discover -s tests/model_gateway -p 'test*.py' -v
python3 run_tests.py
python3 -m compileall -q contracts src tests
```

Run the deployment asset checks and shell syntax checks described in the testing
document when deployment files change.

## Optional live verification

Live work is skipped by default. It requires explicit opt-in and an already
approved private endpoint. Use only sanitized prompts and aggregate evidence.

Required environment contract:

```text
RUN_VLLM_INTEGRATION=1
VLLM_BASE_URL=http(s)://.../v1
VLLM_MODEL=<exact-served-model-id>
# Leave VLLM_API_KEY unset locally; inject it from a secret manager in production.
VLLM_TIMEOUT_SECONDS=60
VLLM_MAX_RETRIES=1
```

Verify exact-model health, plain chat, assistant tool-call replay over two turns,
forced tool selection, structured output, SSE/TTFT, an approved long-context case,
and intended concurrency. A skip is `NOT_RUN`, not `PASS`.

## Definition of done

1. Every fixed contract-v2 decision is implemented and tested.
2. All offline commands pass in a clean environment.
3. Invalid tool and structured outputs are blocked in every negative fixture.
4. Retry calls/delays match the policy exactly.
5. Streaming always ends in one structured terminal event and never retries after
   visible output.
6. Transport security tests cover URL, proxy, redirect, TLS, header, and size
   boundaries.
7. No unit test needs network/GPU/Docker.
8. Deployment/live status and benchmark status are recorded as `PASS`, `FAIL`, or
   `NOT_RUN` without fabricated numbers.
9. A final report states files changed, commands, exact results, live evidence,
   compatibility assumptions, and unresolved risks.

## Required final report from the implementing model

Return:

1. files changed;
2. contract/serialization/security decisions implemented;
3. tests added and exact commands run;
4. pass/fail counts;
5. live tests and benchmark as `PASS`, `FAIL`, or `NOT_RUN`;
6. exact Qwen revision, vLLM image/version, parser/template settings if live work
   occurred;
7. limitations or follow-up work;
8. no performance claim unless the evidence template is fully populated.
