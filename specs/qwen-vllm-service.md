# Qwen / vLLM Model Gateway Specification

## 1. Purpose, status, and proof boundary

This module gives the Agent Runtime one provider-neutral interface to a privately
served Qwen Instruct model. The first backend is vLLM's OpenAI-compatible API.
The public contract version is `MODEL_GATEWAY_CONTRACT_VERSION = "2"`.

The gateway is responsible for:

1. serializing chat history, assistant tool-call history, tool definitions, and
   structured-output schemas;
2. normalizing text, tool calls, structured JSON, token usage, finish reasons,
   health, and provider errors;
3. validating every model-selected tool and every structured result locally;
4. providing deterministic non-streaming and streaming semantics;
5. applying bounded retries before output becomes visible;
6. keeping provider objects, credentials, proxies, redirects, and malformed JSON
   outside the Agent Runtime contract.

It does not execute tools, choose medical algorithms, store CT/Mask data, build
the LangGraph state machine, silently repair invalid JSON, or select a fallback
cloud model.

There are three separate levels of proof:

| Evidence | What it proves | What it does not prove |
|---|---|---|
| Mock tests | deterministic Agent integration against repository contracts | vLLM protocol compatibility |
| Fake-transport adapter tests | request mapping, parsing, retries, security and failure policy without a network/GPU | that a Qwen checkpoint starts or generates correctly |
| Gated live tests and benchmark evidence | compatibility and measured behavior for one exact model/vLLM/hardware profile | compatibility with another revision or deployment profile |

Passing offline tests must never be described as a completed private deployment.
The module may be tagged as a final release only after the release policy decides
whether live evidence is required; otherwise use an ordinary commit or prerelease
and record `LIVE_VALIDATION=NOT_RUN`.

## 2. Files and ownership

Public model contracts:

- `src/puncture_agent/model_gateway/models.py`
- `src/puncture_agent/model_gateway/__init__.py`

Adapter and transport:

- `src/puncture_agent/model_gateway/client.py`
- `src/puncture_agent/model_gateway/http_transport.py`

Development double:

- `src/puncture_agent/model_gateway/mock_qwen.py`
- `mocks/model_responses/responses.json`

Verification and implementation handoff:

- `tests/contract/test_model_rag_contracts.py`
- `tests/model_gateway/test_mock_qwen.py`
- `tests/model_gateway/test_vllm_gateway.py`
- `docs/testing-qwen-vllm.md`
- `tasks/task-01-qwen-vllm.md`

Contract-v2 fields and policies must not be changed implicitly. A later incompatible
change requires a new contract version, migration notes, and synchronized Agent
Runtime tests.

## 3. Runtime topology

```text
LangGraph node
    |
    | ModelRequest
    v
ModelGateway
    |-- MockQwenGateway          deterministic tests
    `-- VllmModelGateway        production protocol adapter
              |
              | bounded OpenAI-compatible HTTP/SSE
              v
       private vLLM endpoint
              |
              v
       pinned Qwen checkpoint
```

Only the gateway knows provider-specific fields. Graph nodes consume
`ModelResponse` and `ModelStreamEvent`, never raw OpenAI/vLLM dictionaries.

## 4. Contract-v2 input

### 4.1 ModelRequest

| Field | Type | Required | Rule |
|---|---|---:|---|
| `request_id` | non-empty string | yes | safe correlation key; CR/LF is forbidden at the HTTP boundary |
| `messages` | tuple of `ChatMessage` | yes | at least one message |
| `tools` | tuple of `ToolDefinition` | no | unique function names with object input schemas |
| `response_schema` | object JSON Schema | no | mutually exclusive with `tools` |
| `temperature` | float | yes | `0.0 <= value <= 2.0` |
| `max_tokens` | positive integer | yes | client and server admission limits still apply |
| `stream` | boolean | yes | must agree with the method called |
| `metadata` | JSON object | no | internal-only; never forwarded wholesale |

`tools` and `response_schema` are mutually exclusive. This is rejected when the
`ModelRequest` is constructed, before network I/O. The gateway does not guess
whether the caller intended tool selection or a final structured answer.

Method semantics are strict:

- `generate(request)` requires `request.stream is False` and sends `stream=false`;
- `stream(request)` requires `request.stream is True` and sends `stream=true` plus
  usage streaming options supported by the selected vLLM version;
- a mismatch fails locally with non-retryable
  `ModelGatewayError(code="MODEL_REQUEST_REJECTED")`,
  `details.attempts=0`, and zero provider calls. `generate()` raises that error;
  iterating `stream()` yields it as the single structured terminal `error` event.

### 4.2 ChatMessage and assistant tool-call replay

Roles are `system`, `user`, `assistant`, and `tool`.

```json
{"role":"user","content":"Inspect the current acceptance procedure"}
```

A tool result requires the provider call ID:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"status\":\"approved\"}"
}
```

An assistant message may replay one or more prior `ToolCall` values. They must be
serialized back to OpenAI function-call shape so the next turn remains valid:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "search_knowledge",
        "arguments": "{\"query\":\"acceptance procedure\"}"
      }
    }
  ]
}
```

Only assistant messages may contain `tool_calls`; assistant call IDs must be
unique within that message. Only tool messages may rely on `tool_call_id`.
Arguments in history are repository objects and are encoded as strict JSON strings.

### 4.3 ToolDefinition provider mapping

Repository definition:

```json
{
  "name": "search_knowledge",
  "description": "Search versioned internal project knowledge.",
  "input_schema": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": false
  }
}
```

Provider request:

```json
{
  "type": "function",
  "function": {
    "name": "search_knowledge",
    "description": "Search versioned internal project knowledge.",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

No internal metadata, API key, trace payload, or arbitrary sampling option may be
copied from `ModelRequest.metadata` into the provider body.

## 5. Provider request policy

Every request uses the exact `VllmGatewayConfig.model` served ID and sends `n=1`.
Accepting multiple alternatives is not part of this contract.

Required fields are:

```json
{
  "model": "<exact-served-model-id>",
  "messages": [],
  "temperature": 0.0,
  "max_tokens": 1024,
  "n": 1,
  "stream": false
}
```

When `response_schema` exists, request OpenAI-style JSON Schema output:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "agent_response",
      "strict": true,
      "schema": {"type":"object"}
    }
  }
}
```

Provider-side constrained decoding is a defense in depth control. The gateway
must parse and validate the final object locally before returning it.

## 6. Non-streaming response contract

`ModelResponse` contains:

- original `request_id`;
- exact configured model ID;
- normalized finish reason;
- optional text;
- zero or more validated `ToolCall` values;
- optional validated structured output;
- `TokenUsage`, including whether the usage is known;
- monotonic gateway latency;
- safe provider response ID when available.

Normalized finish reasons are:

```text
stop
tool_calls
length
content_filter
error
```

### 6.1 Exact choice and model-ID rules

A non-streaming success must have exactly one `choices` entry, and its index must
be the integer `0`. Empty, multiple, non-object, missing-index, or alternative-
index choices are `MODEL_PROTOCOL_ERROR`.

The response `model` must be a non-empty string equal to
`VllmGatewayConfig.model`. A missing or different model ID is
`MODEL_PROTOCOL_ERROR`; aliases are not silently accepted. `/models` health also
requires an exact configured-ID match. Changing the served alias is a deployment
and configuration change, not a response-parser convenience.

### 6.2 Payload rules

At least one usable payload must exist. If no text, tool call, or structured object
is present, fail with `EMPTY_MODEL_RESPONSE`.

Tool calls are accepted only when all of these are true:

1. the call object and function shape are valid;
2. the call ID is non-empty and unique in the response;
3. the function name exactly matches one offered tool;
4. arguments are strict JSON and decode to an object;
5. arguments satisfy the offered tool schema.

Unknown tools and malformed arguments never become executable calls.

The minimum local JSON Schema subset is implemented by
`validate_json_schema_subset` and covers types, `properties`, `required`,
`additionalProperties: false`, array `items`, and `enum`. A complete validator may
be substituted only if it preserves these observable rules.

### 6.3 Token usage sentinel

When the provider returns valid usage, preserve the exact non-negative counts and
set `usage_known=True`. Counts must satisfy:

```text
total_tokens == prompt_tokens + completion_tokens
```

When usage is absent, return:

```python
TokenUsage(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    usage_known=False,
)
```

Zero with `usage_known=False` means “not reported,” not “the request consumed no
tokens.” Telemetry, cost, and benchmark code must branch on the sentinel.

## 7. Streaming contract

`stream()` returns contiguous `ModelStreamEvent` objects beginning at sequence 0.

| Event | Exact payload | Rule |
|---|---|---|
| `text_delta` | non-null `delta` | append in sequence order |
| `tool_call` | one complete validated `ToolCall` | emit only after all fragments form valid JSON |
| `completed` | final `ModelResponse` | exactly one terminal success |
| `error` | structured `error` object | exactly one terminal failure |

The error payload is safe and provider-neutral. It requires non-empty string
`code`, non-empty safe string `message`, boolean `retryable`, and object `details`:

```json
{
  "code": "MODEL_UNAVAILABLE",
  "message": "model stream disconnected",
  "retryable": false,
  "details": {
    "attempts": 1,
    "provider_status": 503,
    "output_visible": true,
    "upstream_retryable": true
  }
}
```

`message` must not contain credentials, prompt text, RAG content, tool arguments,
provider bodies, or raw exception representations. `details` contains allowlisted
scalar diagnostics only.

Streaming invariants:

1. exactly one terminal event is produced: `completed` or `error`;
2. no event follows a terminal event;
3. each event carries only the payload for its own event type; an error event has
   no `delta`, `tool_call`, or `response`;
4. `VllmModelGateway.stream()` does not expose `ModelGatewayError` to the stream
   consumer; terminal failures are normalized to `error`;
5. transient establishment failures may be retried before any visible event;
6. after a `text_delta` or `tool_call` is visible, the gateway never retries the
   provider request because replay would duplicate output or side effects;
7. incomplete tool fragments remain buffered and never leave as executable calls;
8. usage-only provider frames may contain zero choices; every other data frame has
   exactly one choice at index 0;
9. the completed `ModelResponse` is authoritative for LangGraph state and tracing.

The provider may fragment UTF-8, SSE lines, tool names, IDs, and argument JSON at
arbitrary byte boundaries. Parser behavior must not depend on HTTP chunk boundaries.

## 8. Strict protocol and security requirements

### 8.1 JSON and size limits

All provider JSON, SSE JSON, tool arguments, structured output, and schema copies
must reject:

- invalid UTF-8;
- duplicate object keys;
- `NaN`, `Infinity`, and `-Infinity`;
- unexpected top-level types;
- inconsistent token counts;
- bodies/events/streams above explicit bounded limits.

Response limits must be enforced while reading, not after an unbounded `read()`.

### 8.2 HTTP transport

The production transport must:

- reuse a bounded connection pool for the gateway lifetime;
- validate `base_url` as HTTP(S), with host present and no embedded credentials,
  query, or fragment;
- verify TLS certificates for HTTPS;
- disable ambient environment proxies for confidential prompts;
- reject cross-origin redirects and prevent authorization leakage;
- use bounded connect/read/write/pool timeouts and normalize them;
- close response and pooled-client resources deterministically;
- validate header values such as `request_id` against CR/LF injection;
- never include the API key or provider response body in errors or traces.

Loopback/private HTTP may be used only inside the documented trusted development
boundary. Remote production traffic requires the deployment's TLS/mTLS and network
policy controls.

### 8.3 Timeout meaning

`timeout_seconds` is the total gateway-operation deadline. For non-streaming it
includes transport attempts, retry delays, bounded body reading and parsing. For
streaming it includes establishment, retry delays and the complete SSE lifetime;
the remaining deadline is passed to each transport attempt/read. The Agent Runtime
may impose a longer outer run deadline for the full graph. The short health probe
uses `min(timeout_seconds, 5 seconds)` and no retry. The maximum retry count means
retries after the first attempt.

## 9. Error and retry matrix

| Failure | Code | Retry before visible output? |
|---|---|---:|
| connect/read timeout, HTTP 408 | `MODEL_TIMEOUT` | yes |
| DNS/connect/reset/disconnect | `MODEL_UNAVAILABLE` | yes |
| HTTP 429 | `MODEL_RATE_LIMITED` | yes; honor bounded `Retry-After` |
| HTTP 500/502/503/504 | `MODEL_UNAVAILABLE` | yes |
| TLS/certificate failure | `MODEL_TLS_ERROR` | no |
| invalid URL, proxy/redirect/security policy | `MODEL_SECURITY_ERROR` | no |
| HTTP 401/403 | `MODEL_PERMISSION_DENIED` | no |
| other HTTP 4xx, including 400/404/413/422 | `MODEL_REQUEST_REJECTED` | no |
| malformed provider JSON/SSE/choice/model/usage | `MODEL_PROTOCOL_ERROR` | no |
| unknown tool name | `UNKNOWN_TOOL` | no |
| malformed tool argument JSON | `TOOL_ARGUMENT_PARSE_ERROR` | no |
| tool schema mismatch | `TOOL_ARGUMENT_SCHEMA_ERROR` | no |
| structured output missing/invalid/schema mismatch | `STRUCTURED_OUTPUT_SCHEMA_ERROR` | no |
| no usable payload | `EMPTY_MODEL_RESPONSE` | no |

Use bounded exponential backoff with jitter and a maximum delay. `Retry-After` may
be delta seconds or an HTTP date and is clamped to the same maximum. Error details
include attempts, safe provider status when known, retry exhaustion, and whether
output had become visible. The gateway never switches model or asks the model to
repair invalid JSON as an internal retry.

## 10. Health and observability

`health()` performs `GET /models`, not a generation request:

- `UP`: endpoint is reachable, response shape is valid, and the exact configured
  model ID is listed;
- `DEGRADED`: endpoint is reachable but the model-list response is malformed or a
  nonessential check cannot be completed;
- `DOWN`: endpoint is unavailable, fails security/auth checks, or the configured
  model is not served.

Record, without sensitive content:

- request/trace ID;
- exact configured model and deployment revision metadata;
- provider response ID;
- offered and selected tool names, not unrestricted arguments;
- retry count and normalized error code;
- TTFT, total latency, and terminal status;
- token counts plus `usage_known`;
- schema-validation result.

## 11. Qwen/vLLM compatibility baseline

Deployment behavior is version-sensitive. The initial documentation baseline was
reviewed against these immutable upstream commits on 2026-07-10:

- [vLLM `08dfd68610d2e05a0d8ddc99c23488da6163df3f`](https://github.com/vllm-project/vllm/tree/08dfd68610d2e05a0d8ddc99c23488da6163df3f),
  including its [structured-output guide](https://github.com/vllm-project/vllm/blob/08dfd68610d2e05a0d8ddc99c23488da6163df3f/docs/features/structured_outputs.md);
- [Qwen3 `7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e`](https://github.com/QwenLM/Qwen3/tree/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e),
  including its [vLLM deployment guide](https://github.com/QwenLM/Qwen3/blob/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e/docs/source/deployment/vllm.md)
  and [function-calling guide](https://github.com/QwenLM/Qwen3/blob/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e/docs/source/framework/function_call.md).

At those revisions, Qwen's vLLM material documents automatic tool choice with a
model-compatible parser, and vLLM documents OpenAI-style structured outputs. This
justifies the deployment template shape only. Every actual image/checkpoint pair
must re-verify server flags, tool parser, chat template, reasoning parser, and
`response_format` behavior. See `docs/qwen-deployment-runbook.md`.

## 12. Verification and acceptance

The exact commands, matrix, live environment contract, and evidence template are
in `docs/testing-qwen-vllm.md`.

The adapter is acceptable only when:

1. contract-v2 and mock regressions pass;
2. adapter tests pass without network/GPU access;
3. method mismatch, multiple choices, model mismatch, malformed JSON, duplicate
   keys/call IDs, and invalid schemas fail closed;
4. retries match the matrix and never occur after visible streaming output;
5. every stream has contiguous sequences and exactly one structured terminal event;
6. no secret or confidential payload appears in fixtures, exceptions, or evidence;
7. live status is reported honestly as `PASS`, `FAIL`, or `NOT_RUN` for the exact
   deployment profile.
