# Qwen / vLLM Model Gateway Specification

## 1. Purpose and scope

This module provides one provider-neutral model interface to the Agent Runtime.
The first production backend is a privately deployed Qwen Instruct model served
through vLLM's OpenAI-compatible API.

The gateway is responsible for:

1. converting internal chat, tool, and response-schema contracts to provider
   requests;
2. parsing ordinary text, tool calls, structured JSON, token usage, and finish
   reasons into stable internal objects;
3. normalizing transport, timeout, overload, output-parse, and schema failures;
4. exposing non-streaming and streaming methods with the same terminal response;
5. recording request IDs, provider response IDs, latency, model name, and usage;
6. preventing malformed model output from reaching a tool executor.

The gateway is not responsible for:

- deciding which medical algorithm is correct;
- executing MCP tools;
- storing CT or Mask data;
- building the LangGraph state machine;
- silently repairing unsafe or semantically invalid tool arguments;
- selecting a fallback cloud model.

The current code intentionally contains a deterministic `MockQwenGateway` and an
unimplemented `VllmModelGateway`. The mock is for integration development only;
it must not be presented as a real Qwen deployment.

## 2. Files and ownership

Public contracts:

- `src/puncture_agent/model_gateway/models.py`
- `src/puncture_agent/model_gateway/client.py`

Development double:

- `src/puncture_agent/model_gateway/mock_qwen.py`
- `mocks/model_responses/responses.json`

Tests:

- `tests/contract/test_model_rag_contracts.py`
- `tests/model_gateway/test_mock_qwen.py`

Implementation task:

- `tasks/task-01-qwen-vllm.md`

Do not rename public classes or change their required fields without coordinating
a contract version change across the Agent Runtime and tests.

## 3. Runtime topology

```text
LangGraph node
    |
    | ModelRequest
    v
ModelGateway interface
    |-- MockQwenGateway        # local deterministic tests
    `-- VllmModelGateway       # production adapter to implement
              |
              | OpenAI-compatible HTTP/SSE
              v
        private vLLM server
              |
              v
        Qwen Instruct model
```

Only the model gateway knows provider-specific request and response fields. Agent
nodes consume `ModelResponse` and `ModelStreamEvent`, never raw OpenAI dictionaries.

## 4. Fixed input contract

The exact Python definition is `ModelRequest` in `models.py`.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `request_id` | string | yes | Caller-generated idempotency and trace key |
| `messages` | tuple of `ChatMessage` | yes | System/user/assistant/tool conversation |
| `tools` | tuple of `ToolDefinition` | no | Model-visible tools and input JSON Schemas |
| `response_schema` | JSON object schema | no | Required structured result schema |
| `temperature` | float 0..2 | yes | Sampling temperature |
| `max_tokens` | positive integer | yes | Maximum generated tokens |
| `stream` | boolean | yes | Caller streaming preference |
| `metadata` | object | no | Internal trace/mock controls; never blindly forwarded |

`ChatMessage` mapping:

```json
{
  "role": "user",
  "content": "Inspect case case-001"
}
```

Tool messages additionally require `tool_call_id`. A production serializer must
preserve the tool-call ID returned by the previous assistant message.

`ToolDefinition` example:

```json
{
  "name": "search_knowledge",
  "description": "Search versioned internal project knowledge.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "integer"}
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

The provider request must serialize this as an OpenAI-style function tool:

```json
{
  "type": "function",
  "function": {
    "name": "search_knowledge",
    "description": "Search versioned internal project knowledge.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}
```

## 5. Fixed output contract

`ModelResponse` always contains:

- the original `request_id`;
- the actual model identifier;
- a normalized finish reason;
- zero or one text payload;
- zero or more parsed `ToolCall` values;
- zero or one validated structured object;
- normalized token usage;
- measured gateway latency;
- provider response ID when available.

Permitted normalized finish reasons:

```text
stop
tool_calls
length
content_filter
error
```

No successful response may be empty. If the provider returns no text, no tool
call, and no structured object, raise `ModelGatewayError` with code
`EMPTY_MODEL_RESPONSE`.

### Tool call output

```json
{
  "call_id": "call_abc123",
  "name": "search_knowledge",
  "arguments": {
    "query": "current path-planning safety radius"
  }
}
```

The `arguments` field must be parsed JSON and validated against the exact tool
schema supplied in the request. Unknown tool names and invalid arguments are not
passed to MCP.

### Structured output

When `response_schema` is supplied, the adapter must request JSON-constrained
generation using the capability supported by the selected vLLM/Qwen version. The
parsed object must then be validated locally. Provider-side constrained decoding
does not replace local validation.

The scaffold helper `validate_json_schema_subset` covers the project's minimum
schema subset. A production implementation may use a complete JSON Schema
validator but must preserve at least these rules:

- object/array/string/integer/number/boolean/null types;
- `properties` and `required`;
- `additionalProperties: false`;
- array `items`;
- `enum`.

## 6. Streaming contract

`stream()` returns ordered `ModelStreamEvent` objects.

| Event | Required payload | Rule |
|---|---|---|
| `text_delta` | `delta` | Append in sequence order |
| `tool_call` | complete `ToolCall` | Emit only after name and argument JSON are complete |
| `completed` | final `ModelResponse` | Exactly one terminal success event |
| `error` | implementation-defined error text | Terminal; no completed event follows |

Sequence numbers start at zero and are strictly contiguous. Provider fragments of
a tool-call name or argument string must be buffered; incomplete JSON must never be
emitted as a `ToolCall`.

The final `completed.response` is the authoritative result stored in LangGraph
state and trace storage.

## 7. Production implementation procedure

### Step 1: configuration

Construct `VllmModelGateway` from `VllmGatewayConfig`:

```text
base_url          e.g. http://qwen-vllm:8000/v1
model             served model name, not a display alias
api_key           optional internal bearer token
timeout_seconds   full request timeout
max_retries       retries after the first attempt
```

Secrets must come from environment/secrets management and must not be included in
trace payloads, exceptions, fixtures, or source control.

### Step 2: request serialization

1. Map chat messages without dropping tool-call IDs.
2. Map each `ToolDefinition` to an OpenAI function tool.
3. Add temperature and token limit.
4. Add provider-supported structured-output configuration when a response schema
   exists.
5. Forward only an allowlist of provider options. Do not send arbitrary internal
   `metadata` to vLLM.
6. Send the internal `request_id` in an HTTP header or trace context when possible.

### Step 3: response parsing

1. Verify the HTTP and provider response shape.
2. Extract actual model and provider response ID.
3. Normalize finish reason.
4. Parse text without converting missing content into the string `"None"`.
5. Parse every tool argument as JSON.
6. Reject unregistered tool names.
7. Validate arguments against the matching request tool schema.
8. Parse and validate structured output when requested.
9. Normalize usage; if provider usage is absent, mark it in trace details rather
   than inventing exact numbers.
10. Measure elapsed monotonic time in the adapter.

### Step 4: retries

Retry only errors that are both transient and safe before tool execution:

| Failure | Retry? | Normalized code |
|---|---:|---|
| connect/read timeout | yes | `MODEL_TIMEOUT` |
| connection reset | yes | `MODEL_UNAVAILABLE` |
| HTTP 429 | yes, honor Retry-After | `MODEL_RATE_LIMITED` |
| HTTP 500/502/503/504 | yes | `MODEL_UNAVAILABLE` |
| HTTP 400/401/403/404 | no | `MODEL_REQUEST_REJECTED` or `MODEL_PERMISSION_DENIED` |
| malformed provider JSON | no | `MODEL_PROTOCOL_ERROR` |
| malformed tool arguments | no | `TOOL_ARGUMENT_PARSE_ERROR` |
| tool schema mismatch | no | `TOOL_ARGUMENT_SCHEMA_ERROR` |
| structured output mismatch | no | `STRUCTURED_OUTPUT_SCHEMA_ERROR` |

Use bounded exponential backoff with jitter. Include attempt count in error details
and tracing. Do not retry indefinitely and do not silently switch models.

### Step 5: health probe

`health()` must distinguish:

- `UP`: endpoint reachable and configured model is served;
- `DEGRADED`: endpoint reachable but model/metrics checks are partially failing;
- `DOWN`: endpoint unavailable or configured model absent.

The health probe must have a short, separate timeout and must not perform a large
generation request.

### Step 6: telemetry

Record at minimum:

- request ID and trace ID;
- model and prompt-template version;
- tool names offered and selected, but redact sensitive arguments;
- TTFT for streaming requests;
- total latency;
- prompt/completion tokens;
- retry count and normalized error code;
- output-schema validation result.

Never log complete internal documents, CT paths, credentials, or unrestricted tool
arguments by default.

## 8. Qwen/vLLM deployment expectations

The infrastructure implementation should document the exact model revision,
quantization, GPU type/count, tensor-parallel size, maximum context length, and
vLLM version. Relevant runtime topics include:

- continuous batching;
- paged KV cache;
- prefix caching;
- chunked prefill;
- tensor parallelism;
- AWQ/GPTQ when required by available memory;
- tool-call parser/chat template compatibility;
- structured-output compatibility;
- maximum context and concurrency trade-offs.

Do not claim a latency or throughput number without recording the hardware,
prompt/output lengths, concurrency, quantization, and measurement method.

## 9. Correctness verification

### 9.1 Contract tests, no network or GPU

Run:

```bash
python3 tests/contract/test_model_rag_contracts.py -v
python3 -m unittest discover -s tests/model_gateway -p 'test*.py' -v
```

These tests prove:

- invalid request objects are rejected;
- structured output is locally validated;
- unknown tools and invalid tool parameters fail closed;
- mock streaming events are ordered and terminal;
- retryable errors can be injected into graph tests;
- the production adapter remains visibly unimplemented until task 01 is done.

### 9.2 Production adapter unit tests

The implementing model must add tests using a fake HTTP transport or local fake
server. Unit tests must not require a running GPU service. Cover at least:

1. correct message and tool serialization;
2. plain text response parsing;
3. one and multiple tool-call parsing;
4. fragmented streaming tool arguments;
5. structured JSON success and schema failure;
6. absent usage fields;
7. every HTTP/error row in the retry table;
8. retry exhaustion and attempt count;
9. health UP/DEGRADED/DOWN;
10. credential and sensitive-field redaction.

### 9.3 Optional live integration test

Live tests must be skipped unless an explicit variable such as
`RUN_VLLM_INTEGRATION=1` is set. Test against a private vLLM endpoint with:

- one plain chat request;
- one forced tool selection request;
- one structured intent-classification request;
- one streaming request;
- one long-context request near the supported project limit.

Store only aggregate measurements and sanitized outputs.

### 9.4 Failure tests

Inject and verify:

- timeout before response headers;
- disconnect in the middle of SSE;
- HTTP 429 followed by success;
- repeated HTTP 503 until retry exhaustion;
- invalid JSON tool arguments;
- valid JSON with missing required tool fields;
- unknown hallucinated tool name;
- structured output with an unexpected property;
- empty provider response;
- configured model missing from health response.

The assertion is not merely “an exception occurred.” Assert normalized code,
retryability, attempt count, absence of a tool call, and absence of leaked secrets.

## 10. Acceptance gates

Task 01 is complete only when all of the following are true:

1. all existing contract and mock tests pass unchanged;
2. all new production-adapter unit tests pass without network/GPU access;
3. malformed or unknown tool calls are blocked in 100% of negative tests;
4. structured-output schema violations are blocked in 100% of negative tests;
5. streaming emits contiguous sequence numbers and one terminal event;
6. retries exactly follow the documented retry matrix and are bounded;
7. the live integration suite is environment-gated and has a documented command;
8. a benchmark report states hardware, model revision, quantization, prompt/output
   sizes, concurrency, TTFT, TPOT, throughput, P50/P95 latency, and peak GPU memory;
9. no source file or test fixture contains a real API key or confidential document;
10. the public contracts used by Agent Runtime have not changed.

For a 100-150 case internal golden set, recommended release targets are at least
98% parseable structured responses and at least 98% schema-valid tool calls at
temperature zero. Any lower result must route to retry/manual review rather than
tool execution and must be reported as a measured limitation.

## 11. Mock usage examples

Deterministic tool call:

```python
request = ModelRequest(
    request_id="demo-1",
    messages=(ChatMessage(role="user", content="search the rules"),),
    tools=(search_tool,),
    metadata={
        "mock_tool_call": {
            "name": "search_knowledge",
            "arguments": {"query": "safety envelope"},
        }
    },
)
response = MockQwenGateway().generate(request)
```

Failure injection:

```python
request = ModelRequest(
    request_id="demo-timeout",
    messages=(ChatMessage(role="user", content="inspect"),),
    metadata={"force_error": {"code": "MODEL_TIMEOUT", "retryable": True}},
)
```

These controls are intentionally explicit so graph tests do not depend on prompt
wording or model randomness.
