# Task 01 — Implement the Real Qwen / vLLM Model Gateway

## Copyable assignment

You are implementing the production model adapter for an existing contract-first
Agent scaffold. Replace only the explicit `VllmModelGateway` stub. Preserve the
deterministic mock and every public input/output contract so other modules continue
to integrate without changes.

No medical imaging algorithm is part of this task.

## Read these files first

Read completely, in this order:

1. `specs/qwen-vllm-service.md`
2. `src/puncture_agent/model_gateway/models.py`
3. `src/puncture_agent/model_gateway/client.py`
4. `src/puncture_agent/model_gateway/mock_qwen.py`
5. `tests/contract/test_model_rag_contracts.py`
6. `tests/model_gateway/test_mock_qwen.py`

Do not infer a different interface from a framework tutorial. The repository
contracts take precedence.

## Goal

Implement a Qwen model gateway that talks to a private vLLM OpenAI-compatible
endpoint and returns only normalized repository objects:

```python
health() -> GatewayHealth
generate(ModelRequest) -> ModelResponse
stream(ModelRequest) -> Iterator[ModelStreamEvent]
```

## Allowed implementation area

Primary area:

```text
src/puncture_agent/model_gateway/
tests/model_gateway/
```

If a dependency or configuration file outside these paths must change, explain the
reason before changing it. Do not modify RAG, graph, medical tools, or their tests.

Do not delete or weaken mock/contract tests. Do not replace tests with assertions
that merely check “no exception.”

## Contracts that must remain unchanged

- `ChatMessage`
- `ToolDefinition`
- `ModelRequest`
- `ToolCall`
- `TokenUsage`
- `ModelResponse`
- `ModelStreamEvent`
- `GatewayHealth`
- `VllmGatewayConfig`
- `ModelGatewayError` observable fields: `code`, `retryable`, `details`
- the three `ModelGateway` methods

You may add private helpers and new internal files. If you believe a public contract
is insufficient, stop and report the exact missing field and caller impact instead
of silently changing it.

## Required implementation behavior

### 1. Configuration and client lifecycle

- Use `VllmGatewayConfig.base_url`, `model`, `api_key`, `timeout_seconds`, and
  `max_retries`.
- Reuse a client/connection pool rather than constructing one per request.
- Never log the API key.
- Validate that the configured model is available during health probing.
- Dependency choice may be the official OpenAI client, `httpx`, or a small standard
  library transport, but provider objects must not escape this module.

### 2. Request mapping

- Serialize system, user, assistant, and tool messages.
- Preserve `name` and `tool_call_id` when present.
- Convert `ToolDefinition` into OpenAI function-tool format.
- Forward temperature and maximum token count.
- Request provider-supported constrained JSON when `response_schema` is supplied.
- Do not forward arbitrary `ModelRequest.metadata`; use an explicit allowlist if a
  provider option is genuinely needed.

### 3. Non-streaming parsing

- Parse plain assistant text.
- Parse one or multiple tool calls.
- Parse tool arguments from JSON strings to objects.
- Reject an unknown tool name with `UNKNOWN_TOOL`.
- Validate arguments against the corresponding request schema.
- Parse and locally validate structured output.
- Normalize finish reason and usage.
- Measure latency using a monotonic clock.
- Raise `EMPTY_MODEL_RESPONSE` when no usable payload exists.

### 4. Streaming parsing

- Parse SSE/provider chunks without exposing them directly.
- Emit contiguous `sequence` numbers starting at zero.
- Buffer fragmented tool names and argument JSON.
- Emit a `tool_call` event only after a complete valid call is available.
- Emit exactly one `completed` event on success.
- On terminal failure, emit/raise consistently as documented; never emit a success
  event afterward.
- The completed event must carry the normalized final `ModelResponse`.

### 5. Error and retry policy

Implement the table in `specs/qwen-vllm-service.md`. At minimum:

```text
MODEL_TIMEOUT                 retryable
MODEL_UNAVAILABLE             retryable
MODEL_RATE_LIMITED            retryable
MODEL_PERMISSION_DENIED       non-retryable
MODEL_REQUEST_REJECTED        non-retryable
MODEL_PROTOCOL_ERROR          non-retryable
TOOL_ARGUMENT_PARSE_ERROR     non-retryable
TOOL_ARGUMENT_SCHEMA_ERROR    non-retryable
STRUCTURED_OUTPUT_SCHEMA_ERROR non-retryable
EMPTY_MODEL_RESPONSE          non-retryable
```

Use bounded exponential backoff with jitter. Honor `Retry-After` if available.
`max_retries` means retries after the first attempt. Include attempt count and safe
provider status in error details.

Do not retry schema errors by silently asking the model to “fix JSON” inside the
gateway. Higher-level Agent policy decides whether another generation is allowed.

### 6. Health and observability

- Return `UP`, `DEGRADED`, or `DOWN` with configured model and provider.
- Use a short health timeout.
- Record model, provider response ID, request ID, retry count, latency, TTFT when
  streaming, usage, and normalized error.
- Redact secrets and configurable sensitive prompt/tool fields.

## Suggested internal design

This is a suggestion, not a required filename layout:

```text
model_gateway/
├── models.py                 # fixed
├── client.py                 # interface + production class
├── mock_qwen.py              # fixed deterministic mock
├── serialization.py          # internal request mapper
├── parsing.py                # internal response/SSE parser
├── transport.py              # replaceable HTTP abstraction
└── retry.py                  # bounded policy
```

Keep the HTTP transport injectable so unit tests can return scripted responses
without network access.

## Minimum unit tests to add

Use a fake transport or local fake server; unit tests must run without vLLM/GPU.

### Success tests

1. all message roles serialize correctly;
2. tool definitions serialize to function-tool format;
3. plain response maps to `ModelResponse`;
4. single and multiple tool calls parse correctly;
5. structured output passes local schema validation;
6. usage and provider response ID are preserved;
7. fragmented stream text reconstructs correctly;
8. fragmented tool argument JSON reconstructs before emission;
9. stream sequence is contiguous with one terminal event;
10. health reports the configured model.

### Failure tests

1. unknown tool;
2. malformed tool argument JSON;
3. missing required tool property;
4. unexpected property when additional properties are forbidden;
5. structured output schema mismatch;
6. empty provider response;
7. timeout then success;
8. HTTP 429 then success, including `Retry-After` handling;
9. repeated HTTP 503 and retry exhaustion;
10. HTTP 401/403 without retry;
11. disconnect during SSE;
12. missing configured model in health response;
13. error/log representation does not contain a configured secret.

For every failure test assert exact normalized code, `retryable`, number of attempts,
and absence of an executable tool call.

## Test commands

Run from the repository root:

```bash
python3 tests/contract/test_model_rag_contracts.py -v
python3 -m unittest discover -s tests/model_gateway -p 'test*.py' -v
python3 -m unittest discover -s tests -p 'test*.py' -v
```

If the full discovery command does not include `tests/contract` because the test
directory is not yet a package, the direct contract command is mandatory.

Optional live tests must be skipped by default and require:

```text
RUN_VLLM_INTEGRATION=1
VLLM_BASE_URL=...
VLLM_MODEL=...
```

Never embed a real token in a test.

## Manual live verification

Against the private service, verify:

1. `/health` or equivalent model-list probe;
2. plain Chinese and English chat;
3. a forced tool call with valid JSON arguments;
4. a structured intent-classification response;
5. SSE streaming and TTFT collection;
6. a request near the configured context limit;
7. concurrency at 1, 4, and the intended deployment load.

Create a benchmark note containing:

```text
model name and exact revision
vLLM version
GPU type and count
tensor parallel size
quantization
maximum context
prompt/output token distribution
concurrency
TTFT, TPOT, tokens/s
P50/P95 end-to-end latency
peak GPU memory
tool-call and structured-output schema-valid rate
```

## Definition of done

- Existing tests pass unchanged.
- New adapter unit tests cover all required success/failure behavior.
- No unit test requires a GPU or network.
- Live tests are explicitly gated.
- Unknown/malformed tool calls never reach the caller as executable calls.
- Structured output is locally validated.
- Retries are bounded and exactly match policy.
- Streaming has deterministic event framing.
- No credentials or confidential prompts appear in source/test fixtures.
- A short implementation note explains dependency choice and how to run live tests.

## Required final report from the implementing model

Return:

1. files changed;
2. protocol/serialization decisions;
3. tests added and their commands;
4. exact test results;
5. live tests run or explicitly not run;
6. unresolved compatibility assumptions about the chosen Qwen/vLLM version;
7. benchmark results only if actually measured.
