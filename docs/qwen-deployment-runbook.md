# Qwen / vLLM Private Deployment Runbook

## 1. Outcome and boundary

This runbook turns the repository's `VllmModelGateway` target into an operable
private model endpoint. It covers one OpenAI-compatible vLLM server, versioned
Qwen checkpoint selection, tool calling, structured output, health checks,
benchmarking, security, rollout, and rollback.

The committed assets have been tested offline for syntax and contract coverage.
No claim is made that this repository has downloaded a model, reserved a GPU, or
passed live inference. A deployment is complete only when the evidence gates in
section 14 pass on the target hardware.

Deployment assets live in `deploy/qwen-vllm/`. The Agent-facing data contract
remains `src/puncture_agent/model_gateway/`; deployment flags must not leak into
LangGraph nodes.

## 2. Target topology

```text
Agent Runtime
    |
    | internal OpenAI-compatible HTTP/SSE, authenticated
    v
VllmModelGateway
    |
    v
internal gateway / network policy / rate limit
    |
    v
qwen-vllm container ---- pinned Qwen checkpoint cache
    |
    `---- NVIDIA GPU(s), tensor parallel when configured
```

The Compose default publishes to `127.0.0.1`. For a remote private service, do
not change it directly to `0.0.0.0` without an authenticated reverse proxy,
transport encryption, network allowlist, request-size limit, and rate limit.

## 3. Version compatibility is an explicit release artifact

vLLM CLI flags, tool parsers, Qwen chat templates, reasoning parsers, and
structured-output controls are version-sensitive. For each release, record and
approve this matrix:

| Component | Required record |
|---|---|
| Container | exact image tag and digest |
| GPU stack | driver, CUDA reported in container, GPU model/count |
| Model | repository, immutable revision, weight/quantization format |
| Serving | tensor parallel, dtype, KV dtype, context, scheduler limits |
| Chat | template source and checksum if overridden |
| Tools | auto-tool-choice setting, parser name, parser plugin/version |
| Reasoning | parser name or explicitly disabled |
| Structured output | request shape and server flag style/backend |
| Client | gateway commit and OpenAI-compatible request mapping |

Before accepting a new vLLM image, run inside that exact image:

```bash
docker run --rm --entrypoint vllm "$VLLM_IMAGE" serve --help
```

Verify every enabled flag from `.env`; do not assume a parser or legacy guided
decoding flag survived an upgrade. Useful primary references are the
[vLLM OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html),
[vLLM tool-calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html),
[vLLM structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html),
and [Qwen's vLLM deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html).

The initial asset baseline was checked on 2026-07-10 against these immutable
upstream source commits:

- [vLLM `08dfd68610d2e05a0d8ddc99c23488da6163df3f`](https://github.com/vllm-project/vllm/tree/08dfd68610d2e05a0d8ddc99c23488da6163df3f),
  especially its [structured-output guide](https://github.com/vllm-project/vllm/blob/08dfd68610d2e05a0d8ddc99c23488da6163df3f/docs/features/structured_outputs.md);
- [Qwen3 `7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e`](https://github.com/QwenLM/Qwen3/tree/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e),
  especially its [vLLM deployment guide](https://github.com/QwenLM/Qwen3/blob/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e/docs/source/deployment/vllm.md)
  and [function-calling guide](https://github.com/QwenLM/Qwen3/blob/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e/docs/source/framework/function_call.md).

The bootstrap container tag is `vllm/vllm-openai:v0.25.0`, whose upstream tag
commit was verified as
[`e12b91b032daed2afc34d77cca20902cef957b3c`](https://github.com/vllm-project/vllm/tree/e12b91b032daed2afc34d77cca20902cef957b3c)
on that date. A production record must still resolve and pin the registry image
digest. No model commit is invented here: the operator must replace bootstrap
`main` with the immutable revision resolved for the selected Qwen checkpoint.

At that Qwen commit, the documented Qwen3-8B automatic tool-call example uses
`--enable-auto-tool-choice --tool-call-parser hermes`; at that vLLM commit,
`response_format` JSON schema remains supported and structured outputs are enabled
by default with an optional `--structured-outputs-config.backend`. These facts
justify the bootstrap defaults, but they do not replace verification against the
image/model revisions actually deployed.

## 4. Prerequisites

The target host must provide:

- Linux and a supported NVIDIA driver;
- Docker Engine, Compose v2, and NVIDIA Container Toolkit;
- enough local storage for the pinned checkpoint and engine caches;
- access to the approved model source or an internal artifact mirror;
- an internal API key/secret delivery mechanism;
- monitoring for GPU memory/utilization, container health, latency, throughput,
  queueing, and normalized model errors.

Preflight commands:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:YOUR_APPROVED_TAG nvidia-smi
```

Use an approved CUDA image in the last command. Store its tag/digest in the
release evidence.

## 5. Size the deployment before starting it

Complete `deploy/qwen-vllm/gpu-sizing-worksheet.md` using the exact pinned model
configuration and realistic prompt/output/concurrency distributions. Then run the
rough estimator with those values.

The estimator deliberately does not select a GPU or guarantee fit. Live approval
requires startup-peak, steady-state, context-limit, concurrency, and soak tests.
If memory is tight, reduce context, active sequences, batch tokens, or model size
before considering CPU offload. CPU offload can increase latency and must be
benchmarked as a different deployment profile.

## 6. Configure without committing secrets

```bash
cd deploy/qwen-vllm
cp .env.example .env
chmod 600 .env
```

Mandatory production changes:

1. replace the bootstrap `VLLM_IMAGE=vllm/vllm-openai:v0.25.0` tag with its
   approved immutable image digest;
2. replace `VLLM_MODEL_REVISION=main` with an immutable model revision;
3. set `CUDA_VISIBLE_DEVICES`, `GPU_COUNT`, and tensor parallel consistently;
4. set context and scheduler limits from the approved sizing result;
5. verify the selected tool parser against this model/revision;
6. choose the structured-output flag style only after checking `serve --help`;
7. inject API credentials outside Git.

For a checkpoint copied into the mounted model cache, set `VLLM_MODEL_ID` to its
absolute path inside the container (for example,
`/models/huggingface/Qwen2.5-3B-Instruct`). The entrypoint deliberately omits the
Hugging Face `--revision` flag for an absolute local path because no remote
revision is resolved in that mode. `VLLM_MODEL_REVISION` remains mandatory as an
operator-recorded identity: set it to the approved source revision or manifest
hash and record checksums for every local weight file. A mutable directory named
`main` is not an immutable production identity.

For local bootstrap, `VLLM_API_KEY` may be exported in the shell:

```bash
read -rsp 'vLLM API key: ' VLLM_API_KEY
export VLLM_API_KEY
```

Compose mounts `/dev/null` as both secret files for unauthenticated bootstrap.
For production, set `VLLM_API_KEY_FILE_HOST` to an absolute, permission-restricted
host secret path (or replace the bind with an orchestrator secret). The container
already reads `/run/secrets/vllm_api_key`. The entrypoint supports either a direct
environment value or file, never both. Do not add the secret file to this
repository.

For public checkpoints, keep `HF_TOKEN_FILE_HOST=/dev/null`. If a source token is
required, point that setting at a protected file (or use an orchestrator secret)
and ensure the token is absent from Compose rendering, logs, shell history,
benchmark JSON, and traces.

## 7. Understand the compatibility switches

### Tool calling

Set:

```text
VLLM_ENABLE_AUTO_TOOL_CHOICE=true
VLLM_TOOL_CALL_PARSER=<parser verified for exact model and vLLM version>
```

`hermes` in `.env.example` is a bootstrap example, not a universal Qwen parser.
If the selected model needs a custom parser, pin the plugin artifact and set
`VLLM_TOOL_PARSER_PLUGIN`. If it needs a non-default chat template, mount a
read-only, checksum-pinned template and set `VLLM_CHAT_TEMPLATE_PATH`.

Do not approve tool calling merely because the model emitted text resembling
JSON. The smoke test forces a named function and parses its argument object. The
application gateway must still reject unknown tool names and schema-invalid
arguments before MCP execution.

### Structured output

The model gateway sends an OpenAI-style JSON schema response format and validates
the returned object locally. Provider-side constrained decoding is an additional
control, not a replacement for local validation.

The entrypoint exposes three server CLI modes:

- `none`: do not add a server-wide backend flag; use request-level support;
- `legacy`: add `--guided-decoding-backend` for a verified older release;
- `config`: add `--structured-outputs-config.backend` for a verified newer
  release.

Only choose `legacy` or `config` when the exact image help output supports that
syntax. A successful plain-chat request does not prove structured output works.

### Thinking/reasoning models

If the chosen checkpoint emits a separate reasoning channel, set a parser only
when the exact vLLM/Qwen combination documents it. Verify that structured JSON
content and tool-call arguments do not contain reasoning markup. Treat parser or
chat-template changes as release changes requiring the entire smoke suite.

`VLLM_ENABLE_REASONING_LEGACY_FLAG` conditionally adds `--enable-reasoning` for a
verified release that still requires it. Leave it false for releases where the
selected reasoning parser is sufficient. If the exact release requires structured
decoding inside reasoning, verify and enable
`VLLM_STRUCTURED_OUTPUT_ENABLE_IN_REASONING`; it adds the dotted vLLM config flag
documented by the pinned source baseline.

## 8. Validate and start

First run offline repository tests:

```bash
python3 -m unittest discover -s deploy/qwen-vllm/tests -p 'test*.py' -v
bash -n deploy/qwen-vllm/entrypoint.sh
```

Render the Compose model before creating a container:

```bash
cd deploy/qwen-vllm
docker compose --profile serve config
```

Inspect the rendered result for:

- the expected image digest and model revision;
- loopback/private exposure;
- the intended visible GPU count and tensor parallel size;
- absence of credentials;
- the correct cache path and read-only script mounts.

Start and observe model loading:

```bash
docker compose --profile serve up -d qwen-vllm
docker compose --profile serve logs -f --tail=200 qwen-vllm
```

Loading can take several minutes. Do not route Agent traffic until readiness
confirms both `/health` and the expected ID from `/v1/models`.

## 9. Readiness and smoke verification

From the host:

```bash
python3 scripts/health_check.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$VLLM_SERVED_MODEL_NAME" \
  --wait-seconds 900

python3 scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$VLLM_SERVED_MODEL_NAME"
```

Or with Compose:

```bash
docker compose --profile serve --profile verify run --rm smoke-test
```

Expected gates:

1. health endpoint succeeds;
2. configured served model appears in `/v1/models`;
3. ordinary assistant content is non-empty;
4. an SSE request emits content deltas, a supported terminal finish reason, and
   `[DONE]` without an early disconnect;
5. a forced tool call returns the requested name and valid JSON arguments;
6. JSON-schema output parses and contains exactly the required fields.

`--skip-tools` and `--skip-structured` exist only for fault isolation. A release
cannot be approved with either capability skipped because the Agent design uses
both.

## 10. Benchmark and admission limits

Warm up before recording results, then test concurrency 1, 4, intended load, and
an overload case. Example:

```bash
mkdir -p benchmark-results
python3 scripts/benchmark.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$VLLM_SERVED_MODEL_NAME" \
  --requests 40 \
  --concurrency 4 \
  --max-tokens 128 \
  --stream true \
  --output benchmark-results/concurrency-4.json
```

The script records success rate, P50/P95 end-to-end latency, streaming TTFT,
request throughput, provider-reported completion-token throughput when available,
and per-sample failures. It does not measure GPU memory; record `nvidia-smi` or
the approved GPU telemetry source in the sizing worksheet at the same time.

Use production-like prompts separately because the committed benchmark contains
no company or patient data. Admission control must reject requests that exceed
the approved context/output/concurrency envelope before they create GPU OOM risk.

## 11. Connect the Agent model gateway

The production application config should resolve to:

```text
base_url=http://<private-service>/v1
model=<VLLM_SERVED_MODEL_NAME>
api_key=<secret reference>
ca_bundle_path=<approved enterprise CA bundle, when system trust is insufficient>
timeout_seconds=<measured SLO-aware timeout>
max_retries=<bounded transient retry count>
```

Then run the model-gateway contract and adapter tests. Confirm that streaming and
non-streaming terminal responses normalize to the same repository contracts,
unknown tools never reach MCP, and provider errors map to stable error codes.

Do not allow arbitrary request metadata to become vLLM sampling parameters. Keep
an allowlist in the gateway and enforce context/output limits server-side and
client-side.

## 12. Security baseline

- Keep inference on an internal network; use TLS/mTLS at the gateway boundary.
- The production model gateway always verifies TLS and has no `verify=false`
  option. Public certificates use the HTTP client's default CA bundle. For an
  enterprise CA, set `VllmGatewayConfig.ca_bundle_path` explicitly or terminate
  TLS at a trusted internal gateway; certificate failures are non-retryable.
  Ambient `SSL_CERT_FILE` is intentionally not used because the HTTP client
  disables all environment-derived transport configuration.
- Require an API key or service identity and rotate it without rebuilding images.
- Bind directly to loopback for single-host development.
- Pin image digest, model revision, plugins, and custom chat-template checksum.
- Keep `trust_remote_code=false` unless code is reviewed, mirrored, pinned, and
  explicitly approved.
- Do not log prompts, RAG context, tool arguments, model output, API keys, or model
  source tokens by default.
- Use request/body/token limits, concurrency admission, timeouts, and rate limits.
- Run with `no-new-privileges`; use the minimum writable cache/temp paths.
- Scan the image/checkpoint supply chain and retain license/provenance records.
- Separate model-download credentials from inference credentials.
- Treat custom parser/plugin code as executable supply-chain input.
- Do not place confidential inputs in smoke or benchmark artifacts.

The entrypoint never uses `eval`, constructs arguments as a Bash array, and does
not print the API key. Note that a CLI `--api-key` may be visible to a privileged
observer inside the container; prefer a trusted network gateway when the threat
model requires credential isolation beyond process-argument protection.

The committed verification clients ignore ambient HTTP proxy variables, reject
every redirect, reject service URLs containing credentials/query/fragment, bound
JSON and SSE response sizes, and omit provider error bodies/network reasons from
reports. This prevents a host proxy or redirect from receiving the API Bearer
token and request content. If an approved egress proxy is genuinely required,
implement it as a reviewed site-specific transport change instead of restoring
ambient proxy inheritance.

## 13. Observability and incident signals

Collect at minimum:

- readiness, restarts, model-load duration, and current image/model identity;
- GPU utilization, memory, temperature, power, and OOM events;
- active/running/waiting requests and scheduler queue time;
- input/output tokens, TTFT, TPOT, end-to-end latency, and throughput;
- HTTP 4xx/5xx, timeout/disconnect, overload, and schema-invalid rates;
- tool-call and structured-output validity rates by deployment version.

Correlate application `request_id` with safe provider request IDs, but redact
prompts and credentials. Alert on readiness loss, OOM/restart loops, rising P95,
queue growth, schema-validity regression, and repeated model-protocol errors.

## 14. Verification matrix and evidence

Every row is required unless explicitly marked deployment-specific.

| Layer | Test | Pass condition | Evidence |
|---|---|---|---|
| Offline | deployment asset unit tests | all pass without Docker/GPU/network | test log |
| Syntax | `bash -n` and `docker compose config` | exit 0; expected rendered values | command log/redacted config |
| Supply chain | image/model/plugin/template identity | immutable approved digests/revisions | release record |
| GPU | container GPU preflight | intended GPU count/type visible | `nvidia-smi` capture |
| Startup | cold model load | reaches ready without OOM | startup log and duration |
| Readiness | health + model-list probe | expected served ID present | health script output |
| Chat | deterministic plain request | non-empty normalized content | smoke JSON |
| Tools | forced named tool | correct name, parseable schema-valid args | smoke JSON |
| Structured | JSON schema response | locally parseable exact required fields | smoke JSON |
| Streaming | SSE request | content arrives, terminal response completes | gateway/live test |
| Context | near approved input limit | succeeds inside limit, rejects outside limit | test record |
| Load | planned concurrency | latency/error SLO and memory headroom pass | benchmark JSON + GPU metrics |
| Overload | above admission limit | controlled reject/queue; no OOM loop | fault-test record |
| Security | secret/redaction/network review | no secret in repo/config/log; private exposure | review checklist |
| Recovery | restart and dependency interruption | readiness recovers; client gets normalized errors | incident drill |
| Rollback | previous image/model pair | rollback meets readiness and smoke gates | rollback drill |

Store with each deployment version:

```text
Git commit/tag:
Image tag/digest:
Model repository/revision:
GPU/driver/CUDA:
Effective non-secret configuration:
Tool/reasoning parser:
Structured-output mode/backend:
Offline test log:
Smoke result:
Benchmark results:
Peak memory evidence:
Known limitations:
Rollback version:
Operator/approver/time:
```

## 15. Rollout and rollback

Use blue/green or canary rollout when infrastructure permits:

1. keep the current approved service running;
2. start the candidate under a new internal endpoint;
3. run readiness, smoke, contract, context, and benchmark gates;
4. send a small allowlisted traffic percentage to the candidate;
5. compare latency, error, tool validity, and structured validity;
6. promote only after the observation window passes;
7. retain the previous image/model cache and non-secret configuration until the
   rollback window closes.

Rollback immediately for readiness loss, OOM/restart loops, material latency or
error regression, malformed tool arguments, structured-output regression, or an
unexpected model/template/parser identity.

Single-host Compose rollback:

```bash
cd deploy/qwen-vllm
cp .env .env.failed-candidate
cp /secure/release-records/PREVIOUS_APPROVED_ENV .env
docker compose --profile serve up -d --force-recreate qwen-vllm
python3 scripts/health_check.py --wait-seconds 900
python3 scripts/smoke_test.py
```

The approved environment record must contain no secret values; inject current
secrets separately. If model cache contents are mutable, use revision-specific
cache directories or verify the resolved checkpoint before traffic resumes.

## 16. Common failures

| Symptom | Likely check | Safe response |
|---|---|---|
| CLI exits with unknown argument | flag changed between vLLM versions | compare exact `serve --help`; change compatibility switch or revert image |
| Model absent from `/v1/models` | load incomplete, wrong served alias, startup error | inspect startup logs and identity; do not route traffic |
| CUDA OOM at startup | weight/graph reserve or TP mismatch | rollback; revise sizing/topology, context, dtype, or model |
| OOM under load | KV/concurrency/context admission too high | stop traffic growth; reduce limits and re-benchmark |
| Tool call returned as prose | wrong parser/template/model capability | verify parser/template pair; rerun forced-tool smoke |
| Invalid tool JSON | parser/model regression | reject in gateway; rollback if validity SLO fails |
| JSON schema ignored | incompatible request/structured-output mode | verify exact request and CLI support; retain local validation |
| High TTFT with low GPU use | queueing, prefill, CPU offload, tokenizer/network | inspect scheduler and stage timings; benchmark one change at a time |
| Health passes but requests fail | model-list or functional path not covered by liveness | use readiness plus full smoke, not `/health` alone |
| 401 from verifier | server/client key injection mismatch | verify secret reference; never print the key |

Never weaken schema validation or bypass MCP authorization to make a smoke test
pass. A compatibility failure is a deployment failure, not an Agent prompt issue.
