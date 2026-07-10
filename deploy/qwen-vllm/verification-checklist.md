# Qwen / vLLM Verification Checklist

Copy this file into the private release evidence directory and check items only
after recording the linked output. Unchecked live items mean the deployment is
not production-verified.

## A. Offline repository gate

- [ ] `python3 -m unittest discover -s deploy/qwen-vllm/tests -p 'test*.py' -v`
  passes.
- [ ] `python3 run_tests.py` passes and includes `qwen_vllm_asset_tests`.
- [ ] `bash -n deploy/qwen-vllm/entrypoint.sh` passes.
- [ ] `docker compose --profile serve config` passes on the target Compose
  version.
- [ ] Rendered Compose output contains no API key, source token, prompt, or
  internal document content.

## B. Immutable identity gate

- [ ] vLLM image tag and registry digest recorded.
- [ ] Selected Qwen repository and immutable model revision recorded.
- [ ] GPU, driver, CUDA, quantization, dtype, and KV dtype recorded.
- [ ] Tool parser, optional plugin, reasoning parser, and chat-template checksum
  recorded.
- [ ] Context length, tensor parallel size, maximum sequences, and batch-token
  limit recorded.
- [ ] Every enabled entrypoint flag exists in the exact image's
  `vllm serve --help` output.
- [ ] `latest` and `main` are absent from the approved identity record.

## C. Startup and readiness gate

- [ ] Intended GPUs are visible inside the container.
- [ ] Cold start completes without CUDA OOM or restart loop.
- [ ] Model load time and startup peak GPU memory recorded.
- [ ] `health_check.py` returns `READY` for the exact served alias.
- [ ] A wrong alias makes `health_check.py` return non-zero.
- [ ] Agent traffic remains disabled until readiness passes.

Evidence command:

```bash
python3 deploy/qwen-vllm/scripts/health_check.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen-enterprise-agent \
  --wait-seconds 900
```

## D. Functional inference gate

- [ ] Plain chat returns non-empty assistant content.
- [ ] Streaming emits content deltas, a terminal finish reason, and `[DONE]`.
- [ ] Forced `lookup_document` returns the exact tool name.
- [ ] Tool arguments parse as JSON and satisfy the required schema.
- [ ] JSON-schema response contains exactly the required fields and types.
- [ ] Application-side gateway schema validation also passes.
- [ ] Unknown or malformed tool calls are rejected before MCP execution.

Evidence command:

```bash
python3 deploy/qwen-vllm/scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen-enterprise-agent
```

Do not approve a release run with `--skip-tools` or `--skip-structured`.

## E. Capacity and reliability gate

- [ ] `gpu-sizing-worksheet.md` contains exact model architecture inputs.
- [ ] Warm runs at concurrency 1, 4, intended peak, and overload are recorded.
- [ ] TTFT P50/P95, end-to-end P50/P95, token throughput, error rate, and peak GPU
  memory meet the declared SLO/headroom.
- [ ] Near-limit context succeeds and over-limit context is rejected safely.
- [ ] Sustained soak does not leak memory or degrade latency beyond the SLO.
- [ ] Overload produces controlled queue/reject behavior without OOM/restart loop.

Evidence command template:

```bash
python3 deploy/qwen-vllm/scripts/benchmark.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen-enterprise-agent \
  --requests 40 \
  --concurrency 4 \
  --max-tokens 128 \
  --stream true \
  --output benchmark-results/concurrency-4.json
```

## F. Security gate

- [ ] Service is loopback-only or behind an authenticated TLS/mTLS internal
  gateway and network allowlist.
- [ ] Inference credential and model-download credential are separate and
  injected from protected secret files/orchestrator secrets.
- [ ] Secret values are absent from Git, `.env`, rendered Compose, process logs,
  traces, smoke output, and benchmark JSON.
- [ ] `trust_remote_code` remains false, or reviewed/pinned code approval is
  attached.
- [ ] Image, checkpoint, parser plugin, and template provenance/license/scan
  evidence is attached.
- [ ] Request/token/body/concurrency/rate limits and timeouts are enabled.
- [ ] Verification transport bypasses ambient proxies, rejects redirects, and
  enforces response-size bounds.

## G. Recovery and rollback gate

- [ ] Process restart returns to readiness within the recovery SLO.
- [ ] Model-service interruption returns normalized gateway errors to the Agent.
- [ ] Previous approved image/model/config remains available.
- [ ] Rollback was executed and passed readiness plus full smoke verification.
- [ ] Candidate traffic was not fully promoted before the observation window.

## H. Sign-off

```text
Deployment version:
Git commit/tag:
Environment:
Evidence directory:
Known limitations:
Rollback version:
Operator:
Reviewer:
Approval time:
Decision: APPROVE / REJECT
```

