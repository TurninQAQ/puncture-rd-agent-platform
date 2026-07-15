# Local RTX 4090 Qwen/vLLM Validation

Validation date: 2026-07-15 UTC

This is a non-production workstation record. It proves that the checked-in
deployment and model gateway can run against a real local model service. It is
not approval evidence for the default v0.25.0 image, a production cluster, or a
production SLA.

## Exact identity and scope

- Git baseline before evidence update: `f42b1d704aa9c86a9e2cb3b6b6bf5f84b17f4f3e`
- Image: `docker.m.daocloud.io/vllm/vllm-openai@sha256:0b51ec38fb965b44f6aa75d8d847c5f21bc062b7140e1d83444b39b67fc4a2ea`
- vLLM: `0.9.1`; image CUDA runtime: `12.8.1`
- Model: local `Qwen2.5-3B-Instruct`
- Source revision identity: `hf-aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Served model: `qwen-enterprise-agent`
- Host: Ubuntu 22.04, Docker 29.1.3, NVIDIA driver 535.247.01
- Available hardware: two RTX 4090 cards; this service intentionally used GPU 0 only
- Effective limits: 4,096-token context, four active sequences, 0.52 GPU memory utilization
- Exposure: unauthenticated loopback only at `127.0.0.1:8008`

Model artifact checksums:

```text
67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2  model-00001-of-00002.safetensors
a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d  model-00002-of-00002.safetensors
eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b  config.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539  tokenizer.json
```

## Startup and functional gates

- The local-path entrypoint omitted Hugging Face `--revision` while retaining the
  recorded source identity.
- Both safetensors shards loaded successfully. The engine reported 5.79 GiB of
  model weights, 6.19 GiB of available KV cache, and 89.92 seconds for engine
  profile/cache/warmup initialization.
- Readiness confirmed `/health` and the expected ID from `/v1/models`.
- The deployment smoke test passed plain chat, SSE streaming, a forced named tool
  call with schema-valid JSON arguments, and JSON Schema structured output.
- `tests.model_gateway.test_live_vllm` passed 5/5 against the live endpoint,
  including an autonomous tool choice through the production gateway.
- The container remained healthy with zero restarts and no OOM kill.

The selected 3B checkpoint produced malformed tool JSON for one adversarially
worded prompt during bring-up. A direct natural-language tool request passed
deterministically and remains an autonomous choice: the gateway does not force a
provider-specific `tool_choice`. Prompt/model/parser changes therefore require
the live gate again.

## Streaming benchmark

The benchmark prompt contained no company or patient data. All requests used a
128-token output limit and streaming responses.

| Concurrency | Requests | Success | Request/s | Completion token/s | Latency P50/P95 | TTFT P50/P95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20/20 | 2.409 | 107.810 | 430.59/441.39 ms | 32.37/40.85 ms |
| 4 | 40 | 40/40 | 8.689 | 374.295 | 473.39/547.84 ms | 37.78/82.04 ms |

The highest sampled GPU memory value during the concurrency-four run was
14,798 MiB. This was sampled telemetry, not a continuously collected peak.
Machine-readable summaries are stored in
`local-rtx4090-benchmark-summary.json`.

## Environment limitation

The host did not have NVIDIA Container Toolkit registered with Docker and the
operator account could not perform passwordless system installation. For this
workstation run only, an ignored Compose override explicitly passed GPU device
nodes, mounted the matching host driver libraries read-only, and placed the host
driver directory before the image CUDA directory in `LD_LIBRARY_PATH`.

This workaround is reversible and proved the application path, but it is not the
documented production configuration. Install and configure NVIDIA Container
Toolkit, then rerun every gate without the override before treating the host as a
deployment target.

## Remaining gates

- vLLM 0.25.0 and its exact CUDA/driver combination
- authenticated secret delivery instead of loopback bootstrap
- near-context-limit, admission-overload, long soak, restart, and rollback drills
- tensor parallel operation across both GPUs
- production prompts, safety evaluation, and externally defined SLOs
