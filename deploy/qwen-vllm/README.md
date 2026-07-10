# Qwen / vLLM Deployment Assets

This directory starts a private Qwen-compatible model through vLLM's
OpenAI-compatible server. It is an executable deployment template, not evidence
that a GPU deployment has already been completed.

The deployment deliberately keeps model, revision, GPU topology, context length,
tool parser, reasoning parser, and structured-output compatibility flags in
configuration. Those options vary across Qwen and vLLM releases.

## Files

- `compose.yaml`: `serve`, `verify`, and `benchmark` profiles.
- `entrypoint.sh`: safe CLI argument construction without `eval`.
- `.env.example`: non-secret bootstrap configuration.
- `scripts/health_check.py`: readiness and served-model verification.
- `scripts/smoke_test.py`: chat, tool-call, and structured-output checks.
- `scripts/benchmark.py`: dependency-free concurrency/streaming benchmark.
- `scripts/sizing_estimator.py`: rough pre-deployment memory worksheet helper.
- `gpu-sizing-worksheet.md`: capacity-planning and measured-result template.
- `verification-checklist.md`: release-by-release functional and operational sign-off.
- `tests/test_deployment_assets.py`: offline static tests; no Docker or GPU needed.

The detailed production procedure is in
`../../docs/qwen-deployment-runbook.md`.

## Quick start

```bash
cd deploy/qwen-vllm
cp .env.example .env
```

The bootstrap image is the latest non-RC tag verified when these assets were
written. Before production, replace its tag with the tested image digest and
replace `VLLM_MODEL_REVISION=main` with the operator-supplied immutable model
revision that was tested with it.

```bash
docker compose --profile serve config
docker compose --profile serve up -d qwen-vllm
python3 scripts/health_check.py --base-url http://127.0.0.1:8000/v1 \
  --model qwen-enterprise-agent --wait-seconds 900
python3 scripts/smoke_test.py --base-url http://127.0.0.1:8000/v1 \
  --model qwen-enterprise-agent
```

The Compose-based smoke test is equivalent:

```bash
docker compose --profile serve --profile verify run --rm smoke-test
```

Run a small benchmark only after smoke checks pass:

```bash
mkdir -p benchmark-results
docker compose --profile serve --profile benchmark run --rm benchmark
```

## Offline asset tests

```bash
python3 -m unittest discover -s deploy/qwen-vllm/tests -p 'test*.py' -v
```

These tests verify syntax, required controls, examples, and script behavior. They
do not pull images, download a model, reserve a GPU, or prove live inference.
