# GPU Sizing Worksheet

Use this worksheet before deployment and replace every `TBD` with a measured or
model-card value. The formulas provide a conservative starting point; they do not
prove that a model/version/GPU combination is supported.

## 1. Immutable software and model identity

| Item | Planned value | Verified value |
|---|---|---|
| vLLM image tag | TBD | TBD |
| vLLM image digest | TBD | TBD |
| CUDA version reported by container | TBD | TBD |
| NVIDIA driver version | TBD | TBD |
| Model repository | TBD | TBD |
| Model revision/commit | TBD | TBD |
| Served model name | TBD | TBD |
| Quantization/checkpoint format | TBD | TBD |
| Chat template checksum, if overridden | TBD | TBD |
| Tool parser and plugin version | TBD | TBD |

Do not approve production with `latest` or `main` in the verified column.

## 2. Model architecture inputs

Take these values from the exact pinned model configuration, not from a similarly
named model.

| Input | Value |
|---|---:|
| Parameters, billions | TBD |
| Weight bits after loading | TBD |
| Transformer layers | TBD |
| KV attention heads | TBD |
| Head dimension | TBD |
| KV cache bytes per element | TBD |
| Maximum model context | TBD |

## 3. Workload assumptions

Capacity must use token distributions rather than only the configured maximum.

| Workload field | P50 | P95 | Maximum/admission limit |
|---|---:|---:|---:|
| Input tokens | TBD | TBD | TBD |
| Output tokens | TBD | TBD | TBD |
| Concurrent active sequences | TBD | TBD | TBD |
| Requests per second | TBD | TBD | TBD |
| Tool schemas per request | TBD | TBD | TBD |
| RAG context tokens | TBD | TBD | TBD |

## 4. Planning formulas

Use GiB (`1024^3` bytes).

```text
weight_GiB = parameters * 1e9 * weight_bits / 8 / 1024^3

KV_GiB = 2 * layers * kv_heads * head_dim
          * context_tokens * concurrent_sequences * kv_bytes / 1024^3

rough_per_GPU_GiB = (weight_GiB * runtime_overhead_factor + KV_GiB)
                    / tensor_parallel_size
                    + activation_and_graph_reserve_GiB
```

This approximation does not model allocator fragmentation, CUDA graphs, temporary
startup allocations, multimodal tensors, speculative decoding, uneven tensor
parallel partitions, or implementation-specific KV blocks. Add measured safety
headroom and run OOM tests.

The helper performs the same rough calculation:

```bash
python3 deploy/qwen-vllm/scripts/sizing_estimator.py \
  --parameters-billions MODEL_VALUE \
  --weight-bits LOADED_WEIGHT_BITS \
  --layers MODEL_VALUE \
  --kv-heads MODEL_VALUE \
  --head-dim MODEL_VALUE \
  --context-tokens PLANNED_CONTEXT \
  --concurrent-sequences PLANNED_CONCURRENCY \
  --kv-bytes CACHE_ELEMENT_BYTES \
  --tensor-parallel GPU_COUNT \
  --gpu-memory-gib PER_GPU_MEMORY \
  --gpu-count GPU_COUNT
```

Do not copy architecture values from this repository: it intentionally contains
none because the selected Qwen checkpoint may change.

## 5. Candidate topology comparison

| Candidate | GPU type/count | TP | Context | Max sequences | Quantization | Estimated headroom/GPU | Startup result |
|---|---|---:|---:|---:|---|---:|---|
| A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| C | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Reject a candidate if tensor parallel size exceeds visible GPUs, the architecture
cannot divide safely for the selected engine/version, or startup peak causes OOM.

## 6. Measured capacity record

Run each intended concurrency at least three times after warm-up. Keep raw JSON
benchmark output with the release record.

| Concurrency | Prompt P50/P95 | Output P50/P95 | TTFT P50/P95 | E2E P50/P95 | Output tokens/s | Peak GPU GiB | Error rate |
|---:|---|---|---|---|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Intended peak | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Also record:

- model load duration and startup peak memory;
- cold and warm TTFT;
- tool-call JSON schema-valid rate;
- structured-output schema-valid rate;
- behavior at the admission context limit;
- overload response and recovery time;
- memory after a sustained soak period;
- whether a single worker/GPU loss is recoverable in the chosen topology.

## 7. Capacity decision

```text
Decision: APPROVE / REJECT
Approved image digest:
Approved model revision:
Approved GPU topology:
Approved max context:
Approved max sequences:
Approved admission concurrency:
Minimum measured headroom:
Performance SLO:
Approver:
Date:
Evidence paths:
```

Approval is invalid if the image digest, model revision, GPU type, quantization,
tool parser, chat template, or context limit changes.

