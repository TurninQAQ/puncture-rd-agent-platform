# `run_segmentation`

## Purpose

Invoke the existing company C++/TensorRT segmentation pipeline for a versioned
model and register its label map. The Agent never loads the engine or performs
pre/post-processing itself.

## Fixed contract

- Request: `RunSegmentationRequest`
- Result: `SegmentationInferenceResult`
- Tool: `run_segmentation@1.0.0`
- Timeout: 120 seconds

The tuple `requested_labels` is an output-selection/validation request. It does
not change the model's trained label mapping unless the registered model
explicitly supports such selection. `precision` must match a built and validated
engine profile. Metrics use milliseconds and megabytes exactly as named in the
result and envelope.

## Implementation algorithm

1. Resolve CT and model registry entries; verify model ID/version, engine hash,
   input schema, label mapping, precision, GPU compute capability, and allowed
   preprocessing profile.
2. Acquire/reuse the TensorRT engine by `(model_id, version, precision,
   device_id)`; engine creation must not occur on every request.
3. Validate CT geometry and intensity dtype. Apply the model-registered
   orientation, resampling, clipping, normalization, and tensor layout. Record
   every transform so output can be returned to original CT geometry.
4. Allocate buffers using the company inference framework, bind the correct CUDA
   stream, run warm state inference, and synchronize before timing ends.
5. Apply registered post-processing: argmax/thresholding, class mapping,
   connected-component rules, and inverse spatial transform. Never infer label
   values from `requested_labels` order.
6. Verify output size, geometry, integer label set, and nonempty required labels.
7. Atomically write/register the segmentation artifact with CT and model lineage.
8. Calculate `LabelStatistics`, inference latency, and peak GPU memory. Clarify
   whether latency includes preprocessing/postprocessing; preferably expose
   additional envelope metrics while keeping `inference_time_ms` definition
   stable.

## Error mapping

- `MODEL_NOT_FOUND`: model ID/version absent.
- `MODEL_VERSION_MISMATCH`: engine metadata conflicts with requested profile.
- `INVALID_ARGUMENT`: unsupported precision/device/labels.
- `GEOMETRY_MISMATCH`: CT geometry cannot enter registered preprocessing.
- `GPU_OUT_OF_MEMORY`: allocation failed; retryable only if runtime can route to
  another worker or release cache.
- `MODEL_INFERENCE_FAILED`: TensorRT enqueue, CUDA, or postprocessing failed.
- `EMPTY_SEGMENTATION`: inference completed but required output is empty.
- `TIMEOUT`, `DEPENDENCY_FAILED`: service/infrastructure errors.

## How to verify correctness

### Functional golden tests

For each deployed model/engine version, freeze at least five internal CT fixture
IDs and reference outputs from the approved offline inference pipeline. Compare:

- exact output geometry and label set;
- per-label Dice and surface-distance tolerance against the reference;
- per-label voxel/volume statistics;
- deterministic checksum when kernels are deterministic, otherwise metric
  tolerance documented per precision.

Test identity direction, non-identity direction, anisotropic spacing, minimum
input size, and a case containing every required class.

### Adapter tests

Inject missing engine, unsupported precision, CUDA error, OOM, timeout, malformed
output dimensions, and invalid label value. Assert cleanup and error mapping.
Retry the same idempotency key and assert one registered artifact.

### Performance protocol

- record GPU model, driver, CUDA, TensorRT, engine hash, input size, batch, and
  precision;
- perform at least 10 warm-ups and 100 measured requests;
- report P50/P95 inference and end-to-end latency plus peak memory;
- verify known deployment evidence (approximately 158 ms and 2795 MB) only on
  the matching company hardware/profile; do not make those universal CI limits.

Existing tests:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools.SegmentationToolTests.test_inference_returns_versioned_artifact_and_metrics -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools.SegmentationToolTests.test_gpu_oom_is_retryable_structured_error -v
```

Acceptance criteria:

- output meets the model registry's accuracy tolerance on every golden case;
- geometry round-trip is exact within `VolumeGeometry` tolerance;
- no engine is rebuilt per normal request;
- no CUDA allocation or temporary artifact leaks after failures;
- model/engine/hash/profile appear in trace and artifact lineage.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| RS-01 | Contract snapshot, typed result, artifact lineage, and JSON round-trip pass. |
| RS-02 | Normal golden cases satisfy registered per-label accuracy tolerances. |
| RS-03 | Input shape/orientation/spacing, precision, label selection, and threshold boundaries pass. |
| RS-04 | Missing model, invalid profile/output, CUDA error, and OOM map to stable errors. |
| RS-05 | Model service/storage timeout and process failure are injected with resource cleanup. |
| RS-06 | Same idempotency key reuses one committed artifact; engine reuse is verified. |
| RS-07 | Target-hardware warm P50/P95, TTFT-equivalent service wait, throughput, and peak GPU memory are recorded. |
