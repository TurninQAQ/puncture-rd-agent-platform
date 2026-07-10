# Task 04 — Implement segmentation tools

## Goal

Replace the mocks for:

1. `run_segmentation`
2. `validate_segmentation_result`
3. `extract_skin_surface`

Reuse the existing C++/TensorRT inference framework and morphology code through
thin adapters. Do not rewrite trained model behavior in the Agent layer.

## Context package

Provide the implementation model:

```text
contracts/**
src/puncture_agent/tooling/catalog.py
src/puncture_agent/tooling/registry.py
src/puncture_agent/tooling/stubs.py
src/puncture_agent/mocks/tool_mocks.py
specs/tools/README.md
specs/tools/run-segmentation.md
specs/tools/validate-segmentation-result.md
specs/tools/extract-skin-surface.md
tests/contract/test_tool_contracts.py
tests/tools/helpers.py
tests/tools/test_segmentation_tools.py
```

Also provide sanitized model registry metadata: input schema, preprocessing,
label map, output schema, engine hash, supported TensorRT/CUDA/GPU versions, and
approved accuracy tolerances. Do not provide confidential engine binaries to an
external service.

## Allowed implementation area

- segmentation adapters under `src/puncture_agent/tooling/implementations/`;
- production registry binding;
- segmentation tool tests/internal fixture resolver;
- owner-approved build/dependency configuration.

Do not change tool contracts, model label semantics, mocks, unrelated services,
or Agent orchestration.

## Required adapter boundary

Use a narrow port rather than putting TensorRT lifecycle code in the tool:

```python
class SegmentationEnginePort(Protocol):
    def describe(self, model_id: str, version: str, precision: str) -> ModelProfile: ...
    def infer(self, ct: ArtifactRef, profile: ModelProfile, device_id: int) -> EngineResult: ...

class ImageAlgorithmPort(Protocol):
    def label_statistics(self, labelmap: ArtifactRef, schema: tuple[LabelDefinition, ...]): ...
    def extract_external_skin_surface(self, source: ArtifactRef, thickness_mm: float, ...): ...
```

The port may call local pybind11, gRPC, or REST. Map all native exceptions/status
codes at this boundary into `ErrorCode` values.

## Implementation sequence

1. Capture baseline contract/mock tests.
2. Build model-profile validation and long-lived TensorRT engine adapter.
3. Implement inference, inverse geometry transform, artifact commit, statistics,
   and metrics; verify against approved offline inference.
4. Implement segmentation validator using independent image operations where
   practical.
5. Implement external thin-skin extraction in physical units, including cavity
   policy and component cleanup.
6. Add deterministic failure injection around engine, CUDA, storage, and native
   calls. Verify no resource/file leak.
7. Add golden accuracy tests and separate target-hardware benchmark job.
8. Produce acceptance ID → test mapping and run the full suite.

## Required test commands

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools -v
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/tools -p 'test_*.py' -v
```

Native unit tests must be run separately (for example CTest) and referenced in
the verification report. Benchmark output must distinguish TensorRT enqueue
latency from end-to-end preprocessing/inference/postprocessing latency.

## Completion checklist

- [ ] RS acceptance IDs pass for every registered engine profile.
- [ ] VS acceptance IDs pass on synthetic and internal golden masks.
- [ ] ES acceptance IDs pass at isotropic/anisotropic spacing.
- [ ] FP16/other approved engine accuracy stays within model registry tolerance.
- [ ] Matching hardware/profile benchmark documents the approximately 158 ms,
      2795 MB evidence or explains its measured replacement.
- [ ] GPU OOM, timeout, native exception, malformed output, and partial write are
      mapped and cleaned up.
- [ ] Engine reuse and idempotent artifact commit are verified.
- [ ] Contracts and mock registry remain unchanged.

## Prompt to hand to another model

> Implement Task 04 only using the existing TensorRT/C++ framework behind a thin
> port. Do not modify any contract or infer preprocessing/label mapping from
> guesses; use the supplied model profile. Implement and verify one tool at a
> time. Include normal, boundary geometry, corrupt input, dependency failure,
> retry/idempotency, independent oracle, and target-hardware benchmark tests.
> Return an acceptance ID → test → result table and all commands/output summaries.
