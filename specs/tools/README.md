# Tool implementation and verification standard

This directory is the hand-off contract for replacing deterministic mocks with
company implementations.  Read this file and the selected tool specification
before editing code.

## Non-negotiable integration rules

1. Do not rename, add, remove, or reinterpret fields in `contracts/tool_inputs.py`
   or `contracts/tool_outputs.py`.
2. CT voxels, masks, path masks, and reports are passed through `ArtifactRef`.
   Never put a voxel array, base64 file, or signed storage URI into an LLM
   message.
3. Interpret image size as `(i, j, k)` and world coordinates in millimetres.
   Respect `coordinate_system`, `origin_mm`, and `direction_cosines`; do not
   assume identity direction.
4. A successful tool call returns `ToolResponseEnvelope(status=SUCCESS)` with a
   typed result. A domain-invalid result is not a Python exception. Return a
   structured `ErrorDetail` or a successful validation result whose `valid`
   field is false, according to the individual specification.
5. Propagate `request_id`, `trace_id`, case ID, tool version, produced artifacts,
   timings, and warnings. Do not expose an artifact's `uri` outside the internal
   tool boundary; external APIs use `ArtifactPublicView`.
6. Make repeated requests with the same `idempotency_key` return the same
   committed artifact rather than duplicating files or inference jobs.
7. Do not silently resample or relabel data. Such actions must be explicitly
   requested, versioned, recorded in artifact lineage, and covered by tests.
8. The LLM chooses and sequences tools. Numeric geometry, segmentation,
   collision, and risk conclusions remain deterministic algorithm outputs.

## Response semantics

Use the following decision rule:

- `SUCCESS`: execution completed and produced a contract-valid result. A
  validator may still report `valid=False`; that is a valid validation outcome.
- `PARTIAL`: a useful subset exists and every omitted part is listed in
  `warnings`. Do not use PARTIAL for a missing required safety structure.
- `FAILED`: no result is safe to consume. `error` is mandatory.

Errors caused by request data are normally non-retryable. Timeouts, temporary
GPU allocation failures, or transient service failures may be retryable. Keep
the retry decision in `ErrorDetail.retryable`; the Agent runtime owns the retry
policy.

## Required test layers for every real implementation

### 1. Contract tests

Run before and after implementation:

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
```

They detect field drift, wrong result types, broken JSON serialization, and URI
leakage. Never update snapshots merely to make an implementation pass. A schema
change requires a versioned migration approved by all module owners.

### 2. Tool unit tests

```bash
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/tools -p 'test_*.py' -v
```

The checked-in tests exercise the mocks and define basic behavior. When a real
adapter is added, run the same tests against it and add implementation-specific
tests beside the relevant file.

### 3. Golden-data tests

Each algorithm implementation must maintain small, de-identified internal
fixtures with frozen expected outputs:

- one normal case;
- one anisotropic-spacing case;
- one non-identity-direction case;
- one empty/missing-label case;
- tool-specific edge cases listed in its specification.

Store only fixture IDs and expected summaries in the test repository if raw
company data cannot be included. Resolve raw fixtures through an internal test
artifact registry. Golden outputs include checksums, geometry fingerprints,
label statistics, candidate IDs, distances, or risk flags as appropriate.

### 4. Independent-oracle tests

For safety-relevant geometry, compare the production implementation with a
slower independent reference implementation. Examples: brute-force sampled
distance for path clearance, SciPy/SimpleITK morphology for a C++ kernel, or
high-resolution ray sampling for an optimized traversal. Do not validate an
algorithm with a test that calls the same kernel through a second wrapper.

### 5. Negative and failure-injection tests

Every documented error code needs at least one deterministic test. Dependency
timeouts, corrupted artifacts, unavailable GPU, and partial file writes should
be injected at adapter boundaries, not created through random sleeps.

### 6. Performance tests

Record hardware, software version, input size, precision, warm-up count, and
sample count. Report P50/P95 rather than one run. Functional CI should not fail
on workstation-specific latency. A separate benchmark job enforces configured
budgets.

## Definition of done for a replacement

A Mock may be replaced only when all of the following are true:

- the real function keeps the exact request and response types;
- contract and tool unit tests pass;
- every required error has a test;
- at least one internal golden normal case and all listed edge cases pass;
- geometry and coordinate conventions are tested;
- artifact lineage, checksum, and idempotent retry are verified;
- metrics use the names and units documented by the tool;
- an independent oracle agrees within the documented tolerance;
- the task file's acceptance checklist is signed off.

## Mock versus real handler selection

`puncture_agent.tooling.build_mock_registry()` is intentionally dependency-free.
Production code should build a separate registry that binds the same
`TOOL_DEFINITIONS` to real functions from `tooling/stubs.py` or adapter modules.
Do not add environment branches inside the contracts or mocks.
