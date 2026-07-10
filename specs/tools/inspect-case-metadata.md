# `inspect_case_metadata`

## Purpose

Gate a case before conversion, inference, or planning. It verifies artifact
availability, optional checksums, required artifact types, and exact spatial
compatibility with the CT. It does not modify or resample data.

## Fixed contract

- Request: `contracts.tool_inputs.InspectCaseMetadataRequest`
- Result: `contracts.tool_outputs.CaseMetadataResult`
- Tool name/version: `inspect_case_metadata@1.0.0`
- Default timeout: 10 seconds

Required behavior for request fields:

- `ct_artifact` is the geometry reference.
- `related_artifacts` are checked in the supplied order.
- `required_artifact_types` is a set-like requirement across CT plus related
  artifacts.
- `require_same_geometry=False` permits geometry differences but still reports
  `geometry_matches_ct` for every artifact.
- `verify_checksums=True` requires streamed SHA-256 verification against the
  registry value; do not load an entire CT into RAM solely to hash it.

## Implementation algorithm

1. Resolve every `ArtifactRef` through the internal registry and enforce the
   request case ID.
2. Verify status is AVAILABLE and object bytes are readable.
3. If requested, stream each object and calculate SHA-256. Cache a verified
   checksum by immutable object version.
4. Read only image headers to obtain size, spacing, origin, direction, and
   coordinate system.
5. Recreate `VolumeGeometry` and verify its fingerprint. Compare related image
   geometry using the tolerances from `VolumeGeometry.is_compatible_with`.
6. Confirm every required artifact type is present and available.
7. Create one `ArtifactInspection` per input artifact. Add deterministic
   `ValidationIssue` records; do not stop at the first issue.
8. Set `ready_for_next_stage` false if any ERROR issue exists. Return SUCCESS,
   because finding invalid case metadata is a successful validation operation.

## Error mapping

- `MISSING_ARTIFACT`: registry has no referenced artifact.
- `ARTIFACT_NOT_AVAILABLE`: status is not AVAILABLE or bytes cannot be read.
- `CHECKSUM_MISMATCH`: stored bytes do not match the immutable reference.
- `UNSUPPORTED_FORMAT`: header reader cannot identify an input image format.
- `CONTRACT_VIOLATION`: header geometry cannot construct `VolumeGeometry`.
- `TIMEOUT` / `DEPENDENCY_FAILED`: registry or object store failed.

Geometry disagreement and missing required types normally appear as validation
issues in a SUCCESS result, not a FAILED envelope.

## How to verify correctness

Unit fixtures:

1. CT plus same-geometry label map: ready is true.
2. Different size, spacing, origin, direction, and coordinate system: test each
   mismatch independently and assert ready is false.
3. Missing geometry header: assert a structured contract/header error.
4. Corrupted one-byte payload: checksum mismatch is detected.
5. Required type absent: `required_types_present=False`.
6. `require_same_geometry=False`: mismatch is reported but does not alone block.
7. Non-identity direction and anisotropic spacing: no false mismatch.

Golden test: compare parsed geometry and fingerprint with an independently read
header (for example SimpleITK versus the production header parser).

Existing commands:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools.CaseDataToolTests.test_inspect_happy_path_is_ready -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools.CaseDataToolTests.test_inspect_reports_geometry_mismatch -v
```

Acceptance criteria:

- all incompatibilities in the fixture matrix are found;
- no voxel data is loaded when checksum verification is disabled;
- repeated inspection is side-effect free;
- output order matches input order;
- URI and raw metadata do not appear in Agent/API projections.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| IC-01 | Contract snapshot and JSON round-trip pass. |
| IC-02 | Normal same-geometry case is ready and exhaustive inspections are correct. |
| IC-03 | Size/spacing/origin/direction/coordinate boundary matrix is classified correctly. |
| IC-04 | Missing/corrupt/checksum-invalid inputs return the documented issue/error. |
| IC-05 | Registry/object-store timeout and permission failure are deterministically injected and mapped. |
| IC-06 | Repeated identical request is side-effect free and returns equivalent results. |
| IC-07 | Header-only and checksum-enabled P50/P95 plus peak memory are recorded on the largest case. |
