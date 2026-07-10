# `convert_mcs_to_nifti`

## Purpose

Convert a Mimics MCS segmentation export into a single nnU-Net-compatible
NIfTI label map while preserving physical alignment with the reference CT and
applying an explicit label mapping. This tool must use the company-approved
MCS reader/export API; do not reverse-engineer an undocumented binary format in
generated code.

## Fixed contract

- Request: `ConvertMcsToNiftiRequest`
- Result: `McsToNiftiResult`
- Tool: `convert_mcs_to_nifti@1.0.0`
- Timeout: 60 seconds

The mapping is authoritative. Unmapped non-background MCS segments are an
error; two sources cannot map to the same target value in v1. `output_dtype`
must be `uint8`, `uint16`, or `int16` and must represent the largest target
value without overflow.

## Implementation algorithm

1. Resolve and verify both input artifacts. Acquire an idempotency lock.
2. Read reference CT header and MCS segment metadata. Normalize source segment
   names only for matching (trim whitespace and documented case rules); retain
   original names in the audit record.
3. Validate every `LabelMappingEntry`: source exists, values are unique, target
   fits dtype, and background behavior is explicit.
4. Export/read every source binary mask through the approved adapter.
5. Transform mask indices to the reference CT grid using MCS physical geometry.
   Nearest-neighbor interpolation is the only permitted interpolation for
   labels. If the business rule forbids resampling, reject nonmatching geometry
   instead; make this adapter policy explicit and test it.
6. Merge masks into one label volume. Treat overlapping source masks as a
   deterministic error unless an approved precedence table exists.
7. Write NIfTI with the reference size, spacing, origin/direction equivalent,
   and requested LPS/RAS convention. Account for RAS/LPS sign conversion rather
   than editing only the metadata string.
8. Reopen the written file using an independent library, recompute observed
   values and geometry, and verify physical corner coordinates against CT.
9. Atomically commit the output, checksum it, and register lineage to MCS and CT.
10. Return mapping voxel counts and nonzero total. On failure, remove the
    temporary object and mark the registry record INVALID.

## Error mapping

- `UNSUPPORTED_FORMAT`: MCS adapter cannot read the source.
- `LABEL_SCHEMA_ERROR`: source missing, overlap unresolved, or mapping ambiguous.
- `UNKNOWN_LABEL_VALUE`: non-background source is not mapped.
- `GEOMETRY_MISMATCH`: physical alignment cannot be safely established.
- `INVALID_ARGUMENT`: dtype overflow or invalid conversion policy.
- `CHECKSUM_MISMATCH`: input/output integrity check failed.
- `TIMEOUT`, `DEPENDENCY_FAILED`, `PERMISSION_DENIED`: adapter/storage failures.

## How to verify correctness

Create synthetic MCS fixtures through the same approved export API, not by
fabricating binary bytes:

1. Three non-overlapping cuboids with known source/target values. Assert exact
   voxel counts and label values after NIfTI reload.
2. Anisotropic spacing, shifted origin, and rotated direction. Transform the
   eight nonzero-region corners to world space and compare within 0.01 mm.
3. LPS to RAS conversion. Assert world locations are preserved after coordinate
   conversion.
4. Unknown source, duplicate target, overlapping segments, dtype overflow, and
   corrupted MCS. Assert stable errors and no AVAILABLE output artifact.
5. Retry identical idempotency key. Assert one output artifact ID/checksum.

Independent oracle: load output with both SimpleITK and nibabel, accounting for
their axis conventions, and compare label histogram plus physical coordinates.

Existing tests:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools.CaseDataToolTests.test_conversion_preserves_reference_geometry_and_mapping -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools.CaseDataToolTests.test_conversion_injected_parser_failure_has_stable_error -v
```

Acceptance criteria:

- output observed label set exactly equals mapped targets plus background;
- voxel counts agree with the approved exporter/reference within the declared
  resampling policy;
- physical reference points agree within 0.01 mm;
- output checksum and lineage survive retry;
- no partially written output is consumable.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| MC-01 | Contract snapshot and output reload/JSON serialization pass. |
| MC-02 | Normal mapped fixture has exact label histogram and physical alignment. |
| MC-03 | LPS/RAS, anisotropic, rotated, overlap, and dtype limit boundaries match policy. |
| MC-04 | Unknown source, corrupt MCS, overlap, and overflow map to stable errors. |
| MC-05 | MCS SDK/storage timeout, permission, and partial-write failures are injected and cleaned up. |
| MC-06 | Same idempotency key yields one artifact ID/checksum/lineage record. |
| MC-07 | P50/P95 conversion time and peak memory are recorded for the largest supported file. |
