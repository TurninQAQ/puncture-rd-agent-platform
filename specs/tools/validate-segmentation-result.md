# `validate_segmentation_result`

## Purpose

Perform deterministic post-inference quality control before a segmentation can
enter skin extraction or path planning. It detects geometry disagreement,
missing/empty labels, implausible volume, excessive connected components, and
common mask corruption. It reports findings; it does not silently repair them.

## Fixed contract

- Request: `ValidateSegmentationResultRequest`
- Result: `SegmentationValidationResult`
- Tool: `validate_segmentation_result@1.0.0`
- Timeout: 30 seconds

Thresholds are keyed by exact canonical label name. A missing threshold means
schema presence is still checked but no volume/component bound is applied.

## Implementation algorithm

1. Resolve and validate CT and segmentation artifacts.
2. Compare size, spacing, origin, direction, and coordinate system. Do not treat
   equal array shapes as equal geometry.
3. Compute observed label set and one-pass voxel count per label.
4. For each expected label, calculate physical volume using voxel spacing,
   connected-component count with a documented connectivity, and border touch.
5. Apply `LabelQualityThreshold`: minimum voxels/volume, optional maximum volume,
   and maximum components. Required absent labels are errors. Optional absent
   labels may be INFO/WARNING according to the project rule version.
6. Add checks appropriate to the label role without changing v1 output fields:
   empty mask, entirely full-volume mask, tiny isolated islands, non-integer
   values, and body/skin labels unexpectedly failing to reach the body boundary.
7. Return one `LabelValidationResult` per expected label in schema order, all
   issues, and a deterministic `recommended_action`:
   - no ERROR: CONTINUE;
   - transient/read issue: RETRY;
   - model/schema mismatch: MANUAL_REVIEW or STOP according to policy.

## Error mapping

Use FAILED only if validation cannot run: missing/corrupt artifact,
`UNSUPPORTED_FORMAT`, `CHECKSUM_MISMATCH`, `TIMEOUT`, or dependency failure.
Quality problems appear in `valid=False` with issues such as:

- `GEOMETRY_MISMATCH`
- `REQUIRED_LABEL_MISSING`
- `VOXEL_COUNT_TOO_LOW`
- `VOLUME_OUT_OF_RANGE`
- `TOO_MANY_COMPONENTS`
- `MASK_IMPLAUSIBLY_FULL`

## How to verify correctness

Synthetic masks provide a stronger oracle than subjective images:

1. One cuboid per label: analytically verify voxel count and physical volume.
2. Create N disconnected components and assert the exact component count for the
   configured connectivity.
3. Put a structure on each of the six volume faces and test `touches_volume_border`.
4. Remove each required label; test issue and recommendation.
5. Vary spacing while keeping voxel count fixed; physical volume must change.
6. Keep the same array but shift origin/rotate direction; geometry mismatch must
   be detected.
7. All-zero, all-one, one-voxel island, and unknown-value masks.

Independent oracle: compare counts/components/volumes with a trusted
SimpleITK/SciPy implementation on randomized small volumes. Require exact voxel
counts/components and floating volume tolerance below `1e-6 ml` for identical
spacing.

Existing test:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools.SegmentationToolTests.test_segmentation_validation_detects_required_missing_label -v
```

Acceptance criteria:

- all analytic synthetic expectations pass exactly;
- every expected label has exactly one result;
- issues and action are deterministic across runs;
- the validator never modifies the input artifact;
- normal-case runtime/memory budget is measured on the largest supported volume.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| VS-01 | Contract snapshot and one ordered result per expected label pass. |
| VS-02 | Normal analytic/golden masks return correct counts, volumes, components, and valid state. |
| VS-03 | Exact min/max volume, voxel, component, border, and geometry boundaries pass. |
| VS-04 | Empty/full/fractional/corrupt/missing-label masks produce documented findings/errors. |
| VS-05 | Storage/reader timeout, checksum, and permission failures are injected and mapped. |
| VS-06 | Repeated validation is immutable and result-equivalent. |
| VS-07 | Largest-volume P50/P95 and peak working memory are recorded. |
