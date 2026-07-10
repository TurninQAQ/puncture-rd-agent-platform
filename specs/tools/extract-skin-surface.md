# `extract_skin_surface`

## Purpose

Convert a thick body/skin mask into the thin outer body boundary required for
entry-point search and later skin-penetration checks. The intended baseline is
binary erosion followed by set difference: `surface = original - eroded`.

## Fixed contract

- Request: `ExtractSkinSurfaceRequest`
- Result: `SkinSurfaceExtractionResult`
- Tool: `extract_skin_surface@1.0.0`
- Timeout: 30 seconds

`thickness_mm` is a physical thickness, not a fixed number of voxels.
`connectivity` affects component cleanup. The result must preserve the source
grid and geometry exactly.

## Implementation algorithm

1. Resolve the skin/body binary mask, verify geometry, and reject an empty mask.
2. Binarize only the configured skin label from the upstream schema; do not
   treat all nonzero organ labels as skin.
3. If `keep_largest_component`, label connected components using requested
   connectivity and retain the anatomically relevant outer body component.
   Report removed count; do not fill internal anatomy indiscriminately.
4. Convert requested physical thickness to an anisotropic structuring element.
   Preferred implementation uses a physical Euclidean distance transform:
   surface voxels have inward distance in `(0, thickness_mm]`. A morphology
   implementation must calculate per-axis radii from spacing and document its
   approximation.
5. For `EROSION_DIFFERENCE`, erode inward and subtract from the original binary
   mask. Preserve only the desired outer boundary; exclude cavity boundaries if
   the project definition requires the external surface. This may require
   background flood-fill from volume borders before erosion.
6. For `DISTANCE_BAND`, threshold the signed/internal distance field using the
   same external-surface rule.
7. Write an integer binary mask on the identical grid. Reopen and verify geometry,
   binary values, nonzero count, and subset relation `surface <= source`.
8. Register source lineage and algorithm parameters.

## Error mapping

- `EMPTY_SEGMENTATION`: no skin/body voxels.
- `GEOMETRY_MISMATCH`: missing/invalid image geometry.
- `LABEL_SCHEMA_ERROR`: requested skin value absent or input not discrete.
- `QUALITY_CHECK_FAILED`: output empty, not a subset, or geometry changed.
- `INVALID_ARGUMENT`: thickness unsupported for geometry.
- storage/dependency timeout errors as defined globally.

## How to verify correctness

Synthetic fixtures:

1. Solid axis-aligned cube at isotropic spacing. Compare surface voxel count to
   an independent erosion-difference oracle.
2. Same physical cube at anisotropic spacing. Physical surface thickness along
   each axis must approximate the request within one voxel of that axis; result
   must not become an arbitrary fixed two-voxel shell.
3. Hollow cube/body with an internal cavity. Assert whether the configured
   external-only policy excludes the inner cavity boundary.
4. Two disconnected bodies: verify largest-component behavior on/off.
5. Empty, one-voxel-thick, border-touching, and all-volume masks.
6. Rotated direction matrix: output geometry remains bit-for-bit identical.

Independent oracle: compare a C++ implementation with SimpleITK binary erosion
or signed Maurer distance map on small randomized masks. Require exact equality
for the same discrete kernel, or document boundary tolerance for physical EDT.

Existing tests:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools.SegmentationToolTests.test_skin_surface_preserves_geometry -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_segmentation_tools.SegmentationToolTests.test_empty_skin_mask_returns_error -v
```

Acceptance criteria:

- output is binary, nonempty for valid fixtures, and a strict subset of a thick
  source mask;
- output and source geometry fingerprints match exactly;
- thickness behavior is validated in millimetres on anisotropic data;
- internal-cavity policy is explicit and tested;
- repeated request produces the same checksum/artifact.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| ES-01 | Contract snapshot, binary output, geometry, lineage, and JSON result pass. |
| ES-02 | Normal thick-body mask matches the independent morphology/distance oracle. |
| ES-03 | Physical-thickness, anisotropy, cavity, connectivity, and one-voxel boundaries pass. |
| ES-04 | Empty/full/corrupt/wrong-label inputs map to documented findings/errors. |
| ES-05 | Native kernel/storage timeout, exception, and partial write are injected and cleaned up. |
| ES-06 | Same idempotency key returns one deterministic artifact/checksum. |
| ES-07 | Largest-mask P50/P95 and peak memory are recorded for each supported method. |
