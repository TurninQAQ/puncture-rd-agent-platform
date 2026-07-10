# `generate_candidate_paths`

## Purpose

Generate entry-to-target line candidates that satisfy geometric needle length
and insertion-angle constraints. Dangerous-organ collision is intentionally
evaluated by the following `evaluate_path_safety` tool, keeping candidate
generation and safety verification independently testable.

## Fixed contract

- Request: `GenerateCandidatePathsRequest`
- Result: `CandidatePathGenerationResult`
- Tool: `generate_candidate_paths@1.0.0`
- Timeout: 60 seconds

All output points are world millimetres. `target_point_world_mm`, when present,
is authoritative after an in-bounds check; otherwise derive a deterministic
target (for example centroid or configured representative point) from
`target_artifact`. Record the target-selection rule in the planner version.

## Implementation algorithm

1. Resolve CT, skin surface, target, and optional lesion. Require compatible
   geometry and an in-volume target.
2. Extract eligible external skin voxels and convert their voxel centres to world
   coordinates with full origin/direction/spacing transform.
3. Sample entry points deterministically at approximately
   `entry_sampling_step_mm`. Uniform voxel skipping is incorrect on anisotropic
   data; use physical-space downsampling, mesh sampling, or farthest-point/grid
   sampling.
4. For each entry, form the segment to target. Calculate Euclidean physical
   length and reject above `max_needle_length_mm`.
5. Calculate insertion angle according to the exact `angle_reference`:
   - LOCAL_SURFACE_NORMAL: estimate an outward normal from signed distance
     gradient or a smoothed surface mesh and use the acute line-normal angle;
   - AXIAL_PLANE_NORMAL: compare with the physical axial normal after direction
     transform;
   - NEEDLE_DEVICE_AXIS: require the calibrated axis from the adapter context or
     return INVALID_ARGUMENT if unavailable.
6. Reject above `max_insertion_angle_deg`. Avoid unstable normals by smoothing or
   neighborhood fitting; flag invalid boundary points instead of using zero
   vectors.
7. Apply deterministic coarse ranking/diversity only: shorter length, preferred
   angle, and spatial separation between entry points. Do not claim organ safety.
8. Return up to `max_candidates`, stable candidate IDs derived from case,
   planner version, entry, and target, plus rejection counts and elapsed time.
9. Optionally create path artifacts, but `path_artifact_id` must refer to a
   registered immutable artifact if populated.

## Error mapping

- `REQUIRED_LABEL_MISSING`: skin or target mask absent/empty.
- `GEOMETRY_MISMATCH`: image grids incompatible.
- `TARGET_OUT_OF_BOUNDS`: target world point cannot map inside CT.
- `INVALID_ARGUMENT`: unsupported angle reference/calibration.
- `NO_CANDIDATE_PATH`: every entry fails length/angle or no valid surface point.
- `TIMEOUT`, `DEPENDENCY_FAILED`: infrastructure failures.

## How to verify correctness

Analytic synthetic volumes:

1. Planar skin at `x=0`, target at `(d,0,0)`, normal `(1,0,0)`: length is `d`
   and local-normal angle is zero.
2. Known 3-4-5 triangle: length and angle have closed-form expectations.
3. Spherical surface with centre target: radial paths should have near-zero
   local-normal angle.
4. Anisotropic spacing, shifted origin, and rotated direction: the physical
   candidates must match an equivalent identity-grid case after transform.
5. Target outside volume, empty skin, maximum length just below/equal/above
   candidate length, and angle boundary just below/equal/above threshold.
6. Run twice and assert identical candidate IDs/order.

Independent oracle: brute-force all eligible entry voxels on small synthetic
volumes and compare the optimized generator's accepted set before downsampling.
Verify all returned paths independently satisfy constraints with tolerance
`1e-3 mm` and `1e-3 degree`.

Existing tests:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools.PlanningToolTests.test_candidate_generation_obeys_count_length_and_angle -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools.PlanningToolTests.test_no_candidate_has_stable_error -v
```

Acceptance criteria:

- zero returned candidate violates length or angle;
- coordinate-transform fixtures pass;
- sampling covers the external surface to the documented physical tolerance;
- order and IDs are deterministic;
- the function makes no unverified claim of organ safety.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| GP-01 | Contract snapshot, finite typed candidates, and JSON round-trip pass. |
| GP-02 | Normal analytic plane/sphere fixtures produce expected candidates. |
| GP-03 | Exact length/angle/target-boundary, anisotropy, direction, and sampling boundaries pass. |
| GP-04 | Empty skin/target, out-of-bounds target, invalid calibration, and no-path cases map correctly. |
| GP-05 | Artifact/native planner timeout, exception, and unavailable dependency are injected and mapped. |
| GP-06 | Repeated request yields identical IDs/order and one optional path artifact per ID. |
| GP-07 | P50/P95 and peak memory are recorded by sampled-entry/candidate count. |
