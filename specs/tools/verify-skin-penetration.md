# `verify_skin_penetration`

## Purpose

Determine whether the virtual needle segment from planned entry to current tip
actually traverses skin voxels. This avoids the unreliable shortcut of inspecting
only the tip label, because both inside and outside may be background label zero.
If no skin crossing is found after meaningful insertion depth, report suspected
slip/failed penetration.

## Fixed contract

- Request: `VerifySkinPenetrationRequest`
- Result: `SkinPenetrationResult`
- Tool: `verify_skin_penetration@1.0.0`
- Timeout: 10 seconds

Sampling density is expressed as a fraction of one voxel through
`sampling_step_voxel`; v1 accepts `(0, 1]`, with 0.5 as the intended baseline.
The traversal still must account for anisotropic spacing and direction.

## Implementation algorithm

1. Resolve the skin mask and validate geometry/discrete label value.
2. Transform planned entry and current tip from world to continuous image index.
   Retain the world segment length as separate evidence.
3. Traverse the segment using 3D DDA/Amanatides-Woo supercover or dense parametric
   sampling whose step is at most `sampling_step_voxel` in index space. A
   supercover traversal is preferred because it enumerates every touched voxel.
4. For each in-volume sample/voxel in order, read `skin_label_value`. Treat
   out-of-volume portions explicitly; do not clamp them to a border voxel.
5. If any sample traverses skin, return PENETRATED, the first crossing index and
   a world crossing point. For a thick skin mask, the first outside-to-skin
   transition is the crossing evidence.
6. If no crossing and `insertion_depth_mm >= min_depth_for_slip_mm`, return
   SUSPECTED_SLIP. If below threshold, return NOT_PENETRATED.
7. If geometry or segment location makes the state indeterminate, return FAILED
   with `SKIN_PENETRATION_UNDETERMINED`; do not guess UNKNOWN in a success result
   unless the business policy explicitly consumes uncertainty.
8. Report sample count, physical path length, insertion depth, and a concise
   evidence string suitable for trace—not a raw list of voxels.

## Error mapping

- `MISSING_ARTIFACT` / `ARTIFACT_NOT_AVAILABLE`: skin mask unavailable.
- `GEOMETRY_MISMATCH`: invalid/missing geometry.
- `REQUIRED_LABEL_MISSING`: skin value absent.
- `INVALID_ARGUMENT`: segment coordinates or sampling parameters invalid.
- `SKIN_PENETRATION_UNDETERMINED`: traversal cannot establish state safely.
- `TIMEOUT`, `DEPENDENCY_FAILED`: infrastructure failure.

## How to verify correctness

Analytic fixtures:

1. One-voxel planar skin. Segments crossing orthogonally, diagonally, exactly at
   a voxel corner, and running along the boundary must not tunnel through.
2. Segment wholly outside and wholly inside without crossing.
3. No crossing with insertion depth just below/equal/above slip threshold.
4. Thick shell: report first traversal, not only the tip label.
5. Anisotropic spacing, shifted origin, and rotated direction with equivalent
   physical segment.
6. Entry/tip outside volume, zero-length segment, absent skin label, and empty mask.
7. Reverse traversal direction; penetration truth remains equal while first
   crossing appropriately changes.

Independent oracle: very dense parametric sampling at at most 0.05 voxel plus a
voxel-supercover reference on small volumes. The optimized method must have zero
false negatives on all randomized thin-surface fixtures.

Existing test:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools.PlanningToolTests.test_skin_penetration_and_slip_are_distinguishable -v
```

Acceptance criteria:

- zero known crossing is reported as not penetrated/slip;
- threshold equality semantics are tested (`>=` means suspected slip);
- non-identity geometry tests pass;
- evidence contains no raw patient/voxel dump;
- output is deterministic for identical inputs.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| SP-01 | Contract snapshot, finite evidence summary, and JSON round-trip pass. |
| SP-02 | Normal crossing/non-crossing/slip fixtures match supercover and dense oracle. |
| SP-03 | Corner/edge, thick shell, threshold equality, direction, anisotropy, and reverse boundaries pass. |
| SP-04 | Empty/missing label, zero/invalid segment, outside-volume, and corrupt geometry map correctly. |
| SP-05 | Native traversal/storage timeout and exception are injected and fail closed. |
| SP-06 | Repeated read-only request returns an equivalent deterministic result. |
| SP-07 | Target-loop P50/P95, sample count scaling, and peak memory are recorded. |
| SP-08 | Randomized thin-surface suite reports zero false-negative crossings. |
