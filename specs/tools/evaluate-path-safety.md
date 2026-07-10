# `evaluate_path_safety`

## Purpose

Evaluate the entire candidate needle segment, not only its tip, against dangerous
structures. The needle is expanded by its physical radius and each structure's
warning/stop margin to form safety envelopes. Paths intersecting a stop envelope
are rejected.

## Fixed contract

- Request: `EvaluatePathSafetyRequest`
- Result: `PathSafetyEvaluationResult`
- Tool: `evaluate_path_safety@1.0.0`
- Timeout: 60 seconds

`DangerMaskSpec.safety_margin.warning_mm` must be at least `stop_mm`. A required
unavailable danger mask makes the whole call FAILED; it must never be interpreted
as an empty safe mask.

## Recommended implementation

Use a physical Euclidean distance transform (EDT) per danger mask:

1. Resolve CT and every required mask; verify same geometry and discrete values.
2. Compute/cache a physical EDT to each danger structure keyed by artifact
   checksum and algorithm version. The EDT must honor anisotropic spacing.
3. For every candidate, verify endpoints and sample the full line in world space
   at no more than `path_sampling_step_mm`. A voxel traversal/supercover line is
   preferable when it guarantees no skipped intersected voxel.
4. At each path sample, interpolate/query physical distance conservatively.
   The effective clearance is `distance_to_structure - needle_radius_mm`.
5. Stop intersection occurs when clearance is at or below `stop_mm`; warning
   intersection occurs at or below `warning_mm`. Direct mask intersection is
   necessarily a stop condition.
6. Record per-structure minimum clearance and both booleans. The candidate
   minimum is the minimum across required structures.
7. Set disposition:
   - any stop: REJECTED;
   - warning and `reject_warning_intersection=True`: REJECTED;
   - warning otherwise: ACCEPTED_WITH_WARNING;
   - otherwise ACCEPTED.
8. Choose `safest_candidate_id` only among non-rejected paths, using largest
   minimum clearance followed by deterministic length/angle/ID tie-breakers.
9. Optionally generate a path-envelope mask for visualization, but the numerical
   result must not depend on display rasterization.

Alternative morphology implementation: rasterize a supercover path and dilate by
`needle_radius + margin` using a physical structuring element. It must be checked
against the EDT oracle and cannot use a constant voxel radius across anisotropic
axes.

## Error mapping

- `REQUIRED_DANGER_MASK_MISSING`: any required mask missing/unavailable.
- `GEOMETRY_MISMATCH`: CT and masks differ physically.
- `INVALID_ARGUMENT`: candidate endpoint invalid, radius/step unsupported.
- `SAFETY_CHECK_FAILED`: distance/envelope computation produced invalid data.
- `NO_FEASIBLE_PATH`: optional orchestration policy after all candidates reject;
  v1 normally returns SUCCESS with empty accepted IDs so all assessments remain
  inspectable.
- `TIMEOUT`, `DEPENDENCY_FAILED`: infrastructure failures.

## How to verify correctness

Analytic fixtures:

1. Straight path parallel to a planar danger mask at known distances 0, stop,
   warning, and warning+epsilon. Assert boundary inclusivity.
2. Single spherical danger region and line with known closest approach. Compare
   clearance analytically after needle radius subtraction.
3. A one-voxel obstacle between coarse sample points. Ensure supercover/adaptive
   logic detects it; this prevents tunneling.
4. Anisotropic spacing and rotated direction. Physical result must be invariant
   under an equivalent coordinate transform.
5. Multiple structures where different path sections define different minima.
6. Required unavailable versus optional unavailable masks.
7. All paths rejected and tie between two safe paths.

Independent oracle: on small volumes, densely sample at at most one tenth of the
minimum spacing and compare with brute-force distance to every danger voxel.
Production clearance must agree within a conservative tolerance and may never
overestimate safety beyond that tolerance.

Existing test:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools.PlanningToolTests.test_safety_rejects_injected_collision -v
```

Acceptance criteria:

- zero known stop-envelope intersection is accepted;
- false-safe count is zero in the internal safety golden set;
- boundary semantics are exact and documented;
- every candidate receives exactly one assessment;
- caches are invalidated by artifact checksum/algorithm version;
- P95 for configured candidate count/volume is recorded.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| PS-01 | Contract snapshot and exactly one typed assessment per candidate pass. |
| PS-02 | Normal analytic clearances match the independent oracle within conservative tolerance. |
| PS-03 | Stop/warning equality, thin obstacle, radius, anisotropy, and tie-break boundaries pass. |
| PS-04 | Missing mask, mismatched geometry, invalid path/radius, and all-rejected behavior match policy. |
| PS-05 | Distance/native/storage timeout and corrupt cache failures are injected and fail closed. |
| PS-06 | Repeated request is deterministic; cache/idempotent optional artifacts are verified. |
| PS-07 | P50/P95 and peak memory are recorded by candidates, masks, and volume size. |
| PS-08 | Internal safety golden set reports zero false-safe paths. |
