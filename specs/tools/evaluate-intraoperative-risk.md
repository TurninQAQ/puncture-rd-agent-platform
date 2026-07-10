# `evaluate_intraoperative_risk`

## Purpose

Evaluate the current physical needle-tip position during insertion against
warning/stop regions for bone, heart, bronchus, and other masks; detect large
vessel core penetration and whether the tip is in lung. The output is a set of
explicit flags used by the robot safety workflow and report generator.

## Fixed contract

- Request: `EvaluateIntraoperativeRiskRequest`
- Result: `IntraoperativeRiskResult`
- Tool: `evaluate_intraoperative_risk@1.0.0`
- Timeout: 10 seconds

This function evaluates current tip state. Full candidate-trajectory safety is
handled separately. `risk_rule_version` identifies the immutable configuration
that maps margins and flag precedence.

## Implementation algorithm

1. Resolve CT, danger, optional lung, and optional skin artifacts; validate
   geometry and required masks.
2. Transform `current_tip_world_mm` and planned entry to continuous voxel
   coordinates using the full inverse image transform. Reject an out-of-volume
   tip unless the rule version explicitly models outside-body state.
3. For heart/bone/bronchus/other structures, use cached physical distance fields
   or precomputed warning/stop masks. Assign STOP at/directly inside the stop
   boundary, WARNING inside warning but outside stop, otherwise SAFE.
4. For large vessels, create or load a core mask eroded inward by
   `vessel_core_erosion_mm` in physical space. If the tip remains inside the core,
   set `large_vessel_penetration=True` and a STOP flag. Clarify that this is a
   size/core heuristic, not semantic vessel classification.
5. For lung, query the configured lung label at the continuous tip position
   using nearest-neighbor semantics and set `needle_in_lung`; absence produces
   `None`, never false.
6. Add one `RiskFlag` for every evaluated required structure with reason code,
   distance, and evidence artifact ID. Compute overall precedence
   `STOP > WARNING > SAFE > UNKNOWN`.
7. Set manual review for warning, stop, contradictory/missing optional evidence,
   or rule-specified uncertainty.
8. Do not infer skin penetration here; that tool's separate result is joined by
   the LangGraph verifier.

## Error mapping

- `REQUIRED_DANGER_MASK_MISSING`: required mask absent/unavailable.
- `GEOMETRY_MISMATCH`: incompatible grids.
- `TARGET_OUT_OF_BOUNDS` or `INVALID_ARGUMENT`: current tip cannot be evaluated.
- `RISK_EVALUATION_FAILED`: corrupt distance field/rule configuration.
- `TIMEOUT`, `DEPENDENCY_FAILED`: infrastructure failure.

## How to verify correctness

Synthetic fixtures:

1. Concentric spherical stop/warning regions. Place tip just outside/on/inside
   each boundary and assert exact level.
2. Rotated/anisotropic volume with physically equivalent tip coordinates.
3. Vessel cylinders of different radius and erosion distance. Verify only the
   surviving core sets large-vessel penetration.
4. Tip in/out/on lung mask boundary and lung mask absent (`None`).
5. Multiple structures with simultaneous warning and stop; overall must be STOP
   and all individual flags must remain present.
6. Tip outside CT and required mask unavailable.
7. Rule version mismatch or malformed margin configuration.

Independent oracle: direct nearest-neighbor mask query plus brute-force physical
distance on small synthetic volumes. For erosion, compare the optimized kernel
to a trusted physical-distance threshold implementation.

Existing test:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools.PlanningToolTests.test_intraoperative_risk_enters_stop_zone -v
```

Acceptance criteria:

- no STOP fixture produces SAFE/WARNING;
- overall precedence and optional `None` semantics are tested;
- world-to-voxel transforms pass non-identity fixtures;
- all flags contain evidence and stable reason codes;
- P95 meets the configured intraoperative loop budget on target hardware.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| IR-01 | Contract snapshot, stable flags/reason codes, and JSON round-trip pass. |
| IR-02 | Normal safe/warning/stop/vessel/lung fixtures match the independent oracle. |
| IR-03 | Boundary equality, erosion radius, outside-volume, anisotropy, and precedence pass. |
| IR-04 | Missing required mask, invalid rule, corrupt field, and invalid tip map correctly. |
| IR-05 | Native/distance/storage timeout and service exception are injected and fail closed. |
| IR-06 | Repeated read-only evaluation returns an equivalent deterministic result. |
| IR-07 | Target-loop P50/P95, jitter, and peak memory are recorded. |
| IR-08 | Internal risk golden set contains zero STOP-to-safe downgrades. |
