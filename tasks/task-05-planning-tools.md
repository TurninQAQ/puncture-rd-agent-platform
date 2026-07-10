# Task 05 — Implement planning and safety tools

## Goal

Replace the mocks for:

1. `generate_candidate_paths`
2. `evaluate_path_safety`
3. `evaluate_intraoperative_risk`
4. `verify_skin_penetration`

Integrate the existing path planning, mask morphology, distance, and 3D ray
tracing algorithms without allowing the LLM to calculate safety conclusions.

## Context package

Provide only:

```text
contracts/**
src/puncture_agent/tooling/catalog.py
src/puncture_agent/tooling/registry.py
src/puncture_agent/tooling/stubs.py
src/puncture_agent/mocks/tool_mocks.py
specs/tools/README.md
specs/tools/generate-candidate-paths.md
specs/tools/evaluate-path-safety.md
specs/tools/evaluate-intraoperative-risk.md
specs/tools/verify-skin-penetration.md
tests/contract/test_tool_contracts.py
tests/tools/helpers.py
tests/tools/test_planning_tools.py
```

Also provide sanitized definitions of angle reference, required danger labels,
warning/stop margins, path ranking, vessel erosion, and skin-slip threshold for
one immutable rule version. Do not ask a model to invent clinical thresholds.

## Allowed implementation area

- planning/safety adapters under `src/puncture_agent/tooling/implementations/`;
- production registry binding;
- planning tool and internal golden tests;
- owner-approved C++/pybind11/gRPC build configuration.

Do not edit contracts, safety thresholds, Agent graph, report wording, or other
tools. Any ambiguous safety rule blocks implementation until the rule owner
resolves it.

## Required adapter boundary

Keep native geometry independently testable:

```python
class PlanningKernelPort(Protocol):
    def generate(self, geometry, skin, target, constraints) -> NativeCandidates: ...
    def path_clearance(self, geometry, path, danger_masks, rules) -> NativeClearance: ...
    def tip_risk(self, geometry, tip, masks, rules) -> NativeRisk: ...
    def traverse_skin(self, geometry, start, end, skin_value, step) -> NativeTraversal: ...
```

The Python adapter converts typed contracts to native structs, validates every
returned finite value/index/artifact, and maps errors. Native and Python tests
must share synthetic geometry fixtures but use independent expected oracles.

## Implementation sequence

1. Freeze coordinate/angle/safety rule definitions and baseline tests.
2. Implement world/index transforms and candidate generation; prove constraints
   on analytic fixtures before using real masks.
3. Implement full-path safety envelope/physical EDT and a slow independent
   brute-force oracle. Require zero false-safe cases.
4. Implement tip risk, warning/stop precedence, vessel core, and lung state.
5. Implement skin traversal with supercover/DDA and dense-reference oracle.
6. Add geometry mismatch, missing mask, empty mask, boundary equality, timeout,
   and native-exception tests.
7. Verify deterministic IDs/order and idempotent optional path-artifact commit.
8. Run internal regression cases and target-hardware latency benchmark. Produce
   acceptance ID → test mapping.

## Required test commands

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_planning_tools -v
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/tools -p 'test_*.py' -v
```

Run the native C++ suite/CTest and the independent Python-oracle suite as
separate jobs. Safety acceptance cannot rely solely on checked-in mock tests.

## Completion checklist

- [ ] GP acceptance IDs pass for analytic and internal entry/target fixtures.
- [ ] PS acceptance IDs pass with zero false-safe golden paths.
- [ ] IR acceptance IDs pass at every warning/stop/vessel/lung boundary.
- [ ] SP acceptance IDs pass with zero false-negative thin-skin crossings.
- [ ] World/index transforms pass identity, translation, anisotropy, and rotation.
- [ ] Required missing danger masks fail closed.
- [ ] Candidate/path/risk output is deterministic and finite.
- [ ] P50/P95 is recorded for configured candidate count and intraoperative loop.
- [ ] Contracts, thresholds, and mocks remain unchanged.

## Prompt to hand to another model

> Implement Task 05 only. Safety must fail closed. Read all four specifications
> and keep the fixed contracts unchanged. Begin with analytic coordinate tests,
> then implement native adapters and compare each optimized kernel with an
> independent slow oracle. Do not invent warning/stop/angle/vessel thresholds.
> Cover normal, exact boundary, invalid/missing artifact, dependency/native
> failure, retry/idempotency, deterministic ordering, and performance cases.
> Finish with an acceptance ID → test → result table and explicitly report false
> safe/false negative counts.
