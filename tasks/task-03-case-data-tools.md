# Task 03 — Implement case-data tools

## Goal

Replace the mocks for the following tools with company adapters while preserving
all v1 contracts:

1. `inspect_case_metadata`
2. `convert_mcs_to_nifti`
3. `validate_label_schema`

This task ends when real implementations pass contract, unit, golden,
failure-injection, idempotency, and benchmark checks. It does not implement the
Agent graph, RAG, model inference, or path planning.

## Context package to give an implementation model

Send only these files plus the relevant company SDK documentation:

```text
contracts/enums.py
contracts/geometry.py
contracts/artifacts.py
contracts/errors.py
contracts/common.py
contracts/domain.py
contracts/tool_inputs.py
contracts/tool_outputs.py
src/puncture_agent/tooling/catalog.py
src/puncture_agent/tooling/registry.py
src/puncture_agent/tooling/stubs.py
src/puncture_agent/mocks/tool_mocks.py
specs/tools/README.md
specs/tools/inspect-case-metadata.md
specs/tools/convert-mcs-to-nifti.md
specs/tools/validate-label-schema.md
tests/contract/test_tool_contracts.py
tests/tools/helpers.py
tests/tools/test_case_data_tools.py
```

Do not send raw company cases to an external model. Describe the proprietary MCS
SDK through a narrow adapter interface or perform that integration internally.

## Allowed implementation area

- add case-data adapters below `src/puncture_agent/tooling/implementations/`;
- bind them in a production registry factory;
- extend `tests/tools/test_case_data_tools.py` and add internal fixture resolvers;
- add optional dependency declarations only after the owner approves exact
  packages/versions.

Do not modify `contracts/**`, mock semantics, Agent state, API schemas, or other
tools. If a contract appears insufficient, stop and write a migration proposal;
do not improvise another field.

## Required ports

Keep storage and proprietary parsing replaceable. The implementation should
depend on interfaces equivalent to:

```python
class ArtifactStorePort(Protocol):
    def open_read(self, artifact: ArtifactRef) -> BinaryIO: ...
    def begin_atomic_write(self, case_id: str, artifact_type: ArtifactType): ...

class ArtifactRegistryPort(Protocol):
    def resolve(self, artifact_id: str) -> ArtifactRef: ...
    def commit(self, artifact: ArtifactRef, idempotency_key: str) -> ArtifactRef: ...

class McsReaderPort(Protocol):
    def read_header(self, stream: BinaryIO) -> McsHeader: ...
    def iter_segment_masks(self, stream: BinaryIO) -> Iterator[McsSegment]: ...
```

The concrete company SDK stays behind `McsReaderPort`. Tests use an in-memory
fake port; this prevents CI from requiring Mimics.

## Implementation sequence

1. Run all baseline tests and save output.
2. Implement read-only artifact resolution/header extraction/checksum streaming.
3. Implement `inspect_case_metadata`; add independent header-parser golden tests.
4. Implement MCS adapter, label merge, coordinate conversion, atomic NIfTI write,
   output re-open verification, lineage, and idempotency.
5. Implement streaming unique-value/schema validator.
6. Create production registry bindings while retaining `build_mock_registry()`.
7. Add one test for every documented error code and every acceptance ID.
8. Run complete tests and benchmark; produce a short verification report mapping
   acceptance ID to test name and result.

## Required test commands

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools -v
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/tools -p 'test_*.py' -v
```

Internal tests must additionally run normal, geometry-boundary, invalid-data,
dependency-failure, duplicate-idempotency, and benchmark suites described in the
three specifications.

## Completion checklist

- [ ] IC acceptance IDs all pass.
- [ ] MC acceptance IDs all pass with the approved MCS adapter.
- [ ] LS acceptance IDs all pass.
- [ ] Two independent readers agree on NIfTI geometry/histogram.
- [ ] No conversion leaves an AVAILABLE partial artifact.
- [ ] Same idempotency key commits one artifact.
- [ ] Contract snapshots are unchanged.
- [ ] Verification report includes hardware, libraries, fixture IDs, tolerances,
      P50/P95, and every test command.

## Prompt to hand to another model

> Implement Task 03 only. Read every supplied contract and tool specification
> before editing. Keep all public dataclass fields and enum wire values unchanged.
> First summarize the required behavior and propose concrete adapter ports/tests.
> Then implement one tool at a time, running its normal, boundary, invalid,
> dependency, idempotency, and performance checks. Do not invent an MCS parser;
> integrate only through the provided company adapter. Finish with an acceptance
> ID → test → result table and list every modified file.
