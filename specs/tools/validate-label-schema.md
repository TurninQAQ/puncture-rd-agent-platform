# `validate_label_schema`

## Purpose

Check a label map against a versioned label name/value contract before model
training, validation, or planning. This is a read-only exhaustive validator; it
does not rewrite label values.

## Fixed contract

- Request: `ValidateLabelSchemaRequest`
- Result: `LabelSchemaValidationResult`
- Tool: `validate_label_schema@1.0.0`
- Timeout: 15 seconds

`expected_labels` is the resolved schema version supplied by the Agent/RAG
workflow. Required labels are evaluated only when
`require_all_required_labels=True`. Unknown values are issues only when
`allow_unknown_values=False`, but are always listed in the result.

## Implementation algorithm

1. Resolve the label map, verify availability/checksum, and read integer dtype.
2. Stream/chunk the volume to compute the exact unique-value set without
   converting the full volume to float.
3. Reject NaN, infinity, fractional values, negative values not present in the
   expected schema, and integer overflow as schema errors.
4. Compare observed values with `LabelDefinition.value`. Record required labels
   absent from the volume and values absent from the schema.
5. Optionally compare label-name metadata from the artifact with canonical names
   and aliases. A name/value disagreement is a `LABEL_NAME_VALUE_MISMATCH` issue.
6. Return all findings in a deterministic order: missing labels follow expected
   schema order; unknown values are numerically sorted.

## Error mapping

- `MISSING_ARTIFACT` / `ARTIFACT_NOT_AVAILABLE`: cannot inspect input.
- `UNSUPPORTED_FORMAT`: not a readable discrete label image.
- `LABEL_SCHEMA_ERROR`: fractional/NaN/invalid encoded labels.
- `CHECKSUM_MISMATCH`, `TIMEOUT`, `DEPENDENCY_FAILED`: infrastructure errors.

Missing and unknown labels are represented in a successful validation result,
not a FAILED envelope.

## How to verify correctness

Fixture matrix:

1. Exact schema including background: `valid=True`.
2. Each required label removed separately: correct missing name is returned.
3. Optional label removed: remains valid.
4. Unknown value 65535: reported exactly, with no dtype truncation.
5. `allow_unknown_values=True`: unknown remains listed but does not invalidate.
6. `require_all_required_labels=False`: missing remains listed but does not by
   itself invalidate.
7. Sparse high-valued labels and a large volume: memory remains bounded.
8. Float label file containing 1.5 or NaN: structured failure.

For property testing, generate random integer arrays and compare the production
unique-value result to a straightforward `set(flattened_values)` reference.

Existing test:

```bash
PYTHONPATH=.:src:tests python3 -m unittest tests.tools.test_case_data_tools.CaseDataToolTests.test_schema_validation_detects_missing_and_unknown_labels -v
```

Acceptance criteria:

- exact match with reference unique values on all generated fixtures;
- stable deterministic ordering;
- maximum working memory is documented and does not scale as a second full-size
  copy of the image;
- all combinations of the two request flags are tested.

### Acceptance IDs

| ID | Required evidence |
|---|---|
| LS-01 | Contract snapshot and deterministic JSON result pass. |
| LS-02 | Exact normal schema is valid and label ordering is stable. |
| LS-03 | Optional/required/unknown/high-value and flag-combination boundaries pass. |
| LS-04 | Fractional, NaN, corrupt, and unsupported inputs map to documented errors. |
| LS-05 | Registry/storage timeout, unavailable object, and checksum failure are injected and mapped. |
| LS-06 | Repeated read-only request returns an equivalent result with no side effects. |
| LS-07 | Largest-volume P50/P95 and bounded peak memory are recorded. |
