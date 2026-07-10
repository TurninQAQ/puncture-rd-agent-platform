# Graph contract

This directory is the source of truth for workflow topology. The JSON files are
framework-neutral on purpose: the standard-library mock runtime and the future
LangGraph runtime must implement exactly the same nodes, ordering, branches,
and terminal semantics.

## Files

- `main_graph.json`: request parsing, RAG, task routing, verification, retry,
  missing-input handling, and report generation.
- `data_model_subgraph.json`: metadata inspection, optional MCS conversion,
  label validation, segmentation, output validation, and skin-surface creation.
- `planning_safety_subgraph.json`: artifact checks, constraint resolution,
  candidate generation, safety filtering, risk evaluation, and skin-penetration
  verification.

## JSON contract

Every graph contains:

```json
{
  "schema_version": "1.0",
  "graph_id": "unique_name",
  "start": "START",
  "end": "END",
  "max_steps": 40,
  "nodes": [],
  "edges": []
}
```

Node kinds:

- `action`: invokes one registered handler.
- `router`: invokes a deterministic handler, then selects the first matching
  outgoing edge.
- `subgraph`: loads another checked-in JSON graph. It cannot also have a
  handler.

Conditions are data, never executable Python. Supported operators are
`always`, `eq`, `ne`, `in`, `not_in`, `truthy`, `falsy`, `gt`, `gte`, `lt`,
`lte`, `eq_field`, and `lte_field`. Conditions can be combined with `all`,
`any`, and `not`. Edges are evaluated in file order. At most one `always`
fallback is allowed for each source, and it must be last.

## Change policy

Do not rename or remove a node merely to simplify an implementation. A graph
change requires all of the following:

1. Update this JSON source of truth.
2. Update the corresponding runtime handler.
3. Add or update branch tests in `tests/graph`.
4. Update the evaluation cases that assert required and forbidden nodes.
5. Record the semantic change in the implementation notes/commit message.

## Validation

Run:

```bash
python3 -m unittest discover -s tests/graph -p 'test_*.py' -v
```

The validator checks node uniqueness, edge references, subgraph paths,
condition syntax, fallback ordering, reachability, and an accessible `END`.
Tests also execute successful, missing-input, no-feasible-path, and transient
failure branches end to end.
