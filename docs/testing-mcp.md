# Testing the MCP Tool Module

## 1. Local completion gate

Module 3 is accepted on the current demo host when all of the following pass:

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/tools -p 'test_*.py' -v
PYTHONPATH=.:src:tests python3 -m unittest discover -s tests/mcp -p 'test_*.py' -v
python3 examples/local_mcp_demo.py
python3 -m compileall -q contracts src tests examples
python3 run_tests.py
git diff --check
```

Release `v0.5.0` evidence on the Python 3.10 development host:

- full suite: 428 tests executed;
- 421 passed;
- 7 existing model-service tests skipped by explicit environment gates;
- all ten local adapter calls returned `SUCCESS`;
- stdio subprocess initialization and tool discovery passed;
- no network, GPU, OpenSearch, official MCP SDK or company algorithm was
  required.

## 2. Test layers

### 2.1 Frozen contract tests

File: `tests/contract/test_tool_contracts.py`

These tests must remain unchanged when a backend is replaced.  They verify:

- exactly ten tool names;
- exact request field order;
- exact result field order;
- response JSON serialization;
- request/trace/tool identity;
- public artifact projection without URI leakage;
- request/result catalog consistency.

If these tests fail, do not update expected fields merely to make a new adapter
pass.  Write a versioned contract migration proposal first.

### 2.2 Case-data adapter tests

File: `tests/tools/test_case_data_adapters.py`

Coverage includes:

- strong typed handlers and trace propagation;
- header-only inspection versus checksum streaming;
- geometry policy and required artifact types;
- missing, unavailable, checksum, permission and dependency failures;
- regular-file reading and symlink rejection;
- MCS mapping, LPS/RAS output geometry, lineage and histogram checks;
- overlapping/unknown segments and dtype limits;
- backend geometry/histogram drift;
- atomic publication and partial-commit failure;
- sequential/concurrent idempotency;
- exact/unknown/missing labels and flag combinations;
- name/value mismatch, non-integer values and unsupported formats;
- expired and injected slow-operation deadlines.

Expected focused count for the release: 14 tests.

### 2.3 Segmentation adapter tests

File: `tests/tools/test_segmentation_adapters.py`

Coverage includes:

- model profile, engine hash, precision and device validation;
- engine reuse and deterministic artifact lineage;
- idempotent inference/commit and conflicting replay;
- permission, checksum, PENDING and missing state;
- OOM, timeout, native error and partial commit cleanup;
- malformed geometry and unknown labels;
- analytic voxel/volume/component/border statistics;
- geometry mismatch despite equal shape;
- quality finding versus failed execution semantics;
- physical skin thickness under anisotropic spacing;
- component cleanup, empty/fractional masks and malformed native output;
- trace/error URI and checksum redaction.

Expected focused count for the release: 28 tests.

### 2.4 Planning/safety adapter tests

File: `tests/tools/test_planning_adapters.py`

Coverage includes:

- sanitized backend command and typed JSON envelope;
- rotated/anisotropic world-to-index validation;
- geometry/type/out-of-bounds checks before backend execution;
- caller permission and expired deadline;
- sequential/concurrent idempotency and conflicting replay;
- no-candidate and backend failure mapping;
- stop/warning policy, all-rejected safety and safest candidate selection;
- required-mask fail closed and unsafe downgrade rejection;
- risk warning/stop precedence, large vessel and lung state;
- optional unavailable lung -> unknown/manual review;
- penetration, non-penetration and slip threshold equality;
- missing skin label, outside segment and backend timeout/error;
- unsupported planner/risk manifest versions;
- safe trace summaries.

Expected planning-adapter plus legacy planning count for the release: 22 tests.

### 2.5 MCP codec/schema tests

File: `tests/mcp/test_codec_schema.py`

The important invariants are:

1. all ten requests round-trip through safe ArtifactHandle arguments;
2. request JSON contains no `uri`, `checksum_sha256` or artifact metadata;
3. full internal artifact-field injection is rejected;
4. unknown fields, bool-as-int and non-finite values are rejected;
5. every input/output schema is JSON serializable;
6. output schemas and results contain only public artifact views.

### 2.6 MCP runtime/protocol tests

File: `tests/mcp/test_runtime_server.py`

Coverage includes:

- 3/3/4 logical server partition and exactly ten unique tools;
- annotations and `taskSupport=forbidden`;
- structured result plus text compatibility block;
- principal/caller/case/tool denial before execution;
- schema-shaped `INVALID_ARGUMENT` failures;
- expired deadlines;
- request/trace/version response drift;
- exact registry handler catalog;
- initialize -> initialized -> list/call lifecycle;
- JSON-RPC parse/method/params errors;
- explicit optional official SDK behavior.

### 2.7 Local demo and stdio tests

File: `tests/mcp/test_local_mcp_demo.py`

The in-process demo calls all ten tools.  The subprocess test starts
`examples/local_mcp_stdio.py`, sends newline-delimited JSON-RPC messages and
verifies:

- negotiation of `2025-11-25`;
- no response to `notifications/initialized`;
- three tools discovered from the `case-data` server;
- stderr remains empty;
- stdout contains only JSON-RPC responses.

## 3. Manual stdio check

Start a server:

```bash
python3 examples/local_mcp_stdio.py --server case-data
```

Then provide these three lines on stdin:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"manual","version":"1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

Expected behavior:

- line 1 returns server information and the tools capability;
- line 2 returns nothing;
- line 3 returns exactly three descriptors with input/output schema;
- stdout contains no log prefix or non-JSON text.

Use `segmentation` or `planning-safety` to inspect the other server catalogs.

## 4. How to validate a replacement backend

When another model implements a real/company adapter, give it only the selected
contracts, task card, tool specifications, current adapter port and selected
tests.  Require the following sequence.

### Step 1: prove the baseline

Run contract, existing adapter and MCP tests before editing.  Save the command,
test count and result.

### Step 2: implement one port group only

Do not edit `contracts/**`, MCP codec/schema, other adapter groups, Agent graph
or API.  Keep the deterministic backend selectable.

### Step 3: add independent fixtures

For proprietary inputs, check in only sanitized IDs/manifests/expected summaries.
Resolve raw fixtures internally.  Do not upload company cases, engines or SDKs
to an external model.

### Step 4: cover the failure matrix

At minimum inject:

- missing/unavailable artifact;
- permission denied;
- checksum mismatch;
- geometry mismatch;
- malformed native return;
- dependency unavailable;
- timeout;
- partial write/commit;
- identical idempotent retry;
- conflicting idempotency-key reuse.

Safety adapters additionally require missing required mask, unsafe downgrade,
warning/stop boundary and false-safe/false-negative oracle tests.

### Step 5: rebind and run MCP tests

Use `build_adapter_registry` with the new port.  The same ten MCP schemas and
tool names must remain.  Verify no URI/checksum appears in structured/text
results or trace records.

### Step 6: run full regression

```bash
python3 run_tests.py
python3 -m compileall -q contracts src tests examples
git diff --check
```

### Step 7: report evidence honestly

The implementation report must include:

- modified files;
- port implementation and dependency versions;
- fixture IDs and independent oracle;
- every test command/count/result;
- exact hardware for performance evidence;
- P50/P95 and sample/warm-up counts when actually measured;
- every unexecuted item marked `NOT_RUN`.

## 5. Acceptance mapping

The detailed acceptance IDs remain in `specs/tools/*.md`.  This module provides
the following local evidence categories:

| Acceptance family | Local evidence | Requires company environment later |
|---|---|---|
| IC / MC / LS | contract, manifest/header/checksum, mapping, schema, permission, idempotency tests | approved MCS SDK, independent NIfTI readers, large-case benchmark |
| RS / VS / ES | profile/inference adapter, analytic QC, physical-thickness fixtures, error cleanup | real TensorRT engine, internal golden Dice/surface metrics, target GPU benchmark |
| GP / PS / IR / SP | coordinate checks, fail-closed scripted backend, risk precedence, traversal result validation | native planner/EDT/risk/DDA kernels, independent slow oracle, false-safe/false-negative audit |
| MCP protocol | schema, discovery, calls, stdio, structured content, redaction | official SDK live client, Streamable HTTP, OIDC/OAuth, multi-process load |

## 6. Expected failure diagnosis

| Symptom | Likely cause | Check |
|---|---|---|
| `INVALID_ARGUMENT` before handler | wire shape or strict codec failure | field path in error; compare `inputSchema` |
| `PERMISSION_DENIED` | caller mismatch or case/tool policy | authenticated principal versus `ToolCallContext` |
| artifact not registered | resolver lacks authoritative ID | register the artifact; do not send a URI |
| `CONTRACT_VIOLATION` after success | handler returned wrong identity/version/type or malformed native result | adapter response and backend DTO validation |
| replay conflict | same key used with changed semantic request | generate a new idempotency key or restore original request |
| stdio client cannot parse | server/log wrote non-JSON to stdout | move logs to stderr and keep one JSON object per line |
| official SDK import failure | optional dependency absent | install `mcp>=1.27,<2`, or use dependency-free local dispatcher |

## 7. Explicitly not run for v0.5.0

- official SDK package installation and Inspector session;
- Streamable HTTP/SSE transport;
- OAuth/OIDC/JWT/mTLS authentication;
- real MCS/NIfTI conversion;
- real TensorRT/CUDA inference;
- company morphology/planning/safety/ray kernels;
- internal patient/animal golden data;
- target-hardware accuracy, latency, throughput or memory benchmark;
- multi-process/distributed idempotency and load testing.

These are future integration evidence, not blockers for the requested local
Python 3.10 demo.
