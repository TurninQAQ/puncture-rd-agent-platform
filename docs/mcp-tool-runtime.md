# MCP Tool Runtime and Algorithm Adapter Design

## 1. Scope

Module 3 turns the ten frozen algorithm contracts into three locally runnable
MCP servers.  It proves tool discovery, argument validation, authorization,
artifact resolution, deterministic execution, structured results, error
mapping, idempotency and trace propagation.

It deliberately does **not** reimplement company algorithms.  MCS parsing,
NIfTI writing, TensorRT inference, morphology, path planning, safety-envelope
collision and 3D ray traversal remain behind injectable ports.  The checked-in
backends consume explicit manifests and deterministic fixtures so the Agent
engineering can be implemented and interviewed without confidential binaries
or data.

The same architecture transfers to EDA and industrial software:

- `ArtifactRef` becomes a netlist, layout, waveform, log or CAD artifact;
- segmentation tools become simulation/synthesis/diagnosis tools;
- planning safety becomes design-rule, equipment-interlock or sign-off logic;
- the MCP/Agent layer remains unchanged.

## 2. Runtime architecture

```text
MCP client / future LangGraph node
        |
        | tools/list, tools/call
        v
LocalMcpServer or official MCP SDK v1 adapter
        |
        | JSON Schema + strict decoding
        v
McpToolRuntime
  - authenticated principal check
  - case/tool allowlist
  - deadline calculation
  - ArtifactHandle resolution
  - durable replay authorization + SQLite ledger
  - response identity/version verification
        |
        v
ToolRegistry (exactly ten frozen definitions)
        |
        +--> CaseDataToolAdapter ------> CaseDataBackendPort
        +--> SegmentationToolAdapter --> Engine/Image/Artifact ports
        +--> PlanningToolAdapters -----> PlanningKernelPort
                                            |
                                            +--> deterministic local backend
                                            +--> future company C++/TensorRT SDK
```

The model never receives a volume, mask, signed URI, checksum or raw company
metadata.  Large data stays in artifact storage; only stable artifact identity
and public lineage information cross the model-visible boundary.

## 3. Logical MCP servers

| Server | Tools | Responsibility |
|---|---|---|
| `case-data` | `inspect_case_metadata`, `convert_mcs_to_nifti`, `validate_label_schema` | artifact/header readiness, approved format conversion and label contract validation |
| `segmentation` | `run_segmentation`, `validate_segmentation_result`, `extract_skin_surface` | model adapter, deterministic QC and physical-thickness skin extraction adapter |
| `planning-safety` | `generate_candidate_paths`, `evaluate_path_safety`, `evaluate_intraoperative_risk`, `verify_skin_penetration` | candidate generation, full-path clearance, tip risk and skin traversal |

All ten public request/result dataclasses remain in
`contracts/tool_inputs.py` and `contracts/tool_outputs.py`.  Module 3 adds a
wire adapter; it does not rename or reinterpret those fields.

## 4. MCP protocol baseline

The dependency-free dispatcher targets MCP `2025-11-25` and supports the
`2025-06-18` protocol version for local negotiation.  Implemented methods are:

- `initialize`;
- `notifications/initialized`;
- `ping`;
- `tools/list`;
- `tools/call`;
- cancellation/progress notifications as safe no-ops for this synchronous demo.

Each tool descriptor contains:

- `inputSchema`: generated JSON Schema 2020-12 for its exact request contract;
- `outputSchema`: URI-redacted `ToolResponseEnvelope` schema;
- `annotations`: `readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`;
- `execution.taskSupport = "forbidden"` for this demo;
- `_meta`: tool version, logical server and default timeout.

Structured results are returned in `structuredContent`.  A JSON-serialized
`TextContent` block is also returned for backward compatibility, as required by
the stable MCP tool-result guidance.

The optional implementation extra pins the stable official SDK line:

```toml
mcp>=1.27,<2
```

`src/puncture_agent/mcp/official_sdk.py` binds the same runtime to the official
low-level SDK.  The dependency-free dispatcher remains the default local test
surface, so protocol behavior is reproducible without installing packages.

## 5. Safe wire format

### 5.1 Artifact arguments

Internal `ArtifactRef` contains URI, checksum, metadata and lineage.  It is not
accepted from an MCP client.  On the wire it becomes:

```json
{
  "artifact_id": "demo-case-001-ct",
  "case_id": "demo-case-001",
  "artifact_type": "CT_VOLUME"
}
```

Only `artifact_id` is mandatory.  `case_id` and `artifact_type` are optional
assertions.  `InMemoryArtifactResolver` or a future registry resolver loads the
authoritative `ArtifactRef` and rejects case/type disagreement.  A client that
tries to send `uri`, `checksum_sha256`, `metadata` or other internal fields is
rejected as `INVALID_ARGUMENT` before a handler runs.

### 5.2 Call context and principal

The fixed request still carries `ToolCallContext`:

```json
{
  "request_id": "req-001",
  "trace_id": "trace-001",
  "case_id": "case-001",
  "caller": "local-demo",
  "idempotency_key": "idem-001",
  "requested_at": "2026-07-11T00:00:00Z",
  "deadline_epoch_ms": null
}
```

The transport independently supplies `McpPrincipal`.  Execution is permitted
only when:

1. `context.caller == principal.subject`;
2. `context.case_id` is in `principal.allowed_case_ids`;
3. the tool is in `principal.allowed_tools`.

In a future Streamable HTTP deployment, the principal must come from validated
OIDC/JWT/mTLS identity, never from the JSON arguments alone.

### 5.3 Result projection

The internal handler returns the frozen `ToolResponseEnvelope`.  Before it is
made model-visible, every `ArtifactRef` is recursively converted to
`ArtifactPublicView`:

```json
{
  "artifact_id": "demo-case-001-seg-...",
  "case_id": "demo-case-001",
  "artifact_type": "SEGMENTATION_MASK",
  "status": "AVAILABLE",
  "producer_name": "run_segmentation",
  "producer_version": "1.0.0",
  "geometry_fingerprint": "..."
}
```

URI, checksum, parent IDs and arbitrary metadata are absent from both
`structuredContent` and the text fallback.

## 6. Strict contract codec

`src/puncture_agent/mcp/codec.py` performs dependency-free recursive decoding:

- dataclass fields are resolved with `get_type_hints`;
- missing required and unknown fields fail;
- `bool` is not accepted as `int`;
- floats must be finite;
- enum values must be canonical wire values;
- fixed and variable tuples are distinguished;
- nested `ArtifactRef` fields always use the trusted resolver;
- dataclass `__post_init__` validation remains the final contract gate.

The codec is intentionally stricter than ordinary `dataclass(**payload)` and
does not silently coerce strings to numbers or accept extra provider fields.

## 7. Adapter ports

### 7.1 Case-data

The proprietary Mimics/MCS and storage implementation should satisfy the
behavior represented by `CaseDataBackendPort`:

```python
class CaseDataBackendPort(Protocol):
    def resolve(self, artifact_id, invocation) -> ArtifactManifest: ...
    def iter_payload(self, manifest, invocation, *, chunk_size) -> Iterable[bytes]: ...
    def render_conversion(
        self, mcs, reference_ct, mapping, output_geometry, output_dtype, invocation
    ) -> ConversionProduct: ...
    def lookup_conversion(self, scope, invocation) -> ConversionCommit | None: ...
    def commit_conversion(self, scope, candidate, invocation) -> ConversionCommit: ...
```

The checked-in `ManifestCaseDataBackend` implements bytes/regular-file
manifests only.  A company adapter replaces parsing and commit internals while
preserving:

- streaming checksum verification;
- case/tenant/caller permission;
- geometry/header comparison;
- coordinate-system conversion evidence;
- label mapping and output histogram verification;
- one committed artifact per idempotency scope;
- no AVAILABLE partial output on failure.

### 7.2 Segmentation

Segmentation is split into three ports so engine lifecycle, independent QC and
artifact publication can evolve separately:

```python
class SegmentationEnginePort(Protocol):
    def describe(self, model_id, version, precision) -> ModelProfile: ...
    def infer(self, ct, profile, device_id) -> EngineResult: ...

class ImageAlgorithmPort(Protocol):
    def label_statistics(self, artifact, schema) -> LabelAnalysis: ...
    def extract_external_skin_surface(
        self, source, method, thickness_mm, connectivity, keep_largest_component
    ) -> SkinSurfaceKernelResult: ...

class SegmentationArtifactPort(Protocol):
    def resolve(self, artifact, *, caller, purpose) -> VolumeManifest: ...
    def commit(self, ...) -> ArtifactRef: ...
```

The future company implementation should wrap the existing C++/TensorRT and
morphology code.  It must not put CUDA lifecycle or voxel algorithms in the MCP
handler.  `ModelProfile` is authoritative for preprocessing, label mapping,
engine hash, precision and device compatibility.

### 7.3 Planning and safety

All numerical/safety conclusions stay behind `PlanningKernelPort`:

```python
class PlanningKernelPort(Protocol):
    @property
    def manifest(self) -> PlanningKernelManifest: ...
    def generate(self, command: CandidateGenerationCommand) -> NativeCandidateBatch: ...
    def path_clearance(self, command: PathSafetyCommand) -> NativeSafetyBatch: ...
    def tip_risk(self, command: TipRiskCommand) -> NativeRiskState: ...
    def traverse_skin(self, command: SkinTraversalCommand) -> NativeSkinTraversal: ...
```

The Python adapter validates coordinate systems, geometry fingerprints,
finite values, required masks, warning/stop precedence and backend return
types.  It fails closed if a backend attempts to downgrade an unsafe result.
The checked-in deterministic backend reads no voxels; company EDT/collision,
risk and DDA/ray-tracing implementations plug into this port.

## 8. Idempotency and timeout behavior

Every request includes an idempotency key.  Write-like adapters cache or commit
against a scope containing tool, case, caller and idempotency identity.  A
replay with an identical semantic fingerprint returns the same artifact/result;
reuse with changed arguments is a non-retryable contract error.

`SQLiteToolReplayLedger` closes the graph-side crash window after a tool server
has produced a terminal response but before the Agent commits its next
checkpoint. The runtime atomically claims a scope, executes the handler, commits
MCP-safe structured content with `PRAGMA synchronous=FULL`, and only then returns
to the graph. A rebuilt bridge/runtime rebinds the current request/trace identity
and returns the stored response without calling the handler again.

The durable state machine is:

```text
absent -> PENDING -> COMPLETED
                  -> deleted     (retryable failure only)
                  -> UNCERTAIN   (write may have advanced without a durable response)
```

Successful, partial and non-retryable failed business terminals are replayed.
Retryable failures delete their claim so a bounded retry can execute again.
Expired read-only claims may be reclaimed; expired write claims become
`UNCERTAIN` and require manual reconciliation. The configured claim TTL must be
longer than every tool timeout on the logical server.

Every call, including replay, runs an explicitly injected current-request
authorizer. A replay hit additionally runs a response validator over the stored
public result before it can be returned. Deployments must connect both callbacks
to their live case/artifact ACL and Registry source; a durable ledger cannot be
enabled without them. All three logical servers may share one SQLite database
on a single host:

```python
from puncture_agent.mcp import McpToolRuntime, SQLiteToolReplayLedger

with SQLiteToolReplayLedger("/var/lib/puncture-agent/tool-replay.sqlite3") as ledger:
    case_data_runtime = McpToolRuntime(
        registry,
        artifact_resolver,
        server_name="case-data",
        replay_ledger=ledger,
        replay_authorizer=current_acl_allows,
        replay_response_validator=stored_artifacts_are_currently_authorized,
    )
```

SQLite proves same-host/shared-file restart behavior. Multi-host MCP servers need
a PostgreSQL or dedicated shared-ledger implementation with the same protocol.
The tool backend must still make its own side effect and ledger completion
atomic, or use an artifact/operation idempotency record, for the narrower crash
window inside the tool service itself.

The checked-in PostgreSQL process-kill gate exercises the graph-side window with
two fresh Python processes. Process A completes the target tool's SQLite ledger
entry and returns through the MCP bridge, then self-terminates with `SIGKILL`
before the LangGraph node checkpoint. Process B observes that the PostgreSQL
checkpoint lacks the target node result, resumes with the identical replay
identity, receives `idempotentReplay=true`, and reaches the terminal checkpoint
without invoking the target handler a second time. Commit
`9f122782d7f9c6cdee842b84b453dfa99be73840` passed
[GitHub Actions run 29147544527](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29147544527);
artifact `postgres-tool-process-kill-9f122782d7f9c6cdee842b84b453dfa99be73840`
is 2,451 bytes with digest
`sha256:6022a7cf74b5de59a72332c9260826967360fcaaeb748cab3854a3181124a1b3`.
This is evidence for the `COMPLETED`-to-checkpoint window on one host, not for
atomicity between an external side effect and ledger completion, a shared
multi-host ledger, or host power-loss recovery.

Deadlines are enforced at two layers:

1. `McpToolRuntime` bounds the registry call using the smaller of the tool
   default and `deadline_epoch_ms`;
2. adapters pass remaining timeout to backend operations and check the deadline
   between stages.

Python threads cannot safely terminate native work.  After detecting a timeout,
the local runtime waits for the worker to stop before returning, so it never
reports `TIMEOUT` while a write continues invisibly in the background.  A
production C++/gRPC/REST adapter must implement cooperative cancellation or
service-side deadlines so that this convergence is actually bounded.  The
local thread check is not a hard-kill mechanism.

## 9. Error model

Protocol-shape errors use JSON-RPC codes such as `-32700`, `-32600`, `-32601`
and `-32602`.  Tool argument/domain/dependency failures return a normal MCP tool
result with `isError=true` and a schema-valid failed `ToolResponseEnvelope`.

Examples:

- malformed field -> `INVALID_ARGUMENT`, non-retryable;
- caller/case violation -> `PERMISSION_DENIED`, non-retryable;
- missing artifact -> `MISSING_ARTIFACT`;
- checksum/geometry mismatch -> corresponding stable error;
- GPU OOM or dependency timeout -> retryability decided by the adapter;
- missing required danger mask -> fail closed;
- response request/trace/version drift -> `CONTRACT_VIOLATION`.

Raw native/provider exception text is not sent to the model.

## 10. Local execution

Run all ten tools in process:

```bash
python3 examples/local_mcp_demo.py
```

Start one stdio server that an MCP client can spawn:

```bash
python3 examples/local_mcp_stdio.py --server case-data
python3 examples/local_mcp_stdio.py --server segmentation
python3 examples/local_mcp_stdio.py --server planning-safety
```

The stdio transport reads and writes one UTF-8 JSON-RPC object per line and
never writes logs to stdout.

## 11. File map

```text
src/puncture_agent/mcp/
  codec.py          strict request decoding and safe result projection
  schema.py         JSON Schema 2020-12 generation
  runtime.py        principal/deadline/registry boundary
  server.py         dependency-free MCP JSON-RPC dispatcher
  stdio.py          newline-delimited stdio transport
  official_sdk.py   optional stable official SDK v1 binding

src/puncture_agent/tooling/
  factory.py        exact ten-handler registry binding
  case_data.py      case-data ports and deterministic backend
  segmentation.py   engine/image/artifact ports and deterministic backend
  planning.py       planning kernel port and deterministic backend
```

## 12. How another model should replace a deterministic backend

Implement one port group at a time.  Do not change contracts or MCP schemas.

1. Read the corresponding `tasks/task-03..05` and every selected tool spec.
2. Implement a narrow company port in a new module; do not edit the Agent graph.
3. Keep the deterministic backend for offline tests.
4. Add sanitized internal fixture resolvers; never send company data externally.
5. Run contract tests before changing bindings.
6. Run normal, boundary, permission, timeout, dependency, idempotency and
   malformed-backend tests for the selected group.
7. Bind the port through `build_adapter_registry`.
8. Run MCP codec/runtime/stdio tests and the full suite.
9. Record exact software/hardware versions and mark unexecuted evidence
   `NOT_RUN`; do not copy demo timing into a production claim.

## 13. Known limitations

- no proprietary MCS or NIfTI parser/writer is included;
- no real TensorRT engine, CUDA allocation or GPU benchmark is included;
- no real morphology, EDT, collision, risk or ray-traversal kernel is included;
- official SDK and Streamable HTTP/OAuth are optional and not part of the
  dependency-free local gate;
- the local principal is statically injected, not derived from OIDC/JWT;
- task-augmented MCP execution is disabled;
- the runtime is synchronous and intended for a demo; durable async execution
  belongs to later LangGraph/FastAPI modules.
