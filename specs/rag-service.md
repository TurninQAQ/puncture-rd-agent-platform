# Enterprise RAG Service Specification

## 1. Purpose and scope

The RAG service retrieves versioned, permission-filtered project knowledge for the
puncture R&D Agent. It supplies evidence for label validation, model interfaces,
path-planning constraints, safety rules, and historical troubleshooting. It does
not calculate image geometry or medical risk itself.

The production target is a hybrid retrieval chain:

```text
query normalization / rewrite
        |
        +--> BM25 lexical Top-N
        |
        `--> dense-vector Top-N
                  |
             RRF fusion
                  |
        metadata + ACL + version enforcement
                  |
             cross-encoder rerank
                  |
          parent-context expansion
                  |
       normalized chunks + citations
```

The current `MockRagService` is only a deterministic lexical development double.
It proves request/response behavior, filters, empty-evidence handling, and failure
routing; it does not claim BM25, embedding, RRF, or reranker quality.

## 2. Files and ownership

Public contracts:

- `src/puncture_agent/rag/models.py`
- `src/puncture_agent/rag/client.py`

Development double:

- `src/puncture_agent/rag/mock_service.py`
- `mocks/rag/documents.json`

Tests:

- `tests/contract/test_model_rag_contracts.py`
- `tests/rag/test_mock_rag.py`

Production task:

- `tasks/task-02-rag-service.md`

Do not remove ACL, module, version, score, rank, or citation fields when replacing
the mock. These fields are consumed by the Agent verifier and trace layer.

## 3. Knowledge sources

Initial internal sources:

1. MCS/NIfTI/nnU-Net data and label specifications;
2. segmentation model cards and input/output contracts;
3. TensorRT/C++ integration documents;
4. skin-Mask processing rules;
5. path-planning needle-length, angle, collision, and clearance rules;
6. warning/stop boundary and intraoperative safety definitions;
7. historical bug tickets, experiment summaries, and approved resolutions.

The production ingestion pipeline may support PDF, Word, Markdown, Wiki, issue
systems, and approved database exports. Source connectors are outside the current
mock, but indexed chunks must conform to the metadata defined below.

## 4. Fixed retrieval input

`RetrievalRequest` is the stable caller contract.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `request_id` | string | yes | Trace and correlation ID |
| `query` | string | yes | User/task query |
| `modules` | tuple of strings | no | Allowed project modules |
| `required_version` | string/null | no | Exact required document version |
| `access_scopes` | tuple of strings | yes | Effective caller authorization scopes |
| `top_k` | integer 1..100 | yes | Maximum returned chunks |
| `metadata_filters` | object | no | Exact filters such as active status |

Example:

```json
{
  "request_id": "rag-case-001-01",
  "query": "current needle path safety envelope rule",
  "modules": ["path_planning"],
  "required_version": "v3",
  "access_scopes": ["algorithm_team"],
  "top_k": 5,
  "metadata_filters": {"status": "active"}
}
```

Authorization scopes must come from authenticated runtime context. Never let an
LLM invent or broaden `access_scopes`.

## 5. Fixed retrieval output

`RetrievalResponse` contains:

```text
request_id
rewritten_query
chunks[]
retrieval_mode
trace_id
latency_ms
warnings[]
```

Each `RetrievedChunk` contains:

| Field | Meaning |
|---|---|
| `chunk_id` | Stable child chunk identifier |
| `document_id` | Stable parent document identifier |
| `title` | Source title |
| `module` | data_validation/segmentation/path_planning/safety_evaluation/etc. |
| `version` | Exact source version |
| `section` | Human-readable section path |
| `text` | Retrieved evidence text |
| `score` | Final normalized score in [0,1] |
| `rank` | Contiguous rank starting at one |
| `citation` | Deterministic display citation |
| `metadata` | Safe additional fields |

Example citation:

```text
[Puncture Path Planning Rules | v3 | Needle length, angle and safety envelope]
```

Empty retrieval is a valid response, not fabricated evidence:

```json
{
  "chunks": [],
  "warnings": ["NO_RELEVANT_KNOWLEDGE"]
}
```

The Agent must then clarify, refuse, or continue using deterministic tool results;
it must not ask Qwen to answer as if documents were found.

## 6. Production index contract

Recommended parent-document fields:

```json
{
  "document_id": "planning-rules-v3",
  "title": "Puncture Path Planning Rules",
  "source_uri": "internal://knowledge/planning-rules-v3",
  "source_type": "markdown",
  "module": "path_planning",
  "version": "v3",
  "status": "active",
  "access_scopes": ["algorithm_team"],
  "created_at": "2026-03-01T00:00:00Z",
  "updated_at": "2026-04-16T00:00:00Z",
  "checksum_sha256": "...",
  "parser_version": "...",
  "chunker_version": "..."
}
```

Recommended child-chunk fields:

```json
{
  "chunk_id": "planning-rules-v3#2.1#0001",
  "document_id": "planning-rules-v3",
  "section_path": ["2 Path Planning", "2.1 Safety Envelope"],
  "text": "...",
  "token_count": 612,
  "embedding": [0.0],
  "chunk_index": 1,
  "access_scopes": ["algorithm_team"],
  "module": "path_planning",
  "version": "v3",
  "status": "active"
}
```

All chunks inherit authorization, version, lifecycle status, source checksum, and
document ID from their parent. The indexer must reject a child that lacks these
fields.

## 7. Ingestion implementation procedure

### Step 1: source registration

- Assign a stable `document_id`; do not use a temporary filesystem path as identity.
- Record source type, owner, version, status, ACL, and checksum.
- Reject documents without ownership and ACL metadata.
- Keep company documents inside approved private storage.

### Step 2: parsing and normalization

- Parse headings, paragraphs, lists, tables, captions, and code/API blocks.
- Remove repeated headers, footers, table of contents noise, and blank fragments.
- Preserve section hierarchy and table context.
- Normalize Unicode and whitespace without changing identifiers, units, label IDs,
  error codes, or C++ symbol names.
- Save parser version and parsing warnings.

### Step 3: chunking

Use heading-aware parent-child chunks rather than fixed windows alone:

1. create parent sections from document hierarchy;
2. create retrieval children around 500-800 tokens;
3. use approximately 80-120 tokens of overlap where semantic continuity requires it;
4. keep parameter tables, error-code tables, and API signatures intact;
5. link every child to a parent and source section;
6. calculate deterministic chunk IDs from document/version/section/chunk index.

Exact sizes must be selected through retrieval evaluation, not treated as universal
constants.

### Step 4: deduplication and lifecycle

- Use source checksum for exact deduplication.
- Detect near duplicates and record canonical document relationships.
- Mark superseded documents; do not silently delete audit history.
- Use a versioned index plus alias for atomic re-indexing and rollback.
- Never mix chunks produced by incompatible embedding dimensions in one vector field.

### Step 5: embedding and indexing

- Batch embeddings with bounded request sizes.
- L2-normalize vectors if required by the chosen similarity metric.
- Store text, BM25 fields, vector field, filters, and citation metadata together.
- Record embedding model name/revision and dimension.
- Validate indexed count, failed count, and checksum reconciliation after each job.

## 8. Retrieval implementation procedure

### Step 1: query normalization/rewrite

- Trim and normalize whitespace/Unicode.
- Preserve exact terms such as label IDs, error codes, model versions, units, and
  C++ function names.
- Expand only approved aliases (for example, NIfTI/nii.gz), with the original query
  retained in trace data.
- If an LLM rewrite is later used, treat the rewrite as untrusted input and retain
  the original exact tokens in the lexical branch.

The final query used by the service is returned as `rewritten_query`.

### Step 2: mandatory filters

Enforce effective ACL, requested module, exact version, active/deprecated status,
and caller metadata filters before a result can be returned. If the backend cannot
apply a mandatory filter during retrieval, post-filter and over-fetch; never omit
the filter for performance.

### Step 3: parallel recall

Recommended initial values from `EnterpriseRagConfig`:

```text
dense_top_k   = 30
lexical_top_k = 30
rerank_top_k  = 10
```

BM25 is important for exact label names, API symbols, error codes, and numeric
parameters. Dense retrieval is important for semantic problem descriptions.

### Step 4: reciprocal rank fusion

Fuse the two ranked lists with deterministic RRF:

```text
RRF(document) = sum(1 / (k + rank_i))
```

Use a configurable `k` (a common starting point is 60). Deduplicate by `chunk_id`,
not by text equality. Apply deterministic tie breaking, for example final score,
updated time, then chunk ID.

### Step 5: reranking

- Rerank the fused candidates with a cross-encoder/reranker model.
- Batch requests and enforce timeout.
- If reranking is unavailable, either return a clearly marked degraded result or
  fail according to runtime configuration; do not pretend reranking succeeded.
- Record reranker name/revision and pre/post ranks in trace data.

### Step 6: parent expansion and context control

- Expand top child chunks to enough parent context to preserve definitions.
- Deduplicate overlapping parents.
- Respect a configured context/token budget.
- Keep citations attached after expansion.
- Do not concatenate an entire large document by default.

### Step 7: confidence and output normalization

- Normalize final scores to [0,1] with a documented method.
- Sort descending and assign contiguous ranks.
- Enforce caller `top_k` after all filtering and deduplication.
- If the relevance threshold is not met, return zero chunks and
  `NO_RELEVANT_KNOWLEDGE`.

## 9. Error contract

Production `EnterpriseRagClient` raises `RagServiceError` with a stable code:

| Code | Retryable | Meaning |
|---|---:|---|
| `RAG_TIMEOUT` | yes | Backend/embedding/reranker timeout |
| `RAG_BACKEND_UNAVAILABLE` | yes | Search backend unavailable |
| `RAG_EMBEDDING_UNAVAILABLE` | yes | Query embedding service unavailable |
| `RAG_RERANKER_UNAVAILABLE` | configurable | Reranker unavailable |
| `RAG_INVALID_REQUEST` | no | Invalid filter/query contract |
| `RAG_PERMISSION_DENIED` | no | Caller context invalid or unauthorized |
| `RAG_INDEX_INCOMPATIBLE` | no | Mapping/model dimension/version mismatch |
| `RAG_PROTOCOL_ERROR` | no | Malformed backend response |

Empty relevant results are not an exception. Unauthorized documents are removed
without revealing their title, count, or existence.

## 10. Trace and metrics

Record:

- request/trace ID;
- original and rewritten query, with configured redaction;
- effective module/version/status/ACL filters;
- dense and BM25 candidate counts;
- RRF and reranker ranks/scores;
- final chunks and citations;
- embedding/reranker/index versions;
- stage latency and total latency;
- empty-result and degraded-mode warnings;
- normalized failures and retry count.

Do not expose confidential chunk text in general metrics. Trace access must follow
the same authorization controls as document access.

## 11. Correctness verification

### 11.1 Contract and mock tests

Run:

```bash
python3 tests/contract/test_model_rag_contracts.py -v
python3 -m unittest discover -s tests/rag -p 'test*.py' -v
```

Existing tests verify:

- request bounds and normalized response scores/ranks;
- fixture loading and deterministic health;
- module, exact-version, metadata, and ACL filters;
- explicit access to a legacy version only when requested;
- citation/rank shape and `top_k` enforcement;
- empty evidence instead of fabricated content;
- normalized retryable timeout failure;
- visible production stub before task 02 completion.

### 11.2 Ingestion unit tests to add

Use small synthetic documents. Do not require company files. Cover:

1. heading hierarchy and stable chunk IDs;
2. table/API/error-code preservation;
3. removal of headers and footers;
4. parent-child linkage;
5. ACL/version/status inheritance;
6. exact checksum deduplication;
7. changed version produces changed identity/index metadata;
8. malformed or ACL-free document is rejected;
9. embedding dimension mismatch fails before index activation;
10. partial indexing reports failed records and does not swap the live alias.

### 11.3 Retrieval unit tests to add

Use fake lexical, vector, and reranker backends. Cover:

1. both recall branches receive mandatory filters;
2. deterministic RRF calculation and tie breaking;
3. duplicate chunk removal;
4. reranker order replacement;
5. reranker timeout degraded/failure behavior;
6. exact identifier query remains in the lexical branch after rewrite;
7. ACL-negative results never appear at any output stage;
8. requested version is exact, not fuzzy;
9. parent expansion respects context budget;
10. low relevance returns empty chunks and warning;
11. scores normalize to [0,1] and ranks are contiguous;
12. backend errors map to documented codes.

### 11.4 Integration tests

Gate live backend tests behind an explicit variable such as
`RUN_RAG_INTEGRATION=1`. Create a temporary test index containing only synthetic
documents, then verify:

- index mapping and embedding dimension;
- BM25 exact error-code lookup;
- dense semantic lookup;
- hybrid result and RRF order;
- reranker integration;
- ACL and version isolation;
- alias swap and rollback;
- teardown of the temporary index.

The test must refuse to run against the production index name.

### 11.5 Golden-set evaluation

Build an internal labelled set of approximately 80-150 queries containing:

- exact label/API/error-code questions;
- semantic troubleshooting questions;
- multi-document questions;
- active-versus-deprecated version conflicts;
- permission-negative queries;
- unanswerable queries.

Measure retrieval separately from answer generation:

```text
Recall@5 and Recall@10
MRR
NDCG@10
correct-version hit rate
ACL leak count
empty/unanswerable decision accuracy
P50/P95 retrieval latency
```

Compare at minimum:

1. BM25 only;
2. dense only;
3. BM25 + dense RRF;
4. hybrid + reranker;
5. hybrid + reranker + parent expansion.

## 12. Failure-test matrix

Inject:

- search backend timeout;
- embedding timeout and malformed vector dimension;
- reranker timeout;
- one recall branch unavailable;
- stale/deprecated document with higher lexical score;
- unauthorized document with perfect keyword match;
- duplicate chunks from both recall branches;
- missing parent document;
- corrupt metadata or missing version;
- zero relevant documents;
- oversized query/context budget;
- attempted use of an unapproved metadata filter.

For each case assert the exact code/warning, retryability, returned chunk list,
absence of ACL leakage, and trace stage that failed.

## 13. Acceptance gates

Task 02 is complete only when:

1. all current contract and mock tests pass unchanged;
2. ingestion and retrieval unit tests run without external services;
3. ACL leak count is exactly zero in unit, integration, and golden-set tests;
4. requested module/version/status filters are obeyed in 100% of negative tests;
5. all output scores are in [0,1], ranks are contiguous, and citations are complete;
6. no-result queries return an empty list and warning rather than invented evidence;
7. temporary-index integration tests are environment-gated and production-safe;
8. the selected chunking configuration beats or matches fixed-window baseline on
   the internal golden set;
9. hybrid retrieval improves Recall@10 over both single recall branches, or the
   measured exception is documented honestly;
10. recommended initial release target is Recall@10 >= 0.85, NDCG@10 >= 0.75,
    correct-version hit rate >= 0.95, and zero ACL leaks on the labelled set;
11. P50/P95 latency is measured with backend/model/hardware/query-count details;
12. index, embedding, reranker, parser, and chunker versions appear in trace data.

Thresholds are engineering release targets, not numbers to place on a resume until
they have been measured on the actual internal dataset.

## 14. Mock usage example

```python
service = MockRagService.from_default_fixture()
response = service.retrieve(
    RetrievalRequest(
        request_id="demo-rag-1",
        query="needle length and safety envelope",
        modules=("path_planning",),
        required_version="v3",
        access_scopes=("algorithm_team",),
        top_k=5,
    )
)
```

Failure injection:

```python
service = MockRagService.from_default_fixture(failure_mode="timeout")
```

This lets LangGraph retry and verifier branches be implemented before OpenSearch,
an embedding model, or a reranker is available.
