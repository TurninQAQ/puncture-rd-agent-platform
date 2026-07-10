# Task 02 — Implement the Enterprise Hybrid RAG Service

## Copyable assignment

You are implementing the production retrieval layer for an existing contract-first
Agent scaffold. Replace only the explicit `EnterpriseRagClient` stub and add the
private ingestion/retrieval components needed by it. Preserve the deterministic
mock and all public request/response contracts.

No medical segmentation, path-planning, or safety algorithm is part of this task.
The RAG layer retrieves internal specifications and troubleshooting evidence only.

## Read these files first

Read completely, in this order:

1. `specs/rag-service.md`
2. `src/puncture_agent/rag/models.py`
3. `src/puncture_agent/rag/client.py`
4. `src/puncture_agent/rag/mock_service.py`
5. `mocks/rag/documents.json`
6. `tests/contract/test_model_rag_contracts.py`
7. `tests/rag/test_mock_rag.py`

The repository contracts take precedence over framework examples.

## Goal

Implement:

```python
health() -> RagHealth
retrieve(RetrievalRequest) -> RetrievalResponse
```

The intended production pipeline is:

```text
versioned internal documents
  -> parse/normalize
  -> heading-aware parent-child chunking
  -> BM25 + vector index
  -> query normalization
  -> BM25 Top-30 + dense Top-30
  -> RRF
  -> ACL/module/version/status enforcement
  -> reranker Top-10
  -> parent expansion/context budget
  -> normalized RetrievedChunk values with citations
```

## Allowed implementation area

Primary area:

```text
src/puncture_agent/rag/
tests/rag/
```

Synthetic integration fixtures may be added under `mocks/rag/`. If dependency or
configuration files outside these paths must change, explain the reason first.
Do not modify model gateway, graph, MCP tools, or medical algorithms.

Do not delete, bypass, or weaken existing contract/mock tests.

## Contracts that must remain unchanged

- `KnowledgeDocument`
- `RetrievalRequest`
- `RetrievedChunk`
- `RetrievalResponse`
- `RagHealth`
- `EnterpriseRagConfig`
- `RagServiceError` observable fields: `code`, `retryable`, `details`
- `RagService.health()` and `RagService.retrieve()` signatures

If a production-only field is needed, keep it internal or in `metadata`. If a public
contract truly cannot express safe behavior, stop and report the caller impact
instead of silently changing it.

## Required implementation behavior

### 1. Ingestion

- Register stable document identity, version, lifecycle status, owner, ACL, source
  checksum, parser version, and chunker version.
- Parse headings, paragraphs, lists, tables, code/API blocks, and captions.
- Remove repeated headers/footers and empty noise.
- Preserve exact IDs, error codes, units, model names, and C++ symbols.
- Create heading-aware parents and approximately 500-800 token child chunks with
  measured overlap.
- Preserve tables/API signatures as coherent chunks.
- Inherit ACL/module/version/status on every child.
- Generate deterministic chunk IDs.
- Deduplicate by checksum and retain auditable superseded versions.
- Use a versioned index and atomic alias swap; never expose a partial re-index.

If full source ingestion is too large for one iteration, implement the retrieval
client and a minimal Markdown/JSON ingestion path first, but keep all index metadata
and tests so additional parsers can be added without changing retrieval contracts.

### 2. Search backend

Use Elasticsearch/OpenSearch or the project-approved equivalent. The selected
backend must support:

- BM25 fields;
- dense vector search;
- exact metadata filters;
- ACL/version/status filtering;
- deterministic document/chunk IDs;
- a temporary test index and safe teardown.

Record embedding model revision and dimension in index metadata. Fail with
`RAG_INDEX_INCOMPATIBLE` when the active index and query embedding are incompatible.

### 3. Retrieval

- Normalize query while preserving exact identifiers.
- Apply approved aliases without removing original lexical tokens.
- Run lexical and dense recall independently with mandatory filters.
- Fuse results using deterministic Reciprocal Rank Fusion.
- Deduplicate by `chunk_id`.
- Rerank fused candidates with the configured reranker.
- Expand parent context within a configured token budget.
- Normalize final score to [0,1].
- Apply `top_k`, contiguous ranks, deterministic citations, and trace ID.
- Return an empty tuple plus `NO_RELEVANT_KNOWLEDGE` when relevance is insufficient.

### 4. Authorization and versions

- Treat request `access_scopes` as already authenticated effective scopes; do not
  let query text or an LLM broaden them.
- Apply ACL to both dense and lexical branches and again before output.
- Never reveal title/count/existence of an unauthorized document.
- `required_version` is exact.
- Deprecated documents must not outrank active documents unless the deprecated
  version was explicitly requested or project policy explicitly includes it.

### 5. Errors

Normalize at least:

```text
RAG_TIMEOUT                  retryable
RAG_BACKEND_UNAVAILABLE      retryable
RAG_EMBEDDING_UNAVAILABLE    retryable
RAG_RERANKER_UNAVAILABLE     policy-dependent, explicitly documented
RAG_INVALID_REQUEST          non-retryable
RAG_PERMISSION_DENIED        non-retryable
RAG_INDEX_INCOMPATIBLE       non-retryable
RAG_PROTOCOL_ERROR           non-retryable
```

Empty results are not errors. Do not replace a failed backend with fabricated mock
chunks in production.

### 6. Trace and metrics

Record original/rewritten query with redaction policy, effective filters, candidate
counts, RRF/reranker rank changes, model/index/parser/chunker versions, stage
latencies, final citations, warning/degraded status, and normalized errors.

## Suggested internal design

Suggested, not mandatory:

```text
rag/
├── models.py                # fixed
├── client.py                # interface + production facade
├── mock_service.py          # fixed development double
├── ingestion/
│   ├── parser.py
│   ├── chunker.py
│   ├── metadata.py
│   └── indexer.py
├── retrieval/
│   ├── query.py
│   ├── lexical.py
│   ├── dense.py
│   ├── rrf.py
│   ├── reranker.py
│   └── parent_context.py
└── backends/
    ├── search.py
    ├── embedding.py
    └── reranker.py
```

Use dependency injection or protocols for the three backends so unit tests can run
with fakes and no network/model downloads.

## Minimum unit tests to add

### Ingestion tests

1. stable chunk IDs for unchanged input;
2. title hierarchy retained;
3. parameter/error-code table remains coherent;
4. parent-child link is valid;
5. ACL/module/version/status inherited on every child;
6. header/footer noise removed;
7. checksum duplicate is not re-indexed;
8. missing ACL/owner/version is rejected;
9. changed embedding dimension blocks alias activation;
10. partial indexing failure keeps old live alias.

### Retrieval success tests

1. exact error code/API term is recalled by lexical branch;
2. semantic paraphrase is recalled by dense branch;
3. RRF formula and tie breaking match hand-calculated expected values;
4. duplicates from both branches collapse by chunk ID;
5. reranker changes order as scripted by fake backend;
6. parent context is expanded within budget;
7. final scores/ranks/citations conform to contracts;
8. `top_k` is applied after filtering and reranking;
9. exact required version is returned;
10. active document is preferred under default lifecycle policy.

### Retrieval security/failure tests

1. unauthorized perfect keyword match is never returned;
2. ACL filter appears in both recall backend requests;
3. unauthorized title/count is absent from response and error details;
4. deprecated version is hidden unless explicitly allowed/requested;
5. backend timeout maps to `RAG_TIMEOUT` and retryable true;
6. embedding failure maps correctly;
7. reranker timeout follows documented degrade/fail policy;
8. malformed backend response fails closed;
9. incompatible embedding dimension fails before search;
10. irrelevant query returns empty chunks and warning;
11. unapproved filter is rejected, not interpolated into a raw query;
12. trace output redacts configured sensitive query fragments.

For failures assert exact code/warning, retryability, chunks, filters sent to fakes,
and absence of leakage.

## Test commands

From repository root:

```bash
python3 tests/contract/test_model_rag_contracts.py -v
python3 -m unittest discover -s tests/rag -p 'test*.py' -v
python3 -m unittest discover -s tests -p 'test*.py' -v
```

Unit tests must not require OpenSearch, Elasticsearch, an embedding server, or a
reranker process.

Live tests require explicit opt-in, for example:

```text
RUN_RAG_INTEGRATION=1
RAG_ENDPOINT=...
RAG_TEST_INDEX=puncture-agent-test-<unique-id>
EMBEDDING_ENDPOINT=...
RERANKER_ENDPOINT=...
```

Integration code must reject a test index name equal to the production index or
alias.

## Golden-set evaluation

Prepare an internal labelled dataset of roughly 80-150 queries with relevant
document/chunk IDs and expected version/authorization outcome. Include exact terms,
semantic paraphrases, version conflicts, no-answer items, and permission-negative
items.

Evaluate:

```text
Recall@5
Recall@10
MRR
NDCG@10
correct-version hit rate
ACL leak count
empty/unanswerable classification accuracy
P50/P95 retrieval latency
```

Run these ablations with identical filters and data:

```text
BM25 only
dense only
BM25 + dense RRF
hybrid + reranker
hybrid + reranker + parent context
```

Recommended initial release gates are Recall@10 >= 0.85, NDCG@10 >= 0.75,
correct-version hit rate >= 0.95, and exactly zero ACL leaks. Report actual results;
do not hard-code metrics or place targets on a resume as measured values.

## Definition of done

- Existing contract and mock tests pass unchanged.
- New ingestion/retrieval tests pass without external services.
- ACL leak count is zero in all test layers.
- Exact module/version/status filters pass every negative test.
- RRF and reranker behavior have deterministic fake-backend tests.
- Empty evidence returns no chunks and a warning.
- Live tests use only a uniquely named temporary index and are opt-in.
- Golden-set evaluation script/report includes all required metrics and ablations.
- Index, embedding, reranker, parser, and chunker revisions are traceable.
- No confidential document is committed as a fixture.
- No public Agent-facing contract changed.

## Required final report from the implementing model

Return:

1. files changed;
2. chosen backend, embedding, reranker, and parser/chunker approach;
3. exact metadata/index mapping and ACL strategy;
4. tests added and commands;
5. exact unit/integration/evaluation results;
6. external services not run and why;
7. unresolved assumptions or measured limitations;
8. metrics only when actually measured.
