# Testing the local enterprise RAG demo

## Purpose and boundary

`examples/local_rag_demo.py` is the smallest executable handoff for the production-style
RAG contracts. It uses `EnterpriseRagClient.offline`, the deterministic in-memory hybrid
index, deterministic embeddings, and the deterministic reranker already implemented in
the repository.

The demo requires no OpenSearch process, model download, network connection, GPU, pip
package, or third-party Python dependency. It proves the local ingestion and retrieval
control flow; it does not claim production retrieval quality.

The synthetic corpus contains four generic enterprise/industrial documents:

| Document | Module | Effective ACL |
|---|---|---|
| EDA timing signoff rules | `eda_flow` | `eda-team` |
| wafer-equipment maintenance | `equipment_ops` | `fab-ops` |
| engineering change control | `release_management` | `eda-team`, `fab-ops` |
| restricted yield excursion playbook | `yield_analysis` | `yield-secret` |

No company document, medical record, public model, or proprietary algorithm is used.

## Exact Python 3.10 commands

Run all commands from the repository root:

```bash
cd /home/turnin/puncture-rd-agent-platform
python3 --version
```

The verified local interpreter is Python `3.10.12`. Run the executable demo:

```bash
python3 examples/local_rag_demo.py
```

Run only the demo handoff tests:

```bash
python3 -m unittest tests.rag.test_local_demo -v
```

Run all RAG tests:

```bash
python3 -m unittest discover -s tests/rag -p 'test_*.py' -v
```

Run the repository-wide standard-library suite, which automatically discovers the demo
test:

```bash
python3 run_tests.py
```

No environment variable is required. Ambient `HTTP_PROXY` and `HTTPS_PROXY` values are
irrelevant because the demo creates no network client.

## What the demo executes

The script performs this deterministic sequence:

```text
four SourceDocument objects
        |
        v
parse -> heading-aware parent/child chunks -> deterministic embeddings
        |
        v
InMemoryHybridIndex, generations 1..4
        |
        +--> authorized query
        |      BM25 + dense recall -> RRF -> reranker -> parent context
        |      ACL/module/metadata filters -> evidence + citations
        |
        `--> ACL-negative query
               exact restricted identifier in query
               wrong effective scope -> zero evidence + NO_RELEVANT_KNOWLEDGE
```

The authorized request uses scope `eda-team`, restricts modules to `eda_flow` and
`release_management`, and applies approved `category=approved` and `language=en`
metadata filters. Its response mode is `hybrid_rrf_rerank_parent`; returned evidence has
both lexical and dense branch ranks and deterministic citations.

The negative request deliberately contains the exact restricted identifier
`LOT-SECRET-ALPHA`, but its effective scope is still only `eda-team`. The restricted
document is filtered from both recall branches and again before output. The response has
no evidence or citations and contains the warning `NO_RELEVANT_KNOWLEDGE`.

Authorization scopes are supplied by the caller in this local contract demo. In a real
runtime they must come from authenticated identity context, never from the query or an
LLM-generated field.

## Deterministic JSON contract

The script prints one JSON object with sorted keys. It intentionally excludes measured
wall-clock latency and other nondeterministic values. The stable output includes:

- four ingestion reports with monotonically increasing generations;
- index health and document/chunk counts;
- the normalized query, deterministic trace ID, and retrieval mode;
- evidence rank, score, document identity, version, and branch ranks;
- citations and warnings for both queries.

The test executes the demo twice in the same process and twice as separate Python 3.10
processes, then requires byte-for-byte identical subprocess output.

## What each test proves

`tests/rag/test_local_demo.py` contains these checks:

| Test | Evidence provided |
|---|---|
| `test_run_demo_is_deterministic_without_network` | Two executions are equal while socket connection attempts are forced to fail; four documents and seven chunks are indexed. |
| `test_authorized_query_uses_hybrid_branches_and_returns_citations` | Retrieval mode is hybrid/RRF/rerank/parent, ranks are contiguous, citations are complete, and every selected item has lexical and dense branch ranks. |
| `test_acl_negative_query_returns_no_restricted_evidence` | A perfect restricted identifier cannot return the unauthorized document, title, or citation; the result is empty with the correct warning. |
| `test_script_stdout_is_byte_stable_across_python_processes` | The documented command succeeds with no third-party setup and emits identical parseable JSON across separate processes. |
| `test_testing_document_lists_exact_commands_and_boundaries` | This handoff keeps the Python 3.10 commands, test meanings, and non-production boundary explicit. |

## Expected failure interpretation

- Ingestion failure means a synthetic document violates the stable source contract or the
  local embedding/index manifest became incompatible.
- Missing authorized evidence means lexical/dense recall, RRF, reranking, filtering, or
  parent expansion changed and must be reviewed.
- Restricted evidence in the ACL-negative response is a security regression and blocks
  the module release.
- Different JSON across repeated executions means nondeterministic data entered the
  handoff output; remove timestamps, random IDs, unordered collections, or latency before
  accepting the change.

## What this demo does not prove

This local run does **not** prove:

- OpenSearch mappings, TLS, credentials, snapshots, alias promotion, or live rollback;
- production embedding or reranker model quality and availability;
- Qwen answer generation, citations in generated prose, or Agent orchestration;
- golden-set Recall@10, NDCG@10, MRR, ACL leak count, or P50/P95 latency;
- production identity-provider integration or document-level authorization policy.

Those require the separately documented environment-gated deployment, integration, and
evaluation evidence. Do not report local deterministic scores as production metrics.
