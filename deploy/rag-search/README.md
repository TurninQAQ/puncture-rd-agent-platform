# OpenSearch assets for the enterprise RAG module

This directory supplies a secured, version-pinned OpenSearch bootstrap profile,
an explicit BM25 + dense-vector index contract, atomic alias tools, and
dependency-free verification scripts. It is infrastructure for the production
RAG adapter; it does not implement embeddings, reranking, RRF, or Agent logic.

The checked-in assets have offline tests. A 2026-07-15 workstation run also
validated a real secured OpenSearch 3.7.0 node, synthetic ingestion, hybrid/ACL
queries, restart persistence, and an isolated snapshot restore; see the
[local evidence](evidence/local-opensearch-validation.md). Production corpus
quality, load, HA, remote backup, and security approval remain **NOT_RUN**.

## Layout

```text
deploy/rag-search/
├── .env.example
├── compose.yaml
├── entrypoint.sh
├── config/
│   └── index-template.json
└── scripts/
    ├── bootstrap_index.py
    ├── container_healthcheck.sh
    ├── health_check.py
    ├── http_utils.py
    ├── index_contract.py
    ├── integration_test.py
    ├── promote_index.py
    ├── restore_drill.py
    ├── smoke_test.py
    └── snapshot_index.py
```

## Security defaults

- OpenSearch Security remains enabled.
- The host port binds to `127.0.0.1` unless explicitly changed.
- No password, certificate, token, or private document is committed.
- Startup reads the initial admin password from a mounted regular file and
  fails when the default `/dev/null` placeholder is still configured.
- HTTP helpers reject redirects, ignore ambient proxy variables, cap response
  sizes, reject credentials in URLs, and allow plain HTTP only on loopback.
- Compose bootstrap TLS uses the image's demo certificate and explicit
  `OPENSEARCH_INSECURE=true`. Replace it with internal PKI and set the flag to
  false before treating the deployment as production.

The demo `admin` identity is for bootstrap only. The application reader,
indexer, alias promoter, and snapshot operator must use separate least-privilege
identities in an actual deployment.

## Index identity and retrieval contract

The bootstrap creates a concrete index such as `puncture-rag-v000001` and adds:

- `puncture-rag-read` for production retrieval;
- `puncture-rag-write` for ingestion into the active generation.

The mapping stores lexical `text`/`title` fields, a Lucene HNSW
`knn_vector`, stable document/chunk/parent IDs, owner, exact version, lifecycle
status, ACL scopes, source checksum, and parser/chunker/embedding provenance.
The mapping `_meta` records the embedding model/revision/dimension, exact query
and document instructions, vector-normalization policy, immutable tokenizer
revision, and positive maximum input-token limit. The bootstrap and promotion
scripts reject a mapping that differs from the configured manifest.

`metadata` is a source/display `flat_object`; it must not be treated as an
arbitrary exact-filter API. Approved exact filters are canonicalized by the
indexer into `metadata_terms` keyword values such as `equipment_line=line-03`.
The adapter accepts only a configured key allowlist and emits exact `term` or
`terms` clauses against `metadata_terms`. Unknown keys are rejected. If a future
backend cannot push a mandatory filter into both recall branches, it must
over-fetch, post-filter, and apply the same ACL/version checks again before
output; it must never silently omit the filter.

## Minimal local bootstrap

```bash
cd deploy/rag-search
cp .env.example .env
install -m 600 /dev/null secrets/opensearch-admin-password
printf '%s\n' 'replace-with-a-strong-local-password' > secrets/opensearch-admin-password
```

Set `OPENSEARCH_ADMIN_PASSWORD_FILE_HOST` in `.env` to the absolute password
file path. The example selects `Qwen/Qwen3-Embedding-0.6B`; replace the model and
tokenizer revision placeholders with the exact immutable checkpoint revision.
Review the query/document instructions, normalization flag, dimension, and
maximum input-token limit as one versioned manifest. Bootstrap intentionally
rejects `main`, `latest`, `unspecified`, and `SET_*` revisions.

Then follow `docs/rag-deployment-runbook.md`. In particular, render Compose,
pin the image digest, start the service, create the first index, and run the
read-only smoke test. Do not use this single-node profile as a production HA
topology.

## Offline verification

From the repository root:

```bash
python3 -m unittest tests.deployment.test_rag_search_assets -v
python3 run_tests.py
```

The opt-in live test only accepts an isolated index name beginning with
`puncture-rag-test-`, never a production index or alias:

```bash
RUN_RAG_INTEGRATION=1 \
RAG_TEST_INDEX=puncture-rag-test-$(date +%s) \
python3 deploy/rag-search/scripts/integration_test.py
```

It creates synthetic records, verifies BM25, dense retrieval, and an ACL-negative
case, then deletes only that explicitly validated test index.
