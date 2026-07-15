# Enterprise RAG search deployment runbook

## 1. Outcome, boundary, and evidence status

This runbook deploys the search storage and recall substrate for the internal
enterprise RAG module. The selected backend is OpenSearch. The production Agent
adapter performs query normalization, two independent recalls, deterministic
RRF, reranking, parent expansion, final authorization checks, citations, and
trace emission; OpenSearch does not replace those application controls.

Repository evidence covers the offline controls below. A 2026-07-15 workstation
run additionally covered an immutable OpenSearch 3.7.0 image, secured loopback
startup, synthetic parent/child ingestion, live BM25/k-NN/ACL behavior, named
volume restart persistence, and filesystem snapshot/isolated restore. See the
[local evidence](../deploy/rag-search/evidence/local-opensearch-validation.md).
The following remain **NOT_RUN** production gates:

- approved production embedding and reranker generation;
- HA, node loss, disk pressure, upgrade, and rollback drills;
- golden-set Recall@10/NDCG/MRR, ACL leak count, or P50/P95 latency.

Do not turn target thresholds into measured resume claims until these gates have
real evidence.

## 2. Version baseline and primary references

The bootstrap image is explicitly versioned as
`opensearchproject/opensearch:3.7.0`; production must resolve and approve its
registry digest. The baseline was inspected on 2026-07-10 against:

- OpenSearch `3.7.0` tag commit
  [`72121f014083f9ca010fd5a7da83b2ec4886027f`](https://github.com/opensearch-project/OpenSearch/tree/72121f014083f9ca010fd5a7da83b2ec4886027f);
- OpenSearch documentation commit
  [`23666cc24059637feff11502def16cdd2bf8fe91`](https://github.com/opensearch-project/documentation-website/tree/23666cc24059637feff11502def16cdd2bf8fe91);
- Qwen3 Embedding source commit
  [`44548aa5f0a0aed1c76d64e19afe47727a325b8f`](https://github.com/QwenLM/Qwen3-Embedding/tree/44548aa5f0a0aed1c76d64e19afe47727a325b8f),
  whose documented `Qwen/Qwen3-Embedding-0.6B` profile has a 1024-dimensional
  output, supports instructions and normalized embeddings, and supports up to a
  32K model context;
- the official [Docker installation guide](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/),
  [k-NN vector mapping guide](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/),
  [efficient k-NN filtering guide](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/),
  [Aliases API](https://docs.opensearch.org/latest/api-reference/alias/aliases-api/),
  and [Snapshot APIs](https://docs.opensearch.org/latest/api-reference/snapshots/).

OpenSearch CLI, image defaults, Security plugin behavior, vector mappings, and
snapshot compatibility are release-sensitive. Record the exact image digest,
JVM, host kernel, storage class, certificates, plugin list, and client commit in
every deployment evidence package.

## 3. Target topology and identities

```text
Agent Runtime / EnterpriseRagClient
        |
        | mutually authenticated private HTTPS
        v
internal gateway / allowlist / request limits
        |
        v
OpenSearch data nodes ---- versioned puncture-rag-vNNNNNN indexes
        |                         |
        |                         +-- puncture-rag-read
        |                         `-- puncture-rag-write
        v
approved snapshot repository
```

The checked-in Compose file is a one-node bootstrap topology. A production
deployment needs the organization-approved availability design, dedicated
cluster-manager/data roles where appropriate, persistent encrypted storage,
shard sizing, capacity alerts, and tested snapshot recovery.

Create separate Security-plugin identities or upstream service accounts:

| Identity | Minimum responsibility |
|---|---|
| `rag-reader` | search/count/mapping-read on `puncture-rag-read` only |
| `rag-indexer` | bulk/create/update on the staged concrete index only |
| `rag-promoter` | mapping inspection and alias updates only |
| `rag-snapshotter` | snapshot repository/status/create for approved indexes |
| break-glass admin | offline, audited, never used by Agent Runtime |

Never give the LLM credentials or allow it to select an OpenSearch identity.

## 4. Host and secret preflight

Check Docker, Compose, memory map limits, free disk, and time synchronization:

```bash
docker version
docker compose version
sysctl vm.max_map_count
df -h
timedatectl status
```

Set `vm.max_map_count` to the approved OpenSearch host value before startup.
Do not change a shared host without the infrastructure owner.

Prepare configuration:

```bash
cd deploy/rag-search
cp .env.example .env
mkdir -m 700 -p secrets
install -m 600 /dev/null secrets/opensearch-admin-password
```

Write a strong bootstrap password through the organization's secret delivery
mechanism. Set `OPENSEARCH_ADMIN_PASSWORD_FILE_HOST` to its absolute path. The
default `/dev/null` is deliberately rejected, so an accidental start is closed.

No direct password setting exists in `.env.example`. Do not put credentials in
Compose, shell history, command-line arguments, source control, logs, trace
attributes, or benchmark output.

## 5. Pin and render the release

Resolve `opensearchproject/opensearch:3.7.0` to an approved immutable digest and
set `OPENSEARCH_IMAGE=opensearchproject/opensearch@sha256:...`. Also pin the
Python verification image digest when used in a controlled environment.

Set these index-contract values before bootstrap:

```text
RAG_EMBEDDING_MODEL=<approved internal model identifier>
RAG_EMBEDDING_REVISION=<immutable revision>
RAG_EMBEDDING_DIMENSION=<actual output dimension>
RAG_QUERY_INSTRUCTION=<exact query prefix used by the embedding adapter, or empty>
RAG_DOCUMENT_INSTRUCTION=<exact document prefix used at indexing, or empty>
RAG_VECTORS_NORMALIZED=<true only when stored and query vectors are L2-normalized>
RAG_TOKENIZER_REVISION=<immutable tokenizer revision>
RAG_MAX_INPUT_TOKENS=<positive adapter-side input limit>
RAG_SCHEMA_VERSION=<mapping contract version>
RAG_PARSER_VERSION=<parser release>
RAG_CHUNKER_VERSION=<chunker release>
```

The `.env.example` Qwen3 demo uses the 0.6B model, dimension `1024`, an English
retrieval instruction for queries, no document instruction, normalized vectors,
and an adapter-side limit of `8192` tokens. The model repository documents a
larger model context, but raising the deployed input limit is a separate capacity
and quality change. Replace both model and tokenizer revision placeholders with
the exact private checkpoint revision. Do not use `main`, `latest`, `unspecified`,
or any `SET_*` placeholder as an immutable revision. Then render:

```bash
docker compose --profile serve --profile bootstrap --profile verify config
```

Inspect the output for the approved image digest, loopback exposure, Security
plugin enabled, password-file mounts, expected heap, persistent volumes, and no
secret value. `OPENSEARCH_INSECURE=true` is allowed only for the isolated demo
certificate bootstrap. Production must mount internal-PKI certificates, set an
approved CA file, and set it false.

## 6. Understand the index contract

`config/index-template.json` supplies both BM25 and dense fields:

| Area | Contract |
|---|---|
| lexical | `title`/`text` standard fields plus whitespace code-preserving subfields |
| dense | Lucene HNSW `knn_vector`, cosine similarity, explicit dimension |
| identity | stable `document_id`, `chunk_id`, `parent_id`, canonical/supersedes IDs |
| authorization | exact keyword `access_scopes` inherited onto every child |
| lifecycle | exact `version`, `status`, timestamps, checksum |
| embedding manifest | model/revision/dimension, exact query/document instructions, vector-normalization policy, tokenizer revision, and maximum input tokens |
| provenance | owner plus parser/chunker versions |
| exact metadata | allowlisted canonical `metadata_terms` keyword values |
| display metadata | non-authoritative `metadata` flat object |

The mapping is `dynamic: strict`; malformed or unapproved fields fail ingestion.
OpenSearch mappings cannot require non-null fields, so the indexer must reject a
parent or child missing identity, owner, ACL, version, status, checksum, and
provenance before bulk indexing.

The five embedding-manifest controls are compatibility fields, not descriptive
notes. Query and document instructions may be empty, but the keys must exist.
`vectors_normalized` must be a JSON boolean, `max_input_tokens` must be a positive
integer, and tokenizer/model revisions must be immutable. If any field differs
between the active mapping and the running embedding adapter, retrieval fails
closed with an incompatible-index error; do not silently rewrite query inputs.

`metadata` is not an arbitrary safe filter namespace. Convert only approved
keys to canonical `metadata_terms`, for example `equipment_line=line-03`, with a
single normalization function shared by ingestion and query validation. Reject
unknown keys and values containing control characters. Never interpolate a key
or query fragment into raw OpenSearch JSON.

## 7. Start, bootstrap, and verify

Run offline tests first:

```bash
python3 -m unittest tests.deployment.test_rag_search_assets -v
bash -n deploy/rag-search/entrypoint.sh
bash -n deploy/rag-search/scripts/container_healthcheck.sh
```

Start the secured node:

```bash
docker compose --profile serve up -d opensearch
docker compose --profile serve ps
```

Bootstrap installs the template, creates `puncture-rag-v000001`, verifies the
mapping model/revision/dimension, instructions, normalization flag, tokenizer
revision, maximum input tokens, and parser/chunker versions, then adds read/write
aliases only when no live alias already exists:

```bash
docker compose --profile serve --profile bootstrap run --rm bootstrap-index
```

It refuses placeholders, incompatible mappings, unsafe identifiers, and an
attempt to overwrite an existing alias. Run the read-only smoke suite:

```bash
docker compose --profile serve --profile verify run --rm smoke-test
```

The smoke test verifies cluster health, one-to-one read/write alias resolution,
mapping provenance, a BM25 query with mandatory ACL/status/doc-kind filters, a
filtered k-NN query, and count response shape. It uses a reserved scope that
must never be assigned to real documents and requires zero hits. It does not
claim retrieval quality.

## 8. Production adapter query requirements

The adapter runs lexical and dense recall independently. Both requests must
push these filters into OpenSearch:

- effective authenticated `access_scopes` intersection;
- allowed `module` values;
- exact `required_version` when provided;
- active lifecycle policy unless an explicitly authorized old version is asked;
- allowlisted `metadata_terms` filters;
- `doc_kind=child` for retrieval candidates.

The dense branch uses the mapping-compatible vector dimension and a filtered
Lucene k-NN query. The lexical branch preserves exact identifiers through BM25
and the code-preserving subfields. The adapter deduplicates by `chunk_id`, runs
RRF/reranking, and enforces the same ACL/version/lifecycle constraints again
before returning any title, count, text, citation, or trace detail.

If a mandatory filter cannot be pushed into a branch, over-fetch and post-filter
under a bounded policy; never omit the filter for speed. Unauthorized document
existence, title, count, score, or failure details must not be observable.

## 9. Ingestion and reconciliation

Use explicit bulk bounds, timeouts, retries, and dead-letter evidence. Recommended
sequence:

1. validate stable document identity, owner, ACL, version, lifecycle, checksum;
2. parse and heading-chunk while preserving tables, APIs, error codes, and units;
3. inherit mandatory metadata onto every child;
4. generate embeddings with the exact mapping model/revision/dimension,
   query/document instruction policy, normalization policy, tokenizer revision,
   and maximum input-token limit;
5. index into a new concrete generation, never the live alias during rebuild;
6. reconcile source, parent, child, indexed, failed, and duplicate counts;
7. compare checksums and sample parent-child links;
8. run authorization-negative and retrieval golden-set gates;
9. only then promote aliases.

A partial bulk failure never triggers alias promotion. Retain the old live index
and the failed-run manifest for diagnosis.

## 10. Atomic migration and rollback

Create a new monotonically named generation, such as
`puncture-rag-v000002`. Bootstrap the template separately if its schema changed,
then create/index/verify that concrete index without an alias.

Before promotion, require:

- mapping `_meta` exactly matches the full embedding manifest and parser/chunker release;
- zero failed records and checksum/count reconciliation;
- all ACL/version negative tests pass with zero leakage;
- approved golden-set gates pass;
- a successful snapshot of the current concrete index exists;
- monitoring and rollback owner are active.

Promote both aliases in one OpenSearch Aliases API request with a compare-by-name
guard:

```bash
python3 deploy/rag-search/scripts/promote_index.py \
  --new-index puncture-rag-v000002 \
  --expected-current puncture-rag-v000001
```

The script first validates the target mapping and confirms that both live aliases
still point to the expected old index. It sends one atomic remove/add action set
and never deletes the old generation.

Rollback uses the same operation in reverse after confirming the old generation
is still compatible with the running adapter:

```bash
python3 deploy/rag-search/scripts/promote_index.py \
  --new-index puncture-rag-v000001 \
  --expected-current puncture-rag-v000002
```

After any switch, run the smoke suite and authorization-negative probes. Do not
delete either generation until the retention window and recovery review pass.

## 11. Snapshot, restore, and backup evidence

Register an approved `fs`, S3, or other supported repository through the
OpenSearch Snapshot API. For a filesystem repository, every relevant node must
have the same repository path in `path.repo`; the Compose volume alone is not a
production backup because it shares the same failure domain.

Verify the repository, then create a non-partial snapshot of the concrete live
index:

```bash
python3 deploy/rag-search/scripts/snapshot_index.py \
  --repository approved-rag-repository \
  --snapshot puncture-rag-before-v000002-20260710t120000z
```

The script resolves the read alias to one concrete index, sets
`include_global_state=false`, `partial=false`, waits, and requires `SUCCESS` with
no shard failures.

At least once per release cycle, restore into a differently named isolated index,
verify mapping metadata, counts, checksums, BM25/vector queries, ACL negatives,
and then delete only the isolated restore. A backup without a successful restore
drill is not accepted evidence.

The checked-in drill provides the safe mapping/count portion of that rehearsal:

```bash
python3 deploy/rag-search/scripts/restore_drill.py \
  --repository approved-rag-repository \
  --snapshot puncture-rag-before-v000002-20260710t120000z \
  --restore-index puncture-rag-restore-v000001-release-check
```

It refuses targets outside `puncture-rag-restore-*`, refuses an existing target,
sets `include_aliases=false` so production aliases cannot be attached to the
restore, compares the full mapping and document count, and deletes only the
isolated target. Run BM25/vector/ACL probes against the isolated target as an
additional release gate when using real corpus snapshots.

## 12. Opt-in live integration test safety

The live script requires both `RUN_RAG_INTEGRATION=1` and an index beginning
`puncture-rag-test-`. It rejects `puncture-rag-v*`, production aliases, wildcard,
path, uppercase, and ambiguous names before connecting:

```bash
RUN_RAG_INTEGRATION=1 \
RAG_TEST_INDEX=puncture-rag-test-local-001 \
python3 deploy/rag-search/scripts/integration_test.py
```

It writes only synthetic documents, tests exact BM25 recall, dense semantic
recall, and an unauthorized perfect match, then deletes only the validated test
index. Run it with a restricted integration identity that has no permission on
production indexes.

## 13. Monitoring and release gates

Monitor cluster health, JVM heap/GC, disk watermarks, shard count/size, indexing
failures, search rejection/timeout, k-NN latency, request size, alias target,
snapshot age/failure, and application-stage latency. Do not put chunk text or
confidential queries in general metrics.

Release requires measured evidence for:

- Recall@5/10, MRR, NDCG@10, correct-version hit rate;
- exactly zero ACL leaks across unit, live integration, and labelled evaluation;
- unanswerable/empty-result accuracy;
- BM25-only, dense-only, hybrid, reranked, and parent-context ablations;
- P50/P95 latency with corpus size, query count, hardware, model revision, and
  concurrency recorded;
- snapshot/restore, alias promotion, and rollback drills.

Initial targets in the service specification are engineering gates, not current
measurements.

## 14. Incident actions

- **ACL leak:** disable Agent RAG traffic, preserve traces under restricted
  access, roll back the index/adapter if safe, rotate affected credentials, and
  complete security review before reopening.
- **Embedding mismatch:** stop promotion. A dimension, model/tokenizer revision,
  instruction, normalization, or input-limit change creates a different embedding
  manifest. Build and evaluate a new versioned index; never relabel old vectors.
- **Partial indexing:** keep old aliases, quarantine the staged index, reconcile
  failures, and rerun from a deterministic manifest.
- **Latency/disk pressure:** protect availability first with admission control and
  bounded recall; do not remove authorization/version filters.
- **Corrupt live generation:** roll aliases to the retained compatible generation
  or restore a verified snapshot into a new index; never restore over the only
  remaining good copy.
