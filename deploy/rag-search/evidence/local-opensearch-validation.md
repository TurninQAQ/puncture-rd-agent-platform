# Local OpenSearch RAG Validation

Validation date: 2026-07-15 UTC

This is a non-production workstation record. It proves that the checked-in
OpenSearch deployment, strict-TLS provider adapter, hybrid retrieval controls,
and backup/restore path can run together against synthetic data. It does not
prove production retrieval quality, availability, security approval, or an SLA.

## Exact identity and scope

- Git baseline before this evidence update: `b1d15785d05d3d493eebbada3655ddff1b1381bd`
- Image: `docker.m.daocloud.io/opensearchproject/opensearch@sha256:123e6591a47b1d54686890551bdb35739c85193ecded381219fc9e059e18128f`
- OpenSearch: `3.7.0`, build `72121f014083f9ca010fd5a7da83b2ec4886027f`
- Lucene: `10.4.0`; JVM: Temurin OpenJDK `25.0.3+9-LTS`
- Host: Linux `5.15.0-185-generic`, Docker `29.1.3`, 62 GiB RAM
- Topology: one data/cluster-manager node, 1 GiB minimum and maximum JVM heap
- Exposure: Security plugin enabled, HTTPS on loopback `127.0.0.1:9200`,
  provider clients validating a copied local root CA
- Live index: `puncture-rag-v000001`, with one-to-one
  `puncture-rag-read` and `puncture-rag-write` aliases
- Local development embedding manifest: `deterministic-hash` revision `1`,
  64 dimensions, normalized vectors, and tokenizer revision
  `rag-regex-tokenizer-v1`

Only synthetic route-clearance and restricted-yield exercise records were
written. No company document, patient data, or proprietary algorithm output was
used.

## Startup, retrieval, and authorization gates

- Bootstrap installed the strict mapping, created the versioned index, and
  attached both aliases without replacing an existing generation.
- The Compose smoke test passed alias, mapping, BM25 mandatory-filter, and
  filtered k-NN request checks against the live node.
- The isolated integration test created a reserved disposable index, passed
  exact BM25, dense semantic, and ACL-negative assertions, and deleted the index.
- `examples/live_opensearch_rag_demo.py` used the production
  `OpenSearchSearchBackend` and strict provider transport. An authorized hybrid
  query returned `live-demo-clearance-v1-chunk-0`; the exact restricted token
  returned no evidence under the wrong scope; the correct restricted scope
  returned `live-demo-restricted-v1-chunk-0`.
- The service was recreated after adding `path.repo`; the named data volume
  recovered the index, aliases, mapping, and four synthetic parent/child records.
- The container remained healthy with zero restarts and no OOM kill during the
  final gates.

The application reports `DEGRADED`, and cluster health is yellow, because the
checked-in local topology is one node while the index retains one replica. This
is expected for the bootstrap topology and is not accepted production health.

## Snapshot and restore drill

The filesystem repository `local-rag-fs` was registered only after the snapshot
volume was made writable by the OpenSearch UID. Snapshot
`puncture-rag-local-20260715t-validation01` completed with state `SUCCESS` and no
shard failure for `puncture-rag-v000001`.

The first restore attempt exposed a real defect: restoring snapshot aliases
would attach the production write alias to both the live and renamed indexes.
The reusable drill now forces `include_aliases=false`, requires a previously
absent `puncture-rag-restore-*` target, waits for completion, checks zero failed
shards, compares the full source/restored mapping and document count, and deletes
only the isolated target. The final drill passed with four restored records and
confirmed cleanup.

The local filesystem repository shares the host and Docker failure domain with
the data volume. It verifies mechanics only; it is not a production backup.

## Environment limitations and remaining gates

- Host `vm.max_map_count` was `65530`; OpenSearch warns that at least `262144`
  is required. The shared host was not changed without system-owner approval.
- An ignored local Compose override makes the otherwise internal network
  non-internal so Docker publishes the loopback-only port. Production topology
  must retain private networking, approved ingress, and managed certificates.
- The local demo uses the bootstrap admin identity and image demo PKI. Production
  needs least-privilege reader/indexer/promoter/snapshot identities and internal
  PKI or an approved service mesh.
- Deterministic embeddings and reranking prove adapter and authorization wiring,
  not Qwen Embedding/Reranker quality or availability.
- Real corpora, ingestion reconciliation, labelled Recall/NDCG/MRR, latency/load,
  ACL policy evaluation, HA/node loss, disk pressure, upgrade, long soak, remote
  backup, and disaster recovery remain unrun production gates.
