# Benchmarks

Benchmarks are evidence scripts, not correctness tests. Always record the
commit, Python version, OS, storage type, record count and command.

Module 0:

```bash
python3 benchmarks/artifact_registry_benchmark.py --backend memory --records 10000
python3 benchmarks/artifact_registry_benchmark.py --backend sqlite --records 10000
```

The SQLite command measures a single local file and must not be used as a
PostgreSQL/MinIO or multi-node performance claim.

LangGraph PostgreSQL checkpoint evidence:

```bash
export PUNCTURE_TEST_POSTGRES_DSN='postgresql://...'
python3 benchmarks/langgraph_postgres_checkpoint_benchmark.py \
  --warmups 5 \
  --rounds 3 \
  --samples-per-round 50 \
  --environment-label controlled-benchmark-host \
  --storage-label local-postgresql-16 \
  --output benchmark-results/langgraph-postgres-checkpoint.json
```

The script uses the real synchronous `PostgresSaver.put()` path and records:

- checkpoint-save P50/P95 across prepare and resume writes;
- public `LangGraphRuntime.resume()` end-to-end P50/P95 after opening a new
  saver/runtime;
- per-round raw samples, `put_writes`, run-to-interrupt latency and serialized
  checkpoint size diagnostics;
- commit, dirty state, Python/package/PostgreSQL versions, CPU, OS, sample count
  and explicit non-secret environment/storage labels.

`checkpoint_save_ms` is specifically the latency of each successful synchronous
`PostgresSaver.put()` call. `put_writes` latency remains a separate diagnostic
and is not folded into that primary metric; the JSON workload metadata records
this boundary explicitly. P50 uses the median and P95 uses nearest-rank.

The workload uses immediate fake Model/RAG/tools and interrupts immediately
before `report_generator`; completed planning tools must not replay during
resume. Saver connection setup, schema migration and graph compilation are
outside the timed resume interval; production advisory-lease acquisition stays
inside the public `resume()` measurement. The checked-in v1 schema is
`benchmarks/schemas/langgraph-postgres-checkpoint-v1.schema.json`.

Each benchmark thread is deleted after its terminal checkpoint is re-read and
verified. PostgreSQL may still retain dead tuples until VACUUM/autovacuum, so
later rounds can include that shared-runner storage jitter.

The original Task 06 gates remain fixed at save P95 <= 50 ms and resume P95 <=
150 ms. Default `record` mode writes the observed result without failing a
shared runner solely for latency jitter; correctness violations always fail.
Use `--enforce-thresholds` or `PUNCTURE_ENFORCE_PERFORMANCE_GATES=1` only on a
controlled benchmark host. GitHub-hosted runner artifacts are engineering
baselines, not production SLA evidence. Never pass the DSN as a command-line
argument or include it in the environment/storage labels.

Recorded GitHub-hosted baseline on 2026-07-11:

- commit `66f193d1c5d62c2c248c90be5e6ea33c2724c09a`;
- [workflow run 29146879180](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29146879180), Ubuntu 24.04, Python 3.10 and PostgreSQL 16;
- 5 warm-ups, 3 rounds x 50 measured sessions;
- checkpoint `put()` P50/P95: `3.131 / 3.606 ms`;
- public `resume()` P50/P95: `23.429 / 25.829 ms`;
- fixed `50 / 150 ms` thresholds observed as passed in `record` mode;
- artifact `postgres-checkpoint-benchmark-66f193d1c5d62c2c248c90be5e6ea33c2724c09a`,
  29,637 bytes, digest
  `sha256:0a1bd4a5539294ad12c6932238fa97205c4bd24796aa310b9cfea7362e73a44e`.

This is a shared GitHub-runner engineering observation. Run with enforcement on
the intended dedicated PostgreSQL/storage host before treating the limits as a
release or production performance gate.
