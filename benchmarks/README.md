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
