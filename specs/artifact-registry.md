# Artifact Registry Specification

## 1. Goal and boundary

Artifact Registry 以不可变 `artifact_id` 连接 CT、MCS、NIfTI、分割、薄皮肤、路径和报告，避免三维数组、患者信息或内部文件路径进入 LLM 上下文。

本模块只管理元数据、状态、校验和、几何指纹、访问控制和血缘。它不解析 MCS/NIfTI，不执行分割，也不判断医学结果是否正确。

## 2. State model

```text
PENDING -> AVAILABLE
PENDING -> INVALID
AVAILABLE -> INVALID
AVAILABLE -> MISSING
```

规则：

- `AVAILABLE` 后内容不可覆盖；新内容必须产生新 `artifact_id`。
- `INVALID`、`MISSING` 不得被工具解析为可消费输入。
- `MISSING` 表示元数据仍存在，但底层对象已不可访问；生成或校验失败使用 `INVALID`。
- 状态只能沿上图方向变化，不能回退。
- `internal_uri` 只对授权的确定性工具返回，Agent 和 LLM 只能看到 artifact 元数据。
- 父制品必须存在；子制品保存全部直接父节点 ID。

## 3. Fixed public operations

真实实现必须保留 Mock 暴露的名称、参数和返回类型：

1. `begin_registration(...)`
   - 创建 `PENDING` 记录。
   - 输入包括 artifact 类型、内部 URI、创建者、父节点、工具版本、参数和可选几何。
   - 不得把尚未完成写入的对象标成 `AVAILABLE`。
2. `finalize(artifact_id, checksum, size_bytes)`
   - 原子校验记录状态并转为 `AVAILABLE`。
   - 重复提交同一 checksum 应幂等；不同 checksum 应冲突。
3. `fail(artifact_id, reason)`
   - 将 `PENDING` 标为 `INVALID` 并保留失败原因。
4. `invalidate(artifact_id, reason)`
   - 将 `AVAILABLE` 标为 `INVALID`，但保留历史和血缘。
5. `mark_missing(artifact_id, reason)`
   - 将底层对象不可访问的 `AVAILABLE` 记录标为 `MISSING`。
6. `get_metadata(artifact_id)`
   - 返回脱敏元数据，不返回 `internal_uri`。
7. `resolve_uri(artifact_id, principal)`
   - 只有授权 principal 且状态为 `AVAILABLE` 时返回 URI。
8. `find_available_by_idempotency_key(key)`
   - 仅返回 `AVAILABLE` 制品。
9. `get_lineage(artifact_id)`
   - 返回可复现的父/子关系；必须检测循环引用。

## 4. Identifiers and fingerprints

### 4.1 Geometry fingerprint

使用规范化后的以下字段生成稳定 SHA-256：

- `shape`；
- `spacing`；
- `origin`；
- `direction`；
- 坐标系标识。

浮点值先按契约约定精度规范化。字段顺序固定，序列化必须稳定。相同几何跨进程生成相同指纹；任一字段变化都应改变指纹。

### 4.2 Idempotency key

推荐构成：

```text
sha256(tool_name + tool_version + sorted_input_artifact_ids
       + canonical_parameters + geometry_fingerprints)
```

不得只使用病例 ID，否则参数或模型版本变化时会错误复用旧结果。

## 5. Reference production design

- PostgreSQL 保存元数据、状态、ACL 和血缘。
- MinIO/S3/NAS 保存真实二进制对象。
- `finalize` 使用数据库事务和唯一约束，避免并发重复注册。
- 大文件 checksum 可由上传服务计算后传入，Registry 负责持久化与校验策略。
- URI 解析写审计日志，日志不得包含患者信息。

以上是推荐实现，不是 Contract 的一部分；其他存储方案只要通过相同测试即可。

## 6. Test fixtures

至少准备：

- `ct_ready`: 合法 CT artifact；
- `segmentation_ready`: 以 `ct_ready` 为父节点；
- `same_geometry`: 相同 shape/spacing/origin/direction；
- `different_spacing`: 仅 spacing 不同；
- `pending_upload`: 尚未 finalize；
- `unauthorized_principal`: 无 URI 解析权限。

测试不得依赖真实医学文件，可使用固定字节串和伪 URI。

## 7. Correctness test matrix

| ID | Scenario | Expected result |
|---|---|---|
| AR-001 | 创建 registration 后查询 | 状态为 `PENDING`，URI 不出现在 metadata |
| AR-002 | 正常 finalize | 状态转为 `AVAILABLE`，checksum/size 固定 |
| AR-003 | 对 AVAILABLE 使用相同 checksum 再 finalize | 返回同一结果，不新增版本 |
| AR-004 | 对 AVAILABLE 使用不同 checksum 再 finalize | 返回结构化冲突错误，原记录不变 |
| AR-005 | PENDING 标记生成失败 | 状态为 `INVALID`，不能 resolve/reuse |
| AR-006 | AVAILABLE 标记 invalid/missing | 历史和 lineage 保留，不能再作为工具输入 |
| AR-007 | 相同几何计算两次 | fingerprint 完全一致 |
| AR-008 | 仅改变 spacing/direction | fingerprint 必须变化 |
| AR-009 | 未授权 principal resolve | 结构化权限错误，不泄漏 URI |
| AR-010 | 缺失父 artifact | 注册失败，不产生孤立记录 |
| AR-011 | 构造循环 lineage | 操作被拒绝，已有血缘不变 |
| AR-012 | 相同幂等键并发注册 | 只有一个 AVAILABLE 结果可被复用 |
| AR-013 | 不同工具/参数/版本 | 生成不同幂等键，不错误复用 |
| AR-014 | 未知 artifact_id | 返回 NOT_FOUND 类错误而非空对象 |

## 8. Failure and performance tests

- 模拟对象写入成功但数据库 finalize 失败，确认不会暴露半成品。
- 模拟数据库超时，调用方获得可重试错误，不能误判为 NOT_FOUND。
- 对 10,000 条内存记录执行 metadata/idempotency 查询，记录 p50/p95；生产实现目标由部署环境另定，但测试必须输出基线。
- 连续 100 次相同请求验证无额外 AVAILABLE 记录。

## 9. Test locations and commands

- Contract: `tests/contract/test_artifact_contracts.py`
- Unit: `tests/artifacts/test_registry.py`
- Integration: `tests/integration/test_mock_workflow.py`

```bash
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m unittest discover -s tests/artifacts -p 'test_*.py' -v
```

## 10. Acceptance

- 本文 AR-001 至 AR-014 全部通过。
- 所有公共测试保持标准库可运行。
- AVAILABLE 不可覆盖，INVALID/MISSING 制品不可复用，URI 不向 LLM/API 元数据泄漏。
- 任一输出都能追溯到输入 artifact、工具版本、参数和几何指纹。
- 实现者提交测试结果、性能基线、已知限制和契约变更声明。

## 11. v0.2.0 implementation profile

The checked-in implementation now contains:

- `InMemoryArtifactRegistry`: deterministic test double with case-scoped idempotency;
- `SQLiteArtifactRegistry`: durable single-node metadata registry using WAL, foreign keys, busy timeout, `BEGIN IMMEDIATE`, conditional updates and active-scope uniqueness;
- `LocalArtifactStore`: same-filesystem private staging, service-side SHA-256, fsync and atomic immutable publication;
- `ArtifactPublicationService`: one-process coordination of registration, object publication, finalization, authorized reads and redacted access audit;
- canonical JSON and tool invocation idempotency-key generation;
- recursive persisted-lineage cycle detection;
- restart, concurrency, lock-timeout, rollback, traversal, symlink, orphan-cleanup and 10k benchmark evidence.

This profile is intentionally limited to one application node and one local storage volume. It is suitable for development, interview demonstration and controlled internal single-node deployment. It is not HA. Multiple API/Celery workers or shared remote storage require PostgreSQL plus MinIO/S3 leases/transactions while preserving the same contracts and tests.

Benchmark evidence: `benchmarks/results/v0.2.0-artifact-registry.json`.
