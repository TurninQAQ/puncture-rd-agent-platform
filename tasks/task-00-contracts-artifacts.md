# Task 00: Contracts and Artifact Registry

## Copy-paste prompt for another model

在本仓库实现 Artifact Registry。先完整阅读 `contracts/README.md`、所有 `contracts/*.py`、`specs/artifact-registry.md`、`docs/testing-guide.md` 和现有测试。只允许修改下述文件；不得更改公共字段、枚举值、错误码和函数签名。先让现有测试保持通过，再实现代码和新增测试。

## Allowed files

- `src/puncture_agent/artifacts/**`
- `tests/artifacts/**`
- 如现有 Contract Test 与契约明显矛盾，只提交说明，不直接修改 `contracts/**`。

## Inputs and outputs

输入/输出以 `contracts` 中的数据类为唯一真源。禁止另建同义模型。内部可以使用数据库实体，但 API 边界必须转换回固定 Contract。

## Implementation steps

1. 实现 canonical JSON 与 geometry fingerprint。
2. 实现内存 Registry，使全部标准库测试通过。
3. 增加状态机保护和结构化错误映射。
4. 实现 checksum、idempotency lookup 和 lineage。
5. 实现 principal/role 驱动的 URI 解析授权。
6. 若要生产化，再增加 PostgreSQL repository 和对象存储 adapter；内存实现继续保留用于测试。
7. 对 finalize 和相同幂等键并发请求使用事务/唯一约束。
8. 增加失败注入和性能测试。

## Required tests

必须实现 `specs/artifact-registry.md` 的 AR-001 至 AR-014，并额外检查：

- 元数据序列化/反序列化不丢字段；
- shape 相同但 spacing、origin 或 direction 不同时不误判为相同几何；
- PENDING 写入失败不产生 AVAILABLE；
- AVAILABLE 无法覆盖；
- unauthorized principal 无法获得 URI；
- 100 次重复请求只复用一个 AVAILABLE 结果；
- lineage 顺序稳定且无循环。

## Commands

```bash
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m unittest discover -s tests/artifacts -p 'test_*.py' -v
```

## Forbidden

- 不得把 `internal_uri` 暴露到 Agent State、LLM Prompt、API metadata 或普通 trace。
- 不得为了通过测试修改固定错误码。
- 不得把真实文件内容写进 JSON Contract。
- 不得允许 INVALID/MISSING Artifact 作为 AVAILABLE 输入解析。

## Delivery report

最终回答必须列出：修改文件、核心设计、事务/并发策略、测试命令和结果、AR 用例覆盖表、性能数据、已知限制、是否修改契约。没有实际运行测试不得声称完成。

## Done when

全部 Contract/Artifact/Integration 测试通过，`specs/artifact-registry.md` 验收条件满足，且交付报告证据齐全。
