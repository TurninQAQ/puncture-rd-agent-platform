# Contract Policy

`contracts/` 是项目的接口事实源。

当前使用 Python 标准库 dataclass/Enum，保证骨架在无第三方依赖环境中可运行。真实工程可以在不改变字段、枚举和值语义的前提下迁移为 Pydantic v2。

契约变更必须同步：

1. Contract Tests；
2. 所有 Stub/Mock；
3. 图状态字段；
4. API文档；
5. 对应任务卡。

## 工具契约入口

- `tool_inputs.py`：10 个固定请求 dataclass；
- `tool_outputs.py`：10 个固定结果 dataclass；
- `common.py`：`ToolCallContext`、`ToolResponseEnvelope` 和 JSON 序列化；
- `geometry.py`：IJK 索引、毫米世界坐标和几何指纹；
- `artifacts.py`：内部 `ArtifactRef` 与不泄露 URI 的 `ArtifactPublicView`；
- `errors.py` / `enums.py`：稳定错误码和 wire values。

`ArtifactStatus` 只有四种语义：`PENDING`（登记但未完成）、`AVAILABLE`
（校验完成且可消费）、`INVALID`（生成或校验失败）、`MISSING`（引用存在但
底层对象不可访问）。其他模块不得再创建平行的 artifact 状态枚举。

运行契约快照：

```bash
PYTHONPATH=.:src python3 -m unittest discover -s tests/contract -p 'test_tool_contracts.py' -v
```
