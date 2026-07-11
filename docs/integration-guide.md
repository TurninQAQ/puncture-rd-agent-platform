# Integration Guide

## Replacing a mock module

1. 阅读对应 `specs/` 和 `tasks/` 文件。
2. 阅读该模块使用的 `contracts/`。
3. 先运行现有 Contract Tests，记录基线。
4. 只替换 Stub 内部实现，不修改签名和返回类型。
   Module 3 之后优先实现对应 Port，并通过 `build_adapter_registry()` 注入，
   不要在 MCP Runtime 中加入算法或环境分支。
5. 增加算法单元测试和失败测试。
6. 运行 `python3 run_tests.py`。
7. 运行该任务卡规定的验收场景。
8. 保存真实性能和正确性结果。

## Integration order

公共契约 -> Model/RAG -> MCP Tools -> Agent Runtime -> API/Trace/Eval。

## Required implementation report

每个模块实现完成后必须提供：

- 修改文件清单；
- 实现算法说明；
- 配置说明；
- 测试命令；
- 测试结果；
- 已知限制；
- 性能数据；
- 是否修改契约。
