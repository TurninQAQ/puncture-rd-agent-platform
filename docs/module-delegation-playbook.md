# Module Delegation Playbook

本文件用于把仓库中的单个模块交给其他模型实现，同时避免它随意改接口或只给出无法验证的示例代码。

## 1. How to delegate

每次只发送一个 `tasks/task-*.md`，并同时提供任务卡中明确引用的：

- 对应 `specs/*.md`；
- 对应 `contracts/*.py`；
- 现有 Mock/Stub；
- 对应测试目录；
- `docs/testing-guide.md`。

不要一次让同一个模型重写整个仓库。推荐按 `docs/implementation-roadmap.md` 的顺序实现，每完成一个模块先集成和回归，再进入下一个模块。

## 2. Standard instruction appended to every task

可以把下面内容附在发送给模型的消息末尾：

```text
你只能修改任务卡 Allowed files 中的文件。contracts、公共枚举、错误码、图节点名和返回结构不可擅自修改。
先运行基线测试，再实现；实现后运行任务卡全部测试和全量回归。
每个外部依赖都要支持 mock/failure injection。
不要用“示例代码”“伪实现”冒充完成；未运行的测试必须明确标注未运行。
最终按任务卡 Delivery report 输出证据。如果接口有问题，先提交契约变更提案，不要直接改接口。
```

## 3. Review checklist

收到其他模型的实现后，逐项检查：

1. 是否只改了允许文件。
2. 是否保留固定输入输出和错误码。
3. 是否同时覆盖正常、边界、非法、依赖失败、幂等、性能场景。
4. 是否有可重复运行的 fixture，而非依赖作者机器上的绝对路径。
5. 是否真的运行了测试，并给出命令、数量和结果。
6. 是否把 secret、患者信息、内部 URI 写进日志或 Prompt。
7. 是否把确定性安全结论交给 LLM 自由生成。
8. 是否记录模型、Prompt、索引、工具和配置版本。
9. 是否保留 Mock，便于其他模块尚未完成时独立测试。
10. 全量 `python3 run_tests.py` 是否仍通过。

## 4. Rejection conditions

满足任一项就应退回修改：

- 删除或弱化失败测试；
- 为通过测试而硬编码某个 fixture 的完整输出；
- 修改 Contract 但未说明影响范围；
- 捕获所有异常后仍返回成功；
- 依赖失败时给出 SAFE/NORMAL；
- 日志、事件或响应泄漏内部 URI/敏感信息；
- 没有幂等策略导致昂贵工具重复执行；
- 只写 README，没有可执行测试；
- 测试未运行却声称全部通过。

## 5. Integration gate

模块只有同时满足以下条件才可以替换 Mock：

- 对应 Contract Tests 通过；
- 模块 Unit/Failure Tests 通过；
- Mock 端到端 Integration Test 通过；
- 真实依赖的最小集成测试通过；
- 性能基线已记录；
- 已知限制已列出；
- 回滚到 Mock 的配置路径仍可用。
