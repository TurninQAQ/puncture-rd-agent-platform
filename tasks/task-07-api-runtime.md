# Task 07: API and Runtime

## Copy-paste prompt for another model

在本仓库将内存 Run Service 实现为 FastAPI + SSE + 持久化 checkpoint。开始前完整阅读 `specs/fastapi-gateway.md`、`contracts/**`、`graph/**`、`docs/testing-guide.md`、Mock 实现和现有测试。保持 Mock 的公共行为，不得自行修改 Agent State、事件字段、状态枚举、错误码和工具返回结构。

## Allowed files

- `src/puncture_agent/api/**`
- `src/puncture_agent/runtime/**`
- `tests/api/**`
- 与本任务直接相关的部署配置

不得修改其他模块来绕过失败测试。

## Required implementation

1. 使用固定 Contract 创建 Pydantic request/response adapter。
2. 实现本文规定的全部 REST endpoint 和 OpenAPI。
3. 实现 SSE 有序事件、last sequence 断点续传和断连处理。
4. 实现 idempotency key 唯一约束。
5. 实现 PostgreSQL checkpoint repository；保留内存 repository 供测试。
6. 实现 approve/reject、cancel、resume 的状态机保护。
7. 实现 tenant/project/case 级权限检查和输出脱敏。
8. 将 dependency timeout、tool error、graph error 映射为固定错误结构。
9. 增加 health/metrics，但不得在 metrics label 中使用 case_id 等高基数字段。
10. 提供启动配置、数据库迁移和本地测试说明。

## Required tests

实现 `specs/fastapi-gateway.md` 的 API-001 至 API-016，并覆盖：

- HTTP schema 与标准库 Contract 一致；
- 两个不同 tenant 使用相同 idempotency key 不串线；
- SSE 重连不重放已确认事件、不跳序；
- approval ID 错误、过期和重复时不再次执行工具；
- SIGTERM/进程重启后恢复；
- 已完成 tool call 依据 call/idempotency ID 去重；
- 数据库短暂超时可重试且不会产生两个 Run；
- internal URI、Authorization、患者标识不出现在响应和日志。

## Commands

```bash
python3 run_tests.py
python3 -m unittest discover -s tests/api -p 'test_*.py' -v
```

生产依赖安装后，再运行项目新增的 HTTP、SSE 和 PostgreSQL 集成测试命令，并把命令写入交付报告。

## Forbidden

- 不得在 endpoint 中直接写 Agent 业务逻辑。
- 不得绕过 Runtime 直接调用医学算法工具。
- 不得用内存全局变量冒充生产 checkpoint。
- 不得因恢复而重复执行已完成的有副作用工具。
- 不得向未授权调用方泄漏资源是否存在或内部 URI。

## Delivery report

最终回答必须给出：修改文件、OpenAPI endpoint 表、状态机说明、checkpoint schema、恢复/幂等策略、权限模型、API-001 至 API-016 结果、性能结果、失败注入结果、已知限制和契约变更声明。

## Done when

Contract/API/Integration/Restart/Authorization 测试全部通过，Mock 测试仍可运行，且进程重启不会重复已完成工具调用。
