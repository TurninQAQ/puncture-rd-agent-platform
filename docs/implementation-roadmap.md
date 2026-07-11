# Implementation Roadmap

## Phase 0: contracts

冻结 Artifact、Geometry、Tool Request/Response、Error Code、Agent State 与 Graph Spec。

## Phase 1: model gateway

用真实 Qwen/vLLM 替换 Mock；保持客户端和响应契约不变。

状态：代码与部署资产已完成离线验收；目标 GPU 现场启动与性能证据待执行。

## Phase 2: RAG

实现文档摄取、Hybrid Retrieval、Rerank、版本与 ACL、Citation；保持 `RetrievalRequest/Response` 不变。

状态：本地 Python 3.10 Demo、离线 Eval、OpenSearch/Qwen provider 适配器与部署资产已完成；真实服务和内部 Golden Set 待执行。

## Phase 3: algorithm tools

按三个 MCP Server 分批替换 10 个 Mock。

状态：三个本地 MCP Server、十个强类型工具适配器、可注入公司算法端口、
Artifact 安全解析、权限/超时/幂等/Trace 以及 stdio Demo 已完成。当前使用
deterministic manifest backend，不声称重写或验证了公司 MCS、TensorRT、
形态学、路径规划或安全算法。

## Phase 4: LangGraph

根据 `graph/*.json` 构建真实 StateGraph，复用现有 node 输入输出。

状态：已新增生产 `StateGraph` 编译器、两个 compiled subgraph、TypedDict 状态边界、
同步 checkpoint/event 接口、动态 interrupt/resume、MCP/Model/RAG 适配器和并发隔离测试。
隔离安装的 LangGraph 1.2.9 已验证真实图、子图恢复和 MCP trace；PostgreSQL saver、CI 服务
配置与跨 runtime 恢复测试已落地。本机无 PostgreSQL 测试 DSN，真实数据库执行、工具返回后到
图 checkpoint 前的进程崩溃 exactly-once 证据、跨 worker 同 thread 锁、可信输出 Artifact
Registry 校验和 API 接线仍待完成，详见
`docs/langgraph-runtime-implementation.md`。

## Phase 5: runtime and evaluation

完成 PostgreSQL 运行证据与持久幂等账本，并接入 FastAPI、SSE、OpenTelemetry
和生产回归 Harness。

## Change policy

实现者只能修改任务卡允许的文件。如果契约不合理，应先提交“契约变更说明”，列出影响模块和测试，不得直接修改公共字段。
