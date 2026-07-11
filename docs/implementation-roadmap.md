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

## Phase 4: LangGraph

根据 `graph/*.json` 构建真实 StateGraph，复用现有 node 输入输出。

## Phase 5: runtime and evaluation

接入 PostgreSQL Checkpoint、FastAPI、SSE、OpenTelemetry 与回归 Harness。

## Change policy

实现者只能修改任务卡允许的文件。如果契约不合理，应先提交“契约变更说明”，列出影响模块和测试，不得直接修改公共字段。
