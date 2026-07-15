# 学习与秋招使用手册

这份手册面向两件事：真正读懂并能修改这个仓库，以及在秋招中用可验证的工程证据讲清楚它。建议先跑、再读、再改；不要先背技术名词。

开始前先读[项目成熟度与落地边界](project-status.md)。面试时最重要的可信度来自你能主动区分真实服务、独立测试和合成边界。

## 先建立可重复基线

### 1. 离线基线

```bash
cd /home/turnin/puncture-rd-agent-platform
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m puncture_agent.api.demo
python3 examples/local_rag_demo.py
python3 examples/local_mcp_demo.py
```

这组命令适合第一次看代码：不需要 GPU、PostgreSQL 或 OpenSearch，能分别观察 Run API、RAG 和 MCP 的最小闭环。

### 2. 真实本地组合

```bash
./deploy/local-demo/doctor.sh
./deploy/local-demo/run_demo.sh
```

不要跳过 `doctor.sh`。它能把 Python 依赖、PostgreSQL、Qwen/vLLM、OpenSearch 和本地安全配置分别报告出来，而且不会打印 DSN、密码、token 或原始异常。

### 3. 你要能解释输出

完成演示后，不要只说“跑通了”。你至少要回答：

- 为什么 `/health` 是 `DEGRADED` 而演示仍然成功？
- 哪些组件确实执行了网络请求，分别使用什么协议？
- 哪些结果仍由确定性合成工具生成？
- 同一个 idempotency key 为什么返回同一个 Run ID？
- JSON event replay 和 SSE replay 各解决什么问题？
- 为什么最终安全状态不能由 Qwen 直接给出？

## 30 秒项目介绍

> 这是一个 Contract-First 的企业研发 Agent 平台。我把自然语言任务通过 Qwen 做结构化理解，用带 ACL、版本和 Citation 的 Hybrid RAG 补充知识，再由固定图编排强类型工具；确定性 verifier 而不是 LLM 决定安全结论。平台侧实现了 FastAPI/SSE、PostgreSQL Run/Event、幂等与恢复协议、MCP、LangGraph、Trace/Eval。当前本地可以真实连接 Qwen/vLLM、OpenSearch 和 PostgreSQL，但医学算法工具仍是合成实现，所以它是可验证的平台工程 Demo，不是临床系统。

这段话中的每个名词都必须能指到代码或测试。如果你解释不了，就删掉该名词，不要硬背。

## 两分钟架构讲解

```text
HTTP/SSE request
  -> FastAPI: authentication, authorization, body/privacy boundary
  -> Run service: idempotency, lifecycle, event contract
  -> PostgreSQL: run snapshot, ordered events, CAS / execution intent
  -> executor
       -> Qwen/vLLM: task classification and structured output
       -> OpenSearch RAG: ACL-filtered hybrid retrieval and citations
       -> graph: deterministic routing across two fixed workflows
       -> typed tools: case-data / segmentation / planning-safety
       -> deterministic verifier: fail closed
  -> JSON report + replayable events + trace identity
```

本机组合里要补充一句：模型、检索、API 和数据库是真实服务；`IntegratedMockExecutor` 使用锁定 JSON 图和 `build_mock_handlers()`，算法结果是合成的。仓库另有真实 LangGraph `StateGraph` 与 PostgreSQL checkpoint/recovery 测试，但它不是当前一键 HTTP Demo 的 executor。

## 七天代码阅读路线

每天都要产出一页自己的笔记：画调用链、写一个你不懂的问题、运行一个测试、做一个小改动再还原。只读不跑很难形成面试能力。

| 天 | 目标 | 必读文件 | 当天要讲清楚的内容 |
|---|---|---|---|
| Day 1 | 契约优先 | [`contracts/README.md`](../contracts/README.md)、[`contracts/domain.py`](../contracts/domain.py)、[`contracts/tool_inputs.py`](../contracts/tool_inputs.py)、[`contracts/tool_outputs.py`](../contracts/tool_outputs.py) | 为什么 artifact ID、geometry fingerprint、枚举和错误码不能由模块私改 |
| Day 2 | Run 状态与持久化 | [`runtime/models.py`](../src/puncture_agent/runtime/models.py)、[`runtime/service.py`](../src/puncture_agent/runtime/service.py)、[`runtime/repository.py`](../src/puncture_agent/runtime/repository.py)、[`runtime/postgres_repository.py`](../src/puncture_agent/runtime/postgres_repository.py) | idempotency、version fence、CAS、事件序号和 COMMIT-unknown 对账 |
| Day 3 | 图与 verifier | [`graph/main_graph.json`](../graph/main_graph.json)、[`agent/runtime.py`](../src/puncture_agent/agent/runtime.py)、[`agent/verifier.py`](../src/puncture_agent/agent/verifier.py)、[`agent/langgraph_runtime.py`](../src/puncture_agent/agent/langgraph_runtime.py) | 两条子图如何路由、错误如何 fail closed、JSON Runtime 与 LangGraph 的区别 |
| Day 4 | 模型与 RAG | [`model_gateway/client.py`](../src/puncture_agent/model_gateway/client.py)、[`rag/retrieval.py`](../src/puncture_agent/rag/retrieval.py)、[`rag/opensearch.py`](../src/puncture_agent/rag/opensearch.py)、[`rag/client.py`](../src/puncture_agent/rag/client.py) | structured output、超时/重试、BM25+dense+RRF+rerank、ACL/版本/Citation |
| Day 5 | MCP 与工具边界 | [`mcp/runtime.py`](../src/puncture_agent/mcp/runtime.py)、[`mcp/ledger.py`](../src/puncture_agent/mcp/ledger.py)、[`tooling/catalog.py`](../src/puncture_agent/tooling/catalog.py)、[`tooling/factory.py`](../src/puncture_agent/tooling/factory.py) | 强类型工具、artifact 解析、权限、稳定 call ID、幂等账本和副作用边界 |
| Day 6 | API、SSE 与 worker | [`api/fastapi_app.py`](../src/puncture_agent/api/fastapi_app.py)、[`api/sse.py`](../src/puncture_agent/api/sse.py)、[`api/postgres_app.py`](../src/puncture_agent/api/postgres_app.py)、[`runtime/worker.py`](../src/puncture_agent/runtime/worker.py) | 可信身份如何注入、SSE 断点续传、lease/heartbeat/reclaim、SIGTERM 顺序 |
| Day 7 | Eval、证据和演示 | [`observability/tracing.py`](../src/puncture_agent/observability/tracing.py)、[`observability/eval_harness.py`](../src/puncture_agent/observability/eval_harness.py)、[`tests/eval`](../tests/eval)、[`deploy/local-demo`](../deploy/local-demo) | 如何证明质量没有回退、如何区分测试证据和生产 SLA、如何完成十分钟 Demo |

### 每天推荐的小练习

1. Day 1：给一个非法 geometry/enum 构造输入，找到它被拒绝的准确层级和错误类型。
2. Day 2：跟踪一次 `create_run`，画出 request fingerprint、Run version 和第一个 lifecycle event 的产生顺序。
3. Day 3：选择 `NO_FEASIBLE_PATH`，沿 JSON edge 找到终态，解释为什么不能自动缩小安全距离。
4. Day 4：在合成语料中增加一条新模块记录，先写失败断言，再让检索测试通过。
5. Day 5：挑一个工具，从 JSON Schema 一直追到 adapter port，标记哪一行开始进入合成实现。
6. Day 6：断开 SSE 后用同一个 cursor 重放；说明 high-water page 为什么能限制内存。
7. Day 7：修改一条 mock Eval 预期，观察 release gate 失败，再恢复文件并重跑。

## 三条必须亲自跟过的调用链

### HTTP Run 创建

1. [`fastapi_app.py`](../src/puncture_agent/api/fastapi_app.py) 完成 body admission、认证和 case 授权。
2. [`service.py`](../src/puncture_agent/runtime/service.py) 规范化请求并建立幂等执行意图。
3. [`postgres_repository.py`](../src/puncture_agent/runtime/postgres_repository.py) 在事务内保存 Run、事件和 version。
4. 本地同步 executor 进入 [`graph_executor.py`](../src/puncture_agent/runtime/graph_executor.py)。

### 模型与 RAG

1. executor 先通过 `ModelGateway` 请求固定结构的任务分类。
2. 再通过 `RagService` 请求限定 module 与 access scope 的检索。
3. 模型与 RAG 响应只成为计划和证据，不直接给出医学安全结论。
4. 图节点与 verifier 生成可审计的最终报告。

### 事件回放

1. executor 的 `emit` 形成固定事件类型和 node name。
2. Repository 为事件分配单调递增 sequence。
3. JSON endpoint 适合一次性查询；SSE 使用 cursor、heartbeat 和终态事件支持重连。
4. trace ID 贯穿 Run，但高基数或敏感值不会进入公共 metrics 标签。

## 十分钟面试演示脚本

### 0–1 分钟：边界先行

打开[项目成熟度页](project-status.md)，一句话说明哪些是真实服务、哪些是合成工具。主动说边界会提高可信度。

### 1–3 分钟：画架构

按“HTTP → Run service → PostgreSQL → Model/RAG → Graph/Tools → Verifier → Event/SSE”讲一遍。强调 LLM 负责理解与编排，确定性工具和 verifier 负责计算与签字。

### 3–5 分钟：依赖体检

```bash
./deploy/local-demo/doctor.sh
```

指出固定依赖版本、PostgreSQL 16、实际 Qwen model，以及为什么单节点 OpenSearch yellow 在本地被接受。

### 5–8 分钟：跑真实组合

```bash
./deploy/local-demo/run_demo.sh
```

只讲五个结果：两个 `SUCCEEDED`、真实 model 名、RAG chunk、幂等 replay、SSE terminal replay。不要逐行念 JSON。

### 8–10 分钟：打开关键代码

依次打开：

1. [`graph/main_graph.json`](../graph/main_graph.json)：固定路由与 fail-closed edge；
2. [`runtime/graph_executor.py`](../src/puncture_agent/runtime/graph_executor.py)：模型/RAG 注入与合成边界；
3. [`runtime/postgres_repository.py`](../src/puncture_agent/runtime/postgres_repository.py)：事务、CAS 和对账；
4. [`api/sse.py`](../src/puncture_agent/api/sse.py)：有界回放与重连。

最后用“真实公司算法、OIDC、Artifact Gateway 和真实语料是下一阶段”收尾。

## 简历表述模板

这些句子只有在你能回答对应代码追问后再使用，并按你亲自完成的部分改写。

- 设计并实现 Contract-First 企业研发 Agent 平台，将 Qwen/vLLM、Hybrid RAG、MCP 强类型工具、双工作流图编排与确定性 verifier 解耦，固定 10 个工具契约并阻止 LLM 直接作安全判定。
- 打通 FastAPI、PostgreSQL 16、Qwen2.5-3B/vLLM 与安全 OpenSearch 的本地真实链路，验证两条工作流、幂等创建、JSON/SSE 回放和 fail-closed 鉴权；保留合成算法边界并保存可复现证据。
- 实现 PostgreSQL Run/Event Repository 的 version CAS、事务事件序列与 COMMIT-unknown 对账，并以 lease/heartbeat/reclaim 支持恢复型 worker 的多进程故障测试。
- 构建带 ACL、版本过滤、parent-child chunk、BM25+dense、RRF、rerank 和 Citation 的 RAG 管线，完成严格 CA 的 OpenSearch 卷重建与 snapshot/restore 演练。
- 建立 701 项分层回归；在固定生产依赖和真实本地服务配置下达到 700 项通过、仅 1 项反向环境测试按设计跳过，并为环境差异增加不泄密 readiness doctor。

不要写“实现医学路径规划算法”“临床可用”“生产 exactly-once”“达到生产 SLA”。这些都没有对应证据。

## 三个 STAR 故事

### 故事一：把门控跳过变成真实验证

- **S**：大量 FastAPI/PostgreSQL/LangGraph/Qwen 测试因本机缺生产依赖或服务而门控跳过。
- **T**：把项目从离线骨架推进到可重复的真实本地组合，同时不能把凭据写入 Git。
- **A**：固定 Python 依赖，启动 PostgreSQL/Qwen/OpenSearch，逐层执行 no-skip 门；增加私有 `.env`、mode 0600、allowlist parser、readiness doctor 和一键 round trip。
- **R**：完整口径从 642 通过/59 跳过推进到 700 通过/1 个反向门控跳过，两条 HTTP 工作流现场成功，证据和边界均已保存。

### 故事二：恢复不等于口号

- **S**：长任务遇到 API SIGTERM、worker 租约过期或数据库 ACK 丢失时，简单重试可能重复副作用。
- **T**：让 Run 状态可恢复、旧 worker 不能继续提交，并能对不确定提交结果进行核对。
- **A**：采用 PostgreSQL execution intent、generation/owner fence、heartbeat/TTL reclaim、version CAS 和 canonical fingerprint；用稳定 `call_id` 的注入端口做多进程故障门。
- **R**：测试证明恢复进程完成同一 Run、旧 generation 被拒绝、测试副作用只记录一次；同时明确公司工具内部原子性仍需共享幂等/outbox。

### 故事三：安全配置也要可测试

- **S**：本地 Demo 需要数据库 DSN、OpenSearch 密码和 bearer token，shell 配置很容易被误提交或执行恶意展开。
- **T**：让启动足够简单，同时 fail closed 且不泄密。
- **A**：配置文件限定 mode 0600、非 symlink、key allowlist、literal parser；token 使用独占创建、单硬链接和 URL-safe 校验；客户端禁用代理和重定向并强制 loopback。
- **R**：安全资产测试覆盖命令替换保持字面量、公开权限拒绝、symlink/hardlink 拒绝、非 loopback URL 拒绝；真实体检和演示不打印秘密。

## 高频追问与回答框架

### 为什么用 Contract-First？

模型、RAG、工具、图和 API 可以并行演进，但 artifact、geometry、错误码和安全状态必须稳定。先锁定契约与 failure tests，能让真实适配器替换 mock 时不改变上层语义，也让跨语言/跨团队边界可验收。

### 为什么 LLM 不能决定安全结果？

LLM 输出不具备几何精度、确定性和可审计保证。它只负责意图理解、检索和工具路由；路径碰撞、mask、spacing、方向和最终状态由强类型工具与 verifier 判断。缺关键输入时必须 fail closed。

### Hybrid RAG 为什么不是只做向量检索？

工程文档同时含语义表达和精确术语/版本号。BM25 擅长精确 token，dense 擅长语义召回，RRF 在分数不可直接比较时融合排序，再以 reranker 精排；ACL、版本和 Citation 则保证“能查到”不等于“有权使用且能追溯”。

### 为什么本地 RAG 使用确定性 embedding/reranker？

它让 ingestion、过滤、融合、Citation 和 OpenSearch 适配器可以重复验证，不把模型下载或随机质量混入工程门。它不代表真实语义质量；生产前必须换成批准模型并用内部 Golden Set 重新定阈值。

### 幂等是不是 exactly-once？

不是。幂等 key、稳定 call ID 和 ledger 可以让重复请求返回同一结果或避免已完成调用重放；但外部副作用与幂等记录之间仍可能有崩溃窗口。要接近有效 exactly-once，需要公司端事务、outbox/inbox 或可对账的业务键。

### 为什么 Run snapshot 和 event 都要存？

snapshot 提供快速当前状态，event 提供有序审计和流式回放。两者必须在同一事务/version 规则下推进，否则会出现状态已完成但终态事件丢失，或事件存在而 snapshot 未更新。

### 为什么用 SSE 而不是 WebSocket？

这里主要是服务端单向事件流。SSE 原生支持 HTTP、`Last-Event-ID` 语义和代理友好的重连；配合数据库 cursor 可以从持久事件继续回放。双向低延迟交互才更适合 WebSocket。

### 为什么健康状态是 DEGRADED？

本地 profile 没有生产 Artifact Gateway，OpenSearch 也是单节点且有未分配 replica。依赖可用不代表所有生产能力齐全，所以返回 200/`DEGRADED` 比伪装 `UP` 更诚实；真正依赖宕机会导致启动或 round trip 失败。

### 一键 Demo 用的是 LangGraph 吗？

不是直接使用生产 LangGraph executor。它使用锁定 JSON 图的 `GraphRuntime`，便于把真实 Qwen/RAG/API/DB 与确定性工具组合。真实 `StateGraph`、checkpoint、interrupt/resume 和 PostgreSQL 恢复在独立集成/故障测试中验证。面试时必须主动说清这一区别。

### 为什么测试允许 skip？

门控测试必须明确区分“代码失败”和“环境未配置”。无依赖基线允许固定数量 skip；配置对应服务的 CI/live job 则对目标测试禁止 skip。这样离线开发可用，生产路径又不会被假绿色掩盖。

### 这个项目最难的部分是什么？

不是调用模型 API，而是跨模型、检索、数据库和工具副作用维持稳定契约：身份与 artifact 不能越权，失败要可分类，重试不能让旧执行继续提交，事件与状态要一致，安全结论必须可核验。

### 下一步你会先做什么？

先在公司沙箱选两个真实工具做端到端适配，同时接 OIDC/Artifact Gateway；再用真实小语料和 Golden Set 替换合成 embedding。只有真实输入、权限和副作用边界建立后，扩全工具与做 HA/性能才有意义。

## 证据索引

| 你说出的结论 | 应打开的证据 |
|---|---|
| Qwen 在本机 GPU 实际推理 | [`deploy/qwen-vllm/evidence/local-rtx4090-validation.md`](../deploy/qwen-vllm/evidence/local-rtx4090-validation.md) |
| OpenSearch 使用严格 CA 并做过恢复 | [`deploy/rag-search/evidence/local-opensearch-validation.md`](../deploy/rag-search/evidence/local-opensearch-validation.md) |
| HTTP 全栈两条工作流实际成功 | [`deploy/local-demo/evidence/local-full-stack-validation.md`](../deploy/local-demo/evidence/local-full-stack-validation.md) |
| LangGraph 分支、checkpoint 和恢复已实现 | [`docs/langgraph-runtime-implementation.md`](langgraph-runtime-implementation.md) |
| API、SSE、PostgreSQL worker 的实现边界 | [`docs/fastapi-runtime.md`](fastapi-runtime.md) |
| Trace/Eval 的指标和门 | [`docs/eval-and-tracing-implementation.md`](eval-and-tracing-implementation.md) |
| 整体完成度与剩余排期 | [`docs/project-status.md`](project-status.md) |

## 面试前检查清单

- [ ] 能在不看稿的情况下画出七层调用链。
- [ ] 能解释 JSON Runtime 与真实 LangGraph 测试的区别。
- [ ] 能从一次 Run ID 追到 PostgreSQL snapshot、event sequence 和 SSE terminal event。
- [ ] 能解释 RRF、ACL、Citation、version filter，而不只说“向量数据库”。
- [ ] 能解释 idempotency、CAS、lease、fence 和 outbox 的边界。
- [ ] 能现场运行 doctor 与 full-stack Demo，并知道日志位置。
- [ ] 能说出至少三个失败场景及对应 fail-closed 行为。
- [ ] 简历中的每个数字都能打开本仓库证据页或测试定位。
- [ ] 主动声明没有真实医学算法、临床结论和生产 SLA。

真正完成这张清单后，这个项目就不只是“GitHub 上有代码”，而是你能设计、运行、排障、验证并诚实答辩的一套工程作品。
