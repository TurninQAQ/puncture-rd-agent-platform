# 项目成熟度与落地边界

更新时间：2026-07-15 UTC

这份状态页回答三个问题：仓库里哪些能力是真的、哪些只在测试或本机验证过、离不同意义上的“落地”还差什么。所有结论都按当前代码和已保存证据描述，不把合成算法或本机结果包装成生产能力。

## 当前结论

- **学习和秋招演示：已可用。** 可以一条命令跑通真实 FastAPI、PostgreSQL、Qwen/vLLM、OpenSearch、两条 Agent 工作流和 JSON/SSE 事件回放。
- **Agent 平台工程 POC：主体已完成。** 契约、模型网关、RAG、MCP、图编排、Run API、持久化、恢复协议和 Eval/Trace 都有代码与测试。
- **公司内部沙箱接入：尚未完成。** 缺真实公司算法、OIDC/IAM、共享 Artifact Gateway、真实语料和生产 embedding/reranker。
- **生产或临床落地：不能宣称。** 尚无目标集群 HA、生产安全审计、受控性能门、真实数据质量门和合规验收。

## 能力矩阵

| 能力 | 当前实现 | 已有证据 | 必须诚实说明的边界 |
|---|---|---|---|
| 公共契约与 Artifact | 固定领域契约、SQLite Registry、原子本地对象存储、发布与审计 | Contract/Artifact 测试 | 本地全栈演示未注入生产 Artifact Gateway；共享 Registry 的 HA/负载未验证 |
| Qwen 模型网关 | OpenAI 兼容 vLLM 客户端、结构化输出、工具调用、流式、重试与安全解码 | RTX 4090 上 Qwen2.5-3B/vLLM 现场验证 | 是本机单卡结果，不是目标生产模型、集群或容量批准 |
| 企业 RAG | 摄取、BM25+dense、RRF、rerank、ACL/版本过滤、Citation、OpenSearch 适配器 | OpenSearch 3.7、严格 CA、合成语料检索、卷恢复与 snapshot/restore | 本地现场使用确定性 64 维 embedding/reranker；没有内部语料质量结论 |
| MCP 与算法工具 | 3 个逻辑 Server、10 个强类型工具、Schema、鉴权、超时、幂等和 trace 边界 | stdio/official SDK/适配器测试与本地 Demo | 工具后端是确定性合成实现，不读取真实医学体素，也不替代公司算法 |
| Agent 图 | 锁定 JSON 主图/子图、确定性 Runtime；另有真实 LangGraph StateGraph、checkpoint、interrupt/resume 实现和测试 | 完整分支/故障矩阵、20 路隔离、PostgreSQL checkpoint/restart 证据 | 当前本地 HTTP 全栈接的是 JSON `GraphRuntime`；真实 LangGraph 路径是独立集成门，不能说全栈 Demo 已使用它 |
| Run API | FastAPI、Pydantic v2、固定错误、可信授权端口、幂等创建、JSON/SSE 回放、低基数 metrics | 真实 FastAPI/PostgreSQL 组合和本地 HTTP round trip | 本地认证是 loopback bearer token，不是 OIDC；API 没有生产 TLS/反向代理 |
| PostgreSQL 与恢复 | Run/event Repository、CAS、COMMIT-unknown 对账、execution lease、worker heartbeat/reclaim | PostgreSQL 16、多进程 SIGTERM/TTL reclaim、restart/process-kill 测试 | 一键演示因合成 executor 不具备公司副作用恢复契约而使用同步执行 |
| Trace 与 Eval | OpenTelemetry 兼容 facade、传播、脱敏、版本化数据集、RAG/Agent 指标和回归门 | 46 项定向测试、3/3 mock reference Eval | 未连接真实 Langfuse/Phoenix/OTLP 后端，也没有内部 Golden Set |
| 本地部署体验 | Qwen Compose、OpenSearch Compose、私有配置、依赖体检、一键 API 演示和证据页 | `doctor.sh` 返回 ready；两条工作流现场成功 | API 本身未容器化；OpenSearch 是单节点 yellow；三个依赖需先启动 |

## 可复现验收

```bash
cd /home/turnin/puncture-rd-agent-platform
./deploy/local-demo/doctor.sh
./deploy/local-demo/run_demo.sh
```

验收时至少确认：

- `ready=true`，PostgreSQL 与 Qwen 为 `UP`；
- 单节点 OpenSearch 可为预期的 `DEGRADED/yellow`；
- `data_validation` 与 `planning_safety` 都为 `SUCCEEDED`；
- RAG chunk 数不小于 2，规划验证为 `PASS`；
- bearer 反例、幂等重放和终态 SSE 都通过；
- 输出明确保留 `algorithm_backend=deterministic-synthetic`。

当前完整测试口径是 701 项：无生产依赖时 642 通过、59 项按环境门控跳过；固定实现依赖与 PostgreSQL 就绪时 695 通过、6 项互斥环境门控跳过；再连接真实 Qwen/vLLM 后 700 通过、仅 1 项“生产依赖必须缺失”的反向测试按设计跳过。

## 离不同层级的“落地”还有多远

下面是工程排期参考，不是交付承诺。假设 1–2 名熟悉 Python/平台工程的开发者、公司接口文档和脱敏沙箱数据能及时提供；时间只覆盖 Agent 平台集成，不包含医学算法研发、采购和合规审批。

| 目标 | 当前距离 | 主要剩余工作 | 粗略工程时间 |
|---|---|---|---|
| 本地学习/求职演示 | 已达到 | 熟悉代码并反复讲解演示 | 现在即可使用 |
| 公司内部沙箱 POC | 中等 | 10 个工具中选 2–3 个接真实端口、OIDC/IAM、Artifact Gateway、真实 RAG 小语料与 Golden Set | 约 3–6 周 |
| 受控试点 | 较大 | 全工具接入、共享幂等/outbox、异步恢复 executor、TLS/Secret、监控告警、负载与故障演练 | 在 POC 后约 4–8 周 |
| 生产发布 | 很大且依赖外部 | 多主机 HA/DR、容量模型、安全与隐私审计、全量质量门、目标硬件性能门、运维手册与审批 | 在试点后约 8–16 周以上 |

如果真实算法接口、脱敏数据或安全基础设施没有按时提供，后三档不能靠仓库内部代码单独完成，时间也不能继续可靠估算。

## 下一步优先级

### P0：公司沙箱入口

1. 注入真实 OIDC/IAM principal 和 case/project authorizer。
2. 注入共享 Artifact Gateway，禁止空 artifact 的演示捷径。
3. 选一个 case-data 工具和一个 planning 工具实现真实适配器与故障门。
4. 用内部脱敏文档替换合成 RAG 语料，接生产 embedding/reranker 并建立 Golden Set。

### P1：恢复与副作用

1. 让公司工具适配器提供稳定 `call_id`、共享幂等记录或 outbox。
2. 将真实 recovery-safe executor 接入 durable worker，而不是沿用同步演示配置。
3. 做容器删除、宿主机失败、WAL crash、网络分区和重复回调现场演练。

### P2：生产治理

1. API TLS/反向代理、Secret Manager、网络 egress 和审计策略。
2. OpenSearch/PostgreSQL HA、备份恢复目标和容量评估。
3. 受控硬件上的模型/RAG/API 延迟、吞吐、显存和 soak 门。
4. 真实 Trace 后端、告警、Runbook、发布回滚和数据保留策略。

## 证据入口

- [本地全栈现场证据](../deploy/local-demo/evidence/local-full-stack-validation.md)
- [Qwen/vLLM RTX 4090 现场证据](../deploy/qwen-vllm/evidence/local-rtx4090-validation.md)
- [OpenSearch 现场证据](../deploy/rag-search/evidence/local-opensearch-validation.md)
- [实现路线与已知缺口](implementation-roadmap.md)
- [测试指南](testing-guide.md)

## 对外表述红线

- 可以说“真实连接 Qwen/vLLM、OpenSearch、PostgreSQL 和 FastAPI”。
- 必须同时说“算法工具和本地 embedding/reranker 是确定性合成实现”。
- 可以说“实现并测试了 LangGraph durable execution 路径”。
- 不要说“一键全栈演示使用了生产 LangGraph worker”；它目前使用同步 JSON Runtime。
- 不要说“实现了医学算法、验证了临床安全、达到 exactly-once 或生产 SLA”。
- 不要把本机单卡、单节点和 GitHub runner 数据外推为生产容量。
