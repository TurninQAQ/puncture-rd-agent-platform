# Implementation Roadmap

## Phase 0: contracts

冻结 Artifact、Geometry、Tool Request/Response、Error Code、Agent State 与 Graph Spec。

## Phase 1: model gateway

用真实 Qwen/vLLM 替换 Mock；保持客户端和响应契约不变。

状态：代码与部署资产已完成离线验收；2026-07-15 已在单张 RTX 4090 上用
Qwen2.5-3B/vLLM 0.9.1 完成真实启动、四能力 smoke、5 项 live gateway 测试和并发
1/4 基准。该[本机现场证据](../deploy/qwen-vllm/evidence/local-rtx4090-validation.md)
不是默认 v0.25.0 镜像、目标生产 GPU/集群、长时 soak 或生产 SLA 的批准记录。

## Phase 2: RAG

实现文档摄取、Hybrid Retrieval、Rerank、版本与 ACL、Citation；保持 `RetrievalRequest/Response` 不变。

状态：本地确定性 Demo、离线 Eval、OpenSearch/Qwen provider 适配器与部署资产已完成；
2026-07-15 又在真实 OpenSearch 3.7.0 上完成严格 TLS、合成数据摄取、Hybrid/ACL 查询、
服务重建后的数据卷恢复以及 snapshot/隔离 restore 演练，详见
[本机现场证据](../deploy/rag-search/evidence/local-opensearch-validation.md)。Embedding/Reranker
仍为确定性开发实现；真实 Qwen provider、内部 Golden Set、受控性能与生产 HA 待执行。

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
配置与跨 runtime 恢复测试已落地。跨 Runtime 的 `run/resume/stream` 已接 SQLite TTL/CAS
测试替身和生产 PostgreSQL session advisory-lock manager，并覆盖竞争、续租、锁丢失和
fail-closed 行为。MCP 桥接层现已在调用前和接收远端结果后查询可信 Artifact Registry，校验
case、ACL、AVAILABLE 状态、类型、完整几何、生产者版本和直接父链；远端自声明的 Artifact
公开视图不能成为自身的信任来源。GitHub Actions 已在 Python 3.10/3.11/3.12 与 PostgreSQL
16 上分别执行 6 项禁止 skip 的真实数据库测试，覆盖 checkpoint 恢复、interrupt、竞争、
backend termination/takeover 和 stream durability；独立 restart job 也已证明同一 PostgreSQL
cluster 的 postmaster 重启后可由新 Python 进程按相同 checkpoint 哈希恢复，且已完成工具不
重放。真实双进程门还已在 ledger 完成、MCP 响应返回但图 checkpoint 尚未提交时强制
SIGKILL，并证明新进程恢复后不重复目标底层副作用。容器替换/宿主机故障、PostgreSQL
SIGKILL/WAL crash recovery、工具内部副作用与 ledger commit 的原子性、跨主机共享账本和
API 接线仍待完成，详见
`docs/langgraph-runtime-implementation.md`。

Task 06 的关键成功/失败矩阵现已直接运行于 LangGraph 1.2.9 `StateGraph`：覆盖 MCS/规划
成功、缺输入、geometry/label 失败、无可行路径、一次重试、重试耗尽、不可重试错误和
malformed model。20 个 worker 通过同步屏障同时进入真实节点，并交替完成 10 个规划与
10 个 MCS 会话；每个工具序列、case 请求和完整 checkpoint 均已核对，未发现跨会话泄漏。

## Phase 5: runtime and evaluation

Task 07 的第一个生产边界节点已完成：Pydantic v2 request/response/error adapter 不接受
body 伪造的 tenant/principal/role/scope，可信权限只能由认证上下文注入；原始影像字段、
敏感 metadata、任意 URI、Bearer/JWT/token 和非 JSON/非有限值在进入 Runtime 前拒绝。
所有 snapshot/event/error 公共视图执行深度脱敏，未知内部错误只返回固定 500/503 消息。
Pydantic 2.13.4 已显式固定，commit `3c4c6fc` 的三版本 CI 独立 no-skip gate 已通过。

Task 07 的第二个生产边界节点也已完成：新增框架无关 `RunRepository` 协议和线程安全内存参考
实现，以内部单调 version fence 保护 executor 所有事件和终态提交；Run 创建、tenant-scoped
idempotency claim、`RUN_CREATED/RUN_STARTED`，以及 approval/success/failure/cancel 状态事件均
原子提交。普通流只接受 NODE/TOOL 事件，所有写入口执行 1 MiB 有界 JSON 规范化和 detached
copy，Service/Repository 双层拒绝非法 outcome/snapshot 字段组合。100 路同 key 创建只执行一次，
100 个并发事件连续编号，cancel/完成、审批缓冲、approve/resume 竞争和 rollback 均已覆盖。
commit `189040b` 的三版本 CI、PostgreSQL restart、benchmark 与真实 process-kill job 全部通过。
普通执行事件又新增不进入公共契约的 version-scoped `event_key`：同 key/同内容重放返回原
sequence，不同内容固定冲突，取消后的 exact replay 仍会再次检查 fence 并停止旧 executor；
canonical SHA-256 区分 `true`/`1` 等 JSON 类型。commit `15af386` 的完整远端矩阵已通过。

Task 07 的第三个生产边界节点已完成：PostgreSQL Run/event Repository 使用显式 checksum
migration、tenant/idempotency 唯一约束、行锁和单调 sequence/version；snapshot/event 动态
JSON 同时保存 canonical text 与 JSONB，严格 UTC 毫秒时间边界避免数据库 round-trip 改变
fingerprint。CAS mutation journal 与状态、lifecycle event 同事务提交，使 COMMIT 回执丢失后
即使状态已经继续推进，原 mutation 仍可精确对账。commit `93391f4` 的 Python 3.10/3.11/3.12
workflow 各自通过 8 项 PostgreSQL 16 no-skip 测试，包含 20 路幂等创建、100 个并发事件、
tenant 隔离、rollback、backend termination，以及 create/append/CAS 在真实 commit 成功后丢失
ACK 的恢复。该历史节点尚不包含 FastAPI endpoint、SSE、原始 HTTP body 解析前限流、OIDC、
worker reclaim/heartbeat 或 API 进程恢复；其中前四项中的 FastAPI/body/SSE 已由后续节点补齐。

Task 07 的第四个生产边界节点已完成：FastAPI 提供全部 9 条固定 REST/OpenAPI 路径，Bearer
认证在 body 缓冲与 JSON 解析前执行，tenant/project/case 权限、Artifact metadata 和公司
Executor 均为 fail-closed 注入端口。原始 body admission、固定错误、隐私输出、health、低基数
HTTP metrics、PostgreSQL composition 和显式 startup migration 已由 commit `b8da00f` 的三版本
CI 及 7/13/8/1 项精确 no-skip 门验证。

Task 07 的第五个生产边界节点现已实现：同一事件路径默认返回有界 JSON 页，显式
`text/event-stream` 返回 SSE；`Last-Event-ID`/`after_sequence` 使用严格 exclusive cursor，
PostgreSQL 以 high-water + keyset page 无锁读取。流在 200 前完成认证、授权、首屏读取和编码，
启动后每轮重新验证 Bearer 并授权，按固定帧发送、空闲 heartbeat、终态 tail drain，并在超时、
撤权、断连和 ASGI send failure 时释放 per-process 全局/tenant 配额。8 项独立 SSE 测试包含
10,000 事件有界回放；commit `9958756` 的三版本 CI、PostgreSQL restart、真实 process-kill
和 checkpoint benchmark 已全部通过。生产代理慢消费者、集群级配额与 HTTP/PostgreSQL
性能仍需现场证据。

Task 07 的第六个生产边界节点现已实现：PostgreSQL v2 migration 新增私有
`run_execution_jobs`，把 CREATE、完整审批决定和 RESUME intent 与 Run version 持久绑定；
worker 使用 DB clock、generation、owner token、heartbeat 和 lease，以 `SKIP LOCKED` 并发
claim，并在所有事件/终态写事务内验证 active claim。FastAPI production composition 默认延迟
执行，migration 后启动 worker，SIGTERM lifespan 停止新 claim、在 grace 内续租并传递 stop，
超时后停止 heartbeat、等待 TTL 接管而不立即释放。真实双进程 probe 已在注入端口提交稳定
PostgreSQL `call_id` 后终止进程 A，进程 B 以 generation 2 恢复同一 Run；副作用记录为 1、
事件连续且只有一个 RUN_COMPLETED。公司算法仍只保留 `RecoverableRunExecutor` 接口，该证据
不替代公司系统自己的幂等/outbox 或 GPU 取消实现。

SQLite 工具回放账本及本机重启/并发/不确定状态证据已完成，跨 worker 租约代码、
确定性双 Runtime 证据、PostgreSQL 16 三版本 CI 和 service restart/独立进程恢复也已完成。
应用进程在 replay ledger 完成后、graph checkpoint 前遭遇 `SIGKILL` 的恢复门也已完成。
GitHub 托管 Ubuntu 24.04/PostgreSQL 16 的 3×50 checkpoint 基准也已记录，并通过原始 50/150
ms 观测阈值；它不是生产 SLA。
Task 08 已完成：OpenTelemetry 兼容 facade（graph/node/RAG/model/MCP/verifier/checkpoint）、W3C traceparent 与 MCP/gRPC metadata 传播、敏感属性 allowlist/redaction、版本化 Eval 数据集、确定性 RAG/Agent 指标、per-case 诊断、baseline 回归门、故障注入行、离线 CLI 与 tracing overhead 基准。CI 增加 46 项 eval/tracing 零跳过门、mock eval CLI 与 record-only benchmark。Live OTLP/Langfuse/Phoenix 仍为 opt-in 适配器；公司 OIDC principal hashing、远程采样器和真实内部 Golden Set 质量阈值待现场配置。
下一步聚焦公司算法端口接入、共享幂等账本/副作用原子性、容器/宿主机/WAL 故障现场验证，以及受控专用主机 enforce 基准。

## Change policy

实现者只能修改任务卡允许的文件。如果契约不合理，应先提交“契约变更说明”，列出影响模块和测试，不得直接修改公共字段。
