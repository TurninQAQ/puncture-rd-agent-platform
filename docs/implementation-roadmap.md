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
ACK 的恢复。FastAPI endpoint、SSE、原始 HTTP body 解析前限流、OIDC、worker reclaim/heartbeat
和 API 进程恢复尚未完成。

SQLite 工具回放账本及本机重启/并发/不确定状态证据已完成，跨 worker 租约代码、
确定性双 Runtime 证据、PostgreSQL 16 三版本 CI 和 service restart/独立进程恢复也已完成。
应用进程在 replay ledger 完成后、graph checkpoint 前遭遇 `SIGKILL` 的恢复门也已完成。
GitHub 托管 Ubuntu 24.04/PostgreSQL 16 的 3×50 checkpoint 基准也已记录，并通过原始 50/150
ms 观测阈值；它不是生产 SLA。下一步完成共享 PostgreSQL 账本及工具副作用原子性、容器
替换/故障恢复、受控专用存储主机 enforce 基准，并接入 FastAPI、SSE、OpenTelemetry 和
生产回归 Harness。

## Change policy

实现者只能修改任务卡允许的文件。如果契约不合理，应先提交“契约变更说明”，列出影响模块和测试，不得直接修改公共字段。
