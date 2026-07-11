# Puncture R&D Agent Platform

企业内部穿刺机器人研发协同 Agent 的 Contract-First 骨架工程。

本仓库当前提供：

- 固定的输入输出契约；
- 10 个算法工具的固定契约、Stub/Mock 与本地可替换适配器；
- Qwen、RAG、Agent Runtime、API、Trace/Eval 的接口骨架；
- 可运行的 Mock 端到端流程；
- Contract Tests；
- 可单独交给其他模型实现的任务卡与验收文档。
- 已实现的 Module 0：SQLite 持久化 Artifact Registry、原子本地对象存储、幂等发布与访问审计。
- 已实现的 Module 1：Qwen/vLLM OpenAI 兼容网关、严格工具/结构化输出校验、安全流式传输与私有化部署模板。
- 已实现的 Module 2：本地可运行的企业 Hybrid RAG、文档摄取、BM25 + dense + RRF + rerank、ACL/版本过滤、Citation、离线 Eval，以及 OpenSearch/Qwen Embedding/Reranker 可选适配器。
- 已实现的 Module 3：三个 MCP Tool Server、十个强类型工具适配器、JSON Schema/structuredContent、Artifact ID 安全解析、权限/超时/幂等/Trace、stdio JSON-RPC Demo，以及可替换的 MCS/TensorRT/规划算法端口。
- 主体已实现、尚未发布的 Module 4：由锁定 JSON 拓扑编译的 LangGraph 主图与两个子图、TypedDict/checkpoint 安全边界、同步事件流、interrupt/resume、Model/RAG/MCP 节点适配器、SQLite 工具回放账本、PostgreSQL saver 和跨 worker 同 thread advisory-lock 入口。
- Task 07 已完成的三个生产边界：Pydantic v2 HTTP 契约/隐私适配器、原子内存 Run Repository 参考实现，以及带显式 migration、tenant 幂等、连续事件、version CAS 和 COMMIT-unknown 对账的 PostgreSQL Run/event Repository。

当前仍未接入公司真实 MCS 解析、TensorRT 推理、医学图像算法、路径规划和安全评估。Module 3 的 deterministic backend 只验证 Agent/MCP 工程链路，不读取真实体素或替代公司算法。Module 4 已用隔离安装的 LangGraph 1.2.9 验证真实 `StateGraph`、子图 checkpoint、动态中断/恢复、跨 runtime 安全 checkpoint 恢复以及 MCP trace 传播；默认无第三方依赖环境仍会显式跳过这些门控测试。SQLite 工具回放账本已证明在终态工具响应和 ledger 完成后、Agent graph checkpoint 前强制 SIGKILL，重建 bridge/runtime 仍不会再次执行底层工具；工具内部副作用与 ledger commit 之间的原子性、跨主机共享账本仍需生产后端证明。跨 Runtime 的 `run/resume/stream` 同 thread single-flight 已用共享 SQLite 租约文件完成确定性故障测试，生产实现使用每次执行独占连接的 PostgreSQL session advisory lock；[三版本 PostgreSQL 16 现场运行](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29145487743) 已分别通过 6 项禁止 skip 的持久化、竞争、连接终止接管和 stream durability 测试，[service restart 现场运行](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29145858819) 还证明了不同 Python 进程在同一 PostgreSQL cluster 的 postmaster 重启后可按相同 checkpoint 哈希恢复 interrupt，且已完成的 candidate 工具不重放。容器删除后外部卷恢复、宿主机故障、PostgreSQL 进程 SIGKILL/WAL crash recovery 和网络分区仍未验证。远端输入输出 Artifact 已改为查询可信 Registry，并校验 ACL、case、状态、类型、完整几何、生产者版本和直接父链；生产共享 Registry 的 failover/load 仍需现场证明。Qwen/vLLM、OpenSearch、Qwen Embedding/Reranker 的代码与部署资产已经完成离线验收，但本仓库不声称已经在目标 GPU/集群上完成模型下载、服务启动或性能基准；现场步骤见对应 runbook。

## Core rule

后续实现可以替换 Stub 内部逻辑，但不得自行修改 `contracts/`、图节点名称、错误码或公共返回结构。若确需变更，必须先更新契约、测试和所有依赖模块。

## Quick start

环境仅需 Python 3.10+，当前 Mock 和测试不依赖第三方包：

```bash
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m puncture_agent.api.demo
python3 examples/local_rag_demo.py
python3 examples/local_mcp_demo.py
```

当前本地 Python 3.10 标准环境基线：执行 608 项测试，571 项通过、37 项门控跳过；使用隔离的 Pydantic 2.13.4/LangGraph 1.2.9/psycopg 依赖路径重跑同一套测试时，588 项通过、20 项门控跳过。graph/eval 定向套件分别为 135 项（128 通过、6 项 PostgreSQL 和 1 项反向依赖门控跳过）和 10 项全通过，其中 10 项测试实际执行真实 `StateGraph`（9 项 graph 集成/故障测试和 1 项 Eval）。完整关键失败矩阵与 20 路并发隔离现已在真实 `StateGraph` 上验证，确定性的 Fake API 仍保留为快速参考门。14 项租约测试覆盖 SQLite 双 manager 竞争/续租/过期接管、PostgreSQL 参数化 advisory-lock 协议、跨 Runtime `run/resume/stream` 互斥、不同 thread 并行和 lease-lost 人工核对路径；可信 Artifact 校验测试覆盖输入/输出的 case、状态、类型、完整几何、生产者版本和直接父链。5 项 PostgreSQL checkpoint benchmark 合同测试固定 P50 median/P95 nearest-rank、50/150 ms 阈值、阶段计时、无秘密 schema 和缺 DSN fail-closed 行为。Task 07 首个边界节点新增 9 项 API contract/privacy 测试，其中 7 项 Pydantic v2 测试在 CI 中禁止 skip；内存 Repository/并发套件现有 25 项，另有 2 项 dependency-free PostgreSQL 配置合同和 8 项真实 PostgreSQL 数据库测试。CI 已精确固定 Pydantic 2.13.4、LangGraph 1.2.9、PostgreSQL checkpointer 3.1.0 和 httpx 0.28.1，并使用受限版本范围安装 psycopg；commit `7c99712` 的完整 workflow 在三个 Python 版本上全部通过，且每个矩阵任务都使用 PostgreSQL 16 服务执行 6 项独立、禁止 skip 的持久化与锁测试。commit `eec87fe` 进一步通过独立 restart job，上传包含 before/after/tool-call 的可核验证据 artifact；commit `67c214b` 的完整 workflow 又通过真实失败矩阵、20 路同步并发门以及全部数据库恢复/基准任务。本机无 DSN 时仍明确保留门控边界。真实 LangGraph 的 100 ms P95 工程门槛默认只记录，需在受控基准机设置 `PUNCTURE_ENFORCE_PERFORMANCE_GATES=1` 才会硬性执行；重复运行曾出现超过门槛的抖动。`local_rag_demo.py` 可运行企业 RAG 摄取、混合检索、ACL-negative 和 Citation；`local_mcp_demo.py` 可运行三个 MCP Server 的十个强类型工具。两者均不需要网络、GPU 或第三方依赖。

[commit `66f193d` 的 PostgreSQL 16 checkpoint 基准](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29146879180) 使用 5 次预热和 3×50 样本，观测到同步 `PostgresSaver.put()` save P50/P95 `3.131/3.606 ms`、公共 `resume()` P50/P95 `23.429/25.829 ms`，低于固定 `50/150 ms` 工程阈值。该结果是 GitHub 托管 Ubuntu 24.04 runner 的 `record` baseline，不是生产 SLA；生产发布门仍需在受控专用 PostgreSQL/storage 主机启用 enforce 模式。

[commit `9f12278` 的真实进程崩溃门](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29147544527) 在进程 A 完成 SQLite replay ledger、收到 MCP 响应但尚未提交对应 LangGraph node checkpoint 时发送 `SIGKILL`；进程 B 从 PostgreSQL checkpoint 恢复后命中同一 replay identity，目标底层副作用总计只执行一次，并完成终态 checkpoint。该证据只覆盖同主机 ledger 的“`COMPLETED` 后到 graph checkpoint 前”窗口，不覆盖外部副作用与 ledger commit 之间的内部原子性或跨主机共享账本。

[commit `67c214b` 的真实 LangGraph 完整矩阵](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29147890458) 覆盖 MCS/规划成功、缺输入、geometry/label fail-closed、无可行路径、一次重试恢复、重试耗尽、不可重试错误和 malformed model；另用 `Barrier(20)` 强制 20 个 worker 同时进入真实图节点，交替执行 10 个规划流和 10 个 MCS 流，并逐会话核对工具序列、case 请求和完整 checkpoint。该 workflow 的三版本常规测试及 PostgreSQL restart、benchmark、process-kill 任务全部成功。

[commit `3c4c6fc` 的安全 API 契约边界](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29148721224) 新增 Pydantic v2 request/response/error adapter、可信身份注入、递归 authority/raw-image/credential/URI/JWT 拒绝、深度公共视图脱敏和固定 HTTP 错误映射。Python 3.10/3.11/3.12 的专门 Pydantic no-skip 门均成功；这仍不是 FastAPI endpoint、SSE 或 PostgreSQL Run Repository 完成声明。

[commit `189040b` 的原子 Run Repository 边界](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29149762976) 新增 framework-neutral repository 协议、内部 version fence、tenant-scoped idempotency、原子 lifecycle event、全写入口有界 JSON 规范化和 cancel/approve/resume 竞态保护。Python 3.10/3.11/3.12、PostgreSQL restart、checkpoint benchmark 与真实 process-kill job 全部成功；当前仍是内存参考后端，不代表 PostgreSQL Run/event 持久化、FastAPI 或 SSE 已完成。

[commit `15af386` 的私有 stream event identity 边界](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29150409932) 为普通执行事件增加不进入公共响应的 version-scoped key 和 canonical fingerprint，使相同事件重放返回原 sequence、内容变化固定冲突，并在取消后的 exact replay 后再次检查 version fence。该节点只为 PostgreSQL COMMIT 对账提供前置保证，不代表 lifecycle/CAS 对账、worker reclaim 或 PostgreSQL Repository 已完成。

[commit `93391f4` 的 PostgreSQL Run/event Repository](https://github.com/TurninQAQ/puncture-rd-agent-platform/actions/runs/29152000095) 新增显式 checksum migration、canonical JSONB sidecar、严格 UTC 毫秒时间、tenant-scoped 幂等创建、跨实例连续 sequence、version CAS mutation journal 和 create/append/CAS COMMIT-unknown 对账。Python 3.10/3.11/3.12 各自执行 8 项 PostgreSQL 16 no-skip 测试，覆盖 20 路创建、100 个并发事件、tenant 隔离、事务 rollback、backend termination、真实 commit 成功后 ACK 丢失，以及状态已继续推进后的旧 CAS 精确对账；完整 workflow 全部成功。该节点仍不代表 FastAPI、SSE、OIDC、worker reclaim/heartbeat 或 API 进程恢复已经完成。

## Reading order

1. `docs/architecture.md`
2. `docs/testing-guide.md`
3. `docs/technology-stack.md`
4. `docs/open-source-baseline.md`
5. `docs/qwen-deployment-runbook.md`
6. `docs/testing-qwen-vllm.md`
7. `docs/module-delegation-playbook.md`
8. `docs/testing-rag.md`
9. `docs/rag-deployment-runbook.md`
10. `docs/mcp-tool-runtime.md`
11. `docs/testing-mcp.md`
12. `docs/langgraph-runtime-implementation.md`
13. `docs/api-runtime-implementation.md`
14. `docs/versioning.md`
15. `contracts/README.md`
16. 对应的 `specs/*.md`、`tasks/task-*.md` 和 Contract Tests

## Module implementation order

1. Contracts and Artifact Registry
2. Qwen/vLLM Model Gateway
3. Enterprise RAG
4. Case-data tools
5. Segmentation tools
6. Planning and safety tools
7. LangGraph runtime
8. API, Trace and Eval

## Delegation tasks

`tasks/task-00` 至 `task-08` 可分别交给其他模型实现。每张任务卡都限定允许修改的文件，并给出固定接口、实现顺序、故障注入、测试命令、性能/Eval 门槛和交付报告格式。
