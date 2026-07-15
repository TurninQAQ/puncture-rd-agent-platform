# 从零开始：边学边操作的 Agent 项目实操课

这份教材给第一次接触 Agent、RAG、MCP、SSE、LangGraph 的学习者使用。它不要求你先看懂整个仓库，也不会直接从高级概念开始。

学习方式只有一条：

> 每次只学一个概念，立即运行一个项目功能，再用自己的话解释刚才发生了什么。

全课程从第 0 课开始，严格连续到第 12 课。不要跳课。每课约 45–90 分钟，总计约 12–18 小时；零基础建议用两周完成。

第一次打开本文时，只做到“第 0 课”的过关检查就停下，不要继续往下滚。把结果和自己的三句话理解发给我，确认后再进入第 1 课。

## 课程地图

| 顺序 | 本课只学一个核心问题 | 马上操作的功能 |
|---|---|---|
| 第 0 课 | 我在哪里运行项目？ | 环境体检、Git 学习分支 |
| 第 1 课 | 一次任务是什么？ | 离线 Run Demo |
| 第 2 课 | 当前状态和过程记录有什么区别？ | Snapshot 与 9 个 Event |
| 第 3 课 | 重复提交为什么不重复执行？ | 幂等成功与冲突测试 |
| 第 4 课 | Agent 如何决定下一步？ | 两条固定工作流图 |
| 第 5 课 | Agent 如何调用专业能力？ | 3 个 MCP Server、10 个工具 |
| 第 6 课 | Agent 如何查项目资料？ | 离线 Hybrid RAG、ACL 反例 |
| 第 7 课 | Qwen 在系统里负责什么？ | 真实 vLLM 五能力测试 |
| 第 8 课 | 前面组件如何连成一条链？ | 真实本地全栈 Demo |
| 第 9 课 | 页面如何持续看到任务进度？ | FastAPI、JSON Event、SSE |
| 第 10 课 | 进程退出后任务为什么还在？ | PostgreSQL、事务与版本 |
| 第 11 课 | 工作流如何暂停和恢复？ | LangGraph checkpoint/resume |
| 第 12 课 | 如何知道 Agent 修改后没退步？ | Trace 与 Eval |

到第 8 课时，你才会第一次完整运行真实全栈。这样看到每个输出字段时，你已经知道它来自哪一层。

## 每课统一学习动作

每课都做六件事：

1. 读“白话理解”，不背英文定义。
2. 复制命令并运行。
3. 对照“你应该看到什么”。
4. 只读本课指定的 1–3 个文件。
5. 完成一个成功或失败实验。
6. 不看文档，用 60 秒讲清楚本课。

为每课保存下面这张学习卡：

```text
第几课：
我学到的概念：
我的生活比喻：
我运行的命令：
我看到的三个关键字段：
成功路径：
失败路径：
对应代码文件：
我的 60 秒讲解：
仍然不懂的问题：
```

过关不是“命令绿了”，而是你能解释输入、过程、输出和失败行为。

---

## 第 0 课：准备环境，不学 Agent

### 本课目标

你只需要知道：终端在哪个目录、使用哪个 Python、依赖服务是否可用、怎样避免把练习改到主分支。

### 五个最小词汇

- **终端**：输入命令的窗口。
- **仓库**：这个项目的全部代码目录。
- **Python 文件**：以 `.py` 结尾的程序文件。
- **JSON**：由键和值组成、方便程序交换的数据文本。
- **测试**：给程序固定输入，自动核对结果是否符合预期。

现在不需要理解 class、Protocol、事务、LangGraph。

### 步骤 1：进入仓库

```bash
cd /home/turnin/puncture-rd-agent-platform
pwd
```

预期最后一行是：

```text
/home/turnin/puncture-rd-agent-platform
```

后面所有命令都默认从这个目录执行。

### 步骤 2：确认主分支没有未提交改动

```bash
git status --short
```

没有输出表示工作区干净。如果有输出，先不要覆盖或删除文件。

建立专门的学习分支：

```bash
git switch -c learning/agent-basics
```

如果提示分支已经存在，改用：

```bash
git switch learning/agent-basics
```

练习期间不要修改或提交以下私有文件：

```text
deploy/local-demo/.env
deploy/rag-search/secrets/
var/
```

### 步骤 3：检查真实依赖

```bash
./deploy/local-demo/doctor.sh
```

你暂时只看：

```text
"ready": true
"postgresql": { "status": "UP" }
"qwen_vllm": { "status": "UP" }
"opensearch": { "status": "DEGRADED" }
```

本地 OpenSearch 是单节点，`DEGRADED/yellow` 是已知状态；`ready=true` 表示它仍满足本地演示条件。

### 本课过关口述

> 我在独立学习分支操作。doctor 只检查配置和依赖，不执行 Agent 任务，也不会打印密码。当前 PostgreSQL 和 Qwen 正常，单节点 OpenSearch 虽然是 yellow，但可以用于本地演示。

### 过关检查

- [ ] 知道仓库绝对路径。
- [ ] 当前在 `learning/agent-basics` 分支。
- [ ] `doctor.sh` 返回 `ready=true`。
- [ ] 知道 `.env`、secret 和 `var/` 不能提交。

---

## 第 1 课：Run——把一次任务理解成一张工单

### 白话理解

先不管 Agent。用户提交一个请求，系统要给它一个编号并记录结果。这个“一次任务”叫 `Run`。

可以类比外卖订单：

```text
run_id       = 订单号
request      = 你点了什么
status       = 配送状态
final_report = 最终送到的内容
```

### 立即运行

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m puncture_agent.api.demo
```

### 你应该看到什么

每次 `run_id` 会变化，但下面这些结构应存在：

```json
{
  "run_id": "run-...",
  "status": "SUCCEEDED",
  "final_report": {
    "case_id": "case-demo-001",
    "task_type": "DATA_MODEL_VALIDATION",
    "verification_status": "PASS"
  },
  "event_count": 9
}
```

先回答四个问题：

1. 输入病例是谁？`case-demo-001`。
2. 任务类型是什么？`DATA_MODEL_VALIDATION`。
3. 最终状态是什么？`SUCCEEDED`。
4. 最终验证是什么？`PASS`。

### 只读这一份小文件

打开 [`src/puncture_agent/api/demo.py`](../src/puncture_agent/api/demo.py)。

按顺序找：

1. `RunRequest(...)`：输入工单；
2. `service.create_run(...)`：提交工单；
3. `snapshot`：提交后的当前结果；
4. `print(...)`：输出到终端。

现在不需要继续追进 service 内部。

### 第一次小改动

在学习分支中，只把 `demo.py` 的 `user_query` 改成你自己的中文问题，例如：

```text
检查 case-demo-001 的标签与分割结果，并说明数据是否可继续使用
```

重新运行同一个命令，观察输出结构是否仍然稳定。然后查看自己的改动：

```bash
git diff -- src/puncture_agent/api/demo.py
```

完成观察后还原练习：

```bash
git restore src/puncture_agent/api/demo.py
```

### 本课过关口述

> Run 是平台中的一次任务工单。RunRequest 是输入，create_run 创建任务，返回的 snapshot 表示当前结果，run_id 用来继续查询同一任务。

### 过关检查

- [ ] 能指出输入、Run ID、状态和最终报告。
- [ ] 改过一次 query 并重新运行。
- [ ] 已还原 `demo.py`，`git diff` 不再显示它。

---

## 第 2 课：Snapshot 和 Event——当前状态与过程记录

### 白话理解

同一张工单有两种视角：

- **Snapshot**：现在订单进行到哪一步，是一张“当前截图”。
- **Event**：从创建到完成发生过什么，是连续“物流轨迹”。

快照回答“现在怎样”，事件回答“之前发生了什么”。

### 再运行一次第 1 课命令

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m puncture_agent.api.demo
```

找到 `visited_event_types`。当前成功路径共有 9 个公共事件：

```text
RUN_CREATED
RUN_STARTED
NODE_STARTED
NODE_COMPLETED
NODE_STARTED
NODE_COMPLETED
NODE_STARTED
NODE_COMPLETED
RUN_COMPLETED
```

`NODE_STARTED/NODE_COMPLETED` 重复出现，是因为模型、RAG、Agent 图各自形成一段节点证据。

### 运行事件测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.api.test_run_service.RunServiceTests.test_happy_path_has_ordered_events_and_report
```

预期：

```text
... ok
Ran 1 test ...
OK
```

再运行“只读取上次位置之后的新事件”：

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.api.test_run_service.RunServiceTests.test_event_cursor_returns_only_new_events
```

### 只读代码

在 [`src/puncture_agent/runtime/models.py`](../src/puncture_agent/runtime/models.py) 中搜索：

```bash
rg -n "class RunStatus|class RunEvent|class RunSnapshot" \
  src/puncture_agent/runtime/models.py
```

不用逐行看实现，只比较三个对象各自保存什么字段。

### 画图练习

在学习卡中画：

```text
一个 Run
├── 一个当前 Snapshot
└── Event 1 → Event 2 → ... → Event 9
```

### 本课过关口述

> Snapshot 用于快速读取当前状态，Event 用于重放和审计执行过程。cursor 表示客户端已经看到的事件位置，重新查询时只需要读取它之后的事件。

---

## 第 3 课：幂等——重复提交不能重复干活

### 白话理解

用户可能双击按钮，网络超时后也可能重试。如果每次都创建新任务，工具就可能重复执行。

`idempotency_key` 可以类比“这笔业务的唯一订单号”。规则是：

| 相同 key | 请求内容 | 结果 |
|---|---|---|
| 相同 | 相同 | 返回原 Run，不重复执行 |
| 相同 | 不同 | 拒绝，防止一个 key 代表两件事 |
| 不同 | 任意 | 可以创建新 Run |

### 运行成功实验

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.api.test_run_service.RunServiceTests.test_duplicate_create_is_idempotent
```

### 运行冲突实验

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.api.test_run_service.RunServiceTests.test_same_key_with_different_payload_conflicts
```

两个测试都应为 `ok`。第二个 `ok` 的含义不是“请求成功”，而是“系统按预期拒绝了冲突”。

### 只读代码

- [`src/puncture_agent/runtime/service.py`](../src/puncture_agent/runtime/service.py)
- [`src/puncture_agent/runtime/repository.py`](../src/puncture_agent/runtime/repository.py)

只搜索 `idempotency`，不要通读整份文件：

```bash
rg -n "idempotency" \
  src/puncture_agent/runtime/service.py \
  src/puncture_agent/runtime/repository.py | head -40
```

### 本课过关口述

> 幂等不等于所有副作用天然只执行一次。这里先保证相同业务请求复用同一 Run，并拒绝相同 key 对应不同内容；真实外部工具还需要稳定 call ID 和自己的幂等记录。

---

## 第 4 课：工作流图——Agent 是受约束的调度员

### 白话理解

到这一课才引入 `Agent`：它是一个根据任务和当前状态选择下一步的调度员，不是一个可以任意行动的聊天机器人。

图里有四种常见东西：

- **Node**：一个处理步骤；
- **Edge**：步骤之间的连线；
- **Router**：根据条件选择分支；
- **Subgraph**：一组较小的流程。

### 查看主图

```bash
sed -n '1,240p' graph/main_graph.json
```

先找节点，不看全部条件：

```bash
rg -n '"id":|"source":|"target":' graph/main_graph.json
```

画出最小主干：

```text
START
  → parse_request
  → retrieve_project_knowledge
  → resolve_case_context
  → task_router
       ├── data_model_subgraph
       └── planning_safety_subgraph
  → result_verifier
  → report_generator
  → END
```

### 立即运行两条分支

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v tests.integration.test_mock_workflow
```

预期运行两个测试：数据验证流程和规划安全流程。

### 失败分支实验

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.graph.test_mock_runtime.MockRuntimeTests.test_missing_case_id_stops_before_any_algorithm_tool
```

它证明缺少 case ID 时不会继续调用算法工具。

### 只读代码

- [`graph/main_graph.json`](../graph/main_graph.json)
- [`src/puncture_agent/agent/runtime.py`](../src/puncture_agent/agent/runtime.py)
- [`src/puncture_agent/agent/verifier.py`](../src/puncture_agent/agent/verifier.py)

### 重要边界

本地一键全栈使用这套锁定 JSON 图的 `GraphRuntime`。仓库还实现了真实 LangGraph，但要到第 11 课再学，不要现在混在一起。

### 本课过关口述

> Agent 通过固定节点和条件边选择流程。数据任务和规划任务进入不同子图，但最后都经过确定性 verifier；缺关键输入时会提前阻断。

---

## 第 5 课：Tool 和 MCP——Agent 自己不做专业计算

### 白话理解

Agent 像项目经理，Tool 像专业设备或工程师。Agent 可以决定“调用哪个工具”，但不能假装自己完成三维计算。

MCP 可以先理解成“模型/Agent 调用工具时使用的一套统一插座”：工具用固定名称、固定输入和固定输出接入。

### 运行全部工具 Demo

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python examples/local_mcp_demo.py
```

你会看到 3 个逻辑 Server、共 10 个工具：

```text
case-data：3 个
  inspect_case_metadata
  convert_mcs_to_nifti
  validate_label_schema

segmentation：3 个
  run_segmentation
  validate_segmentation_result
  extract_skin_surface

planning-safety：4 个
  generate_candidate_paths
  evaluate_path_safety
  evaluate_intraoperative_risk
  verify_skin_penetration
```

每次调用都有 `status`、`summary` 和同一个 `trace_id`。

### 运行未知工具反例

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.graph.test_tool_bridge.ToolBridgeTests.test_unknown_tool_fails_before_calling_runtime
```

### 只读代码

- [`src/puncture_agent/tooling/catalog.py`](../src/puncture_agent/tooling/catalog.py)
- [`contracts/tool_inputs.py`](../contracts/tool_inputs.py)
- [`contracts/tool_outputs.py`](../contracts/tool_outputs.py)

在 catalog 中找 10 个工具名，在 contracts 中任选一个工具比较输入和输出。

### 重要边界

Demo 中 10 个工具是确定性合成实现。它们验证接口、流程和失败行为，不读取真实患者体素，也不是公司医学算法。

### 本课过关口述

> MCP 让 Agent 通过统一契约发现和调用工具。工具输入输出是强类型的，未知工具或非法参数会在执行前拒绝。当前工具后端用于工程演示，不是临床算法。

---

## 第 6 课：RAG——做任务前先查有权限的资料

### 白话理解

RAG 可以理解为“带权限的项目资料搜索”。用户提问后，系统先检索相关文档片段，再把少量证据交给 Agent。

先掌握四个词：

- **Query**：搜索问题；
- **Chunk**：从文档切出的一个小片段；
- **Citation**：证据来自哪份文档；
- **ACL**：谁有权限看到哪份文档。

BM25、dense、RRF、rerank 暂时只理解成“几种召回和排序方法”，不用背公式。

### 运行离线 RAG

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python examples/local_rag_demo.py
```

重点比较输出里的两个查询。

`authorized_hybrid` 应返回 3 条 evidence 和 Citation，例如：

```text
EDA Timing Signoff Rules
Engineering Release Change Control
```

`acl_negative` 即使查询中包含秘密文档的精确词，也应得到：

```text
"citations": []
"evidence": []
"warnings": ["NO_RELEVANT_KNOWLEDGE"]
```

这比“搜索到了但前端不显示”更安全，因为未授权文档在排序前就被过滤。

### 运行 RAG 测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v tests.rag.test_local_demo
```

### 只读代码

- [`examples/local_rag_demo.py`](../examples/local_rag_demo.py)
- [`src/puncture_agent/rag/retrieval.py`](../src/puncture_agent/rag/retrieval.py)
- [`src/puncture_agent/rag/models.py`](../src/puncture_agent/rag/models.py)

先在 example 中搜索两个 request ID：

```bash
rg -n "local-demo-authorized|local-demo-acl-negative" \
  examples/local_rag_demo.py
```

### 本课过关口述

> RAG 不等于让模型记住全部文档。它根据 query 检索有权限、版本有效的 chunk，返回 Citation；无权限或无相关证据时应返回空，而不是让模型编造。

---

## 第 7 课：Qwen/vLLM——模型负责理解和生成结构

### 白话理解

Qwen 是语言模型，vLLM 是把模型作为网络服务运行的推理引擎。`ModelGateway` 是项目内部统一调用模型的适配层。

在这个项目里，模型负责：

- 理解用户问题；
- 生成固定 JSON；
- 选择工具并填写参数；
- 生成文本说明。

模型不负责：

- 判断几何是否真的一致；
- 计算路径碰撞；
- 给出最终安全签字。

### 步骤 1：确认模型在线

```bash
./deploy/local-demo/doctor.sh
```

找到：

```text
"qwen_vllm": {
  "model": "qwen-enterprise-agent",
  "provider": "vllm",
  "status": "UP"
}
```

### 步骤 2：运行五项真实模型测试

```bash
. deploy/local-demo/common.sh
load_local_demo_env

PYTHONPATH="$PWD/src:$PWD" \
RUN_VLLM_INTEGRATION=1 \
"${PYTHON_BIN}" -m unittest -v tests.model_gateway.test_live_vllm
```

应运行五项：

```text
health
plain_chat
one_tool_call
structured_output
sse_has_one_terminal_completed_event
```

测试不会打印模型原始内容或秘密，只报告每种能力是否通过。

### 只读代码

- [`src/puncture_agent/model_gateway/models.py`](../src/puncture_agent/model_gateway/models.py)
- [`src/puncture_agent/model_gateway/client.py`](../src/puncture_agent/model_gateway/client.py)

先看 `ModelRequest` 和 `ModelResponse`，再在 client 中搜索 `generate`、`stream`、`health`。

### 本课过关口述

> vLLM 把本地 Qwen 暴露成 OpenAI 兼容服务，ModelGateway 隔离了 HTTP、超时和响应校验。Qwen负责语言理解和结构化计划，工具与 verifier 负责确定性结果。

---

## 第 8 课：真实全栈——把前七课连起来

你现在已经依次学过 Run、Event、幂等、图、Tool、RAG 和 Qwen，所以这时才运行完整链路。

### 先画调用链

```text
用户请求
  → FastAPI 接收请求
  → PostgreSQL 保存 Run/Event
  → Qwen 生成结构化任务理解
  → OpenSearch RAG 检索证据
  → 固定工作流选择分支
  → 合成工具执行
  → Verifier 检查
  → JSON 报告与 SSE 事件
```

### 运行

```bash
./deploy/local-demo/doctor.sh
./deploy/local-demo/run_demo.sh
```

### 逐字段解释输出

数据验证任务应包含：

```text
status                 = SUCCEEDED
rag_chunk_count        = 2
tool_call_count        = 6
visited_node_count     = 18
event_count            = 9
idempotent_replay      = true
sse_terminal_replay    = true
```

规划安全任务应包含：

```text
status                 = SUCCEEDED
rag_chunk_count        = 2
tool_call_count        = 4
visited_node_count     = 16
verification_status    = PASS
```

安全边界应包含：

```text
invalid_bearer_rejected = true
algorithm_backend       = deterministic-synthetic
company_or_patient_data = false
loopback_only           = true
```

Run ID 和 trace ID 每次变化是正常的。

### 为什么 health 是 DEGRADED

本地演示没有生产 Artifact Gateway，OpenSearch 也是单节点。依赖可用不等于生产能力齐全，所以健康接口诚实返回 `DEGRADED`；这不等于本次任务失败。

### 只读组合入口

- [`examples/live_api_server.py`](../examples/live_api_server.py)
- [`examples/live_api_demo.py`](../examples/live_api_demo.py)
- [`src/puncture_agent/runtime/graph_executor.py`](../src/puncture_agent/runtime/graph_executor.py)

先在 `graph_executor.py` 中搜索：

```bash
rg -n "model_gateway|rag_service|GraphRuntime|build_mock_handlers|runtime_evidence" \
  src/puncture_agent/runtime/graph_executor.py
```

这几处正好证明：模型和 RAG 可以注入真实实现，而工作流工具仍是 mock handlers。

### 本课过关口述

> 这次 HTTP、PostgreSQL、Qwen/vLLM 和 OpenSearch 都是真实服务；固定 JSON 图与算法工具是确定性实现。演示证明平台组件能组合，不证明医学算法或临床安全。

---

## 第 9 课：FastAPI 与 SSE——持续查看任务进度

### 白话理解

- **HTTP API**：客户端提交和查询任务的入口；
- **JSON Event**：一次性读取一批事件；
- **SSE**：服务器保持连接，持续向客户端推送新事件。

SSE 像物流页面自动刷新。连接中断后，客户端带上已经看到的 event cursor，可以从断点继续。

### 两个终端操作

终端 1 启动 API：

```bash
cd /home/turnin/puncture-rd-agent-platform
./deploy/local-demo/serve.sh
```

终端 2 查看健康：

```bash
curl -sS http://127.0.0.1:8010/health
```

预期：

```json
{"status":"DEGRADED"}
```

终端 2 执行完整客户端验证：

```bash
./deploy/local-demo/verify.sh
```

完成后回到终端 1 按 `Ctrl+C` 停止 API。

如需查看最近一次 `run_demo.sh` 保存的日志：

```bash
tail -n 40 var/local-demo/api.log
```

不要读取、复制或打印 `var/local-demo/bearer-token`。

### 运行 SSE 单测

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.api.test_fastapi_app.FastApiSseTests.test_terminal_sse_frames_headers_and_openapi_content
```

### 只读代码

- [`src/puncture_agent/api/fastapi_app.py`](../src/puncture_agent/api/fastapi_app.py)
- [`src/puncture_agent/api/sse.py`](../src/puncture_agent/api/sse.py)

### 本课过关口述

> FastAPI 提供任务创建和查询入口，Event 保存在数据库，SSE 按 event cursor 持续回放。SSE 适合服务器到客户端的单向进度流，断线不需要重新执行任务。

---

## 第 10 课：PostgreSQL——状态、事件、事务和版本

### 白话理解

进程内存像草稿纸，程序退出后会消失；PostgreSQL 像正式工单系统。

本课只掌握三个词：

- **Transaction（事务）**：一组数据库修改要么一起成功，要么一起失败；
- **Version（版本）**：每次状态推进都增加编号；
- **CAS**：只有我看到的旧版本仍然有效时，才允许我更新。

它们共同避免“状态已经成功但终态事件没保存”或两个执行者同时覆盖结果。

### 加载本地测试数据库配置

```bash
. deploy/local-demo/common.sh
load_local_demo_env
export PUNCTURE_TEST_POSTGRES_DSN="${PUNCTURE_API_POSTGRES_DSN}"
```

该赋值只在当前终端生效，不要 `echo` DSN。

### 持久化测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
"${PYTHON_BIN}" -m unittest -v \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests.test_pg_run_01_persists_across_instances_and_cursor
```

### 并发事件测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
"${PYTHON_BIN}" -m unittest -v \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests.test_pg_run_04_concurrent_events_are_contiguous_across_instances
```

### CAS 竞争测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
"${PYTHON_BIN}" -m unittest -v \
  tests.api.test_postgres_run_repository.PostgresRunRepositoryTests.test_pg_run_05_cas_has_one_winner_and_fences_old_version
```

### 只读代码

- [`src/puncture_agent/runtime/postgres_repository.py`](../src/puncture_agent/runtime/postgres_repository.py)
- [`src/puncture_agent/runtime/worker.py`](../src/puncture_agent/runtime/worker.py)

不要手工修改演示数据库。先通过测试理解行为。

### 本课过关口述

> PostgreSQL 同时保存当前 snapshot 和连续 event。事务保证它们一致，version/CAS 保证并发更新只有一个获胜，旧执行者不能覆盖新状态。

---

## 第 11 课：LangGraph checkpoint——暂停、存档与恢复

### 先澄清两个 Runtime

仓库有两条图执行路径：

1. 第 8 课全栈 Demo 使用锁定 JSON 的 `GraphRuntime`；
2. 这一课测试真实 LangGraph `StateGraph`、checkpoint 和 resume。

不能把两者混成一句“全栈 Demo 已使用生产 LangGraph worker”。

### 白话理解

- **State**：工作流当前携带的数据；
- **Checkpoint**：工作流存档；
- **Interrupt**：在安全位置暂停；
- **Resume**：从存档继续，而不是从头执行。

### 运行真实 LangGraph 暂停/恢复

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.graph.test_langgraph_runtime.RealLangGraphSmokeTests.test_real_interrupt_stream_resume_reuses_trace
```

### 运行新 Runtime 恢复测试

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m unittest -v \
  tests.graph.test_langgraph_runtime.RealLangGraphSmokeTests.test_real_new_runtime_resumes_child_from_safe_checkpoint
```

测试为 `ok` 表示：旧 Runtime 创建的存档能被新 Runtime 读取，已完成的安全步骤不会随意重放。

### 只读代码

- [`src/puncture_agent/agent/langgraph_state.py`](../src/puncture_agent/agent/langgraph_state.py)
- [`src/puncture_agent/agent/langgraph_runtime.py`](../src/puncture_agent/agent/langgraph_runtime.py)
- [`docs/langgraph-runtime-implementation.md`](langgraph-runtime-implementation.md)

### 本课过关口述

> Event 是审计轨迹，checkpoint 是恢复执行所需的状态。LangGraph 可以在固定位置 interrupt，把 state 保存后由同一或新的 Runtime resume；工具是否重放还要依赖稳定调用身份和幂等边界。

---

## 第 12 课：Trace 和 Eval——看过程并给 Agent 考试

### 白话理解

- **Trace**：一次请求跨模型、RAG、图和工具的完整调用链；
- **Span**：Trace 中一个步骤的时间和结果；
- **Eval**：给固定题目，检查修改后成功率、工具选择和安全结果是否退步。

Trace 回答“这次为什么这样运行”，Eval 回答“改完以后整体能力有没有变差”。

### 运行带 Trace 的 Eval

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m puncture_agent.observability.eval_runner run \
  --dataset tests/eval/fixtures/mock_reference_v1.json \
  --traced \
  --fail-on-release-block
```

预期摘要：

```text
passed=3/3
release_blocked=False
```

### 查看测试题

打开 [`tests/eval/fixtures/mock_reference_v1.json`](../tests/eval/fixtures/mock_reference_v1.json)，任选一条 case，找出：

- 输入问题；
- 预期走过的节点；
- 预期工具；
- 预期验证状态。

### 只读代码

- [`src/puncture_agent/observability/tracing.py`](../src/puncture_agent/observability/tracing.py)
- [`src/puncture_agent/observability/eval_harness.py`](../src/puncture_agent/observability/eval_harness.py)
- [`docs/eval-and-tracing-implementation.md`](eval-and-tracing-implementation.md)

### 本课过关口述

> Trace 用同一个 trace ID 串联一次请求的各个步骤，并对敏感属性做限制；Eval 用版本化数据集自动核对路径、工具和安全结果，失败时可以阻断发布。

---

## 毕业任务：做一次十分钟讲解

完成第 0–12 课后，按这个顺序演示。

### 第 1 分钟：说明问题

> 这个项目让 Qwen 理解研发任务，用 RAG 查项目知识，用固定工作流调用强类型工具，并由确定性 verifier 给出结果；Run、Event 和 checkpoint 负责持久化与恢复。

### 第 2–3 分钟：画调用链

```text
FastAPI → Run/PostgreSQL → Qwen → RAG → Graph → Tool → Verifier → Event/SSE
```

### 第 4 分钟：环境证据

```bash
./deploy/local-demo/doctor.sh
```

只讲 `ready`、PostgreSQL、Qwen 和 OpenSearch 状态。

### 第 5–7 分钟：真实全栈

```bash
./deploy/local-demo/run_demo.sh
```

只讲：两个 `SUCCEEDED`、RAG chunk、Tool call、幂等、SSE 和 verifier。

### 第 8–9 分钟：打开代码

打开三个文件：

1. [`graph/main_graph.json`](../graph/main_graph.json)：说明固定分支；
2. [`src/puncture_agent/runtime/graph_executor.py`](../src/puncture_agent/runtime/graph_executor.py)：说明模型/RAG 与合成工具边界；
3. [`src/puncture_agent/runtime/postgres_repository.py`](../src/puncture_agent/runtime/postgres_repository.py)：说明状态和事件持久化。

### 第 10 分钟：主动说明边界

> FastAPI、PostgreSQL、Qwen/vLLM 和 OpenSearch 是实际本地服务；医学工具和本地 embedding/reranker 是确定性合成实现。真实 LangGraph 已在独立集成测试中验证，但当前一键 HTTP Demo 使用 JSON GraphRuntime。项目不是临床系统，也没有生产 SLA。

## 两周建议安排

| 天 | 内容 | 当天交付 |
|---|---|---|
| Day 1 | 第 0、1 课 | 能运行离线 Run 并解释四个字段 |
| Day 2 | 第 2 课 | 画 Snapshot/Event 图 |
| Day 3 | 第 3 课 | 解释两个幂等测试 |
| Day 4 | 第 4 课 | 手画两条工作流 |
| Day 5 | 第 5 课 | 任选一个 Tool 讲输入输出 |
| Day 6 | 第 6 课 | 对比 authorized 与 ACL negative |
| Day 7 | 复习 | 不看文档串讲第 1–6 课 |
| Day 8 | 第 7 课 | 解释 Qwen 能做与不能做的事 |
| Day 9 | 第 8 课 | 完成真实全栈并标注每个字段来源 |
| Day 10 | 第 9 课 | 两终端演示 API/SSE |
| Day 11 | 第 10 课 | 解释事务、version、CAS |
| Day 12 | 第 11 课 | 解释 Event 与 checkpoint 的区别 |
| Day 13 | 第 12 课 | 跑通 3/3 Eval |
| Day 14 | 毕业任务 | 录一段十分钟讲解并复盘 |

## 常见卡点

### `ModuleNotFoundError`

确认命令前带有：

```bash
PYTHONPATH="$PWD/src:$PWD"
```

并使用：

```bash
.venv/bin/python
```

### `doctor.sh` 返回 `ready=false`

不要继续第 7–10 课。根据 JSON 中唯一 `DOWN` 的组件定位问题，不要复制或公开 `.env`。

### 端口 8010 已占用

检查是否忘记停止上一轮 `serve.sh`。回到它的终端按 `Ctrl+C`，再重试。

### 测试显示 `ok`，但不知道自己学到了什么

回到本课的失败实验，回答：

1. 测试给了什么输入？
2. 如果没有这层保护，会产生什么错误？
3. 是哪一个文件负责拒绝它？

答不出来就不要进入下一课。

## 与我一起学习时的提交格式

每完成一课，把下面内容发给我：

```text
我完成了第 N 课。
命令结果：粘贴最后 10–20 行，不粘贴 token/密码/DSN。
我的理解：三句话。
我不懂的地方：最多三个问题。
```

我会先检查你的理解和输出，再给下一课的小实验。这样每一步都能形成“我学会了什么、我怎样在项目里证明”的完整证据链。
