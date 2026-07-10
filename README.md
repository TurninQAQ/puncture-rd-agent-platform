# Puncture R&D Agent Platform

企业内部穿刺机器人研发协同 Agent 的 Contract-First 骨架工程。

本仓库当前提供：

- 固定的输入输出契约；
- 10 个算法工具的 Stub/Mock；
- Qwen、RAG、Agent Runtime、API、Trace/Eval 的接口骨架；
- 可运行的 Mock 端到端流程；
- Contract Tests；
- 可单独交给其他模型实现的任务卡与验收文档。
- 已实现的 Module 0：SQLite 持久化 Artifact Registry、原子本地对象存储、幂等发布与访问审计。
- 已实现的 Module 1：Qwen/vLLM OpenAI 兼容网关、严格工具/结构化输出校验、安全流式传输与私有化部署模板。

当前仍未实现：真实 MCS 解析、TensorRT 推理、医学图像算法、路径规划、安全评估、Elasticsearch/OpenSearch RAG 或真实 LangGraph。Qwen/vLLM 的代码与部署资产已经完成离线验收，但本仓库不声称已经在目标 GPU 上完成模型下载、启动和性能基准；现场步骤见 `docs/qwen-deployment-runbook.md`。

## Core rule

后续实现可以替换 Stub 内部逻辑，但不得自行修改 `contracts/`、图节点名称、错误码或公共返回结构。若确需变更，必须先更新契约、测试和所有依赖模块。

## Quick start

环境仅需 Python 3.10+，当前 Mock 和测试不依赖第三方包：

```bash
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m puncture_agent.api.demo
```

当前基线：本机执行 248 项测试，241 项通过、7 项门控跳过（2 项真实 `httpx` 传输集成测试由 CI 安装依赖后执行，5 项私有 vLLM 测试需显式端点）；Mock Demo 会串联 Model Gateway、RAG、Agent Graph、工具调用、确定性 Verifier 和 Run Event。

## Reading order

1. `docs/architecture.md`
2. `docs/testing-guide.md`
3. `docs/technology-stack.md`
4. `docs/open-source-baseline.md`
5. `docs/qwen-deployment-runbook.md`
6. `docs/testing-qwen-vllm.md`
7. `docs/module-delegation-playbook.md`
8. `docs/versioning.md`
9. `contracts/README.md`
10. 对应的 `specs/*.md`、`tasks/task-*.md` 和 Contract Tests

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
