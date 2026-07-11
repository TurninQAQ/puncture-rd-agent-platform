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
- 已实现的 Module 2：本地可运行的企业 Hybrid RAG、文档摄取、BM25 + dense + RRF + rerank、ACL/版本过滤、Citation、离线 Eval，以及 OpenSearch/Qwen Embedding/Reranker 可选适配器。

当前仍未实现：真实 MCS 解析、TensorRT 推理、医学图像算法、路径规划、安全评估或真实 LangGraph。Qwen/vLLM、OpenSearch、Qwen Embedding/Reranker 的代码与部署资产已经完成离线验收，但本仓库不声称已经在目标 GPU/集群上完成模型下载、服务启动或性能基准；现场步骤见对应 runbook。

## Core rule

后续实现可以替换 Stub 内部逻辑，但不得自行修改 `contracts/`、图节点名称、错误码或公共返回结构。若确需变更，必须先更新契约、测试和所有依赖模块。

## Quick start

环境仅需 Python 3.10+，当前 Mock 和测试不依赖第三方包：

```bash
python3 run_tests.py
PYTHONPATH="$PWD/src:$PWD" python3 -m puncture_agent.api.demo
python3 examples/local_rag_demo.py
```

当前本地 Python 3.10 基线：执行 346 项测试，339 项通过、7 项门控跳过（2 项模型网关真实 `httpx` 集成测试和 5 项私有 vLLM 测试）；`local_rag_demo.py` 无需网络或第三方依赖即可运行企业 RAG 摄取、混合检索、ACL-negative 和 Citation 流程。

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
10. `docs/versioning.md`
11. `contracts/README.md`
12. 对应的 `specs/*.md`、`tasks/task-*.md` 和 Contract Tests

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
