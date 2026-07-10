# Open-source Baseline and Adaptation Plan

本项目以通用开源能力为基线，不采用医疗场景特化 Agent。下面的仓库/文档是实现者应优先阅读的上游项目；真正编码前应再次核对最新 release、兼容矩阵和许可证，并在 `pyproject.toml`/部署清单中固定版本。

> 当前工程环境无法稳定访问外网，因此本文件不声称已锁定 2026 年的具体最新版本。任务卡故意按接口与行为锁定，不依赖某个瞬时版本号。

## 1. Selected baselines

| Area | Upstream baseline | What to reuse | What this project adds |
|---|---|---|---|
| Agent graph | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | state graph, conditional edge, checkpoint, interrupt/resume concepts | fixed medical/industrial tool contracts, two deterministic subgraphs, fail-closed verifier, artifact-only large-data flow |
| Tool protocol | [modelcontextprotocol/specification](https://github.com/modelcontextprotocol/specification) and MCP SDKs | tool schema, discovery and transport conventions | three internal server boundaries, case/tenant context, artifact permission, idempotency and safety error taxonomy |
| Private LLM serving | [vllm-project/vllm](https://github.com/vllm-project/vllm) | OpenAI-compatible serving, batching, streaming, distributed inference | normalized gateway, Qwen-specific deployment profile, structured-output validation, retry/circuit-breaker and trace fields |
| Qwen model family | [QwenLM](https://github.com/QwenLM) | tokenizer/model usage and deployment guidance | enterprise prompt/tool policy, version pinning, internal evaluation and fallback behavior |
| Search/RAG | [elastic/elasticsearch](https://github.com/elastic/elasticsearch) or [opensearch-project/OpenSearch](https://github.com/opensearch-project/OpenSearch) | BM25, vector search, filters | version-aware project corpus, ACL before ranking, RRF/rerank, citation and evidence sufficiency |
| Embedding/rerank | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) or an approved equivalent | multilingual embedding and reranking patterns | domain Golden Set, latency budget, model/index version trace |
| Telemetry | [open-telemetry/opentelemetry-specification](https://github.com/open-telemetry/opentelemetry-specification) | trace/span/context and semantic conventions | run/node/tool/RAG/model attributes, redaction policy and domain regression linkage |
| LLM trace UI | [langfuse/langfuse](https://github.com/langfuse/langfuse) or [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | trace visualization and evaluation storage | vendor-neutral adapter, artifact-safe payload, graph/tool correctness metrics |

## 2. Why LangGraph is the primary orchestrator

This project needs explicit state, deterministic branches, retry loops, subgraphs, human approval and process recovery. A graph runtime makes those behaviors inspectable and testable. A free-form multi-agent chat abstraction is not the default because tool safety and exact branch behavior matter more than agent role-play.

Other frameworks can still be studied, but the resume-facing implementation should stay on one primary runtime to avoid a shallow “framework collection” project.

## 3. Adaptation boundaries

Do not fork or rewrite upstream frameworks. Implement adapters:

- `model_gateway` isolates vLLM/OpenAI-compatible details;
- `rag` isolates Elasticsearch/OpenSearch/embedding/reranker details;
- `tooling` isolates MCP and algorithm services;
- `agent` isolates LangGraph from graph/state contracts;
- `observability` isolates Langfuse/Phoenix/OpenTelemetry exporters;
- `runtime` isolates FastAPI and checkpoint repositories.

Mock classes remain available so every module can be developed without all upstream dependencies running.

## 4. Upstream verification checklist

Before implementing each task, record in the delivery report:

1. upstream repository and documentation URL;
2. selected release/tag/commit and release date;
3. Python/CUDA/driver/GPU compatibility where applicable;
4. license and enterprise-use review;
5. breaking changes affecting the fixed contract;
6. a minimal upstream example reproduced locally;
7. why an adapter is sufficient or why a small patch is necessary;
8. dependency vulnerability scan result.

## 5. Improvements that make the project interview-worthy

- not merely “call an LLM”: Qwen is privately served, normalized and evaluated;
- not merely “vector search”: retrieval combines lexical/dense/rerank with ACL/version/citation;
- not merely “ten functions”: tools are strongly typed, idempotent, traceable and fail closed;
- not merely “LangGraph demo”: graph has two subgraphs, retries, approval and checkpoint recovery;
- not merely “logs”: traces connect model/RAG/tool versions to regression metrics;
- not merely “medical”: artifacts, rules, tools, verifier and sign-off map to EDA/industrial workflows.
