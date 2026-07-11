# Open-source Baseline and Adaptation Plan

本项目以通用开源能力为基线，不采用医疗场景特化 Agent。下面的仓库/文档是实现者应优先阅读的上游项目；真正编码前应再次核对最新 release、兼容矩阵和许可证，并在 `pyproject.toml`/部署清单中固定版本。

> 接口与行为始终由仓库契约锁定。Module 1 已在 Release Record 中保存所核对的 vLLM/Qwen 源码提交与 vLLM 镜像基线；后续模块仍需在实现时重新核对上游 release、许可证和兼容矩阵，不能把一次核对结果永久当作“最新版本”。

## 1. Selected baselines

| Area | Upstream baseline | What to reuse | What this project adds |
|---|---|---|---|
| Agent graph | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | state graph, conditional edge, checkpoint, interrupt/resume concepts | fixed medical/industrial tool contracts, two deterministic subgraphs, fail-closed verifier, artifact-only large-data flow |
| Tool protocol | [modelcontextprotocol/specification](https://github.com/modelcontextprotocol/modelcontextprotocol) 2025-11-25 and stable Python SDK v1.x | tool schema, discovery, structured content, stdio and Streamable HTTP conventions | three internal server boundaries, safe artifact handles, principal/case/tool permission, idempotency and safety error taxonomy |
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

Module 3 deliberately pins the optional Python SDK to `mcp>=1.27,<2`.  The
official SDK's `main` branch documents a pre-release v2 line, while v1.x remains
the stable production recommendation as of the recorded release audit.  The
dependency-free local dispatcher targets MCP `2025-11-25`; installing the
optional SDK changes the transport implementation, not the internal tool
contracts or adapters.

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
