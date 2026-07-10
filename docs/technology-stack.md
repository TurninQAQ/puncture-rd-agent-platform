# Technology Stack and Interview Keywords

本项目的重点不是医学算法本身，而是把既有算法能力改造成可检索、可编排、可恢复、可评测、可审计的企业 Agent Runtime。下面列出实际实现与后续模块使用的技术栈；每一项是否完成代码验收或现场部署，以对应 Release Record 为准。

## 1. Core Agent stack

| Layer | Primary choice | Keywords to understand and explain |
|---|---|---|
| Agent orchestration | LangGraph | StateGraph, typed state, node/router, conditional edge, subgraph, checkpoint, interrupt, resume, human-in-the-loop, durable execution |
| Tool protocol | MCP + internal tool registry | MCP Server/Client, tool discovery, JSON Schema, stdio/Streamable HTTP transport, capability negotiation, tool timeout, idempotency, permission boundary |
| Model gateway | Qwen served by vLLM | OpenAI-compatible API, tensor parallelism, KV cache, continuous batching, structured output, tool calling, streaming, timeout/retry, model warm-up |
| Runtime/API | FastAPI + SSE | request/response schema, idempotency key, async lifecycle, SSE reconnect cursor, approval API, cancel/resume, dependency injection |
| Persistence | PostgreSQL | LangGraph checkpoint, run/event store, optimistic concurrency, transaction, unique constraint, recovery after restart |
| Cache/short state | Redis (optional) | distributed lock, rate limit, response/cache TTL; do not make Redis the only durable checkpoint |

## 2. RAG stack

| Capability | Suggested implementation | Keywords |
|---|---|---|
| Ingestion | Python pipeline + object storage | parser, normalization, parent-child chunk, metadata enrichment, document/version hash, incremental indexing |
| Dense retrieval | BGE/Qwen embedding + vector index | embedding, cosine similarity, ANN, HNSW, top-k |
| Lexical retrieval | Elasticsearch/OpenSearch BM25 | inverted index, analyzer, filters, BM25 |
| Fusion | Reciprocal Rank Fusion | dense + lexical, RRF, candidate pool |
| Rerank | BGE reranker or equivalent | cross-encoder, rerank top-n, latency/quality trade-off |
| Governance | ACL + version filter | tenant/project/module ACL, effective version, superseded docs, deletion propagation |
| Answer grounding | citation contract | chunk ID, document ID, version, section, evidence sufficiency, abstention |
| Evaluation | Golden Set | Recall@K, MRR, nDCG@K, citation precision/recall, answer faithfulness, no-answer accuracy |

## 3. Observability and evaluation

- OpenTelemetry: trace/span/context propagation, attributes, metrics, logs correlation.
- Langfuse/Phoenix/LangSmith: choose one production trace backend; keep the internal trace adapter vendor-neutral.
- Agent evaluation: task success rate, tool selection accuracy, argument exact match, graph branch accuracy, recovery success rate.
- Safety regression: missing-mask fail-closed rate, unsafe-path false acceptance, verifier override count.
- Operational metrics: first-token latency, end-to-end latency, tool latency, token usage, GPU memory, queue depth, retry rate, error rate.
- Version dimensions: model, tokenizer, prompt, graph, tool, RAG index, document, embedding and reranker versions.

## 4. Algorithm/tool integration

These are company-background capabilities, not the Agent focus:

- Python/C++ adapters and optional pybind11/gRPC boundary;
- TensorRT/nnU-Net inference adapter;
- MCS/NIfTI, SimpleITK/nibabel concepts;
- geometry fingerprint: shape, spacing, origin, direction, coordinate system;
- morphology, candidate path generation, safety-envelope collision, 3D ray traversal;
- artifact ID and object storage instead of passing arrays through LLM context.

Real algorithms can remain behind the fixed 10-tool interface. Interview preparation should focus on why deterministic verification stays outside the LLM.

## 5. Engineering and delivery

- Python 3.10+, dataclasses/Pydantic v2, type hints, Protocol/adapter pattern.
- pytest can supplement but must not replace the standard-library Contract Tests.
- Docker/Compose for local dependencies; Kubernetes/Helm for production discussion.
- NVIDIA Container Toolkit, CUDA/NCCL for multi-GPU Qwen deployment.
- GitLab CI/Jenkins/GitHub Actions equivalents: lint, contract, unit, failure, integration, eval and performance gates.
- Configuration and secrets: environment/config files, Vault/KMS/Secret manager; never hard-code tokens.
- Security: OIDC/JWT, RBAC/ABAC, audit logs, prompt-injection defense, tool allowlist, egress control, PII redaction.

## 6. Priority for autumn recruitment

### Must implement and be able to explain

1. Qwen + vLLM private deployment and model gateway.
2. Hybrid RAG with ACL/version/citation and measurable retrieval evaluation.
3. LangGraph state, conditional routing, checkpoint and interrupt/resume.
4. Strongly typed MCP tools, timeout/retry/idempotency and deterministic verifier.
5. Trace and Eval pipeline with one reproducible regression set.
6. FastAPI/SSE run service and restart recovery.

### Understand but can keep lightweight

- Redis/Celery background jobs;
- Kubernetes autoscaling;
- complete medical file parsing;
- real image/segmentation/planning implementations;
- multiple model providers;
- a polished public front end.

## 7. Transfer to semiconductor, EDA and industrial software

The same architecture maps directly:

| This scaffold | Semiconductor/EDA/industrial equivalent |
|---|---|
| CT/Mask artifact | netlist, layout, waveform, log, CAD/model artifact |
| Medical label schema | PDK/rule/config schema |
| Segmentation/planning tool | simulation, synthesis, DRC/LVS, diagnosis tool |
| Safety constraint | design rule, process constraint, equipment interlock |
| Project RAG | PDK manuals, tool manuals, SOP, issue database, design standards |
| Deterministic verifier | rule checker, regression checker, sign-off gate |
| Human approval | engineer review/sign-off |

Interview wording should emphasize this general pattern: the LLM understands intent and coordinates evidence; deterministic domain tools calculate and sign off the result.
