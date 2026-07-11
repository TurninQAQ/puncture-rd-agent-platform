# Architecture

## 1. Goal

在联影现有 MCS/NIfTI、nnU-Net/TensorRT、皮肤 Mask、路径规划、安全边界和三维射线追踪算法之上增加研发协同 Agent。

LLM 只负责任务理解、RAG、计划、工具路由和报告；所有三维计算仍由确定性工具完成。

## 2. Logical architecture

```text
User request
  -> API Gateway
  -> Agent Runtime
       -> Model Gateway (Qwen/vLLM)
       -> RAG Service
       -> MCP Tool Registry
            -> Case-data tools
            -> Segmentation tools
            -> Planning/safety tools
       -> Deterministic Verifier
       -> Checkpoint / Human approval
  -> Structured report and trace
```

## 3. Two subgraphs

### Data/model validation

```text
inspect metadata
  -> optional MCS conversion
  -> label validation
  -> optional segmentation
  -> segmentation QC
  -> optional skin surface extraction
  -> readiness verifier
```

### Planning/safety

```text
resolve constraints
  -> preflight check
  -> candidate path generation
  -> full-path safety evaluation
  -> optional intraoperative risk evaluation
  -> optional skin penetration verification
  -> safety verifier
```

## 4. Replaceable modules

所有外部能力均保留 Mock 以支持离线测试；生产适配器按模块逐步替换，并保持稳定输入输出：

- `model_gateway`: Mock Qwen + 已实现的 vLLM/Qwen 生产适配器
- `rag`: in-memory documents -> Elasticsearch/OpenSearch RAG
- `tooling`: three implemented MCP/local adapter groups -> future company C++/TensorRT/Python algorithms
- `agent`: mock state runner -> LangGraph StateGraph
- `observability`: in-memory trace -> OpenTelemetry/Langfuse/Phoenix

## 5. Non-negotiable rules

- 三维数据使用 artifact ID，不进入 LLM 上下文。
- 几何不一致时必须阻断。
- 必要危险 Mask 缺失时不能输出 SAFE/NORMAL。
- 无可行路径时不能自动缩小安全距离。
- 最终安全状态由 deterministic verifier 决定。
