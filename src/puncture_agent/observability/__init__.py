"""Framework-neutral tracing and evaluation helpers."""

from .eval_harness import AgentEvalHarness, EvalCase, EvalCaseResult, EvalReport
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from .tracing import (
    InMemoryTraceExporter,
    JsonLinesTraceExporter,
    SpanRecord,
    TraceRecorder,
)

__all__ = [
    "AgentEvalHarness",
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "InMemoryTraceExporter",
    "JsonLinesTraceExporter",
    "SpanRecord",
    "TraceRecorder",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
