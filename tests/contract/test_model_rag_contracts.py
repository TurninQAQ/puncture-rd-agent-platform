"""Cross-module contract tests for model gateway and RAG data types."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.model_gateway import (  # noqa: E402
    ChatMessage,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolDefinition,
    validate_json_schema_subset,
)
from puncture_agent.rag import RetrievalRequest, RetrievalResponse, RetrievedChunk  # noqa: E402


class ModelContractTests(unittest.TestCase):
    def test_request_normalizes_sequences_and_serializes(self) -> None:
        request = ModelRequest(
            request_id="model-contract-1",
            messages=[ChatMessage(role="user", content="inspect case")],
            tools=[
                ToolDefinition(
                    name="inspect_case_metadata",
                    description="Inspect registered case metadata.",
                    input_schema={
                        "type": "object",
                        "properties": {"case_id": {"type": "string"}},
                        "required": ["case_id"],
                        "additionalProperties": False,
                    },
                )
            ],
        )

        self.assertIsInstance(request.messages, tuple)
        self.assertIsInstance(request.tools, tuple)
        self.assertEqual(request.to_dict()["request_id"], "model-contract-1")

    def test_invalid_tool_message_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool_call_id"):
            ChatMessage(role="tool", content="result")

    def test_schema_subset_rejects_missing_and_unknown_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {"task_type": {"type": "string"}},
            "required": ["task_type"],
            "additionalProperties": False,
        }
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_json_schema_subset({}, schema)
        with self.assertRaisesRegex(ValueError, "unexpected properties"):
            validate_json_schema_subset({"task_type": "RAG", "extra": 1}, schema)

    def test_model_response_requires_payload(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        with self.assertRaisesRegex(ValueError, "must contain"):
            ModelResponse(
                request_id="response-1",
                model="qwen",
                finish_reason="stop",
                text=None,
                tool_calls=(),
                structured_output=None,
                usage=usage,
                latency_ms=1.0,
            )


class RagContractTests(unittest.TestCase):
    def test_retrieval_request_validates_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            RetrievalRequest(request_id="rag-1", query="rules", top_k=0)

    def test_response_requires_contiguous_ranks(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="doc#1",
            document_id="doc",
            title="Title",
            module="planning",
            version="v1",
            section="Section",
            text="Text",
            score=0.8,
            rank=2,
            citation="[Title | v1 | Section]",
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            RetrievalResponse(
                request_id="rag-2",
                rewritten_query="rules",
                chunks=(chunk,),
                retrieval_mode="mock",
                trace_id="trace-rag-2",
                latency_ms=1.0,
            )

    def test_chunk_score_must_be_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized"):
            RetrievedChunk(
                chunk_id="doc#1",
                document_id="doc",
                title="Title",
                module="planning",
                version="v1",
                section="Section",
                text="Text",
                score=2.5,
                rank=1,
                citation="[Title | v1 | Section]",
            )


if __name__ == "__main__":
    unittest.main()
