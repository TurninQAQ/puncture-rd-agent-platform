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
    MODEL_GATEWAY_CONTRACT_VERSION,
    ChatMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    TokenUsage,
    ToolDefinition,
    validate_json_schema_subset,
    VllmGatewayConfig,
)
from puncture_agent.rag import RetrievalRequest, RetrievalResponse, RetrievedChunk  # noqa: E402


class ModelContractTests(unittest.TestCase):
    def test_contract_version_records_tool_history_extension(self) -> None:
        self.assertEqual(MODEL_GATEWAY_CONTRACT_VERSION, "2")

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
        with self.assertRaisesRegex(ValueError, "only tool messages"):
            ChatMessage(role="user", content="result", tool_call_id="call-1")

    def test_request_id_rejects_header_control_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "control characters"):
            ModelRequest(
                request_id="safe-id\r\nX-Injected: value",
                messages=(ChatMessage(role="user", content="inspect"),),
            )

    def test_assistant_tool_call_history_is_representable(self) -> None:
        call = ToolCall(
            call_id="call-1",
            name="inspect_case_metadata",
            arguments={"case_id": "case-1"},
        )
        assistant = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[call],
        )
        tool_result = ChatMessage(
            role="tool",
            content='{"status":"ok"}',
            tool_call_id="call-1",
        )

        self.assertEqual(assistant.tool_calls, (call,))
        self.assertEqual(tool_result.tool_call_id, "call-1")

    def test_tool_call_history_requires_assistant_role_and_unique_ids(self) -> None:
        call = ToolCall(call_id="call-1", name="inspect", arguments={})
        with self.assertRaisesRegex(ValueError, "only assistant"):
            ChatMessage(role="user", content="inspect", tool_calls=(call,))
        with self.assertRaisesRegex(ValueError, "unique"):
            ChatMessage(role="assistant", content="", tool_calls=(call, call))

    def test_unknown_token_usage_has_explicit_sentinel(self) -> None:
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            usage_known=False,
        )
        self.assertFalse(usage.usage_known)
        with self.assertRaisesRegex(ValueError, "zero-count sentinels"):
            TokenUsage(
                prompt_tokens=1,
                completion_tokens=0,
                total_tokens=1,
                usage_known=False,
            )

    def test_gateway_config_repr_redacts_api_key(self) -> None:
        config = VllmGatewayConfig(
            base_url="https://vllm.internal/v1",
            model="qwen-private",
            api_key="contract-secret-token",
            ca_bundle_path="/etc/ssl/certs/enterprise-agent-ca.pem",
        )
        self.assertNotIn("contract-secret-token", repr(config))
        self.assertIn("enterprise-agent-ca.pem", repr(config))

    def test_gateway_config_rejects_unsafe_types_and_nonfinite_timeout(self) -> None:
        invalid_options = (
            {"api_key": 123},
            {"timeout_seconds": True},
            {"timeout_seconds": float("nan")},
            {"max_retries": True},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                VllmGatewayConfig(
                    base_url="https://vllm.internal/v1",
                    model="qwen-private",
                    **options,
                )

    def test_stream_error_requires_one_structured_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured error"):
            ModelStreamEvent(
                request_id="stream-1",
                event_type="error",
                sequence=0,
            )
        event = ModelStreamEvent(
            request_id="stream-1",
            event_type="error",
            sequence=0,
            error={
                "code": "MODEL_TIMEOUT",
                "message": "model request timed out",
                "retryable": True,
                "details": {"attempts": 3},
            },
        )
        self.assertEqual(event.error["code"], "MODEL_TIMEOUT")
        with self.assertRaisesRegex(ValueError, "unexpected payloads"):
            ModelStreamEvent(
                request_id="stream-1",
                event_type="error",
                sequence=0,
                delta="should-not-coexist",
                error={
                    "code": "MODEL_PROTOCOL_ERROR",
                    "message": "malformed provider response",
                    "retryable": False,
                    "details": {},
                },
            )

    def test_stream_error_payload_is_strongly_validated(self) -> None:
        invalid_payloads = (
            {},
            {"code": "MODEL_TIMEOUT"},
            {
                "code": "MODEL_TIMEOUT",
                "message": "timeout",
                "retryable": "yes",
                "details": {},
            },
            {
                "code": "MODEL_TIMEOUT",
                "message": "timeout",
                "retryable": True,
                "details": "not-an-object",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ModelStreamEvent(
                    request_id="stream-invalid-error",
                    event_type="error",
                    sequence=0,
                    error=payload,
                )

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

    def test_tools_and_structured_response_are_mutually_exclusive(self) -> None:
        tool = ToolDefinition(
            name="inspect",
            description="Inspect one registered record.",
            input_schema={"type": "object", "properties": {}},
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ModelRequest(
                request_id="model-contract-exclusive-output",
                messages=(ChatMessage(role="user", content="inspect"),),
                tools=(tool,),
                response_schema={"type": "object", "properties": {}},
            )

    def test_request_rejects_duplicate_tools_and_mistyped_controls(self) -> None:
        tool = ToolDefinition(
            name="inspect",
            description="Inspect one record.",
            input_schema={"type": "object", "properties": {}},
        )
        with self.assertRaisesRegex(ValueError, "tool names must be unique"):
            ModelRequest(
                request_id="duplicate-tools",
                messages=(ChatMessage(role="user", content="inspect"),),
                tools=(tool, tool),
            )
        with self.assertRaisesRegex(ValueError, "stream must be a boolean"):
            ModelRequest(
                request_id="mistyped-stream",
                messages=(ChatMessage(role="user", content="inspect"),),
                stream=1,
            )
        with self.assertRaisesRegex(ValueError, "max_tokens must be an integer"):
            ModelRequest(
                request_id="mistyped-tokens",
                messages=(ChatMessage(role="user", content="inspect"),),
                max_tokens=True,
            )

    def test_usage_and_schema_numbers_reject_non_integer_or_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "integers"):
            TokenUsage(prompt_tokens=1.5, completion_tokens=1, total_tokens=2.5)
        with self.assertRaisesRegex(ValueError, "expected number"):
            validate_json_schema_subset(float("nan"), {"type": "number"})

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
