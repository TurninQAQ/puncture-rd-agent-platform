"""Behavior and failure tests for the deterministic Qwen gateway."""

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
    MockQwenGateway,
    ModelGatewayError,
    ModelRequest,
    ToolDefinition,
    VllmGatewayConfig,
    VllmModelGateway,
)


TOOL = ToolDefinition(
    name="search_knowledge",
    description="Search internal project knowledge.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


class MockQwenGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = MockQwenGateway()

    def test_health_does_not_require_gpu(self) -> None:
        health = self.gateway.health()
        self.assertEqual(health.status, "UP")
        self.assertFalse(health.details["gpu_required"])

    def test_plain_text_generation_is_deterministic(self) -> None:
        request = ModelRequest(
            request_id="text-1",
            messages=(ChatMessage(role="user", content="hello"),),
        )
        first = self.gateway.generate(request)
        second = self.gateway.generate(request)
        self.assertEqual(first.text, "MOCK_QWEN_RESPONSE: hello")
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.usage.total_tokens, first.usage.prompt_tokens + first.usage.completion_tokens)

    def test_structured_output_is_validated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": ["DATA", "PLANNING"]},
                "confidence": {"type": "number"},
            },
            "required": ["task_type", "confidence"],
            "additionalProperties": False,
        }
        request = ModelRequest(
            request_id="structured-1",
            messages=(ChatMessage(role="user", content="classify"),),
            response_schema=schema,
            metadata={"mock_structured_output": {"task_type": "PLANNING", "confidence": 0.9}},
        )
        response = self.gateway.generate(request)
        self.assertEqual(response.structured_output["task_type"], "PLANNING")
        self.assertEqual(response.finish_reason, "stop")

    def test_invalid_structured_output_fails_closed(self) -> None:
        schema = {
            "type": "object",
            "properties": {"task_type": {"type": "string"}},
            "required": ["task_type"],
        }
        request = ModelRequest(
            request_id="structured-bad",
            messages=(ChatMessage(role="user", content="classify"),),
            response_schema=schema,
            metadata={"mock_structured_output": {}},
        )
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.gateway.generate(request)

    def test_tool_call_validates_name_and_arguments(self) -> None:
        request = ModelRequest(
            request_id="tool-1",
            messages=(ChatMessage(role="user", content="search"),),
            tools=(TOOL,),
            metadata={"mock_tool_call": {"name": "search_knowledge", "arguments": {"query": "spacing"}}},
        )
        response = self.gateway.generate(request)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].name, "search_knowledge")
        self.assertEqual(response.tool_calls[0].arguments["query"], "spacing")

    def test_unknown_tool_and_invalid_arguments_are_normalized_errors(self) -> None:
        unknown_request = ModelRequest(
            request_id="tool-unknown",
            messages=(ChatMessage(role="user", content="search"),),
            tools=(TOOL,),
            metadata={"mock_tool_call": {"name": "delete_case", "arguments": {}}},
        )
        with self.assertRaises(ModelGatewayError) as unknown_context:
            self.gateway.generate(unknown_request)
        self.assertEqual(unknown_context.exception.code, "UNKNOWN_TOOL")

        invalid_request = ModelRequest(
            request_id="tool-invalid",
            messages=(ChatMessage(role="user", content="search"),),
            tools=(TOOL,),
            metadata={"mock_tool_call": {"name": "search_knowledge", "arguments": {}}},
        )
        with self.assertRaises(ModelGatewayError) as invalid_context:
            self.gateway.generate(invalid_request)
        self.assertEqual(invalid_context.exception.code, "TOOL_ARGUMENT_SCHEMA_ERROR")

    def test_stream_has_monotonic_sequence_and_terminal_event(self) -> None:
        request = ModelRequest(
            request_id="stream-1",
            messages=(ChatMessage(role="user", content="hello stream"),),
            metadata={"mock_text": "one two three"},
            stream=True,
        )
        events = list(self.gateway.stream(request))
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual("".join(event.delta or "" for event in events), "one two three")
        self.assertEqual(events[-1].event_type, "completed")
        self.assertEqual(events[-1].response.text, "one two three")

    def test_forced_retryable_failure_is_available_for_graph_tests(self) -> None:
        request = ModelRequest(
            request_id="error-1",
            messages=(ChatMessage(role="user", content="fail"),),
            metadata={"force_error": {"code": "MODEL_TIMEOUT", "retryable": True}},
        )
        with self.assertRaises(ModelGatewayError) as context:
            self.gateway.generate(request)
        self.assertEqual(context.exception.code, "MODEL_TIMEOUT")
        self.assertTrue(context.exception.retryable)

    def test_production_adapter_remains_an_explicit_stub(self) -> None:
        adapter = VllmModelGateway(VllmGatewayConfig(base_url="http://127.0.0.1:8000/v1", model="qwen"))
        with self.assertRaises(NotImplementedError):
            adapter.health()


if __name__ == "__main__":
    unittest.main()
