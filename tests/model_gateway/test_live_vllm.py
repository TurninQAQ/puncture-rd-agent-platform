"""Environment-gated smoke tests for a private Qwen/vLLM deployment.

These tests are skipped in ordinary CI.  They read connection settings only from
the process environment and intentionally avoid printing request/response bodies
or credentials.
"""

from __future__ import annotations

import os
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
    ToolDefinition,
    VllmGatewayConfig,
    VllmModelGateway,
)


RUN_LIVE = os.environ.get("RUN_VLLM_INTEGRATION") == "1"


@unittest.skipUnless(RUN_LIVE, "set RUN_VLLM_INTEGRATION=1 to run private vLLM tests")
class LiveVllmGatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_url = os.environ.get("VLLM_BASE_URL")
        model = os.environ.get("VLLM_MODEL")
        if not base_url or not model:
            raise unittest.SkipTest("VLLM_BASE_URL and VLLM_MODEL are required for live tests")
        cls.gateway = VllmModelGateway(
            VllmGatewayConfig(
                base_url=base_url,
                model=model,
                api_key=os.environ.get("VLLM_API_KEY"),
                ca_bundle_path=os.environ.get("VLLM_CA_BUNDLE_PATH"),
                timeout_seconds=float(os.environ.get("VLLM_TIMEOUT_SECONDS", "60")),
                max_retries=int(os.environ.get("VLLM_MAX_RETRIES", "1")),
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        gateway = getattr(cls, "gateway", None)
        if gateway is not None:
            gateway.close()

    def test_health_serves_the_configured_model(self) -> None:
        health = self.gateway.health()
        self.assertEqual(health.status, "UP")
        self.assertEqual(health.model, self.gateway.config.model)

    def test_plain_chat(self) -> None:
        response = self.gateway.generate(
            ModelRequest(
                request_id="live-plain-chat",
                messages=(ChatMessage(role="user", content="Reply with exactly: READY"),),
                temperature=0.0,
                max_tokens=16,
            )
        )
        self.assertIsInstance(response.text, str)
        self.assertTrue(response.text.strip())
        self.assertEqual(response.model, self.gateway.config.model)

    def test_one_tool_call(self) -> None:
        tool = ToolDefinition(
            name="lookup_component",
            description="Look up one component by its exact identifier.",
            input_schema={
                "type": "object",
                "properties": {"component_id": {"type": "string"}},
                "required": ["component_id"],
                "additionalProperties": False,
            },
        )
        response = self.gateway.generate(
            ModelRequest(
                request_id="live-tool-call",
                messages=(
                    ChatMessage(
                        role="user",
                        content=(
                            "You must call lookup_component exactly once with "
                            "component_id equal to CHIP-001. Do not answer in text."
                        ),
                    ),
                ),
                tools=(tool,),
                temperature=0.0,
                max_tokens=128,
            )
        )
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "lookup_component")
        self.assertEqual(response.tool_calls[0].arguments, {"component_id": "CHIP-001"})

    def test_structured_output(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["LOOKUP", "EXPLAIN"]},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }
        response = self.gateway.generate(
            ModelRequest(
                request_id="live-structured-output",
                messages=(
                    ChatMessage(
                        role="user",
                        content="Classify 'look up CHIP-001' as LOOKUP or EXPLAIN.",
                    ),
                ),
                response_schema=schema,
                temperature=0.0,
                max_tokens=64,
            )
        )
        self.assertEqual(response.structured_output["intent"], "LOOKUP")
        self.assertIsInstance(response.structured_output["confidence"], (int, float))

    def test_sse_has_one_terminal_completed_event(self) -> None:
        events = list(
            self.gateway.stream(
                ModelRequest(
                    request_id="live-sse",
                    messages=(ChatMessage(role="user", content="Reply briefly with READY"),),
                    temperature=0.0,
                    max_tokens=16,
                    stream=True,
                )
            )
        )
        self.assertTrue(events)
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual(sum(event.event_type == "completed" for event in events), 1)
        self.assertEqual(sum(event.event_type == "error" for event in events), 0)
        self.assertEqual(events[-1].event_type, "completed")


if __name__ == "__main__":
    unittest.main()
