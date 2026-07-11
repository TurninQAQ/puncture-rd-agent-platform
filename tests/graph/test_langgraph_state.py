from __future__ import annotations

import math
import sys
import unittest
from dataclasses import fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent.langgraph_state import (  # noqa: E402
    DEFAULT_MAX_STATE_BYTES,
    NonFiniteFloatError,
    NonStringMappingKeyError,
    ProductionAgentState,
    RawBytesStateError,
    StateSerializationError,
    StateSizeLimitError,
    UnknownStateFieldError,
    state_from_mapping,
    state_to_mapping,
)
from puncture_agent.agent.state import AgentState  # noqa: E402


class LangGraphStateTests(unittest.TestCase):
    def test_typed_dict_fields_exactly_match_agent_state(self) -> None:
        self.assertEqual(
            {item.name for item in fields(AgentState)},
            set(ProductionAgentState.__annotations__),
        )
        self.assertFalse(ProductionAgentState.__total__)

    def test_round_trip_preserves_all_state_fields(self) -> None:
        original = AgentState(
            user_query="检查 Case-601 的路径",
            case_id="Case-601",
            retry_count=1,
            max_retries=3,
            metadata={"nested": {"score": 0.25}, "enabled": True},
            candidate_paths=[{"candidate_id": "path-1", "length_mm": 82.5}],
        )

        mapping = state_to_mapping(original)
        restored = state_from_mapping(mapping)

        self.assertEqual(original.to_dict(), mapping)
        self.assertEqual(original.to_dict(), restored.to_dict())

    def test_unknown_top_level_field_is_rejected_explicitly(self) -> None:
        payload = AgentState(user_query="test").to_dict()
        payload["future_unmigrated_field"] = "value"

        with self.assertRaises(UnknownStateFieldError):
            state_from_mapping(payload)

    def test_deep_raw_binary_values_are_rejected_before_copying(self) -> None:
        factories = (
            lambda: b"raw voxels",
            lambda: bytearray(b"raw voxels"),
            lambda: memoryview(b"raw voxels"),
        )
        for make_value in factories:
            with self.subTest(binary_type=type(make_value()).__name__):
                state = AgentState(
                    user_query="test",
                    metadata={"level_1": [{"level_2": make_value()}]},
                )
                with self.assertRaises(RawBytesStateError):
                    state_to_mapping(state)

    def test_non_string_mapping_key_is_rejected_at_any_depth(self) -> None:
        state = AgentState(
            user_query="test",
            metadata={"nested": {1: "not a JSON object key"}},
        )

        with self.assertRaises(NonStringMappingKeyError):
            state_to_mapping(state)

    def test_non_finite_float_is_rejected_at_any_depth(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                state = AgentState(
                    user_query="test",
                    tool_results=[{"metrics": {"value": value}}],
                )
                with self.assertRaises(NonFiniteFloatError):
                    state_to_mapping(state)

    def test_non_json_value_has_a_serialization_error(self) -> None:
        state = AgentState(user_query="test", metadata={"invalid": object()})

        with self.assertRaises(StateSerializationError):
            state_to_mapping(state)

    def test_default_limit_rejects_state_larger_than_one_mebibyte(self) -> None:
        self.assertEqual(1024 * 1024, DEFAULT_MAX_STATE_BYTES)
        state = AgentState(
            user_query="test",
            metadata={"oversized": "x" * DEFAULT_MAX_STATE_BYTES},
        )

        with self.assertRaises(StateSizeLimitError) as raised:
            state_to_mapping(state)

        self.assertGreater(raised.exception.actual_bytes, DEFAULT_MAX_STATE_BYTES)
        self.assertEqual(DEFAULT_MAX_STATE_BYTES, raised.exception.max_bytes)

    def test_conversion_does_not_share_nested_mutable_references(self) -> None:
        original = AgentState(
            user_query="test",
            metadata={"nested": {"items": [{"value": 1}]}},
        )

        mapping = state_to_mapping(original)
        mapping["metadata"]["nested"]["items"][0]["value"] = 2
        self.assertEqual(1, original.metadata["nested"]["items"][0]["value"])

        restored = state_from_mapping(mapping)
        restored.metadata["nested"]["items"][0]["value"] = 3
        self.assertEqual(2, mapping["metadata"]["nested"]["items"][0]["value"])


if __name__ == "__main__":
    unittest.main()
