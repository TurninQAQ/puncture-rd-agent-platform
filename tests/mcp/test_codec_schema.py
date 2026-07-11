from __future__ import annotations

import json
import math
import unittest

from contracts.tool_inputs import TOOL_REQUEST_TYPES
from puncture_agent.mcp import (
    ContractDecodeError,
    decode_tool_request,
    to_mcp_arguments,
    to_mcp_safe_primitive,
)
from puncture_agent.mcp.schema import envelope_schema, request_schema
from puncture_agent.tooling import TOOL_DEFINITIONS, build_mock_registry

from mcp.helpers import REQUEST_FACTORIES, resolver_for


class McpCodecSchemaTests(unittest.TestCase):
    def test_all_ten_request_contracts_round_trip_through_safe_artifact_handles(self) -> None:
        for name, factory in REQUEST_FACTORIES.items():
            with self.subTest(tool=name):
                request = factory()
                arguments = to_mcp_arguments(request)
                encoded = json.dumps(arguments, sort_keys=True)
                self.assertNotIn('"uri"', encoded)
                self.assertNotIn("checksum_sha256", encoded)
                self.assertNotIn('"metadata"', encoded)
                decoded = decode_tool_request(
                    TOOL_REQUEST_TYPES[name],
                    arguments,
                    artifact_resolver=resolver_for(request),
                )
                self.assertEqual(request, decoded)

    def test_artifact_handle_rejects_internal_fields_even_when_values_are_valid(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        arguments = to_mcp_arguments(request)
        arguments["ct_artifact"]["uri"] = "file:///secret/internal.nii.gz"
        with self.assertRaisesRegex(ContractDecodeError, "forbidden fields: uri"):
            decode_tool_request(
                type(request),
                arguments,
                artifact_resolver=resolver_for(request),
            )

    def test_decoder_rejects_unknown_fields_and_bool_for_integer(self) -> None:
        request = REQUEST_FACTORIES["run_segmentation"]()
        arguments = to_mcp_arguments(request)
        arguments["unexpected"] = True
        with self.assertRaisesRegex(ContractDecodeError, "unknown fields"):
            decode_tool_request(type(request), arguments, artifact_resolver=resolver_for(request))

        arguments = to_mcp_arguments(request)
        arguments["device_id"] = True
        with self.assertRaisesRegex(ContractDecodeError, "expected an integer"):
            decode_tool_request(type(request), arguments, artifact_resolver=resolver_for(request))

    def test_decoder_rejects_non_finite_numbers(self) -> None:
        request = REQUEST_FACTORIES["generate_candidate_paths"]()
        arguments = to_mcp_arguments(request)
        arguments["max_needle_length_mm"] = math.inf
        with self.assertRaisesRegex(ContractDecodeError, "finite"):
            decode_tool_request(type(request), arguments, artifact_resolver=resolver_for(request))

    def test_schemas_are_json_serializable_and_never_expose_internal_artifact_fields(self) -> None:
        for name, definition in TOOL_DEFINITIONS.items():
            with self.subTest(tool=name):
                input_schema = request_schema(definition.request_type)
                output_schema = envelope_schema(definition.result_type)
                encoded_input = json.dumps(input_schema, sort_keys=True)
                encoded_output = json.dumps(output_schema, sort_keys=True)
                self.assertIn("ArtifactHandle", encoded_input)
                self.assertNotIn('"uri"', encoded_input)
                self.assertNotIn("checksum_sha256", encoded_input)
                self.assertNotIn('"uri"', encoded_output)
                self.assertNotIn("checksum_sha256", encoded_output)
                self.assertEqual("object", output_schema["type"])
                self.assertIn("structured", "structured output")

    def test_model_visible_tool_result_redacts_artifact_storage_details(self) -> None:
        request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        response = build_mock_registry().execute("convert_mcs_to_nifti", request)
        safe = to_mcp_safe_primitive(response)
        encoded = json.dumps(safe, sort_keys=True)
        self.assertNotIn('"uri"', encoded)
        self.assertNotIn("checksum_sha256", encoded)
        self.assertNotIn('"metadata"', encoded)
        self.assertEqual(
            "case-001-labels-nifti",
            safe["result"]["output_artifact"]["artifact_id"],
        )


if __name__ == "__main__":
    unittest.main()
