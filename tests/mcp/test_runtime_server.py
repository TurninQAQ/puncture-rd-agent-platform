from __future__ import annotations

from dataclasses import replace
import json
from threading import Event
import time
import unittest

from contracts.common import ToolResponseEnvelope
from contracts.enums import ErrorCode, ToolExecutionStatus
from puncture_agent.mcp import (
    LocalMcpServer,
    McpPrincipal,
    McpToolRuntime,
    OfficialMcpSdkUnavailable,
    build_official_sdk_server,
    to_mcp_arguments,
)
from puncture_agent.tooling import TOOL_DEFINITIONS, bind_tool_handlers, build_mock_registry
from puncture_agent.tooling.registry import ToolRegistry
from puncture_agent.mocks.tool_mocks import MOCK_HANDLERS

from mcp.helpers import REQUEST_FACTORIES, resolver_for


def _initialized_server(server_name: str, requests: tuple[object, ...]) -> LocalMcpServer:
    runtime = McpToolRuntime(
        build_mock_registry(),
        resolver_for(*requests),
        server_name=server_name,
    )
    server = LocalMcpServer(
        runtime,
        principal=McpPrincipal("unit-test", ("case-001",)),
    )
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "unit-test", "version": "1"},
            },
        }
    )
    assert response is not None and response["result"]["protocolVersion"] == "2025-11-25"
    assert server.handle(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    ) is None
    return server


class McpRuntimeServerTests(unittest.TestCase):
    def test_registry_factory_requires_exactly_the_frozen_ten_handlers(self) -> None:
        handlers = dict(MOCK_HANDLERS)
        rebound = bind_tool_handlers(handlers)
        self.assertEqual(
            tuple(sorted(TOOL_DEFINITIONS)),
            tuple(definition.name for definition in rebound.list_definitions()),
        )
        handlers.pop("inspect_case_metadata")
        with self.assertRaisesRegex(ValueError, "missing=inspect_case_metadata"):
            bind_tool_handlers(handlers)

    def test_three_servers_discover_exactly_ten_strongly_typed_tools(self) -> None:
        requests = tuple(factory() for factory in REQUEST_FACTORIES.values())
        counts = {}
        discovered: set[str] = set()
        read_only: dict[str, bool] = {}
        for server_name in ("case-data", "segmentation", "planning-safety"):
            runtime = McpToolRuntime(
                build_mock_registry(),
                resolver_for(*requests),
                server_name=server_name,
            )
            tools = runtime.list_tools()
            counts[server_name] = len(tools)
            for tool in tools:
                discovered.add(tool["name"])
                read_only[tool["name"]] = tool["annotations"]["readOnlyHint"]
                self.assertEqual("object", tool["outputSchema"]["type"])
                self.assertEqual("forbidden", tool["execution"]["taskSupport"])
                self.assertTrue(tool["annotations"]["idempotentHint"])
                self.assertFalse(tool["annotations"]["openWorldHint"])
        self.assertEqual({"case-data": 3, "segmentation": 3, "planning-safety": 4}, counts)
        self.assertEqual(set(REQUEST_FACTORIES), discovered)
        self.assertTrue(read_only["inspect_case_metadata"])
        self.assertFalse(read_only["convert_mcs_to_nifti"])
        self.assertFalse(read_only["run_segmentation"])
        self.assertTrue(read_only["evaluate_path_safety"])

    def test_successful_call_returns_structured_and_text_compatible_content(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver_for(request),
            server_name="case-data",
        )
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        self.assertFalse(result.is_error)
        self.assertEqual("SUCCESS", result.structured_content["status"])
        self.assertEqual(result.structured_content, json.loads(result.content[0]["text"]))
        self.assertEqual("trace-001", result.meta["com.turninqaq/traceId"])

    def test_principal_mismatch_fails_closed_without_executing_tool(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        delegate = resolver_for(request)

        class CountingResolver:
            def __init__(self):
                self.calls = 0

            def resolve(self, artifact_id, *, expected_case_id=None, expected_type=None):
                self.calls += 1
                return delegate.resolve(
                    artifact_id,
                    expected_case_id=expected_case_id,
                    expected_type=expected_type,
                )

        resolver = CountingResolver()
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver,
            server_name="case-data",
        )
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("another-user", ("case-001",)),
        )
        self.assertTrue(result.is_error)
        self.assertEqual("PERMISSION_DENIED", result.structured_content["error"]["code"])
        self.assertEqual(0, resolver.calls)

    def test_invalid_arguments_still_return_output_schema_shaped_error(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        arguments = to_mcp_arguments(request)
        arguments["ct_artifact"]["checksum_sha256"] = "a" * 64
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver_for(request),
            server_name="case-data",
        )
        result = runtime.call_tool(
            "inspect_case_metadata",
            arguments,
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        payload = result.structured_content
        self.assertTrue(result.is_error)
        self.assertEqual("FAILED", payload["status"])
        self.assertEqual("INVALID_ARGUMENT", payload["error"]["code"])
        self.assertEqual(
            {
                "request_id",
                "trace_id",
                "tool_name",
                "tool_version",
                "status",
                "result",
                "artifacts",
                "metrics",
                "warnings",
                "error",
                "started_at",
                "finished_at",
            },
            set(payload),
        )

    def test_expired_deadline_maps_to_retryable_timeout(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        request = replace(request, context=replace(request.context, deadline_epoch_ms=1))
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver_for(request),
            server_name="case-data",
        )
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        self.assertEqual("TIMEOUT", result.structured_content["error"]["code"])
        self.assertTrue(result.structured_content["error"]["retryable"])

    def test_registry_response_identity_mismatch_is_contract_violation(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        original = MOCK_HANDLERS["inspect_case_metadata"]

        def wrong_identity(value):
            response = original(value)
            return ToolResponseEnvelope(
                request_id="wrong",
                trace_id=response.trace_id,
                tool_name=response.tool_name,
                tool_version=response.tool_version,
                status=response.status,
                result=response.result,
                artifacts=response.artifacts,
                metrics=response.metrics,
                warnings=response.warnings,
                error=response.error,
                started_at=response.started_at,
                finished_at=response.finished_at,
            )

        registry = ToolRegistry()
        for name, definition in TOOL_DEFINITIONS.items():
            registry.register(
                definition,
                wrong_identity if name == "inspect_case_metadata" else MOCK_HANDLERS[name],
            )
        runtime = McpToolRuntime(registry, resolver_for(request), server_name="case-data")
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        self.assertEqual("CONTRACT_VIOLATION", result.structured_content["error"]["code"])

    def test_unhandled_handler_exception_does_not_leak_internal_location(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()

        def explode(value):
            del value
            raise RuntimeError("memory://private/secret.nii.gz checksum=deadbeef")

        registry = ToolRegistry()
        for name, definition in TOOL_DEFINITIONS.items():
            registry.register(
                definition,
                explode if name == "inspect_case_metadata" else MOCK_HANDLERS[name],
            )
        runtime = McpToolRuntime(registry, resolver_for(request), server_name="case-data")
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        encoded = json.dumps(result.to_protocol_result(), sort_keys=True)
        self.assertEqual("INTERNAL_ERROR", result.structured_content["error"]["code"])
        self.assertNotIn("memory://private", encoded)
        self.assertNotIn("deadbeef", encoded)

    def test_json_rpc_handshake_list_call_and_protocol_errors(self) -> None:
        requests = tuple(
            REQUEST_FACTORIES[name]()
            for name in ("inspect_case_metadata", "convert_mcs_to_nifti", "validate_label_schema")
        )
        server = _initialized_server("case-data", requests)
        listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        self.assertEqual(3, len(listed["result"]["tools"]))

        request = requests[0]
        called = server.handle(
            {
                "jsonrpc": "2.0",
                "id": "call-1",
                "method": "tools/call",
                "params": {
                    "name": "inspect_case_metadata",
                    "arguments": to_mcp_arguments(request),
                },
            }
        )
        self.assertEqual("SUCCESS", called["result"]["structuredContent"]["status"])
        self.assertFalse(called["result"]["isError"])

        unknown = server.handle({"jsonrpc": "2.0", "id": 3, "method": "unknown", "params": {}})
        self.assertEqual(-32601, unknown["error"]["code"])
        parse_error = json.loads(server.handle_json_line("{not-json"))
        self.assertEqual(-32700, parse_error["error"]["code"])

    def test_tools_call_notification_is_ignored_without_execution(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        calls = []

        def counted(value):
            calls.append(value.context.request_id)
            return MOCK_HANDLERS["inspect_case_metadata"](value)

        registry = ToolRegistry()
        for name, definition in TOOL_DEFINITIONS.items():
            registry.register(
                definition,
                counted if name == "inspect_case_metadata" else MOCK_HANDLERS[name],
            )
        runtime = McpToolRuntime(registry, resolver_for(request), server_name="case-data")
        server = LocalMcpServer(runtime, principal=McpPrincipal("unit-test", ("case-001",)))
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            }
        )
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "inspect_case_metadata",
                    "arguments": to_mcp_arguments(request),
                },
            }
        )
        self.assertIsNone(response)
        self.assertEqual([], calls)

    def test_runtime_timeout_returns_only_after_handler_has_stopped(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        request = replace(
            request,
            context=replace(
                request.context,
                deadline_epoch_ms=int(time.time() * 1000) + 10,
            ),
        )
        finished = Event()

        def slow(value):
            time.sleep(0.04)
            finished.set()
            return MOCK_HANDLERS["inspect_case_metadata"](value)

        registry = ToolRegistry()
        for name, definition in TOOL_DEFINITIONS.items():
            registry.register(
                definition,
                slow if name == "inspect_case_metadata" else MOCK_HANDLERS[name],
            )
        runtime = McpToolRuntime(registry, resolver_for(request), server_name="case-data")
        result = runtime.call_tool(
            "inspect_case_metadata",
            to_mcp_arguments(request),
            principal=McpPrincipal("unit-test", ("case-001",)),
        )
        self.assertEqual("TIMEOUT", result.structured_content["error"]["code"])
        self.assertTrue(finished.is_set())

    def test_tools_cannot_be_called_before_initialized_notification(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver_for(request),
            server_name="case-data",
        )
        server = LocalMcpServer(runtime, principal=McpPrincipal("unit-test", ("case-001",)))
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        self.assertEqual(-32002, response["error"]["code"])

    def test_official_sdk_adapter_is_explicitly_optional(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        runtime = McpToolRuntime(
            build_mock_registry(),
            resolver_for(request),
            server_name="case-data",
        )
        with self.assertRaisesRegex(OfficialMcpSdkUnavailable, "mcp>=1.27,<2"):
            build_official_sdk_server(
                runtime,
                principal=McpPrincipal("unit-test", ("case-001",)),
            )


if __name__ == "__main__":
    unittest.main()
