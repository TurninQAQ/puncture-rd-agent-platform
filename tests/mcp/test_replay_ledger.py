from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest

from contracts.common import ToolResponseEnvelope
from contracts.enums import ErrorCode, ToolExecutionStatus
from contracts.errors import ErrorDetail
from puncture_agent.agent.tool_bridge import McpToolExecutor
from puncture_agent.mcp import (
    McpPrincipal,
    McpToolRuntime,
    SQLiteToolReplayLedger,
    ToolReplayLedgerError,
    ToolReplayUncertain,
    to_mcp_arguments,
)
from puncture_agent.mocks.tool_mocks import MOCK_HANDLERS
from puncture_agent.tooling import TOOL_DEFINITIONS
from puncture_agent.tooling.registry import ToolRegistry

from mcp.helpers import REQUEST_FACTORIES, resolver_for


PRINCIPAL = McpPrincipal("unit-test", ("case-001",))
NOW = "2026-07-11T00:00:00Z"


def counted_registry(
    tool_name: str,
    calls: list[str],
    *,
    started: Event | None = None,
    release: Event | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    def counted(request):
        calls.append(request.context.request_id)
        if started is not None:
            started.set()
        if release is not None:
            release.wait(timeout=5)
        return MOCK_HANDLERS[tool_name](request)

    for name, definition in TOOL_DEFINITIONS.items():
        registry.register(
            definition,
            counted if name == tool_name else MOCK_HANDLERS[name],
        )
    return registry


def failure_registry(
    tool_name: str,
    calls: list[str],
    *,
    code: ErrorCode,
    retryable: bool,
) -> ToolRegistry:
    registry = ToolRegistry()

    def failed(request):
        calls.append(request.context.request_id)
        return ToolResponseEnvelope(
            request_id=request.context.request_id,
            trace_id=request.context.trace_id,
            tool_name=tool_name,
            tool_version=TOOL_DEFINITIONS[tool_name].version,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(),
            warnings=(),
            error=ErrorDetail(
                code=code,
                message="deterministic test failure",
                retryable=retryable,
            ),
            started_at=NOW,
            finished_at=NOW,
        )

    for name, definition in TOOL_DEFINITIONS.items():
        registry.register(
            definition,
            failed if name == tool_name else MOCK_HANDLERS[name],
        )
    return registry


class ReplayLedgerTests(unittest.TestCase):
    def test_success_replays_after_runtime_and_ledger_restart(self) -> None:
        request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        calls: list[str] = []
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tool-replay.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                first_runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first = first_runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
            replay_request = replace(
                request,
                context=replace(
                    request.context,
                    request_id="req-after-graph-restart",
                    trace_id="trace-after-graph-restart",
                    requested_at="2026-07-11T00:00:00Z",
                    deadline_epoch_ms=1,
                ),
            )
            with SQLiteToolReplayLedger(database) as restarted_ledger:
                restarted_runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(replay_request),
                    server_name="case-data",
                    replay_ledger=restarted_ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                replay = restarted_runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(replay_request),
                    principal=PRINCIPAL,
                )

            self.assertFalse(first.is_error)
            self.assertFalse(replay.is_error)
            self.assertEqual([request.context.request_id], calls)
            self.assertEqual(
                "req-after-graph-restart",
                replay.structured_content["request_id"],
            )
            self.assertEqual(
                "trace-after-graph-restart",
                replay.structured_content["trace_id"],
            )
            self.assertTrue(replay.meta["com.turninqaq/idempotentReplay"])
            self.assertEqual(
                first.structured_content["result"],
                replay.structured_content["result"],
            )
            self.assertEqual(
                0o600,
                database.stat().st_mode & 0o777,
            )

    def test_same_scope_with_changed_payload_is_rejected(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        calls: list[str] = []
        with TemporaryDirectory() as directory:
            with SQLiteToolReplayLedger(Path(directory) / "ledger.sqlite3") as ledger:
                runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
                changed = to_mcp_arguments(request)
                changed["require_same_geometry"] = False
                conflict = runtime.call_tool(
                    "inspect_case_metadata",
                    changed,
                    principal=PRINCIPAL,
                )

        self.assertFalse(first.is_error)
        self.assertTrue(conflict.is_error)
        self.assertEqual("INVALID_ARGUMENT", conflict.structured_content["error"]["code"])
        self.assertFalse(conflict.structured_content["error"]["retryable"])
        self.assertEqual([request.context.request_id], calls)

    def test_concurrent_runtime_is_busy_then_replays_without_second_execution(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        arguments = to_mcp_arguments(request)
        calls: list[str] = []
        started = Event()
        release = Event()
        with TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            with (
                SQLiteToolReplayLedger(database) as first_ledger,
                SQLiteToolReplayLedger(database) as second_ledger,
            ):
                first_runtime = McpToolRuntime(
                    counted_registry(
                        "inspect_case_metadata",
                        calls,
                        started=started,
                        release=release,
                    ),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=first_ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                second_runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=second_ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        first_runtime.call_tool,
                        "inspect_case_metadata",
                        arguments,
                        principal=PRINCIPAL,
                    )
                    self.assertTrue(started.wait(timeout=5))
                    busy = second_runtime.call_tool(
                        "inspect_case_metadata",
                        arguments,
                        principal=PRINCIPAL,
                    )
                    release.set()
                    first = future.result(timeout=5)
                replay = second_runtime.call_tool(
                    "inspect_case_metadata",
                    arguments,
                    principal=PRINCIPAL,
                )

        self.assertFalse(first.is_error)
        self.assertEqual("DEPENDENCY_FAILED", busy.structured_content["error"]["code"])
        self.assertTrue(busy.structured_content["error"]["retryable"])
        self.assertFalse(replay.is_error)
        self.assertTrue(replay.meta["com.turninqaq/idempotentReplay"])
        self.assertEqual([request.context.request_id], calls)

    def test_success_is_not_acknowledged_when_ledger_commit_fails(self) -> None:
        request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        calls: list[str] = []

        class FailingCompleteLedger(SQLiteToolReplayLedger):
            def complete(self, decision, response):
                del decision, response
                raise ToolReplayLedgerError("private database detail")

        with TemporaryDirectory() as directory:
            database = Path(directory) / "failed-complete.sqlite3"
            with FailingCompleteLedger(database) as ledger:
                runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                result = runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
            with SQLiteToolReplayLedger(database) as restarted:
                restarted_runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=restarted,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                still_uncertain = restarted_runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertTrue(result.is_error)
        self.assertEqual("DEPENDENCY_FAILED", result.structured_content["error"]["code"])
        self.assertFalse(result.structured_content["error"]["retryable"])
        self.assertNotIn("private database detail", json.dumps(result.to_protocol_result()))
        self.assertTrue(still_uncertain.is_error)
        self.assertFalse(still_uncertain.structured_content["error"]["retryable"])
        self.assertEqual([request.context.request_id], calls)

    def test_nonretryable_failure_replays_but_retryable_failure_executes_again(self) -> None:
        request = REQUEST_FACTORIES["generate_candidate_paths"]()
        with TemporaryDirectory() as directory:
            terminal_calls: list[str] = []
            terminal_database = Path(directory) / "terminal.sqlite3"
            with SQLiteToolReplayLedger(terminal_database) as ledger:
                runtime = McpToolRuntime(
                    failure_registry(
                        "generate_candidate_paths",
                        terminal_calls,
                        code=ErrorCode.NO_CANDIDATE_PATH,
                        retryable=False,
                    ),
                    resolver_for(request),
                    server_name="planning-safety",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first_terminal = runtime.call_tool(
                    "generate_candidate_paths",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
            with SQLiteToolReplayLedger(terminal_database) as restarted:
                replay_runtime = McpToolRuntime(
                    failure_registry(
                        "generate_candidate_paths",
                        terminal_calls,
                        code=ErrorCode.NO_CANDIDATE_PATH,
                        retryable=False,
                    ),
                    resolver_for(request),
                    server_name="planning-safety",
                    replay_ledger=restarted,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                replay_terminal = replay_runtime.call_tool(
                    "generate_candidate_paths",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

            retryable_calls: list[str] = []
            retryable_database = Path(directory) / "retryable.sqlite3"
            with SQLiteToolReplayLedger(retryable_database) as ledger:
                retry_runtime = McpToolRuntime(
                    failure_registry(
                        "generate_candidate_paths",
                        retryable_calls,
                        code=ErrorCode.TIMEOUT,
                        retryable=True,
                    ),
                    resolver_for(request),
                    server_name="planning-safety",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                retry_runtime.call_tool(
                    "generate_candidate_paths",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
                retry_runtime.call_tool(
                    "generate_candidate_paths",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertTrue(first_terminal.is_error)
        self.assertTrue(replay_terminal.is_error)
        self.assertEqual("NO_CANDIDATE_PATH", replay_terminal.structured_content["error"]["code"])
        self.assertTrue(replay_terminal.meta["com.turninqaq/idempotentReplay"])
        self.assertEqual([request.context.request_id], terminal_calls)
        self.assertEqual(2, len(retryable_calls))

    def test_replay_rechecks_current_authorization(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        calls: list[str] = []
        allowed = [True]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "authorization.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: allowed[0],
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
                allowed[0] = False
                denied = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertFalse(first.is_error)
        self.assertTrue(denied.is_error)
        self.assertEqual("PERMISSION_DENIED", denied.structured_content["error"]["code"])
        self.assertNotIn("com.turninqaq/idempotentReplay", denied.meta)
        self.assertEqual([request.context.request_id], calls)

    def test_replay_rechecks_stored_response_authorization(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        calls: list[str] = []
        response_allowed = [True]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "response-authorization.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=(
                        lambda name, value, response, principal: response_allowed[0]
                    ),
                )
                first = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
                response_allowed[0] = False
                denied = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertFalse(first.is_error)
        self.assertTrue(denied.is_error)
        self.assertEqual("PERMISSION_DENIED", denied.structured_content["error"]["code"])
        self.assertNotIn("com.turninqaq/idempotentReplay", denied.meta)
        self.assertEqual([request.context.request_id], calls)

    def test_timed_out_write_becomes_uncertain_and_never_executes_twice(self) -> None:
        request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        calls: list[str] = []
        registry = ToolRegistry()

        def slow_write(value):
            calls.append(value.context.request_id)
            time.sleep(0.03)
            return MOCK_HANDLERS["convert_mcs_to_nifti"](value)

        for name, definition in TOOL_DEFINITIONS.items():
            registry.register(
                replace(definition, default_timeout_ms=10)
                if name == "convert_mcs_to_nifti"
                else definition,
                slow_write if name == "convert_mcs_to_nifti" else MOCK_HANDLERS[name],
            )

        with TemporaryDirectory() as directory:
            with SQLiteToolReplayLedger(Path(directory) / "timeout.sqlite3") as ledger:
                runtime = McpToolRuntime(
                    registry,
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first = runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
                second = runtime.call_tool(
                    "convert_mcs_to_nifti",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertTrue(first.is_error)
        self.assertEqual("DEPENDENCY_FAILED", first.structured_content["error"]["code"])
        self.assertFalse(first.structured_content["error"]["retryable"])
        self.assertTrue(second.is_error)
        self.assertFalse(second.structured_content["error"]["retryable"])
        self.assertEqual([request.context.request_id], calls)

    def test_bridge_replays_after_graph_state_is_lost(self) -> None:
        frozen_request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        legacy_request = {
            "case_id": "case-001",
            "source_artifact_id": frozen_request.mcs_artifact.artifact_id,
            "reference_ct_artifact_id": frozen_request.reference_ct_artifact.artifact_id,
        }
        calls: list[str] = []
        with TemporaryDirectory() as directory:
            database = Path(directory) / "bridge-crash-window.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(frozen_request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                bridge = McpToolExecutor(
                    runtime,
                    principal=PRINCIPAL,
                    session_id="graph-session-before-checkpoint",
                    trace_id="graph-trace-before-checkpoint",
                    clock=lambda: datetime.fromisoformat(
                        "2026-07-11T00:00:00+00:00"
                    ),
                )
                first = bridge.execute("convert_mcs_to_nifti", legacy_request)

            # Simulate losing every AgentState mutation after the MCP response:
            # rebuild bridge, runtime, registry and SQLite connection from disk.
            with SQLiteToolReplayLedger(database) as restarted:
                restarted_runtime = McpToolRuntime(
                    counted_registry("convert_mcs_to_nifti", calls),
                    resolver_for(frozen_request),
                    server_name="case-data",
                    replay_ledger=restarted,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                restarted_bridge = McpToolExecutor(
                    restarted_runtime,
                    principal=PRINCIPAL,
                    session_id="graph-session-before-checkpoint",
                    trace_id="graph-trace-before-checkpoint",
                    clock=lambda: datetime.fromisoformat(
                        "2026-07-11T00:00:00+00:00"
                    ),
                )
                replay = restarted_bridge.execute(
                    "convert_mcs_to_nifti",
                    legacy_request,
                )

        self.assertEqual(first["result"], replay["result"])
        self.assertEqual(1, len(calls))

    def test_expired_write_claim_becomes_persistently_uncertain(self) -> None:
        clock = [100.0]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "uncertain.sqlite3"
            with SQLiteToolReplayLedger(
                database,
                lease_seconds=1.0,
                clock=lambda: clock[0],
            ) as ledger:
                ledger.begin(
                    "write-scope",
                    "write-fingerprint",
                    reclaim_expired=False,
                )
                clock[0] = 102.0
                with self.assertRaises(ToolReplayUncertain):
                    ledger.begin(
                        "write-scope",
                        "write-fingerprint",
                        reclaim_expired=False,
                    )
                status = ledger._connection.execute(
                    "SELECT status FROM tool_replay_ledger WHERE scope_key = ?",
                    ("write-scope",),
                ).fetchone()[0]
                self.assertEqual("UNCERTAIN", status)
                with self.assertRaises(ToolReplayUncertain):
                    ledger.begin(
                        "write-scope",
                        "write-fingerprint",
                        reclaim_expired=False,
                    )

    def test_runtime_rejects_lease_shorter_than_server_timeout(self) -> None:
        request = REQUEST_FACTORIES["convert_mcs_to_nifti"]()
        with TemporaryDirectory() as directory:
            with SQLiteToolReplayLedger(
                Path(directory) / "missing-validator.sqlite3"
            ) as ledger:
                with self.assertRaisesRegex(
                    ValueError,
                    "replay_response_validator is required",
                ):
                    McpToolRuntime(
                        counted_registry("convert_mcs_to_nifti", []),
                        resolver_for(request),
                        server_name="case-data",
                        replay_ledger=ledger,
                        replay_authorizer=lambda name, value, principal: True,
                    )
            with SQLiteToolReplayLedger(
                Path(directory) / "short-lease.sqlite3",
                lease_seconds=1.0,
            ) as ledger:
                with self.assertRaisesRegex(ValueError, "TTL must exceed"):
                    McpToolRuntime(
                        counted_registry("convert_mcs_to_nifti", []),
                        resolver_for(request),
                        server_name="case-data",
                        replay_ledger=ledger,
                        replay_authorizer=lambda name, value, principal: True,
                        replay_response_validator=lambda name, value, response, principal: True,
                    )

    def test_corrupt_stored_response_fails_closed_without_handler_reexecution(self) -> None:
        request = REQUEST_FACTORIES["inspect_case_metadata"]()
        calls: list[str] = []
        with TemporaryDirectory() as directory:
            database = Path(directory) / "corrupt.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=ledger,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                first = runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )
            connection = sqlite3.connect(database)
            connection.execute(
                "UPDATE tool_replay_ledger SET response_json = ?",
                ('{"status":"SUCCESS","result":{"forged":true}}',),
            )
            connection.commit()
            connection.close()
            with SQLiteToolReplayLedger(database) as restarted:
                restarted_runtime = McpToolRuntime(
                    counted_registry("inspect_case_metadata", calls),
                    resolver_for(request),
                    server_name="case-data",
                    replay_ledger=restarted,
                    replay_authorizer=lambda name, value, principal: True,
                    replay_response_validator=lambda name, value, response, principal: True,
                )
                corrupt = restarted_runtime.call_tool(
                    "inspect_case_metadata",
                    to_mcp_arguments(request),
                    principal=PRINCIPAL,
                )

        self.assertFalse(first.is_error)
        self.assertTrue(corrupt.is_error)
        self.assertEqual("DEPENDENCY_FAILED", corrupt.structured_content["error"]["code"])
        self.assertFalse(corrupt.structured_content["error"]["retryable"])
        self.assertEqual([request.context.request_id], calls)

    def test_ledger_uses_full_sync_and_enforces_response_size_limit(self) -> None:
        with SQLiteToolReplayLedger(":memory:") as ledger:
            synchronous = ledger._connection.execute("PRAGMA synchronous").fetchone()[0]
            self.assertEqual(2, synchronous)
            decision = ledger.begin(
                "scope-size",
                "fingerprint-size",
                reclaim_expired=True,
            )
            with self.assertRaisesRegex(ToolReplayLedgerError, "exceeds 1 MiB"):
                ledger.complete(decision, {"payload": "x" * (1024 * 1024)})

    def test_three_logical_servers_share_one_durable_ledger(self) -> None:
        scenarios = (
            ("case-data", "inspect_case_metadata"),
            ("segmentation", "run_segmentation"),
            ("planning-safety", "generate_candidate_paths"),
        )
        requests = {
            tool_name: REQUEST_FACTORIES[tool_name]()
            for _, tool_name in scenarios
        }
        calls = {tool_name: [] for _, tool_name in scenarios}
        with TemporaryDirectory() as directory:
            database = Path(directory) / "three-servers.sqlite3"
            with SQLiteToolReplayLedger(database) as ledger:
                for server_name, tool_name in scenarios:
                    runtime = McpToolRuntime(
                        counted_registry(tool_name, calls[tool_name]),
                        resolver_for(requests[tool_name]),
                        server_name=server_name,
                        replay_ledger=ledger,
                        replay_authorizer=lambda name, value, principal: True,
                        replay_response_validator=lambda name, value, response, principal: True,
                    )
                    result = runtime.call_tool(
                        tool_name,
                        to_mcp_arguments(requests[tool_name]),
                        principal=PRINCIPAL,
                    )
                    self.assertFalse(result.is_error)
            with SQLiteToolReplayLedger(database) as restarted:
                for server_name, tool_name in scenarios:
                    runtime = McpToolRuntime(
                        counted_registry(tool_name, calls[tool_name]),
                        resolver_for(requests[tool_name]),
                        server_name=server_name,
                        replay_ledger=restarted,
                        replay_authorizer=lambda name, value, principal: True,
                        replay_response_validator=lambda name, value, response, principal: True,
                    )
                    replay = runtime.call_tool(
                        tool_name,
                        to_mcp_arguments(requests[tool_name]),
                        principal=PRINCIPAL,
                    )
                    self.assertTrue(replay.meta["com.turninqaq/idempotentReplay"])

        self.assertEqual(
            {tool_name: [requests[tool_name].context.request_id] for _, tool_name in scenarios},
            calls,
        )

    def test_expired_pending_claim_can_be_recovered(self) -> None:
        clock = [100.0]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            with SQLiteToolReplayLedger(
                database,
                lease_seconds=1.0,
                clock=lambda: clock[0],
            ) as ledger:
                first = ledger.begin(
                    "scope",
                    "fingerprint",
                    reclaim_expired=True,
                )
                clock[0] = 102.0
                recovered = ledger.begin(
                    "scope",
                    "fingerprint",
                    reclaim_expired=True,
                )
                self.assertNotEqual(first.owner_token, recovered.owner_token)
                with self.assertRaisesRegex(ToolReplayLedgerError, "claim was lost"):
                    ledger.complete(first, {"status": "SUCCESS"})
                ledger.complete(recovered, {"status": "SUCCESS"})
                replay = ledger.begin(
                    "scope",
                    "fingerprint",
                    reclaim_expired=True,
                )
                self.assertTrue(replay.is_replay)
                self.assertEqual("SUCCESS", replay.response["status"])


if __name__ == "__main__":
    unittest.main()
