"""Model Context Protocol adapters for the deterministic tool registry.

The dependency-free runtime is used by local tests and demos.  The optional
official SDK adapter is loaded only when the ``implementation`` extra is
installed.
"""

from .codec import (
    ContractDecodeError,
    InMemoryArtifactResolver,
    decode_tool_context,
    decode_tool_request,
    to_mcp_arguments,
    to_mcp_safe_primitive,
)
from .official_sdk import (
    OfficialMcpSdkUnavailable,
    build_official_sdk_server,
    run_official_stdio,
    run_official_stdio_sync,
)
from .ledger import (
    MAX_LEDGER_RESPONSE_BYTES,
    SQLiteToolReplayLedger,
    ToolReplayBusy,
    ToolReplayConflict,
    ToolReplayDecision,
    ToolReplayLedger,
    ToolReplayLedgerError,
    ToolReplayUncertain,
)
from .runtime import (
    McpCallResult,
    McpPrincipal,
    McpToolRuntime,
    SERVER_TOOL_NAMES,
)
from .server import JsonRpcError, LocalMcpServer
from .stdio import serve_stdio

__all__ = [
    "ContractDecodeError",
    "InMemoryArtifactResolver",
    "JsonRpcError",
    "LocalMcpServer",
    "MAX_LEDGER_RESPONSE_BYTES",
    "McpCallResult",
    "McpPrincipal",
    "McpToolRuntime",
    "OfficialMcpSdkUnavailable",
    "SQLiteToolReplayLedger",
    "SERVER_TOOL_NAMES",
    "ToolReplayBusy",
    "ToolReplayConflict",
    "ToolReplayDecision",
    "ToolReplayLedger",
    "ToolReplayLedgerError",
    "ToolReplayUncertain",
    "build_official_sdk_server",
    "decode_tool_context",
    "decode_tool_request",
    "run_official_stdio",
    "run_official_stdio_sync",
    "serve_stdio",
    "to_mcp_arguments",
    "to_mcp_safe_primitive",
]
