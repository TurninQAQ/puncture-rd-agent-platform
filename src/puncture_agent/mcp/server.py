"""Small MCP 2025-11-25 JSON-RPC dispatcher for dependency-free local demos."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from puncture_agent import __version__

from .runtime import McpPrincipal, McpToolRuntime


SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")
_REQUEST_METHODS = {"initialize", "ping", "tools/list", "tools/call"}


@dataclass(frozen=True, slots=True)
class JsonRpcError(Exception):
    code: int
    message: str
    data: Any | None = None

    def as_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


class LocalMcpServer:
    """Stateful single-connection MCP dispatcher.

    The transport is intentionally separate: tests call :meth:`handle` directly
    while ``stdio.py`` feeds one UTF-8 JSON-RPC object per line, matching the
    official Python SDK stdio framing.
    """

    def __init__(self, runtime: McpToolRuntime, *, principal: McpPrincipal) -> None:
        self._runtime = runtime
        self._principal = principal
        self._initialized = False
        self._negotiated_protocol: str | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def handle_json_line(self, line: str) -> str | None:
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeError):
            return self._encode_error(None, JsonRpcError(-32700, "Parse error"))
        response = self.handle(message)
        if response is None:
            return None
        return json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def handle(self, message: Any) -> dict[str, Any] | None:
        request_id: str | int | None = None
        try:
            if not isinstance(message, Mapping):
                raise JsonRpcError(-32600, "Invalid Request")
            if message.get("jsonrpc") != "2.0":
                raise JsonRpcError(-32600, "Invalid Request", {"reason": "jsonrpc must be 2.0"})
            method = message.get("method")
            if not isinstance(method, str) or not method:
                raise JsonRpcError(-32600, "Invalid Request", {"reason": "method is required"})
            if "id" in message:
                raw_id = message["id"]
                if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
                    raise JsonRpcError(-32600, "Invalid Request", {"reason": "invalid request id"})
                request_id = raw_id
            params = message.get("params", {})
            if not isinstance(params, Mapping):
                raise JsonRpcError(-32602, "Invalid params", {"reason": "params must be an object"})
            if method in _REQUEST_METHODS and request_id is None:
                # MCP tool operations are request/response interactions.  Never
                # execute a side-effecting call disguised as a JSON-RPC notification.
                return None

            if method == "initialize":
                result = self._initialize(params)
            elif method == "notifications/initialized":
                self._require_notification(request_id)
                if self._negotiated_protocol is None:
                    raise JsonRpcError(-32002, "Server not initialized")
                self._initialized = True
                return None
            elif method in {"notifications/cancelled", "notifications/progress"}:
                self._require_notification(request_id)
                return None
            elif method == "ping":
                self._require_initialized()
                result = {}
            elif method == "tools/list":
                self._require_initialized()
                cursor = params.get("cursor")
                if cursor not in (None, ""):
                    raise JsonRpcError(-32602, "Invalid params", {"reason": "pagination cursor is not supported"})
                result = {"tools": list(self._runtime.list_tools())}
            elif method == "tools/call":
                self._require_initialized()
                name = params.get("name")
                arguments = params.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise JsonRpcError(-32602, "Invalid params", {"reason": "tool name is required"})
                if not isinstance(arguments, Mapping):
                    raise JsonRpcError(-32602, "Invalid params", {"reason": "arguments must be an object"})
                try:
                    result = self._runtime.call_tool(
                        name,
                        arguments,
                        principal=self._principal,
                    ).to_protocol_result()
                except KeyError as exc:
                    raise JsonRpcError(-32602, "Invalid params", {"reason": str(exc)}) from exc
            else:
                if request_id is None:
                    return None
                raise JsonRpcError(-32601, "Method not found")

            if request_id is None:
                return None
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except JsonRpcError as exc:
            if request_id is None and isinstance(message, Mapping) and "id" not in message:
                return None
            return self._error_response(request_id, exc)
        except Exception as exc:  # never expose exception text across the protocol
            return self._error_response(
                request_id,
                JsonRpcError(-32603, "Internal error", {"exception": type(exc).__name__}),
            )

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise JsonRpcError(-32602, "Invalid params", {"reason": "protocolVersion is required"})
        negotiated = (
            protocol_version
            if protocol_version in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        self._negotiated_protocol = negotiated
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": f"puncture-rd-{self._runtime.server_name}",
                "version": __version__,
            },
            "instructions": (
                "Use opaque artifact IDs only. Deterministic tools own geometry, "
                "segmentation, planning and safety conclusions."
            ),
        }

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise JsonRpcError(-32002, "Server not initialized")

    @staticmethod
    def _require_notification(request_id: str | int | None) -> None:
        if request_id is not None:
            raise JsonRpcError(-32600, "Invalid Request", {"reason": "notification must not include id"})

    @staticmethod
    def _error_response(request_id: str | int | None, error: JsonRpcError) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": error.as_object()}

    @classmethod
    def _encode_error(cls, request_id: str | int | None, error: JsonRpcError) -> str:
        return json.dumps(
            cls._error_response(request_id, error),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
