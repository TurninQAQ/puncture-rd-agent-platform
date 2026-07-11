"""Optional adapter for the stable v1 official MCP Python SDK.

The project remains runnable without third-party packages.  Installing the
``implementation`` extra activates this adapter for official stdio or
Streamable HTTP transports while reusing the same dependency-free runtime.
"""

from __future__ import annotations

import asyncio
from typing import Any

from puncture_agent import __version__

from .runtime import McpPrincipal, McpToolRuntime


class OfficialMcpSdkUnavailable(RuntimeError):
    pass


def build_official_sdk_server(runtime: McpToolRuntime, *, principal: McpPrincipal) -> Any:
    try:
        import mcp.types as types
        from mcp.server.lowlevel import Server
    except ModuleNotFoundError as exc:
        raise OfficialMcpSdkUnavailable(
            'install the implementation extra with stable "mcp>=1.27,<2"'
        ) from exc

    server = Server(f"puncture-rd-{runtime.server_name}")

    @server.list_tools()
    async def list_tools() -> list[Any]:
        result = []
        for descriptor in runtime.list_tools():
            annotations = descriptor["annotations"]
            result.append(
                types.Tool(
                    name=descriptor["name"],
                    title=descriptor["title"],
                    description=descriptor["description"],
                    inputSchema=descriptor["inputSchema"],
                    outputSchema=descriptor["outputSchema"],
                    annotations=types.ToolAnnotations(**annotations),
                    execution=types.ToolExecution(**descriptor["execution"]),
                    _meta=descriptor["_meta"],
                )
            )
        return result

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        try:
            result = runtime.call_tool(name, arguments, principal=principal)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return types.CallToolResult(
            content=[types.TextContent(**item) for item in result.content],
            structuredContent=result.structured_content,
            isError=result.is_error,
            _meta=result.meta,
        )

    return server


async def run_official_stdio(
    runtime: McpToolRuntime,
    *,
    principal: McpPrincipal,
) -> None:
    try:
        import mcp.server.stdio
        from mcp.server.lowlevel import NotificationOptions
        from mcp.server.models import InitializationOptions
    except ModuleNotFoundError as exc:
        raise OfficialMcpSdkUnavailable(
            'install the implementation extra with stable "mcp>=1.27,<2"'
        ) from exc

    server = build_official_sdk_server(runtime, principal=principal)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=f"puncture-rd-{runtime.server_name}",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_official_stdio_sync(
    runtime: McpToolRuntime,
    *,
    principal: McpPrincipal,
) -> None:
    asyncio.run(run_official_stdio(runtime, principal=principal))


__all__ = [
    "OfficialMcpSdkUnavailable",
    "build_official_sdk_server",
    "run_official_stdio",
    "run_official_stdio_sync",
]
