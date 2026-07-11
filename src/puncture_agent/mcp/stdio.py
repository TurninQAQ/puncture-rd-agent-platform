"""UTF-8 newline-delimited stdio transport for the local MCP dispatcher."""

from __future__ import annotations

import sys
from typing import TextIO

from .server import LocalMcpServer


def serve_stdio(
    server: LocalMcpServer,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    """Serve one JSON-RPC message per line without writing logs to stdout."""

    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    for line in input_stream:
        if not line.strip():
            continue
        response = server.handle_json_line(line)
        if response is None:
            continue
        output_stream.write(response + "\n")
        output_stream.flush()


__all__ = ["serve_stdio"]
