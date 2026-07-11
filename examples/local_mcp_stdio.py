#!/usr/bin/env python3
"""Start one dependency-free demo MCP server over stdio."""

from __future__ import annotations

import argparse
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.local_mcp_demo import (  # noqa: E402
    CALLER,
    CASE_ID,
    _build_requests,
    _collect_artifacts,
)
from puncture_agent.mcp import (  # noqa: E402
    InMemoryArtifactResolver,
    LocalMcpServer,
    McpPrincipal,
    McpToolRuntime,
    serve_stdio,
)
from puncture_agent.tooling import build_adapter_registry  # noqa: E402


def build_server(server_name: str) -> LocalMcpServer:
    requests, case_backend = _build_requests()
    bundle = build_adapter_registry(case_data_backend=case_backend)
    artifacts = {}
    for request in requests.values():
        for artifact in _collect_artifacts(request):
            artifacts[artifact.artifact_id] = artifact
    runtime = McpToolRuntime(
        bundle.registry,
        InMemoryArtifactResolver(tuple(artifacts.values())),
        server_name=server_name,
    )
    return LocalMcpServer(
        runtime,
        principal=McpPrincipal(CALLER, (CASE_ID,)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        choices=("case-data", "segmentation", "planning-safety"),
        required=True,
    )
    args = parser.parse_args()
    serve_stdio(build_server(args.server))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
