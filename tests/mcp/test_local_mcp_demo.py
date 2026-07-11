from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from examples.local_mcp_demo import run_demo  # noqa: E402


class LocalMcpDemoTests(unittest.TestCase):
    def test_demo_executes_all_ten_tools_across_three_servers(self) -> None:
        result = run_demo()
        self.assertEqual(10, len(result["calls"]))
        self.assertEqual(
            {"case-data": 3, "segmentation": 3, "planning-safety": 4},
            {name: value["tool_count"] for name, value in result["servers"].items()},
        )
        self.assertTrue(all(call["status"] == "SUCCESS" for call in result["calls"]))
        self.assertFalse(result["security"]["uri_or_checksum_visible_to_model"])
        self.assertFalse(result["runtime"]["company_algorithms_reimplemented"])

    def test_demo_output_is_json_and_contains_no_private_artifact_location(self) -> None:
        completed = subprocess.run(
            [sys.executable, "examples/local_mcp_demo.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual("local-strongly-typed-mcp-tools", payload["demo"])
        self.assertNotIn("memory://private", completed.stdout)
        self.assertNotIn("checksum_sha256", completed.stdout)

    def test_stdio_process_completes_mcp_handshake_and_tool_discovery(self) -> None:
        messages = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "subprocess-test", "version": "1"},
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
            + "\n"
            + json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            )
            + "\n"
        )
        completed = subprocess.run(
            [sys.executable, "examples/local_mcp_stdio.py", "--server", "case-data"],
            cwd=ROOT,
            input=messages,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(2, len(responses))
        self.assertEqual("2025-11-25", responses[0]["result"]["protocolVersion"])
        self.assertEqual(3, len(responses[1]["result"]["tools"]))
        self.assertEqual("", completed.stderr)


if __name__ == "__main__":
    unittest.main()
