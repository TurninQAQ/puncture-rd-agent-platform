from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import py_compile
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEPLOY_DIR = pathlib.Path(__file__).resolve().parents[1]
PROJECT_DIR = DEPLOY_DIR.parents[1]
SCRIPTS_DIR = DEPLOY_DIR / "scripts"
RUNBOOK = PROJECT_DIR / "docs" / "qwen-deployment-runbook.md"


class FakeVllmHandler(BaseHTTPRequestHandler):
    served_model = "qwen-enterprise-agent"

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _send_json(self, value: dict[str, Any], content_type: str = "application/json") -> None:
        encoded = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": self.served_model}]})
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/redirected")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/error":
            encoded = b"provider-secret-error-body"
            self.send_response(500)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if self.path == "/large":
            encoded = b"x" * 256
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if payload.get("stream"):
            events = [
                {"id": "stream-1", "choices": [{"delta": {"content": "contract "}}]},
                {"id": "stream-1", "choices": [{"delta": {"content": "ready"}}]},
                {
                    "id": "stream-1",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": "stream-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            if payload.get("model") != "missing-done":
                body += "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if payload.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup_document",
                            "arguments": json.dumps({"query": "equipment acceptance procedure"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif payload.get("response_format"):
            message = {
                "role": "assistant",
                "content": json.dumps({"intent": "search", "confidence": 0.99}),
            }
            finish_reason = "stop"
        else:
            message = {"role": "assistant", "content": "READY"}
            finish_reason = "stop"
        self._send_json(
            {
                "id": "chatcmpl-test",
                "model": self.served_model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        )


class DeploymentAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def clean_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.pop("VLLM_API_KEY", None)
        environment.pop("VLLM_API_KEY_FILE", None)
        environment.pop("HF_TOKEN", None)
        environment.pop("HF_TOKEN_FILE", None)
        environment["HTTP_PROXY"] = "http://127.0.0.1:9"
        environment["HTTPS_PROXY"] = "http://127.0.0.1:9"
        environment["NO_PROXY"] = ""
        environment["no_proxy"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def test_required_assets_exist(self) -> None:
        required = {
            ".env.example",
            ".gitignore",
            "README.md",
            "compose.yaml",
            "entrypoint.sh",
            "gpu-sizing-worksheet.md",
            "verification-checklist.md",
            "scripts/http_utils.py",
            "scripts/health_check.py",
            "scripts/smoke_test.py",
            "scripts/benchmark.py",
            "scripts/sizing_estimator.py",
        }
        missing = [name for name in required if not (DEPLOY_DIR / name).is_file()]
        self.assertEqual([], missing)
        self.assertTrue(RUNBOOK.is_file())

    def test_example_environment_contains_no_secret_value(self) -> None:
        values: dict[str, str] = {}
        for line in (DEPLOY_DIR / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        for key in ("VLLM_API_KEY", "VLLM_API_KEY_FILE", "HF_TOKEN", "HF_TOKEN_FILE"):
            self.assertIn(key, values)
            self.assertEqual("", values[key])
        self.assertEqual("/dev/null", values["VLLM_API_KEY_FILE_HOST"])
        self.assertEqual("/dev/null", values["HF_TOKEN_FILE_HOST"])
        self.assertEqual("vllm/vllm-openai:v0.25.0", values["VLLM_IMAGE"])
        self.assertEqual("main", values["VLLM_MODEL_REVISION"])

    def test_compose_has_profiles_private_binding_and_required_controls(self) -> None:
        compose = (DEPLOY_DIR / "compose.yaml").read_text(encoding="utf-8")
        for expected in (
            'profiles: ["serve"]',
            'profiles: ["verify"]',
            'profiles: ["benchmark"]',
            "VLLM_MODEL_REVISION",
            "VLLM_TENSOR_PARALLEL_SIZE",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_TOOL_CALL_PARSER",
            "VLLM_STRUCTURED_OUTPUT_FLAG_STYLE",
            "VLLM_API_KEY_FILE",
            "127.0.0.1",
            "no-new-privileges:true",
        ):
            self.assertIn(expected, compose)
        self.assertNotIn("VLLM_API_KEY: ${VLLM_API_KEY", compose)
        self.assertNotIn("HF_TOKEN: ${HF_TOKEN", compose)

    def test_shell_entrypoint_syntax_and_no_eval(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_DIR / "entrypoint.sh")],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        entrypoint = (DEPLOY_DIR / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertNotIn("eval ", entrypoint)
        self.assertIn('exec "${args[@]}"', entrypoint)

    def test_entrypoint_builds_configurable_flags_without_printing_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory)
            capture_file = temporary / "arguments.json"
            fake_vllm = temporary / "vllm"
            fake_vllm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            environment = self.clean_environment()
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture_file),
                    "VLLM_API_KEY": "test-secret-must-not-be-logged",
                    "VLLM_MODEL_ID": "Qwen/test-model",
                    "VLLM_MODEL_REVISION": "immutable-revision",
                    "VLLM_SERVED_MODEL_NAME": "served-test",
                    "VLLM_TOOL_CALL_PARSER": "hermes",
                    "VLLM_REASONING_PARSER": "qwen3",
                    "VLLM_STRUCTURED_OUTPUT_FLAG_STYLE": "config",
                    "VLLM_STRUCTURED_OUTPUT_BACKEND": "auto",
                    "VLLM_STRUCTURED_OUTPUT_ENABLE_IN_REASONING": "true",
                }
            )
            result = subprocess.run(
                ["bash", str(DEPLOY_DIR / "entrypoint.sh")],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertNotIn("test-secret-must-not-be-logged", result.stdout + result.stderr)
            arguments = json.loads(capture_file.read_text(encoding="utf-8"))
            for expected in (
                "serve",
                "Qwen/test-model",
                "--revision",
                "immutable-revision",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "qwen3",
                "--structured-outputs-config.backend",
                "auto",
                "--structured-outputs-config.enable_in_reasoning=True",
                "--api-key",
                "test-secret-must-not-be-logged",
            ):
                self.assertIn(expected, arguments)

    def test_entrypoint_omits_hugging_face_revision_for_local_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory)
            capture_file = temporary / "arguments.json"
            fake_vllm = temporary / "vllm"
            fake_vllm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            environment = self.clean_environment()
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture_file),
                    "VLLM_MODEL_ID": "/models/huggingface/Qwen2.5-3B-Instruct",
                    "VLLM_MODEL_REVISION": "local-snapshot-sha256",
                }
            )

            result = subprocess.run(
                ["bash", str(DEPLOY_DIR / "entrypoint.sh")],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            arguments = json.loads(capture_file.read_text(encoding="utf-8"))
            self.assertIn("/models/huggingface/Qwen2.5-3B-Instruct", arguments)
            self.assertNotIn("--revision", arguments)
            self.assertIn("Using local model path; omitting Hugging Face --revision", result.stderr)
            self.assertIn("identity=local-snapshot-sha256", result.stderr)

    def test_entrypoint_accepts_empty_bootstrap_secret_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory)
            capture_file = temporary / "arguments.json"
            empty_api_key = temporary / "empty-api-key"
            empty_hf_token = temporary / "empty-hf-token"
            empty_api_key.touch(mode=0o600)
            empty_hf_token.touch(mode=0o600)
            fake_vllm = temporary / "vllm"
            fake_vllm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            environment = self.clean_environment()
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture_file),
                    "VLLM_API_KEY_FILE": str(empty_api_key),
                    "HF_TOKEN_FILE": str(empty_hf_token),
                }
            )
            result = subprocess.run(
                ["bash", str(DEPLOY_DIR / "entrypoint.sh")],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            arguments = json.loads(capture_file.read_text(encoding="utf-8"))
            self.assertNotIn("--api-key", arguments)

    def test_entrypoint_accepts_compose_dev_null_secret_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory)
            capture_file = temporary / "arguments.json"
            fake_vllm = temporary / "vllm"
            fake_vllm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            environment = self.clean_environment()
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture_file),
                    "VLLM_API_KEY_FILE": "/dev/null",
                    "HF_TOKEN_FILE": "/dev/null",
                }
            )
            result = subprocess.run(
                ["bash", str(DEPLOY_DIR / "entrypoint.sh")],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            arguments = json.loads(capture_file.read_text(encoding="utf-8"))
            self.assertNotIn("--api-key", arguments)

    def test_python_scripts_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for script in sorted(SCRIPTS_DIR.glob("*.py")):
                target = pathlib.Path(directory) / f"{script.stem}.pyc"
                py_compile.compile(str(script), cfile=str(target), doraise=True)

    def test_health_check_against_fake_openai_server(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "health_check.py"),
                "--base-url",
                self.base_url,
                "--model",
                FakeVllmHandler.served_model,
                "--once",
            ],
            env=self.clean_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("READY", result.stdout)

    def test_health_check_rejects_wrong_served_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "health_check.py"),
                "--base-url",
                self.base_url,
                "--model",
                "wrong-model",
                "--once",
            ],
            env=self.clean_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("configured model is not ready", result.stderr)

    def test_full_smoke_test_against_fake_openai_server(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "smoke_test.py"),
                "--base-url",
                self.base_url,
                "--model",
                FakeVllmHandler.served_model,
            ],
            env=self.clean_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual("PASS", report["status"])
        self.assertEqual("lookup_document", report["tool_call"]["tool"])
        self.assertEqual("search", report["structured_output"]["intent"])
        self.assertEqual("stop", report["streaming_chat"]["finish_reason"])

    def test_streaming_benchmark_against_fake_openai_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "benchmark.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "benchmark.py"),
                    "--base-url",
                    self.base_url,
                    "--model",
                    FakeVllmHandler.served_model,
                    "--requests",
                    "4",
                    "--concurrency",
                    "2",
                    "--stream",
                    "true",
                    "--output",
                    str(output),
                ],
                env=self.clean_environment(),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("PASS", report["status"])
            self.assertEqual(4, report["summary"]["successful"])
            self.assertIsNotNone(report["summary"]["ttft_ms_p95"])
            self.assertEqual(8, report["summary"]["known_completion_tokens"])

    def test_http_helpers_reject_unsafe_urls_redirects_and_error_bodies(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        import http_utils

        for unsafe in (
            "ftp://127.0.0.1/v1",
            "http://user:password@127.0.0.1/v1",
            "http://127.0.0.1/v1?token=secret",
            "http://127.0.0.1/v1#fragment",
            "http://127.0.0.1/v1\nInjected: value",
        ):
            with self.assertRaises(ValueError, msg=unsafe):
                http_utils.normalize_base_url(unsafe)
        root = self.base_url[:-3]
        with self.assertRaises(http_utils.ServiceError) as redirect_error:
            http_utils.request_bytes("GET", f"{root}/redirect")
        self.assertIn("redirect rejected", str(redirect_error.exception))
        with self.assertRaises(http_utils.ServiceError) as provider_error:
            http_utils.request_bytes("GET", f"{root}/error")
        self.assertEqual(500, provider_error.exception.status)
        self.assertNotIn("provider-secret-error-body", str(provider_error.exception))

        original_json_limit = http_utils.MAX_JSON_RESPONSE_BYTES
        original_event_limit = http_utils.MAX_SSE_EVENT_BYTES
        try:
            http_utils.MAX_JSON_RESPONSE_BYTES = 64
            with self.assertRaisesRegex(http_utils.ServiceError, "size limit"):
                http_utils.request_bytes("GET", f"{root}/large")
            with self.assertRaisesRegex(http_utils.ServiceError, r"before \[DONE\]"):
                list(
                    http_utils.iter_sse_json(
                        f"{self.base_url}/chat/completions",
                        {"model": "missing-done", "messages": [], "stream": True},
                    )
                )
            http_utils.MAX_SSE_EVENT_BYTES = 16
            with self.assertRaisesRegex(http_utils.ServiceError, "event exceeds"):
                list(
                    http_utils.iter_sse_json(
                        f"{self.base_url}/chat/completions",
                        {"model": FakeVllmHandler.served_model, "messages": [], "stream": True},
                    )
                )
        finally:
            http_utils.MAX_JSON_RESPONSE_BYTES = original_json_limit
            http_utils.MAX_SSE_EVENT_BYTES = original_event_limit

    def test_benchmark_persists_only_normalized_failure_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "failed-benchmark.json"
            environment = self.clean_environment()
            environment["HTTP_PROXY"] = "http://user:ambient-proxy-secret@127.0.0.1:9"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "benchmark.py"),
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--model",
                    FakeVllmHandler.served_model,
                    "--requests",
                    "1",
                    "--concurrency",
                    "1",
                    "--stream",
                    "false",
                    "--timeout",
                    "0.1",
                    "--output",
                    str(output),
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(1, result.returncode)
            report_text = output.read_text(encoding="utf-8")
            self.assertNotIn("ambient-proxy-secret", report_text)
            report = json.loads(report_text)
            self.assertEqual("ServiceError", report["samples"][0]["error"])

    def test_sizing_estimator_formula_and_invalid_parallelism(self) -> None:
        module_path = SCRIPTS_DIR / "sizing_estimator.py"
        specification = importlib.util.spec_from_file_location("sizing_estimator", module_path)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        arguments = {
            "parameters_billions": 1.0,
            "weight_bits": 16.0,
            "layers": 2,
            "kv_heads": 2,
            "head_dim": 4,
            "context_tokens": 16,
            "concurrent_sequences": 1,
            "kv_bytes": 2.0,
            "tensor_parallel": 1,
            "runtime_overhead_percent": 0.0,
            "activation_reserve_gib": 0.0,
            "gpu_memory_gib": 4.0,
            "gpu_count": 1,
            "gpu_memory_utilization": 0.9,
        }
        report = module.estimate_memory(**arguments)
        self.assertAlmostEqual(1.863, report["estimate"]["raw_weights_gib"], places=3)
        self.assertTrue(report["estimate"]["fits_estimate"])
        arguments.update({"tensor_parallel": 2, "gpu_count": 1})
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            module.estimate_memory(**arguments)

    def test_runbook_pins_sources_and_defines_all_evidence_gates(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")
        for expected in (
            "08dfd68610d2e05a0d8ddc99c23488da6163df3f",
            "e12b91b032daed2afc34d77cca20902cef957b3c",
            "7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e",
            "No claim is made",
            "bootstrap defaults",
            "Tool calling",
            "Structured output",
            "streaming",
            "Security baseline",
            "Verification matrix and evidence",
            "Rollout and rollback",
        ):
            self.assertIn(expected, runbook)


if __name__ == "__main__":
    unittest.main(verbosity=2)
