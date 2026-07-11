"""Offline, deterministic verification for the OpenSearch RAG deployment assets."""

from __future__ import annotations

import base64
import copy
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
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_DIR / "deploy" / "rag-search"
SCRIPTS_DIR = DEPLOY_DIR / "scripts"
TEMPLATE_FILE = DEPLOY_DIR / "config" / "index-template.json"
RUNBOOK = PROJECT_DIR / "docs" / "rag-deployment-runbook.md"

http_spec = importlib.util.spec_from_file_location(
    "rag_search_asset_http_utils", SCRIPTS_DIR / "http_utils.py"
)
if http_spec is None or http_spec.loader is None:
    raise RuntimeError("cannot load RAG search HTTP helper")
http_utils = importlib.util.module_from_spec(http_spec)
sys.modules[http_spec.name] = http_utils
http_spec.loader.exec_module(http_utils)


class FakeOpenSearchHandler(BaseHTTPRequestHandler):
    password = "offline-test-password"
    template: dict[str, Any] | None = None
    indices: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    requests: list[tuple[str, str, dict[str, Any] | None]] = []
    search_payloads: list[dict[str, Any]] = []

    def log_message(self, _format: str, *args: Any) -> None:
        return

    @classmethod
    def reset(cls) -> None:
        cls.template = None
        cls.indices = {}
        cls.aliases = {}
        cls.requests = []
        cls.search_payloads = []

    def _authorized(self) -> bool:
        expected = base64.b64encode(f"admin:{self.password}".encode()).decode()
        return self.headers.get("Authorization") == f"Basic {expected}"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise AssertionError("test request body must be an object")
        return value

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _authenticate(self) -> bool:
        if self._authorized():
            return True
        self._send_json({"error": "unauthorized"}, status=401)
        return False

    def do_HEAD(self) -> None:
        if not self._authenticate():
            return
        path = urllib.parse.urlsplit(self.path).path
        index = urllib.parse.unquote(path.lstrip("/"))
        self.requests.append(("HEAD", path, None))
        self._send_json({}, status=200 if index in self.indices else 404)

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/credential-leak")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self._authenticate():
            return
        self.requests.append(("GET", path, None))
        if path == "/_cluster/health":
            self._send_json(
                {
                    "cluster_name": "puncture-rag-search",
                    "status": "yellow",
                    "timed_out": False,
                    "number_of_nodes": 1,
                }
            )
            return
        if path == "/duplicate-json":
            encoded = b'{"value":1,"value":2}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if path.startswith("/_alias/"):
            alias = urllib.parse.unquote(path.removeprefix("/_alias/"))
            index = self.aliases.get(alias)
            if index is None:
                self._send_json({"error": "alias missing"}, status=404)
            else:
                self._send_json({index: {"aliases": {alias: {}}}})
            return
        if path.endswith("/_mapping"):
            index = urllib.parse.unquote(path[1 : -len("/_mapping")])
            mapping = self.indices.get(index)
            if mapping is None:
                self._send_json({"error": "index missing"}, status=404)
            else:
                self._send_json({index: {"mappings": copy.deepcopy(mapping)}})
            return
        if path.endswith("/_count"):
            self._send_json({"count": 7, "_shards": {"failed": 0}})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_PUT(self) -> None:
        if not self._authenticate():
            return
        path = urllib.parse.urlsplit(self.path).path
        payload = self._read_json()
        self.requests.append(("PUT", path, payload))
        if path.startswith("/_index_template/"):
            self.__class__.template = copy.deepcopy(payload)
            self._send_json({"acknowledged": True})
            return
        if path.startswith("/_snapshot/"):
            self._send_json({"snapshot": {"state": "SUCCESS", "failures": []}})
            return
        index = urllib.parse.unquote(path.lstrip("/"))
        if "/_doc/" in path:
            self._send_json({"result": "created"}, status=201)
            return
        if self.template is None:
            self._send_json({"error": "template not installed"}, status=400)
            return
        mapping = copy.deepcopy(self.template["template"]["mappings"])
        self.indices[index] = mapping
        self._send_json({"acknowledged": True, "index": index})

    def do_POST(self) -> None:
        if not self._authenticate():
            return
        path = urllib.parse.urlsplit(self.path).path
        payload = self._read_json()
        self.requests.append(("POST", path, payload))
        if path == "/_aliases":
            actions = payload.get("actions")
            if not isinstance(actions, list):
                self._send_json({"error": "actions missing"}, status=400)
                return
            for action in actions:
                if "remove" in action:
                    body = action["remove"]
                    if self.aliases.get(body["alias"]) == body["index"]:
                        del self.aliases[body["alias"]]
                if "add" in action:
                    body = action["add"]
                    self.aliases[body["alias"]] = body["index"]
            self._send_json({"acknowledged": True})
            return
        if path.endswith("/_search"):
            self.search_payloads.append(copy.deepcopy(payload))
            self._send_json({"took": 1, "hits": {"total": {"value": 0}, "hits": []}})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_DELETE(self) -> None:
        if not self._authenticate():
            return
        path = urllib.parse.urlsplit(self.path).path
        index = urllib.parse.unquote(path.lstrip("/"))
        self.requests.append(("DELETE", path, None))
        existed = self.indices.pop(index, None) is not None
        self._send_json({"acknowledged": True}, status=200 if existed else 404)


class RagSearchDeploymentAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenSearchHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        FakeOpenSearchHandler.reset()

    def _environment(self, password_file: pathlib.Path) -> dict[str, str]:
        environment = os.environ.copy()
        for name in (
            "OPENSEARCH_INITIAL_ADMIN_PASSWORD",
            "OPENSEARCH_PASSWORD",
            "OPENSEARCH_CA_FILE",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "OPENSEARCH_ENDPOINT": self.endpoint,
                "OPENSEARCH_USERNAME": "admin",
                "OPENSEARCH_PASSWORD_FILE": str(password_file),
                "OPENSEARCH_INSECURE": "false",
                "RAG_REQUEST_TIMEOUT_SECONDS": "5",
                "RAG_INDEX_TEMPLATE_NAME": "puncture-rag-template-v1",
                "RAG_INDEX_NAME": "puncture-rag-v000001",
                "RAG_READ_ALIAS": "puncture-rag-read",
                "RAG_WRITE_ALIAS": "puncture-rag-write",
                "RAG_SCHEMA_VERSION": "1",
                "RAG_EMBEDDING_MODEL": "internal-embedding-model",
                "RAG_EMBEDDING_REVISION": "revision-0123456789abcdef",
                "RAG_EMBEDDING_DIMENSION": "1024",
                "RAG_QUERY_INSTRUCTION": "Retrieve internal engineering evidence.",
                "RAG_DOCUMENT_INSTRUCTION": "",
                "RAG_VECTORS_NORMALIZED": "true",
                "RAG_TOKENIZER_REVISION": "tokenizer-revision-0123456789abcdef",
                "RAG_MAX_INPUT_TOKENS": "8192",
                "RAG_PARSER_VERSION": "parser-1",
                "RAG_CHUNKER_VERSION": "chunker-1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _password_file(self, directory: pathlib.Path) -> pathlib.Path:
        password_file = directory / "admin-password"
        password_file.write_text(FakeOpenSearchHandler.password + "\n", encoding="utf-8")
        password_file.chmod(0o600)
        return password_file

    def _run(self, script: str, environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script), *arguments],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )

    def _bootstrap_server_state(self) -> None:
        template = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
        metadata = {
            "schema_version": "1",
            "embedding_model": "internal-embedding-model",
            "embedding_revision": "revision-0123456789abcdef",
            "embedding_dimension": 1024,
            "query_instruction": "Retrieve internal engineering evidence.",
            "document_instruction": "",
            "vectors_normalized": True,
            "tokenizer_revision": "tokenizer-revision-0123456789abcdef",
            "max_input_tokens": 8192,
            "parser_version": "parser-1",
            "chunker_version": "chunker-1",
        }
        template["_meta"].update(metadata)
        template["template"]["mappings"]["_meta"].update(metadata)
        template["template"]["mappings"]["properties"]["embedding"]["dimension"] = 1024
        FakeOpenSearchHandler.template = template
        FakeOpenSearchHandler.indices["puncture-rag-v000001"] = copy.deepcopy(
            template["template"]["mappings"]
        )
        FakeOpenSearchHandler.aliases.update(
            {
                "puncture-rag-read": "puncture-rag-v000001",
                "puncture-rag-write": "puncture-rag-v000001",
            }
        )

    def test_required_assets_exist_and_compile(self) -> None:
        required = {
            ".env.example",
            ".gitignore",
            "README.md",
            "compose.yaml",
            "entrypoint.sh",
            "config/index-template.json",
            "scripts/bootstrap_index.py",
            "scripts/container_healthcheck.sh",
            "scripts/health_check.py",
            "scripts/http_utils.py",
            "scripts/index_contract.py",
            "scripts/integration_test.py",
            "scripts/promote_index.py",
            "scripts/smoke_test.py",
            "scripts/snapshot_index.py",
        }
        self.assertEqual([], sorted(name for name in required if not (DEPLOY_DIR / name).is_file()))
        self.assertTrue(RUNBOOK.is_file())
        for script in (DEPLOY_DIR / "scripts").glob("*.py"):
            py_compile.compile(str(script), doraise=True)
        for script in (DEPLOY_DIR / "entrypoint.sh", SCRIPTS_DIR / "container_healthcheck.sh"):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)

    def test_compose_is_versioned_loopback_secured_and_secret_file_only(self) -> None:
        compose = (DEPLOY_DIR / "compose.yaml").read_text(encoding="utf-8")
        for expected in (
            "opensearchproject/opensearch:3.7.0",
            'profiles: ["serve"]',
            'profiles: ["bootstrap"]',
            'profiles: ["verify"]',
            "127.0.0.1",
            'DISABLE_SECURITY_PLUGIN: "false"',
            "OPENSEARCH_ADMIN_PASSWORD_FILE",
            "RAG_QUERY_INSTRUCTION",
            "RAG_DOCUMENT_INSTRUCTION",
            "RAG_VECTORS_NORMALIZED",
            "RAG_TOKENIZER_REVISION",
            "RAG_MAX_INPUT_TOKENS",
            "no-new-privileges:true",
            "internal: true",
        ):
            self.assertIn(expected, compose)
        self.assertNotIn("opensearch:latest", compose)
        self.assertNotIn("OPENSEARCH_INITIAL_ADMIN_PASSWORD:", compose)
        self.assertNotIn("DISABLE_SECURITY_PLUGIN: \"true\"", compose)

    def test_example_environment_has_no_secret_value_and_requires_release_identity(self) -> None:
        values: dict[str, str] = {}
        for line in (DEPLOY_DIR / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual("opensearchproject/opensearch:3.7.0", values["OPENSEARCH_IMAGE"])
        self.assertEqual("/dev/null", values["OPENSEARCH_ADMIN_PASSWORD_FILE_HOST"])
        self.assertEqual("Qwen/Qwen3-Embedding-0.6B", values["RAG_EMBEDDING_MODEL"])
        self.assertEqual(
            "SET_IMMUTABLE_QWEN3_EMBEDDING_REVISION", values["RAG_EMBEDDING_REVISION"]
        )
        self.assertEqual("1024", values["RAG_EMBEDDING_DIMENSION"])
        self.assertEqual("true", values["RAG_VECTORS_NORMALIZED"])
        self.assertEqual(
            "SET_IMMUTABLE_QWEN3_TOKENIZER_REVISION", values["RAG_TOKENIZER_REVISION"]
        )
        self.assertEqual("8192", values["RAG_MAX_INPUT_TOKENS"])
        self.assertEqual("", values["RAG_DOCUMENT_INSTRUCTION"])
        self.assertNotIn("OPENSEARCH_INITIAL_ADMIN_PASSWORD", values)
        self.assertNotIn("OPENSEARCH_PASSWORD", values)

    def test_index_mapping_has_bm25_vector_acl_version_and_exact_metadata_contract(self) -> None:
        template = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
        settings = template["template"]["settings"]
        mappings = template["template"]["mappings"]
        properties = mappings["properties"]
        self.assertIs(True, settings["index.knn"])
        self.assertEqual("strict", mappings["dynamic"])
        self.assertEqual("text", properties["text"]["type"])
        self.assertEqual("code_preserving", properties["text"]["fields"]["code"]["analyzer"])
        embedding = properties["embedding"]
        self.assertEqual("knn_vector", embedding["type"])
        self.assertEqual(1024, embedding["dimension"])
        self.assertEqual("hnsw", embedding["method"]["name"])
        self.assertEqual("lucene", embedding["method"]["engine"])
        self.assertEqual("cosinesimil", embedding["method"]["space_type"])
        for field in (
            "document_id",
            "chunk_id",
            "parent_id",
            "owner",
            "module",
            "version",
            "status",
            "access_scopes",
            "metadata_terms",
            "embedding_model",
            "embedding_revision",
            "parser_version",
            "chunker_version",
        ):
            self.assertEqual("keyword", properties[field]["type"], field)
        self.assertEqual("flat_object", properties["metadata"]["type"])
        self.assertEqual(1024, mappings["_meta"]["embedding_dimension"])
        self.assertIn("query_instruction", mappings["_meta"])
        self.assertEqual("", mappings["_meta"]["document_instruction"])
        self.assertIs(True, mappings["_meta"]["vectors_normalized"])
        self.assertEqual("SET_DURING_BOOTSTRAP", mappings["_meta"]["tokenizer_revision"])
        self.assertEqual(8192, mappings["_meta"]["max_input_tokens"])

    def test_entrypoint_reads_regular_secret_file_and_does_not_log_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            password_file = self._password_file(directory)
            capture_file = directory / "captured-password"
            fake_entrypoint = directory / "original-entrypoint"
            fake_entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "printf '%s' \"$OPENSEARCH_INITIAL_ADMIN_PASSWORD\" > \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            fake_entrypoint.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "OPENSEARCH_ADMIN_PASSWORD_FILE": str(password_file),
                    "OPENSEARCH_ORIGINAL_ENTRYPOINT": str(fake_entrypoint),
                    "CAPTURE_FILE": str(capture_file),
                }
            )
            environment.pop("OPENSEARCH_INITIAL_ADMIN_PASSWORD", None)
            result = subprocess.run(
                ["bash", str(DEPLOY_DIR / "entrypoint.sh"), "opensearch"],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(FakeOpenSearchHandler.password, capture_file.read_text(encoding="utf-8"))
            self.assertNotIn(FakeOpenSearchHandler.password, result.stdout + result.stderr)

    def test_http_helper_ignores_proxy_rejects_redirect_and_strictly_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            environment = self._environment(password_file)
            previous = os.environ.copy()
            try:
                os.environ.update(environment)
                status, health = http_utils.request_json(
                    "GET",
                    self.endpoint,
                    "/_cluster/health",
                    username="admin",
                    password=FakeOpenSearchHandler.password,
                )
                self.assertEqual(200, status)
                self.assertEqual("yellow", health["status"])
                with self.assertRaises(http_utils.ServiceError):
                    http_utils.request_json(
                        "GET",
                        self.endpoint,
                        "/redirect",
                        username="admin",
                        password=FakeOpenSearchHandler.password,
                    )
                with self.assertRaises(http_utils.ServiceError):
                    http_utils.request_json(
                        "GET",
                        self.endpoint,
                        "/duplicate-json",
                        username="admin",
                        password=FakeOpenSearchHandler.password,
                    )
                with self.assertRaises(ValueError):
                    http_utils.validate_endpoint("http://search.internal:9200")
                with self.assertRaises(ValueError):
                    http_utils.validate_endpoint("https://admin:secret@search.internal:9200")
            finally:
                os.environ.clear()
                os.environ.update(previous)

    def test_bootstrap_renders_contract_creates_index_and_adds_aliases_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run("bootstrap_index.py", self._environment(password_file))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn(FakeOpenSearchHandler.password, result.stdout + result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual("READY", output["status"])
        self.assertEqual("puncture-rag-v000001", output["index"])
        self.assertEqual("puncture-rag-v000001", FakeOpenSearchHandler.aliases["puncture-rag-read"])
        self.assertEqual("puncture-rag-v000001", FakeOpenSearchHandler.aliases["puncture-rag-write"])
        mapping = FakeOpenSearchHandler.indices["puncture-rag-v000001"]
        self.assertEqual("internal-embedding-model", mapping["_meta"]["embedding_model"])
        self.assertEqual(1024, mapping["properties"]["embedding"]["dimension"])
        self.assertEqual(
            "Retrieve internal engineering evidence.", mapping["_meta"]["query_instruction"]
        )
        self.assertEqual("", mapping["_meta"]["document_instruction"])
        self.assertIs(True, mapping["_meta"]["vectors_normalized"])
        self.assertEqual(
            "tokenizer-revision-0123456789abcdef",
            mapping["_meta"]["tokenizer_revision"],
        )
        self.assertEqual(8192, mapping["_meta"]["max_input_tokens"])
        self.assertIn('"max_input_tokens": 8192', result.stdout)
        alias_requests = [body for method, path, body in FakeOpenSearchHandler.requests if method == "POST" and path == "/_aliases"]
        self.assertEqual(1, len(alias_requests))
        self.assertEqual(2, len(alias_requests[0]["actions"]))

    def test_bootstrap_dry_run_rejects_placeholder_and_never_contacts_backend(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "RAG_EMBEDDING_MODEL": "SET_ME",
                "RAG_EMBEDDING_REVISION": "SET_ME",
                "RAG_EMBEDDING_DIMENSION": "1024",
            }
        )
        result = self._run("bootstrap_index.py", environment, "--dry-run")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("explicit immutable release value", result.stderr)
        self.assertEqual([], FakeOpenSearchHandler.requests)

    def test_bootstrap_rejects_invalid_embedding_manifest_controls_offline(self) -> None:
        base = os.environ.copy()
        base.update(
            {
                "RAG_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                "RAG_EMBEDDING_REVISION": "immutable-model-revision",
                "RAG_EMBEDDING_DIMENSION": "1024",
                "RAG_QUERY_INSTRUCTION": "Retrieve internal engineering evidence.",
                "RAG_DOCUMENT_INSTRUCTION": "",
                "RAG_TOKENIZER_REVISION": "immutable-tokenizer-revision",
                "RAG_PARSER_VERSION": "parser-1",
                "RAG_CHUNKER_VERSION": "chunker-1",
            }
        )
        cases = (
            ({"RAG_VECTORS_NORMALIZED": "maybe", "RAG_MAX_INPUT_TOKENS": "8192"}, "true or false"),
            ({"RAG_VECTORS_NORMALIZED": "true", "RAG_MAX_INPUT_TOKENS": "0"}, "positive integer"),
            (
                {
                    "RAG_VECTORS_NORMALIZED": "true",
                    "RAG_MAX_INPUT_TOKENS": "8192",
                    "RAG_TOKENIZER_REVISION": "SET_IMMUTABLE_TOKENIZER_REVISION",
                },
                "explicit immutable release value",
            ),
        )
        for overrides, expected in cases:
            environment = {**base, **overrides}
            result = self._run("bootstrap_index.py", environment, "--dry-run")
            with self.subTest(overrides=overrides):
                self.assertNotEqual(0, result.returncode)
                self.assertIn(expected, result.stderr)
        self.assertEqual([], FakeOpenSearchHandler.requests)

    def test_bootstrap_allows_empty_instructions_and_renders_typed_manifest(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "RAG_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                "RAG_EMBEDDING_REVISION": "immutable-model-revision",
                "RAG_EMBEDDING_DIMENSION": "1024",
                "RAG_QUERY_INSTRUCTION": "",
                "RAG_DOCUMENT_INSTRUCTION": "",
                "RAG_VECTORS_NORMALIZED": "true",
                "RAG_TOKENIZER_REVISION": "immutable-tokenizer-revision",
                "RAG_MAX_INPUT_TOKENS": "8192",
                "RAG_PARSER_VERSION": "parser-1",
                "RAG_CHUNKER_VERSION": "chunker-1",
            }
        )
        result = self._run("bootstrap_index.py", environment, "--dry-run")
        self.assertEqual(0, result.returncode, result.stderr)
        mapping_meta = json.loads(result.stdout)["template"]["mappings"]["_meta"]
        self.assertEqual("", mapping_meta["query_instruction"])
        self.assertEqual("", mapping_meta["document_instruction"])
        self.assertIs(True, mapping_meta["vectors_normalized"])
        self.assertEqual("immutable-tokenizer-revision", mapping_meta["tokenizer_revision"])
        self.assertEqual(8192, mapping_meta["max_input_tokens"])
        self.assertEqual([], FakeOpenSearchHandler.requests)

    def test_bootstrap_refuses_existing_other_alias_before_any_write(self) -> None:
        self._bootstrap_server_state()
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            environment = self._environment(password_file)
            environment["RAG_INDEX_NAME"] = "puncture-rag-v000002"
            result = self._run("bootstrap_index.py", environment)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("refuses to replace an existing live alias", result.stderr)
        writes = [request for request in FakeOpenSearchHandler.requests if request[0] != "GET"]
        self.assertEqual([], writes)
        self.assertEqual("puncture-rag-v000001", FakeOpenSearchHandler.aliases["puncture-rag-read"])

    def test_smoke_sends_acl_status_filters_to_bm25_and_dense_branches(self) -> None:
        self._bootstrap_server_state()
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run("smoke_test.py", self._environment(password_file))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn(FakeOpenSearchHandler.password, result.stdout + result.stderr)
        self.assertEqual(2, len(FakeOpenSearchHandler.search_payloads))
        lexical, dense = FakeOpenSearchHandler.search_payloads
        self.assertIs(False, lexical["_source"])
        self.assertIs(False, dense["_source"])
        expected_filters = [
            {"term": {"doc_kind": "child"}},
            {"term": {"status": "active"}},
            {"terms": {"access_scopes": ["__rag_health_probe__"]}},
        ]
        self.assertEqual(expected_filters, lexical["query"]["bool"]["filter"])
        dense_body = dense["query"]["knn"]["embedding"]
        self.assertEqual(expected_filters, dense_body["filter"]["bool"]["filter"])
        self.assertEqual(1024, len(dense_body["vector"]))

    def test_smoke_rejects_embedding_manifest_mismatch_before_search(self) -> None:
        self._bootstrap_server_state()
        FakeOpenSearchHandler.indices["puncture-rag-v000001"]["_meta"][
            "query_instruction"
        ] = "Different query instruction."
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run("smoke_test.py", self._environment(password_file))
        self.assertNotEqual(0, result.returncode)
        self.assertIn("metadata mismatch: query_instruction", result.stderr)
        self.assertEqual([], FakeOpenSearchHandler.search_payloads)

    def test_promotion_uses_one_compare_guarded_atomic_alias_request(self) -> None:
        self._bootstrap_server_state()
        FakeOpenSearchHandler.indices["puncture-rag-v000002"] = copy.deepcopy(
            FakeOpenSearchHandler.indices["puncture-rag-v000001"]
        )
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run(
                "promote_index.py",
                self._environment(password_file),
                "--new-index",
                "puncture-rag-v000002",
                "--expected-current",
                "puncture-rag-v000001",
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("puncture-rag-v000002", FakeOpenSearchHandler.aliases["puncture-rag-read"])
        self.assertEqual("puncture-rag-v000002", FakeOpenSearchHandler.aliases["puncture-rag-write"])
        alias_requests = [body for method, path, body in FakeOpenSearchHandler.requests if method == "POST" and path == "/_aliases"]
        self.assertEqual(1, len(alias_requests))
        self.assertEqual(4, len(alias_requests[0]["actions"]))
        self.assertIn("--new-index puncture-rag-v000001", result.stdout)

    def test_promotion_fails_closed_when_expected_current_is_stale(self) -> None:
        self._bootstrap_server_state()
        FakeOpenSearchHandler.indices["puncture-rag-v000002"] = copy.deepcopy(
            FakeOpenSearchHandler.indices["puncture-rag-v000001"]
        )
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run(
                "promote_index.py",
                self._environment(password_file),
                "--new-index",
                "puncture-rag-v000002",
                "--expected-current",
                "puncture-rag-v000003",
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("expected-current check failed", result.stderr)
        alias_writes = [
            request
            for request in FakeOpenSearchHandler.requests
            if request[0] == "POST" and request[1] == "/_aliases"
        ]
        self.assertEqual([], alias_writes)
        self.assertEqual("puncture-rag-v000001", FakeOpenSearchHandler.aliases["puncture-rag-read"])

    def test_snapshot_resolves_concrete_index_and_disallows_partial_snapshot(self) -> None:
        self._bootstrap_server_state()
        with tempfile.TemporaryDirectory() as directory_name:
            password_file = self._password_file(pathlib.Path(directory_name))
            result = self._run(
                "snapshot_index.py",
                self._environment(password_file),
                "--repository",
                "approved-rag-repository",
                "--snapshot",
                "puncture-rag-before-v000002-20260710t120000z",
            )
        self.assertEqual(0, result.returncode, result.stderr)
        requests = [
            body
            for method, path, body in FakeOpenSearchHandler.requests
            if method == "PUT" and path.startswith("/_snapshot/")
        ]
        self.assertEqual(1, len(requests))
        self.assertEqual("puncture-rag-v000001", requests[0]["indices"])
        self.assertIs(False, requests[0]["partial"])
        self.assertIs(False, requests[0]["include_global_state"])

    def test_live_integration_index_guard_refuses_production_names_before_connecting(self) -> None:
        for unsafe in (
            "puncture-rag-v000001",
            "puncture-rag-read",
            "puncture-rag-write",
            "puncture-rag-test-*",
            "PUNCTURE-RAG-TEST-001",
        ):
            environment = os.environ.copy()
            environment["RUN_RAG_INTEGRATION"] = "1"
            result = self._run("integration_test.py", environment, "--test-index", unsafe)
            with self.subTest(unsafe=unsafe):
                self.assertNotEqual(0, result.returncode)
        safe_result = self._run(
            "integration_test.py",
            {**os.environ, "RUN_RAG_INTEGRATION": "false"},
            "--test-index",
            "puncture-rag-test-local-001",
        )
        self.assertEqual(0, safe_result.returncode, safe_result.stderr)
        self.assertIn("SKIPPED", safe_result.stdout)
        self.assertEqual([], FakeOpenSearchHandler.requests)

    def test_documentation_is_honest_and_covers_migration_rollback_backup_and_metadata_filters(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        for expected in (
            "**NOT_RUN**",
            "72121f014083f9ca010fd5a7da83b2ec4886027f",
            "23666cc24059637feff11502def16cdd2bf8fe91",
            "metadata_terms",
            "Qwen/Qwen3-Embedding-0.6B",
            "44548aa5f0a0aed1c76d64e19afe47727a325b8f",
            "RAG_QUERY_INSTRUCTION",
            "RAG_DOCUMENT_INSTRUCTION",
            "RAG_VECTORS_NORMALIZED",
            "RAG_TOKENIZER_REVISION",
            "RAG_MAX_INPUT_TOKENS",
            "Atomic migration and rollback",
            "Snapshot, restore, and backup evidence",
            "exactly zero ACL leaks",
            "puncture-rag-test-",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
