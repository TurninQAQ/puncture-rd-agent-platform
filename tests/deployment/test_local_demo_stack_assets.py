"""Offline safety and syntax checks for the opt-in local full-stack demo."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import py_compile
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "local-demo"
SERVER = ROOT / "examples" / "live_api_server.py"
CLIENT = ROOT / "examples" / "live_api_demo.py"
DOCTOR = DEPLOY / "doctor.py"


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


server = _load("live_api_server_asset_test", SERVER)
client = _load("live_api_demo_asset_test", CLIENT)
doctor = _load("local_demo_doctor_asset_test", DOCTOR)


class LocalDemoStackAssetTests(unittest.TestCase):
    def test_assets_compile_and_shell_scripts_parse(self) -> None:
        self.assertTrue((DEPLOY / "README.md").is_file())
        self.assertTrue(
            (DEPLOY / "evidence" / "local-full-stack-validation.md").is_file()
        )
        for path in (SERVER, CLIENT, DOCTOR):
            py_compile.compile(str(path), doraise=True)
        for name in (
            "common.sh",
            "doctor.sh",
            "serve.sh",
            "verify.sh",
            "run_demo.sh",
        ):
            path = DEPLOY / name
            self.assertTrue(path.is_file())
            result = subprocess.run(
                ["bash", "-n", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_example_configuration_is_closed_and_secret_free(self) -> None:
        values = {}
        for line in (DEPLOY / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual("0", values["RUN_FULL_STACK_DEMO"])
        self.assertEqual("127.0.0.1", values["PUNCTURE_DEMO_HOST"])
        self.assertIn("REPLACE_ME", values["PUNCTURE_API_POSTGRES_DSN"])
        self.assertIn("/absolute/path/", values["OPENSEARCH_PASSWORD_FILE"])
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            marker = directory / "must-not-exist"
            literal = f"$(touch {marker})"
            env_file = directory / ".env"
            env_file.write_text(
                "RUN_FULL_STACK_DEMO=1\n"
                f"PUNCTURE_API_POSTGRES_DSN={literal}\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            environment = os.environ.copy()
            environment.update(
                {
                    "PUNCTURE_DEMO_ENV_FILE": str(env_file),
                    "PUNCTURE_DEMO_PYTHON": sys.executable,
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    '. "$1"; load_local_demo_env; [[ "$PUNCTURE_API_POSTGRES_DSN" == "$2" ]]',
                    "local-demo-test",
                    str(DEPLOY / "common.sh"),
                    literal,
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertFalse(marker.exists())

            env_file.chmod(0o644)
            public = subprocess.run(
                [
                    "bash",
                    "-c",
                    '. "$1"; load_local_demo_env',
                    "local-demo-test",
                    str(DEPLOY / "common.sh"),
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(0, public.returncode)
            self.assertIn("group/world", public.stderr)

    def test_server_requires_explicit_opt_in_before_dependencies_or_network(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                socket.socket,
                "connect",
                autospec=True,
                side_effect=AssertionError("server attempted a network connection"),
            ):
                with self.assertRaisesRegex(ValueError, "RUN_FULL_STACK_DEMO=1"):
                    server.build_local_demo_app()
                report = doctor.run_checks()
                self.assertFalse(report["ready"])
                self.assertEqual(
                    "LOCAL_CONFIGURATION_INVALID",
                    report["components"]["configuration"]["error_code"],
                )

    def test_token_is_private_stable_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name) / "private"
            directory.mkdir(mode=0o700)
            token_file = directory / "bearer-token"
            first = server.load_or_create_demo_token(token_file)
            second = server.load_or_create_demo_token(token_file)
            self.assertEqual(first, second)
            self.assertRegex(first, r"^[A-Za-z0-9_-]{32,256}$")
            self.assertEqual(0o600, stat.S_IMODE(token_file.stat().st_mode))

            symlink = directory / "token-link"
            symlink.symlink_to(token_file)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                server.load_or_create_demo_token(symlink)

            hardlink = directory / "token-hardlink"
            os.link(token_file, hardlink)
            with self.assertRaisesRegex(ValueError, "one hard link"):
                server.load_or_create_demo_token(token_file)

    def test_client_refuses_nonloopback_and_credential_urls(self) -> None:
        for unsafe in (
            "http://0.0.0.0:8010",
            "http://example.com:8010",
            "http://user:password@127.0.0.1:8010",
            "https://127.0.0.1:8010",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    client.validate_base_url(unsafe)
        self.assertEqual(
            "http://127.0.0.1:8010",
            client.validate_base_url("http://127.0.0.1:8010"),
        )

    def test_documentation_keeps_the_one_command_and_synthetic_boundary_explicit(self) -> None:
        text = (DEPLOY / "README.md").read_text(encoding="utf-8")
        for expected in (
            "./deploy/local-demo/doctor.sh",
            "./deploy/local-demo/run_demo.sh",
            "deterministic synthetic tools",
            "worker_enabled=false",
            "Why health is DEGRADED",
            "Never point the demo at a shared production schema",
        ):
            self.assertIn(expected, text)
        self.assertIn(
            "deploy/local-demo/doctor.py --quiet",
            (DEPLOY / "run_demo.sh").read_text(encoding="utf-8"),
        )
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in (
            "beginner-hands-on-tutorial.md",
            "learning-and-interview-guide.md",
            "project-status.md",
        ):
            self.assertTrue((ROOT / "docs" / name).is_file())
            self.assertIn(f"docs/{name}", root_readme)


if __name__ == "__main__":
    unittest.main()
