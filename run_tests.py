"""Run all standard-library unittest suites without third-party dependencies."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _github_command_value(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _emit_github_failure_annotations(result: unittest.TestResult) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for outcome, failures in (("failure", result.failures), ("error", result.errors)):
        for test, _ in failures:
            test_id = test.id() if callable(getattr(test, "id", None)) else str(test)
            title = _github_command_value(f"unittest {outcome}")
            message = _github_command_value(test_id)
            print(f"::error title={title}::{message}", flush=True)
    for test in result.unexpectedSuccesses:
        test_id = test.id() if callable(getattr(test, "id", None)) else str(test)
        print(
            "::error title=unittest unexpected success::"
            + _github_command_value(test_id),
            flush=True,
        )


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    _emit_github_failure_annotations(result)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
