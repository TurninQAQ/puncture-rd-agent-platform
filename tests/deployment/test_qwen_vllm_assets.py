"""Expose the colocated Qwen/vLLM deployment tests to `run_tests.py`."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
ASSET_TEST = PROJECT_DIR / "deploy" / "qwen-vllm" / "tests" / "test_deployment_assets.py"
SCRIPTS_DIR = ASSET_TEST.parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

specification = importlib.util.spec_from_file_location("qwen_vllm_asset_tests", ASSET_TEST)
if specification is None or specification.loader is None:
    raise RuntimeError(f"cannot load deployment test module: {ASSET_TEST}")
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)

DeploymentAssetTests = module.DeploymentAssetTests

