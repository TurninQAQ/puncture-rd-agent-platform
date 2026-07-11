"""Shared fixtures for dependency-free MCP tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from contracts.artifacts import ArtifactRef
from puncture_agent.mcp import InMemoryArtifactResolver
from tools.helpers import (
    candidate_request,
    conversion_request,
    inspect_request,
    label_validation_request,
    penetration_request,
    risk_request,
    safety_request,
    segmentation_request,
    segmentation_validation_request,
    skin_request,
)


REQUEST_FACTORIES = {
    "inspect_case_metadata": inspect_request,
    "convert_mcs_to_nifti": conversion_request,
    "validate_label_schema": label_validation_request,
    "run_segmentation": segmentation_request,
    "validate_segmentation_result": segmentation_validation_request,
    "extract_skin_surface": skin_request,
    "generate_candidate_paths": candidate_request,
    "evaluate_path_safety": safety_request,
    "evaluate_intraoperative_risk": risk_request,
    "verify_skin_penetration": penetration_request,
}


def collect_artifacts(value: Any) -> tuple[ArtifactRef, ...]:
    found: dict[str, ArtifactRef] = {}

    def visit(item: Any) -> None:
        if isinstance(item, ArtifactRef):
            found[item.artifact_id] = item
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def resolver_for(*requests: Any) -> InMemoryArtifactResolver:
    artifacts: dict[str, ArtifactRef] = {}
    for request in requests:
        for artifact in collect_artifacts(request):
            artifacts[artifact.artifact_id] = artifact
    return InMemoryArtifactResolver(tuple(artifacts[key] for key in sorted(artifacts)))
