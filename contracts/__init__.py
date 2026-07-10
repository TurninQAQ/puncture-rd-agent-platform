"""Stable, standard-library-only contracts shared by all project modules.

The objects exported here are the integration boundary.  Implementations may
change internally, but field names, enum values, and response semantics must
not be changed without a contract-version migration.
"""

from .artifacts import ArtifactPublicView, ArtifactRef
from .common import MetricValue, ToolCallContext, ToolResponseEnvelope, to_json, to_primitive
from .domain import (
    CandidatePath,
    DangerMaskSpec,
    LabelDefinition,
    LabelQualityThreshold,
    RiskFlag,
    SafetyMargin,
    ValidationIssue,
)
from .enums import *  # noqa: F401,F403 - convenience export for contract consumers
from .errors import ErrorDetail
from .geometry import VolumeGeometry, VoxelPoint, WorldPoint

__all__ = [
    "ArtifactRef",
    "ArtifactPublicView",
    "CandidatePath",
    "DangerMaskSpec",
    "ErrorDetail",
    "LabelDefinition",
    "LabelQualityThreshold",
    "MetricValue",
    "RiskFlag",
    "SafetyMargin",
    "ToolCallContext",
    "ToolResponseEnvelope",
    "ValidationIssue",
    "VolumeGeometry",
    "VoxelPoint",
    "WorldPoint",
    "to_json",
    "to_primitive",
]
