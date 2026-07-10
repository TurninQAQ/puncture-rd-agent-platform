"""References to large files that must never be embedded in LLM context."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import ArtifactStatus, ArtifactType
from .geometry import VolumeGeometry


@dataclass(frozen=True, slots=True)
class ArtifactPublicView:
    """API-safe artifact projection with no storage URI or internal metadata."""

    artifact_id: str
    case_id: str
    artifact_type: ArtifactType
    status: ArtifactStatus
    producer_name: str
    producer_version: str
    geometry_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    case_id: str
    artifact_type: ArtifactType
    uri: str
    checksum_sha256: str
    status: ArtifactStatus
    geometry: VolumeGeometry | None = None
    producer_name: str = "unknown"
    producer_version: str = "0"
    parent_artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.case_id:
            raise ValueError("artifact_id and case_id are required")
        if not self.uri:
            raise ValueError("artifact uri is required")
        if len(self.checksum_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.checksum_sha256.lower()
        ):
            raise ValueError("checksum_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "checksum_sha256", self.checksum_sha256.lower())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def require_available(self) -> None:
        if self.status is not ArtifactStatus.AVAILABLE:
            raise ValueError(f"artifact {self.artifact_id} is not AVAILABLE")

    def to_public_view(self) -> ArtifactPublicView:
        """Return the only artifact representation allowed in external APIs."""

        return ArtifactPublicView(
            artifact_id=self.artifact_id,
            case_id=self.case_id,
            artifact_type=self.artifact_type,
            status=self.status,
            producer_name=self.producer_name,
            producer_version=self.producer_version,
            geometry_fingerprint=(
                self.geometry.geometry_fingerprint if self.geometry is not None else None
            ),
        )
