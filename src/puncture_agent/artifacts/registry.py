"""Standard-library reference implementation of the Artifact Registry.

The registry stores no volume bytes.  It proves lifecycle, idempotency,
authorization, URI redaction, and lineage before a database/object-store
implementation is introduced.  Its lifecycle uses the canonical
``ArtifactStatus`` enum and does not define a second state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Mapping
from uuid import uuid4

from contracts.artifacts import ArtifactPublicView, ArtifactRef
from contracts.enums import ArtifactStatus, ArtifactType
from contracts.geometry import VolumeGeometry


class ArtifactRegistryError(RuntimeError):
    """Stable registry error consumed by API/runtime adapters."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    roles: tuple[str, ...] = ()
    allowed_case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("principal subject is required")

    def may_resolve_uri(self, case_id: str) -> bool:
        role_allowed = "system" in self.roles or "artifact_uri_reader" in self.roles
        case_allowed = "system" in self.roles or case_id in self.allowed_case_ids
        return role_allowed and case_allowed


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    artifact_id: str
    parent_artifact_ids: tuple[str, ...]
    child_artifact_ids: tuple[str, ...]


@dataclass(slots=True)
class _ArtifactRecord:
    artifact_id: str
    case_id: str
    artifact_type: ArtifactType
    status: ArtifactStatus
    internal_uri: str
    created_by: str
    idempotency_key: str
    producer_name: str
    producer_version: str
    parent_artifact_ids: tuple[str, ...]
    geometry: VolumeGeometry | None
    metadata: dict[str, str] = field(default_factory=dict)
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    failure_reason: str | None = None

    def to_public_view(self) -> ArtifactPublicView:
        return ArtifactPublicView(
            artifact_id=self.artifact_id,
            case_id=self.case_id,
            artifact_type=self.artifact_type,
            status=self.status,
            producer_name=self.producer_name,
            producer_version=self.producer_version,
            geometry_fingerprint=(self.geometry.geometry_fingerprint if self.geometry else None),
        )


class InMemoryArtifactRegistry:
    """Thread-safe deterministic registry double.

    Status meaning follows the canonical contract:

    - PENDING: registered but not finalized;
    - AVAILABLE: checksum-validated and consumable;
    - INVALID: generation/validation failed or was invalidated;
    - MISSING: metadata exists but the underlying object is inaccessible.
    """

    def __init__(self) -> None:
        self._records: dict[str, _ArtifactRecord] = {}
        self._artifact_by_idempotency_scope: dict[tuple[str, str], str] = {}
        self._children: dict[str, set[str]] = {}
        self._lock = RLock()

    @property
    def coordination_key(self) -> str:
        """Stable only for this in-process registry instance."""

        return f"memory-registry:{id(self)}"

    def begin_registration(
        self,
        *,
        case_id: str,
        artifact_type: ArtifactType,
        internal_uri: str,
        created_by: str,
        idempotency_key: str,
        producer_name: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...] = (),
        geometry: VolumeGeometry | None = None,
        metadata: Mapping[str, str] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactPublicView:
        required = (case_id, internal_uri, created_by, idempotency_key, producer_name, producer_version)
        if any(not value.strip() for value in required):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "required registration field is empty")
        if len(parent_artifact_ids) != len(set(parent_artifact_ids)):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "parent artifact IDs must be unique")
        if not isinstance(artifact_type, ArtifactType):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "artifact_type must be canonical ArtifactType")
        metadata_value = dict(metadata or {})
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata_value.items()
        ):
            raise ArtifactRegistryError("INVALID_ARGUMENT", "metadata keys and values must be strings")

        with self._lock:
            scope = (case_id, idempotency_key)
            existing_id = self._artifact_by_idempotency_scope.get(scope)
            if existing_id is not None:
                existing = self._records[existing_id]
                if not self._registration_matches(
                    existing,
                    artifact_type=artifact_type,
                    producer_name=producer_name,
                    producer_version=producer_version,
                    parent_artifact_ids=parent_artifact_ids,
                    geometry=geometry,
                    metadata=metadata_value,
                ):
                    raise ArtifactRegistryError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency key was already used with different registration inputs",
                    )
                if existing.status in {ArtifactStatus.PENDING, ArtifactStatus.AVAILABLE}:
                    return existing.to_public_view()
                self._artifact_by_idempotency_scope.pop(scope, None)

            new_id = artifact_id or f"art-{uuid4().hex}"
            if new_id in self._records:
                raise ArtifactRegistryError("CONFLICT", f"artifact {new_id} already exists")
            if new_id in parent_artifact_ids:
                raise ArtifactRegistryError("LINEAGE_CYCLE", "artifact cannot be its own parent")

            for parent_id in parent_artifact_ids:
                parent = self._records.get(parent_id)
                if parent is None:
                    raise ArtifactRegistryError("PARENT_NOT_FOUND", f"parent {parent_id} does not exist")
                if parent.status is not ArtifactStatus.AVAILABLE:
                    raise ArtifactRegistryError("PARENT_NOT_AVAILABLE", f"parent {parent_id} is not AVAILABLE")
                if parent.case_id != case_id:
                    raise ArtifactRegistryError("CASE_MISMATCH", "parent belongs to a different case")

            record = _ArtifactRecord(
                artifact_id=new_id,
                case_id=case_id,
                artifact_type=artifact_type,
                status=ArtifactStatus.PENDING,
                internal_uri=internal_uri,
                created_by=created_by,
                idempotency_key=idempotency_key,
                producer_name=producer_name,
                producer_version=producer_version,
                parent_artifact_ids=tuple(sorted(parent_artifact_ids)),
                geometry=geometry,
                metadata=metadata_value,
            )
            self._records[new_id] = record
            self._artifact_by_idempotency_scope[scope] = new_id
            for parent_id in parent_artifact_ids:
                self._children.setdefault(parent_id, set()).add(new_id)
            self._children.setdefault(new_id, set())
            return record.to_public_view()

    def finalize(self, artifact_id: str, checksum_sha256: str, size_bytes: int) -> ArtifactRef:
        checksum = checksum_sha256.lower()
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ArtifactRegistryError("INVALID_CHECKSUM", "checksum must be a SHA-256 hex digest")
        if size_bytes < 0:
            raise ArtifactRegistryError("INVALID_ARGUMENT", "size_bytes must be non-negative")

        with self._lock:
            record = self._require_record(artifact_id)
            if record.status is ArtifactStatus.AVAILABLE:
                if record.checksum_sha256 == checksum and record.size_bytes == size_bytes:
                    return self._to_ref(record)
                raise ArtifactRegistryError("CONFLICT", "AVAILABLE artifact cannot be overwritten")
            if record.status is not ArtifactStatus.PENDING:
                raise ArtifactRegistryError("INVALID_STATE", f"cannot finalize {record.status.value} artifact")

            scope = (record.case_id, record.idempotency_key)
            existing_id = self._artifact_by_idempotency_scope.get(scope)
            if existing_id is not None and existing_id != artifact_id:
                existing = self._records[existing_id]
                if existing.checksum_sha256 != checksum:
                    record.status = ArtifactStatus.INVALID
                    record.failure_reason = "IDEMPOTENCY_CHECKSUM_CONFLICT"
                    raise ArtifactRegistryError("IDEMPOTENCY_CONFLICT", "idempotency key maps to different content")
                record.status = ArtifactStatus.INVALID
                record.failure_reason = "DUPLICATE_OUTPUT"
                return self._to_ref(existing)

            record.status = ArtifactStatus.AVAILABLE
            record.checksum_sha256 = checksum
            record.size_bytes = size_bytes
            self._artifact_by_idempotency_scope[scope] = artifact_id
            return self._to_ref(record)

    def fail(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        """Mark generation/validation failure as canonical INVALID."""

        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "failure reason is required")
        with self._lock:
            record = self._require_record(artifact_id)
            if record.status is not ArtifactStatus.PENDING:
                raise ArtifactRegistryError("INVALID_STATE", "only PENDING artifacts can fail")
            record.status = ArtifactStatus.INVALID
            record.failure_reason = reason
            self._release_idempotency_claim(record)
            return record.to_public_view()

    def invalidate(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "invalidation reason is required")
        with self._lock:
            record = self._require_record(artifact_id)
            if record.status is not ArtifactStatus.AVAILABLE:
                raise ArtifactRegistryError("INVALID_STATE", "only AVAILABLE artifacts can be invalidated")
            record.status = ArtifactStatus.INVALID
            record.failure_reason = reason
            self._release_idempotency_claim(record)
            return record.to_public_view()

    def mark_missing(self, artifact_id: str, reason: str) -> ArtifactPublicView:
        """Record that metadata exists but its underlying object is unavailable."""

        if not reason.strip():
            raise ArtifactRegistryError("INVALID_ARGUMENT", "missing reason is required")
        with self._lock:
            record = self._require_record(artifact_id)
            if record.status is not ArtifactStatus.AVAILABLE:
                raise ArtifactRegistryError("INVALID_STATE", "only AVAILABLE artifacts can become MISSING")
            record.status = ArtifactStatus.MISSING
            record.failure_reason = reason
            self._release_idempotency_claim(record)
            return record.to_public_view()

    def get_metadata(self, artifact_id: str) -> ArtifactPublicView:
        with self._lock:
            return self._require_record(artifact_id).to_public_view()

    def resolve_uri(self, artifact_id: str, principal: Principal) -> str:
        with self._lock:
            record = self._require_record(artifact_id)
            if record.status is not ArtifactStatus.AVAILABLE:
                raise ArtifactRegistryError("ARTIFACT_NOT_AVAILABLE", "artifact is not AVAILABLE")
            if not principal.may_resolve_uri(record.case_id):
                raise ArtifactRegistryError("PERMISSION_DENIED", "principal cannot resolve artifact URI")
            return record.internal_uri

    def find_available_by_idempotency_key(
        self,
        key: str,
        *,
        case_id: str | None = None,
    ) -> ArtifactRef | None:
        with self._lock:
            if case_id is not None:
                artifact_ids = [self._artifact_by_idempotency_scope.get((case_id, key))]
            else:
                artifact_ids = [
                    artifact_id
                    for (scoped_case_id, scoped_key), artifact_id
                    in self._artifact_by_idempotency_scope.items()
                    if scoped_key == key
                ]
            artifact_ids = [artifact_id for artifact_id in artifact_ids if artifact_id is not None]
            if not artifact_ids:
                return None
            if len(set(artifact_ids)) > 1:
                raise ArtifactRegistryError(
                    "AMBIGUOUS_IDEMPOTENCY_KEY",
                    "case_id is required when an idempotency key exists in multiple cases",
                )
            record = self._records.get(artifact_ids[0])
            if record is None or record.status is not ArtifactStatus.AVAILABLE:
                return None
            return self._to_ref(record)

    # Backward-compatible name used by the written specification.
    def find_ready_by_idempotency_key(
        self,
        key: str,
        *,
        case_id: str | None = None,
    ) -> ArtifactRef | None:
        return self.find_available_by_idempotency_key(key, case_id=case_id)

    def get_lineage(self, artifact_id: str) -> ArtifactLineage:
        with self._lock:
            record = self._require_record(artifact_id)
            return ArtifactLineage(
                artifact_id=artifact_id,
                parent_artifact_ids=record.parent_artifact_ids,
                child_artifact_ids=tuple(sorted(self._children.get(artifact_id, set()))),
            )

    def _require_record(self, artifact_id: str) -> _ArtifactRecord:
        record = self._records.get(artifact_id)
        if record is None:
            raise ArtifactRegistryError("NOT_FOUND", f"artifact {artifact_id} was not found")
        return record

    def _release_idempotency_claim(self, record: _ArtifactRecord) -> None:
        scope = (record.case_id, record.idempotency_key)
        if self._artifact_by_idempotency_scope.get(scope) == record.artifact_id:
            self._artifact_by_idempotency_scope.pop(scope, None)

    @staticmethod
    def _registration_matches(
        record: _ArtifactRecord,
        *,
        artifact_type: ArtifactType,
        producer_name: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...],
        geometry: VolumeGeometry | None,
        metadata: Mapping[str, str] | None,
    ) -> bool:
        geometry_fingerprint = geometry.geometry_fingerprint if geometry else None
        record_fingerprint = record.geometry.geometry_fingerprint if record.geometry else None
        return (
            record.artifact_type is artifact_type
            and record.producer_name == producer_name
            and record.producer_version == producer_version
            and tuple(sorted(record.parent_artifact_ids)) == tuple(sorted(parent_artifact_ids))
            and record_fingerprint == geometry_fingerprint
            and record.metadata == dict(metadata or {})
        )

    @staticmethod
    def _to_ref(record: _ArtifactRecord) -> ArtifactRef:
        if record.status is not ArtifactStatus.AVAILABLE or record.checksum_sha256 is None:
            raise ArtifactRegistryError("ARTIFACT_NOT_AVAILABLE", "artifact is not finalized")
        return ArtifactRef(
            artifact_id=record.artifact_id,
            case_id=record.case_id,
            artifact_type=record.artifact_type,
            uri=record.internal_uri,
            checksum_sha256=record.checksum_sha256,
            status=record.status,
            geometry=record.geometry,
            producer_name=record.producer_name,
            producer_version=record.producer_version,
            parent_artifact_ids=record.parent_artifact_ids,
            metadata=dict(record.metadata),
        )
