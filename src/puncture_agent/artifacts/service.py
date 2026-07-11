"""Atomic single-node coordination between artifact metadata and object bytes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock, RLock
from typing import Any, Protocol
from uuid import uuid4

from contracts.artifacts import ArtifactRef
from contracts.enums import ArtifactStatus, ArtifactType
from contracts.geometry import VolumeGeometry

from .registry import ArtifactRegistryError, ArtifactValidationRecord, Principal
from .store import ArtifactStore, ArtifactStoreError, Payload


_PUBLICATION_LOCK_STRIPES = tuple(Lock() for _ in range(257))


class ArtifactRegistry(Protocol):
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
    ) -> Any: ...

    def finalize(self, artifact_id: str, checksum_sha256: str, size_bytes: int) -> ArtifactRef: ...

    def fail(self, artifact_id: str, reason: str) -> Any: ...

    def get_metadata(self, artifact_id: str) -> Any: ...

    def get_validation_record(self, artifact_id: str) -> ArtifactValidationRecord: ...

    def resolve_uri(self, artifact_id: str, principal: Principal) -> str: ...

    def find_available_by_idempotency_key(
        self,
        key: str,
        *,
        case_id: str | None = None,
    ) -> ArtifactRef | None: ...


class ArtifactPublicationError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ArtifactAccessEvent:
    timestamp: str
    action: str
    artifact_id: str
    principal_subject: str
    allowed: bool
    error_code: str | None = None


class ArtifactAccessAudit(Protocol):
    def record(self, event: ArtifactAccessEvent) -> None: ...


class InMemoryArtifactAccessAudit:
    """Thread-safe audit sink for tests and local single-node deployments."""

    def __init__(self) -> None:
        self._events: list[ArtifactAccessEvent] = []
        self._lock = RLock()

    def record(self, event: ArtifactAccessEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> tuple[ArtifactAccessEvent, ...]:
        with self._lock:
            return tuple(self._events)


class _NullArtifactAccessAudit:
    def record(self, event: ArtifactAccessEvent) -> None:
        return


class ArtifactPublicationService:
    """Publish immutable payloads and metadata as one recoverable workflow.

    The implementation serializes a scoped idempotency key inside this process.
    SQLite plus a local store is therefore intended for a single application
    node. Multi-node publication must use a database-backed lease and remote
    object store while preserving this public behavior.
    """

    URI_PREFIX = "artifact-store:"

    def __init__(
        self,
        registry: ArtifactRegistry,
        store: ArtifactStore,
        *,
        access_audit: ArtifactAccessAudit | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.access_audit = access_audit or _NullArtifactAccessAudit()
        self._registry_coordination_key = str(
            getattr(registry, "coordination_key", f"registry-instance:{id(registry)}")
        )
        self._store_coordination_key = str(
            getattr(store, "coordination_key", f"store-instance:{id(store)}")
        )

    def publish(
        self,
        payload: Payload,
        *,
        case_id: str,
        artifact_type: ArtifactType,
        created_by: str,
        idempotency_key: str,
        producer_name: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...] = (),
        geometry: VolumeGeometry | None = None,
        metadata: Mapping[str, str] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        scope = (case_id, idempotency_key)
        with self._lock_for(scope):
            proposed_id = artifact_id or f"art-{uuid4().hex}"
            proposed_key = self._object_key(case_id, artifact_type, proposed_id)
            try:
                public_view = self.registry.begin_registration(
                    artifact_id=proposed_id,
                    case_id=case_id,
                    artifact_type=artifact_type,
                    internal_uri=self.URI_PREFIX + proposed_key,
                    created_by=created_by,
                    idempotency_key=idempotency_key,
                    producer_name=producer_name,
                    producer_version=producer_version,
                    parent_artifact_ids=parent_artifact_ids,
                    geometry=geometry,
                    metadata=metadata,
                )
            except ArtifactRegistryError as exc:
                raise ArtifactPublicationError(
                    exc.code, str(exc), retryable=exc.retryable
                ) from exc

            if public_view.status is ArtifactStatus.AVAILABLE:
                try:
                    existing = self.registry.find_available_by_idempotency_key(
                        idempotency_key,
                        case_id=case_id,
                    )
                except ArtifactRegistryError as exc:
                    raise ArtifactPublicationError(
                        exc.code, str(exc), retryable=exc.retryable
                    ) from exc
                if existing is None:
                    raise ArtifactPublicationError(
                        "REGISTRY_INCONSISTENT",
                        "AVAILABLE registration cannot be resolved",
                    )
                return existing

            object_key = self._object_key(case_id, artifact_type, public_view.artifact_id)
            try:
                stored = self.store.put(object_key, payload)
                return self.registry.finalize(
                    public_view.artifact_id,
                    stored.checksum_sha256,
                    stored.size_bytes,
                )
            except ArtifactStoreError as exc:
                if not exc.retryable:
                    self._fail_pending(public_view.artifact_id, f"{exc.code}: {exc}")
                raise ArtifactPublicationError(exc.code, str(exc), retryable=exc.retryable) from exc
            except ArtifactRegistryError as exc:
                if not exc.retryable:
                    self._fail_pending(public_view.artifact_id, f"{exc.code}: {exc}")
                raise ArtifactPublicationError(exc.code, str(exc), retryable=exc.retryable) from exc

    def read(self, artifact_id: str, principal: Principal) -> bytes:
        try:
            uri = self.registry.resolve_uri(artifact_id, principal)
            object_key = self._key_from_uri(uri)
            payload = self.store.read(object_key)
        except (ArtifactRegistryError, ArtifactStoreError, ArtifactPublicationError) as exc:
            self._record_access(
                artifact_id,
                principal,
                allowed=False,
                error_code=getattr(exc, "code", type(exc).__name__),
            )
            if isinstance(exc, ArtifactStoreError):
                raise ArtifactPublicationError(exc.code, str(exc), retryable=exc.retryable) from exc
            raise
        else:
            self._record_access(artifact_id, principal, allowed=True)
            return payload

    def _record_access(
        self,
        artifact_id: str,
        principal: Principal,
        *,
        allowed: bool,
        error_code: str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        self.access_audit.record(
            ArtifactAccessEvent(
                timestamp=timestamp,
                action="READ_OBJECT",
                artifact_id=artifact_id,
                principal_subject=principal.subject,
                allowed=allowed,
                error_code=error_code,
            )
        )

    def verify_integrity(self, artifact: ArtifactRef, principal: Principal) -> bool:
        """Recompute object integrity against an authorized internal reference."""

        payload = self.read(artifact.artifact_id, principal)
        return sha256(payload).hexdigest() == artifact.checksum_sha256

    def _fail_pending(self, artifact_id: str, reason: str) -> None:
        try:
            if self.registry.get_metadata(artifact_id).status is ArtifactStatus.PENDING:
                self.registry.fail(artifact_id, reason)
        except ArtifactRegistryError:
            # Preserve the publication failure; reconciliation reports any
            # competing state transition separately.
            return

    def _lock_for(self, scope: tuple[str, str]) -> Lock:
        payload = "\x1f".join(
            (
                self._registry_coordination_key,
                self._store_coordination_key,
                scope[0],
                scope[1],
            )
        )
        stripe = int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")
        return _PUBLICATION_LOCK_STRIPES[stripe % len(_PUBLICATION_LOCK_STRIPES)]

    @classmethod
    def _key_from_uri(cls, uri: str) -> str:
        if not uri.startswith(cls.URI_PREFIX):
            raise ArtifactPublicationError("UNSUPPORTED_URI", "registry URI is not handled by this store")
        key = uri[len(cls.URI_PREFIX) :]
        if not key:
            raise ArtifactPublicationError("UNSUPPORTED_URI", "registry URI has an empty object key")
        return key

    @staticmethod
    def _object_key(case_id: str, artifact_type: ArtifactType, artifact_id: str) -> str:
        case_hash = sha256(case_id.encode("utf-8")).hexdigest()[:20]
        artifact_hash = sha256(artifact_id.encode("utf-8")).hexdigest()
        return f"cases/{case_hash}/{artifact_type.value.lower()}/{artifact_hash}.bin"
