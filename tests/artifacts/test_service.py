from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from contracts.enums import ArtifactStatus, ArtifactType, CoordinateSystem
from contracts.geometry import VolumeGeometry
from puncture_agent.artifacts.registry import (
    ArtifactRegistryError,
    InMemoryArtifactRegistry,
    Principal,
)
from puncture_agent.artifacts.service import (
    ArtifactPublicationError,
    ArtifactPublicationService,
    InMemoryArtifactAccessAudit,
)
from puncture_agent.artifacts.sqlite_registry import SQLiteArtifactRegistry
from puncture_agent.artifacts.store import LocalArtifactStore


def geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(32, 32, 16),
        spacing_mm=(0.8, 0.8, 2.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=CoordinateSystem.LPS,
    )


class ArtifactPublicationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.registry = InMemoryArtifactRegistry()
        self.store = LocalArtifactStore(Path(self.temp.name) / "store")
        self.audit = InMemoryArtifactAccessAudit()
        self.service = ArtifactPublicationService(
            self.registry,
            self.store,
            access_audit=self.audit,
        )
        self.principal = Principal(
            "tool-worker",
            roles=("artifact_uri_reader",),
            allowed_case_ids=("case-001",),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def publish(self, payload=b"mock-volume", *, key="publish-key"):
        return self.service.publish(
            payload,
            artifact_id="artifact-001",
            case_id="case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            created_by="test",
            idempotency_key=key,
            producer_name="fixture-loader",
            producer_version="1.0.0",
            geometry=geometry(),
        )

    def test_publish_uses_store_checksum_and_registry_becomes_available(self) -> None:
        reference = self.publish()
        self.assertEqual(ArtifactStatus.AVAILABLE, reference.status)
        self.assertEqual(b"mock-volume", self.service.read(reference.artifact_id, self.principal))
        self.assertTrue(self.service.verify_integrity(reference, self.principal))
        public = self.registry.get_metadata(reference.artifact_id)
        self.assertFalse(hasattr(public, "uri"))
        self.assertFalse(hasattr(public, "checksum_sha256"))

    def test_repeated_publication_reuses_one_artifact(self) -> None:
        first = self.publish()
        second = self.publish()
        self.assertEqual(first, second)

    def test_concurrent_same_publication_is_serialized(self) -> None:
        def upload(_: int):
            return self.publish()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(upload, range(20)))
        self.assertEqual(1, len({result.artifact_id for result in results}))
        self.assertEqual(b"mock-volume", self.service.read("artifact-001", self.principal))

    def test_retryable_store_failure_keeps_pending_and_can_retry(self) -> None:
        def broken_payload():
            yield b"partial"
            raise OSError("source disappeared")

        with self.assertRaises(ArtifactPublicationError) as raised:
            self.publish(broken_payload(), key="broken-key")
        self.assertEqual("WRITE_FAILED", raised.exception.code)
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(ArtifactStatus.PENDING, self.registry.get_metadata("artifact-001").status)
        recovered = self.publish(b"recovered-volume", key="broken-key")
        self.assertEqual(ArtifactStatus.AVAILABLE, recovered.status)
        self.assertEqual(
            b"recovered-volume",
            self.service.read(recovered.artifact_id, self.principal),
        )

    def test_retryable_finalize_failure_reuses_committed_object_on_retry(self) -> None:
        with patch.object(
            self.registry,
            "finalize",
            side_effect=ArtifactRegistryError(
                "REGISTRY_BUSY",
                "simulated database lock",
                retryable=True,
            ),
        ):
            with self.assertRaises(ArtifactPublicationError) as raised:
                self.publish(b"durable-before-db", key="finalize-retry-key")
        self.assertEqual("REGISTRY_BUSY", raised.exception.code)
        self.assertEqual(ArtifactStatus.PENDING, self.registry.get_metadata("artifact-001").status)

        # The second call sees the same immutable object and only needs to
        # complete the metadata transition; it does not conflict on artifact ID.
        recovered = self.publish(b"durable-before-db", key="finalize-retry-key")
        self.assertEqual(ArtifactStatus.AVAILABLE, recovered.status)
        self.assertTrue(self.service.verify_integrity(recovered, self.principal))

    def test_terminal_object_mismatch_invalidates_claim_and_allows_new_artifact(self) -> None:
        artifact_id = "preseeded-artifact"
        object_key = self.service._object_key(  # internal deterministic storage identity
            "case-001", ArtifactType.CT_VOLUME, artifact_id
        )
        self.store.put(object_key, b"unexpected-existing-content")
        with self.assertRaises(ArtifactPublicationError) as raised:
            self.service.publish(
                b"expected-content",
                artifact_id=artifact_id,
                case_id="case-001",
                artifact_type=ArtifactType.CT_VOLUME,
                created_by="test",
                idempotency_key="mismatch-key",
                producer_name="fixture-loader",
                producer_version="1.0.0",
                geometry=geometry(),
            )
        self.assertEqual("OBJECT_EXISTS", raised.exception.code)
        self.assertEqual(ArtifactStatus.INVALID, self.registry.get_metadata(artifact_id).status)

        recovered = self.service.publish(
            b"expected-content",
            case_id="case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            created_by="test",
            idempotency_key="mismatch-key",
            producer_name="fixture-loader",
            producer_version="1.0.0",
            geometry=geometry(),
        )
        self.assertEqual(ArtifactStatus.AVAILABLE, recovered.status)
        self.assertNotEqual(artifact_id, recovered.artifact_id)

    def test_retryable_begin_failure_uses_publication_error_contract(self) -> None:
        with patch.object(
            self.registry,
            "begin_registration",
            side_effect=ArtifactRegistryError(
                "REGISTRY_BUSY",
                "simulated begin lock",
                retryable=True,
            ),
        ):
            with self.assertRaises(ArtifactPublicationError) as raised:
                self.publish(key="begin-busy-key")
        self.assertEqual("REGISTRY_BUSY", raised.exception.code)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(any((Path(self.temp.name) / "store" / "objects").rglob("*.bin")))

    def test_begin_idempotency_conflict_is_normalized_without_mutating_owner(self) -> None:
        self.registry.begin_registration(
            artifact_id="owner-artifact",
            case_id="case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            internal_uri="artifact-store:owner",
            created_by="owner",
            idempotency_key="owned-key",
            producer_name="different-producer",
            producer_version="1.0.0",
            geometry=geometry(),
        )
        with self.assertRaises(ArtifactPublicationError) as raised:
            self.publish(key="owned-key")
        self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)
        self.assertEqual(
            ArtifactStatus.PENDING,
            self.registry.get_metadata("owner-artifact").status,
        )

    def test_unauthorized_read_never_returns_payload(self) -> None:
        reference = self.publish()
        denied = Principal("user", roles=(), allowed_case_ids=("case-001",))
        with self.assertRaises(Exception) as raised:
            self.service.read(reference.artifact_id, denied)
        self.assertNotIn("mock-volume", str(raised.exception))
        denied_event = self.audit.events()[-1]
        self.assertFalse(denied_event.allowed)
        self.assertEqual("PERMISSION_DENIED", denied_event.error_code)
        self.assertNotIn("artifact-store:", repr(denied_event))

    def test_access_audit_records_success_without_uri_or_payload(self) -> None:
        reference = self.publish()
        self.service.read(reference.artifact_id, self.principal)
        event = self.audit.events()[-1]
        self.assertTrue(event.allowed)
        self.assertEqual("READ_OBJECT", event.action)
        self.assertNotIn("mock-volume", repr(event))
        self.assertNotIn("artifact-store:", repr(event))

    def test_sqlite_and_local_store_survive_service_restart(self) -> None:
        database_path = Path(self.temp.name) / "registry.sqlite3"
        store = LocalArtifactStore(Path(self.temp.name) / "persistent-store")
        registry = SQLiteArtifactRegistry(database_path)
        service = ArtifactPublicationService(registry, store)
        reference = service.publish(
            b"durable-volume",
            artifact_id="durable-artifact",
            case_id="case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            created_by="test",
            idempotency_key="durable-key",
            producer_name="fixture-loader",
            producer_version="1.0.0",
            geometry=geometry(),
        )
        registry.close()

        reopened_registry = SQLiteArtifactRegistry(database_path)
        self.addCleanup(reopened_registry.close)
        reopened_service = ArtifactPublicationService(reopened_registry, store)
        restored = reopened_registry.find_available_by_idempotency_key(
            "durable-key", case_id="case-001"
        )
        self.assertEqual(reference, restored)
        self.assertEqual(
            b"durable-volume",
            reopened_service.read("durable-artifact", self.principal),
        )

    def test_two_service_instances_cannot_invalidate_shared_pending_claim(self) -> None:
        database_path = Path(self.temp.name) / "shared-registry.sqlite3"
        store_root = Path(self.temp.name) / "shared-store"
        registry_a = SQLiteArtifactRegistry(database_path)
        registry_b = SQLiteArtifactRegistry(database_path)
        self.addCleanup(registry_a.close)
        self.addCleanup(registry_b.close)
        service_a = ArtifactPublicationService(registry_a, LocalArtifactStore(store_root))
        service_b = ArtifactPublicationService(registry_b, LocalArtifactStore(store_root))

        finalize_entered = Event()
        allow_finalize = Event()
        original_finalize = registry_a.finalize
        results = []
        errors: list[BaseException] = []

        def paused_finalize(*args, **kwargs):
            finalize_entered.set()
            allow_finalize.wait(timeout=3)
            return original_finalize(*args, **kwargs)

        def publish(service: ArtifactPublicationService, payload: bytes) -> None:
            try:
                results.append(
                    service.publish(
                        payload,
                        case_id="case-001",
                        artifact_type=ArtifactType.CT_VOLUME,
                        created_by="test",
                        idempotency_key="shared-service-key",
                        producer_name="fixture-loader",
                        producer_version="1.0.0",
                        geometry=geometry(),
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        with patch.object(registry_a, "finalize", side_effect=paused_finalize):
            first = Thread(target=publish, args=(service_a, b"writer-a"))
            second = Thread(target=publish, args=(service_b, b"writer-b"))
            first.start()
            self.assertTrue(finalize_entered.wait(timeout=3))
            second.start()
            allow_finalize.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        self.assertEqual(1, len({result.artifact_id for result in results}))
        principal = Principal(
            "tool-worker",
            roles=("artifact_uri_reader",),
            allowed_case_ids=("case-001",),
        )
        self.assertEqual(b"writer-a", service_b.read(results[0].artifact_id, principal))
        self.assertEqual(
            ArtifactStatus.AVAILABLE,
            registry_b.get_metadata(results[0].artifact_id).status,
        )


if __name__ == "__main__":
    unittest.main()
