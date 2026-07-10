from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from contracts.enums import ArtifactStatus, ArtifactType, CoordinateSystem
from contracts.geometry import VolumeGeometry
from puncture_agent.artifacts import ArtifactRegistryError, InMemoryArtifactRegistry, Principal


CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64


def geometry(spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)) -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(64, 64, 32),
        spacing_mm=spacing,
        origin_mm=(0.0, 0.0, 0.0),
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=CoordinateSystem.LPS,
    )


def begin(
    registry: InMemoryArtifactRegistry,
    *,
    artifact_id: str,
    key: str,
    parents: tuple[str, ...] = (),
    volume_geometry: VolumeGeometry | None = None,
):
    return registry.begin_registration(
        artifact_id=artifact_id,
        case_id="case-001",
        artifact_type=ArtifactType.CT_VOLUME,
        internal_uri=f"mock://private/{artifact_id}",
        created_by="unit-test",
        idempotency_key=key,
        producer_name="mock-tool",
        producer_version="1.0.0",
        parent_artifact_ids=parents,
        geometry=volume_geometry or geometry(),
    )


class ArtifactRegistryTests(unittest.TestCase):
    def test_pending_metadata_is_public_and_uri_free(self) -> None:
        registry = InMemoryArtifactRegistry()
        view = begin(registry, artifact_id="ct-1", key="key-1")
        self.assertEqual(ArtifactStatus.PENDING, view.status)
        self.assertFalse(hasattr(view, "uri"))
        self.assertFalse(hasattr(view, "checksum_sha256"))

    def test_finalize_is_idempotent_but_cannot_overwrite(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="key-1")
        first = registry.finalize("ct-1", CHECKSUM_A, 123)
        second = registry.finalize("ct-1", CHECKSUM_A, 123)
        self.assertEqual(first, second)
        self.assertEqual(ArtifactStatus.AVAILABLE, first.status)
        with self.assertRaises(ArtifactRegistryError) as raised:
            registry.finalize("ct-1", CHECKSUM_B, 123)
        self.assertEqual("CONFLICT", raised.exception.code)

    def test_failed_pending_artifact_is_invalid_and_unresolvable(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="key-1")
        view = registry.fail("ct-1", "upload failed")
        self.assertEqual(ArtifactStatus.INVALID, view.status)
        principal = Principal("tool", ("artifact_uri_reader",), ("case-001",))
        with self.assertRaises(ArtifactRegistryError):
            registry.resolve_uri("ct-1", principal)
        self.assertIsNone(registry.find_available_by_idempotency_key("key-1"))

    def test_mark_missing_preserves_metadata_but_prevents_consumption(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="key-1")
        registry.finalize("ct-1", CHECKSUM_A, 123)
        missing = registry.mark_missing("ct-1", "object store key disappeared")
        self.assertEqual(ArtifactStatus.MISSING, missing.status)
        self.assertEqual("ct-1", registry.get_metadata("ct-1").artifact_id)
        self.assertIsNone(registry.find_available_by_idempotency_key("key-1"))

    def test_uri_resolution_requires_role_and_case_access(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="key-1")
        registry.finalize("ct-1", CHECKSUM_A, 123)
        denied = Principal("user", (), ("case-001",))
        wrong_case = Principal("tool", ("artifact_uri_reader",), ("case-999",))
        allowed = Principal("tool", ("artifact_uri_reader",), ("case-001",))
        for principal in (denied, wrong_case):
            with self.assertRaises(ArtifactRegistryError) as raised:
                registry.resolve_uri("ct-1", principal)
            self.assertEqual("PERMISSION_DENIED", raised.exception.code)
        self.assertEqual("mock://private/ct-1", registry.resolve_uri("ct-1", allowed))

    def test_lineage_requires_available_same_case_parent(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="key-1")
        with self.assertRaises(ArtifactRegistryError):
            begin(registry, artifact_id="seg-1", key="key-2", parents=("ct-1",))
        registry.finalize("ct-1", CHECKSUM_A, 123)
        begin(registry, artifact_id="seg-1", key="key-2", parents=("ct-1",))
        self.assertEqual(("ct-1",), registry.get_lineage("seg-1").parent_artifact_ids)
        self.assertEqual(("seg-1",), registry.get_lineage("ct-1").child_artifact_ids)

    def test_missing_parent_and_self_cycle_are_rejected(self) -> None:
        registry = InMemoryArtifactRegistry()
        with self.assertRaises(ArtifactRegistryError) as missing:
            begin(registry, artifact_id="seg-1", key="key-1", parents=("missing",))
        self.assertEqual("PARENT_NOT_FOUND", missing.exception.code)
        with self.assertRaises(ArtifactRegistryError) as cycle:
            begin(registry, artifact_id="seg-1", key="key-1", parents=("seg-1",))
        self.assertEqual("LINEAGE_CYCLE", cycle.exception.code)

    def test_geometry_fingerprint_changes_with_spacing(self) -> None:
        same_a = geometry()
        same_b = geometry()
        changed = geometry((1.0, 1.0, 2.1))
        self.assertEqual(same_a.geometry_fingerprint, same_b.geometry_fingerprint)
        self.assertNotEqual(same_a.geometry_fingerprint, changed.geometry_fingerprint)

    def test_ready_result_is_reused_by_idempotency_key(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="same-key")
        registry.finalize("ct-1", CHECKSUM_A, 123)
        reused = begin(registry, artifact_id="ct-2", key="same-key")
        self.assertEqual("ct-1", reused.artifact_id)
        self.assertEqual(ArtifactStatus.AVAILABLE, reused.status)

    def test_same_key_is_scoped_by_case_and_never_leaks_metadata(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="case-1-ct", key="shared-key")
        registry.finalize("case-1-ct", CHECKSUM_A, 123)
        second = registry.begin_registration(
            artifact_id="case-2-ct",
            case_id="case-002",
            artifact_type=ArtifactType.CT_VOLUME,
            internal_uri="mock://private/case-2-ct",
            created_by="unit-test",
            idempotency_key="shared-key",
            producer_name="mock-tool",
            producer_version="1.0.0",
            geometry=geometry(),
        )
        self.assertEqual("case-002", second.case_id)
        self.assertEqual("case-2-ct", second.artifact_id)
        with self.assertRaises(ArtifactRegistryError) as raised:
            registry.find_available_by_idempotency_key("shared-key")
        self.assertEqual("AMBIGUOUS_IDEMPOTENCY_KEY", raised.exception.code)
        self.assertEqual(
            "case-1-ct",
            registry.find_available_by_idempotency_key(
                "shared-key", case_id="case-001"
            ).artifact_id,
        )

    def test_same_scoped_key_with_different_registration_inputs_conflicts(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="same-key")
        with self.assertRaises(ArtifactRegistryError) as raised:
            registry.begin_registration(
                artifact_id="ct-2",
                case_id="case-001",
                artifact_type=ArtifactType.NIFTI_LABELMAP,
                internal_uri="mock://private/ct-2",
                created_by="unit-test",
                idempotency_key="same-key",
                producer_name="different-tool",
                producer_version="2.0.0",
                geometry=geometry(),
            )
        self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)

    def test_memory_and_persistent_registry_share_input_validation(self) -> None:
        registry = InMemoryArtifactRegistry()
        with self.assertRaises(ArtifactRegistryError) as raised:
            registry.begin_registration(
                artifact_id="ct-1",
                case_id="case-001",
                artifact_type=ArtifactType.CT_VOLUME,
                internal_uri="mock://private/ct-1",
                created_by="unit-test",
                idempotency_key="key-1",
                producer_name="mock-tool",
                producer_version="1.0.0",
                metadata={"invalid": 1},  # type: ignore[dict-item]
            )
        self.assertEqual("INVALID_ARGUMENT", raised.exception.code)

    def test_concurrent_begin_claims_one_pending_registration(self) -> None:
        registry = InMemoryArtifactRegistry()

        def register(index: int):
            return registry.begin_registration(
                artifact_id=f"ct-{index}",
                case_id="case-001",
                artifact_type=ArtifactType.CT_VOLUME,
                internal_uri=f"mock://private/ct-{index}",
                created_by="unit-test",
                idempotency_key="concurrent-key",
                producer_name="mock-tool",
                producer_version="1.0.0",
                geometry=geometry(),
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            views = list(pool.map(register, range(100)))
        self.assertEqual(1, len({view.artifact_id for view in views}))
        self.assertEqual({ArtifactStatus.PENDING}, {view.status for view in views})

    def test_invalidated_artifact_keeps_lineage_and_releases_claim(self) -> None:
        registry = InMemoryArtifactRegistry()
        begin(registry, artifact_id="ct-1", key="parent-key")
        registry.finalize("ct-1", CHECKSUM_A, 123)
        begin(registry, artifact_id="seg-1", key="child-key", parents=("ct-1",))
        registry.finalize("seg-1", CHECKSUM_B, 456)
        registry.invalidate("seg-1", "quality regression")
        self.assertEqual(("ct-1",), registry.get_lineage("seg-1").parent_artifact_ids)
        replacement = begin(
            registry,
            artifact_id="seg-2",
            key="child-key",
            parents=("ct-1",),
        )
        self.assertEqual("seg-2", replacement.artifact_id)

    def test_unknown_artifact_is_structured_not_found(self) -> None:
        registry = InMemoryArtifactRegistry()
        with self.assertRaises(ArtifactRegistryError) as raised:
            registry.get_metadata("unknown")
        self.assertEqual("NOT_FOUND", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
