from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Lock, Thread
import unittest
from unittest.mock import patch

from contracts.enums import ArtifactStatus, ArtifactType, CoordinateSystem
from contracts.geometry import VolumeGeometry
from puncture_agent.artifacts.registry import ArtifactRegistryError, Principal
from puncture_agent.artifacts.registry import InMemoryArtifactRegistry
from puncture_agent.artifacts.sqlite_registry import SQLiteArtifactRegistry


CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64
PRIVATE_URI = "s3://private-medical-bucket/case-001/ct.nii.gz?secret=never-log"


def geometry() -> VolumeGeometry:
    return VolumeGeometry(
        size_ijk=(67, 71, 29),
        spacing_mm=(0.7421875, 0.7421875, 2.5),
        origin_mm=(-126.25, 17.125, 42.5),
        direction_cosines=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        coordinate_system=CoordinateSystem.LPS,
    )


def begin(
    registry: SQLiteArtifactRegistry,
    *,
    artifact_id: str,
    key: str,
    case_id: str = "case-001",
    parents: tuple[str, ...] = (),
    internal_uri: str | None = None,
    volume_geometry: VolumeGeometry | None = None,
    metadata: dict[str, str] | None = None,
    artifact_type: ArtifactType = ArtifactType.CT_VOLUME,
    producer_name: str = "mock-tool",
    producer_version: str = "1.2.3",
):
    return registry.begin_registration(
        artifact_id=artifact_id,
        case_id=case_id,
        artifact_type=artifact_type,
        internal_uri=internal_uri or f"mock://private/{artifact_id}",
        created_by="sqlite-unit-test",
        idempotency_key=key,
        producer_name=producer_name,
        producer_version=producer_version,
        parent_artifact_ids=parents,
        geometry=volume_geometry or geometry(),
        metadata=metadata,
    )


class SQLiteArtifactRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "artifact-registry.sqlite3"
        self.registry = SQLiteArtifactRegistry(self.database_path)

    def tearDown(self) -> None:
        self.registry.close()
        self.temporary_directory.cleanup()

    def test_public_method_signatures_match_memory_reference(self) -> None:
        methods = (
            "begin_registration",
            "finalize",
            "fail",
            "invalidate",
            "mark_missing",
            "get_metadata",
            "resolve_uri",
            "find_available_by_idempotency_key",
            "find_ready_by_idempotency_key",
            "get_lineage",
        )
        for method in methods:
            with self.subTest(method=method):
                self.assertEqual(
                    inspect.signature(getattr(InMemoryArtifactRegistry, method)),
                    inspect.signature(getattr(SQLiteArtifactRegistry, method)),
                )

    def test_multi_parent_order_is_canonical_in_both_registries(self) -> None:
        memory = InMemoryArtifactRegistry()
        for registry in (memory, self.registry):
            registry.begin_registration(
                artifact_id="parent-z",
                case_id="case-001",
                artifact_type=ArtifactType.CT_VOLUME,
                internal_uri="mock://private/parent-z",
                created_by="test",
                idempotency_key="parent-z-key",
                producer_name="mock-tool",
                producer_version="1.0.0",
                geometry=geometry(),
            )
            registry.finalize("parent-z", CHECKSUM_A, 10)
            registry.begin_registration(
                artifact_id="parent-a",
                case_id="case-001",
                artifact_type=ArtifactType.CT_VOLUME,
                internal_uri="mock://private/parent-a",
                created_by="test",
                idempotency_key="parent-a-key",
                producer_name="mock-tool",
                producer_version="1.0.0",
                geometry=geometry(),
            )
            registry.finalize("parent-a", CHECKSUM_B, 10)
            registry.begin_registration(
                artifact_id="child",
                case_id="case-001",
                artifact_type=ArtifactType.NIFTI_LABELMAP,
                internal_uri="mock://private/child",
                created_by="test",
                idempotency_key="child-key",
                producer_name="mock-tool",
                producer_version="1.0.0",
                parent_artifact_ids=("parent-z", "parent-a"),
                geometry=geometry(),
            )
            reference = registry.finalize("child", "c" * 64, 10)
            self.assertEqual(("parent-a", "parent-z"), reference.parent_artifact_ids)
            self.assertEqual(
                ("parent-a", "parent-z"),
                registry.get_lineage("child").parent_artifact_ids,
            )

    def test_restart_preserves_artifact_geometry_metadata_and_lineage(self) -> None:
        expected_geometry = geometry()
        begin(
            self.registry,
            artifact_id="ct-1",
            key="ct-key",
            volume_geometry=expected_geometry,
            metadata={"modality": "CT", "series": "venous"},
        )
        self.registry.finalize("ct-1", CHECKSUM_A, 4096)
        begin(
            self.registry,
            artifact_id="seg-1",
            key="seg-key",
            parents=("ct-1",),
            volume_geometry=expected_geometry,
        )
        self.registry.finalize("seg-1", CHECKSUM_B, 2048)
        self.registry.close()

        self.registry = SQLiteArtifactRegistry(self.database_path)
        restored = self.registry.find_available_by_idempotency_key("ct-key")
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual("ct-1", restored.artifact_id)
        self.assertEqual(expected_geometry, restored.geometry)
        self.assertEqual({"modality": "CT", "series": "venous"}, restored.metadata)
        self.assertEqual(("ct-1",), self.registry.get_lineage("seg-1").parent_artifact_ids)
        self.assertEqual(("seg-1",), self.registry.get_lineage("ct-1").child_artifact_ids)

    def test_public_views_do_not_leak_uri_checksum_or_private_metadata(self) -> None:
        pending = begin(
            self.registry,
            artifact_id="ct-secret",
            key="secret-key",
            internal_uri=PRIVATE_URI,
            metadata={"access_token": "also-never-log"},
        )
        available_ref = self.registry.finalize("ct-secret", CHECKSUM_A, 32)
        available = self.registry.get_metadata("ct-secret")

        for public_view in (pending, available, available_ref.to_public_view()):
            projection = asdict(public_view)
            self.assertNotIn("uri", projection)
            self.assertNotIn("internal_uri", projection)
            self.assertNotIn("checksum_sha256", projection)
            self.assertNotIn("metadata", projection)
            self.assertNotIn(PRIVATE_URI, repr(public_view))
            self.assertNotIn("also-never-log", repr(public_view))

    def test_uri_resolution_requires_both_role_and_case_authorization(self) -> None:
        begin(
            self.registry,
            artifact_id="ct-secret",
            key="secret-key",
            internal_uri=PRIVATE_URI,
        )
        self.registry.finalize("ct-secret", CHECKSUM_A, 32)

        denied_principals = (
            Principal("no-role", (), ("case-001",)),
            Principal("wrong-case", ("artifact_uri_reader",), ("case-999",)),
        )
        for principal in denied_principals:
            with self.assertRaises(ArtifactRegistryError) as raised:
                self.registry.resolve_uri("ct-secret", principal)
            self.assertEqual("PERMISSION_DENIED", raised.exception.code)
            self.assertNotIn(PRIVATE_URI, str(raised.exception))

        allowed = Principal("worker", ("artifact_uri_reader",), ("case-001",))
        system = Principal("runtime", ("system",), ())
        self.assertEqual(PRIVATE_URI, self.registry.resolve_uri("ct-secret", allowed))
        self.assertEqual(PRIVATE_URI, self.registry.resolve_uri("ct-secret", system))

    def test_finalize_is_idempotent_and_rejects_overwrite(self) -> None:
        begin(self.registry, artifact_id="ct-1", key="key-1")
        first = self.registry.finalize("ct-1", CHECKSUM_A.upper(), 123)
        second = self.registry.finalize("ct-1", CHECKSUM_A, 123)
        self.assertEqual(first, second)
        self.assertIs(ArtifactStatus.AVAILABLE, first.status)
        self.assertEqual(CHECKSUM_A, first.checksum_sha256)

        with self.assertRaises(ArtifactRegistryError) as raised:
            self.registry.finalize("ct-1", CHECKSUM_B, 123)
        self.assertEqual("CONFLICT", raised.exception.code)
        self.assertEqual(CHECKSUM_A, self.registry.find_available_by_idempotency_key("key-1").checksum_sha256)

    def test_available_registration_is_reused_after_restart(self) -> None:
        begin(self.registry, artifact_id="ct-1", key="same-key")
        self.registry.finalize("ct-1", CHECKSUM_A, 10)
        self.registry.close()
        self.registry = SQLiteArtifactRegistry(self.database_path)

        reused = begin(self.registry, artifact_id="ct-unused", key="same-key")
        self.assertEqual("ct-1", reused.artifact_id)
        self.assertIs(ArtifactStatus.AVAILABLE, reused.status)
        with self.assertRaises(ArtifactRegistryError) as raised:
            self.registry.get_metadata("ct-unused")
        self.assertEqual("NOT_FOUND", raised.exception.code)

    def test_lifecycle_transitions_release_idempotency_key(self) -> None:
        begin(self.registry, artifact_id="failed", key="failed-key")
        failed = self.registry.fail("failed", "object upload failed")
        self.assertIs(ArtifactStatus.INVALID, failed.status)
        self.assertIsNone(self.registry.find_available_by_idempotency_key("failed-key"))

        begin(self.registry, artifact_id="invalidated", key="reusable-key")
        self.registry.finalize("invalidated", CHECKSUM_A, 10)
        invalidated = self.registry.invalidate("invalidated", "quality gate rejected output")
        self.assertIs(ArtifactStatus.INVALID, invalidated.status)
        self.assertIsNone(self.registry.find_available_by_idempotency_key("reusable-key"))
        replacement = begin(self.registry, artifact_id="replacement", key="reusable-key")
        self.assertEqual("replacement", replacement.artifact_id)

        begin(self.registry, artifact_id="missing", key="missing-key")
        self.registry.finalize("missing", CHECKSUM_B, 10)
        missing = self.registry.mark_missing("missing", "object store key disappeared")
        self.assertIs(ArtifactStatus.MISSING, missing.status)
        self.assertIsNone(self.registry.find_available_by_idempotency_key("missing-key"))

    def test_registration_failure_rolls_back_artifact_and_all_lineage(self) -> None:
        begin(self.registry, artifact_id="parent-ready", key="parent-ready-key")
        self.registry.finalize("parent-ready", CHECKSUM_A, 10)
        begin(self.registry, artifact_id="parent-pending", key="parent-pending-key")

        with self.assertRaises(ArtifactRegistryError) as raised:
            begin(
                self.registry,
                artifact_id="rolled-back-child",
                key="child-key",
                parents=("parent-ready", "parent-pending"),
            )
        self.assertEqual("PARENT_NOT_AVAILABLE", raised.exception.code)

        with self.assertRaises(ArtifactRegistryError) as absent:
            self.registry.get_metadata("rolled-back-child")
        self.assertEqual("NOT_FOUND", absent.exception.code)
        self.assertEqual((), self.registry.get_lineage("parent-ready").child_artifact_ids)

    def test_registration_fingerprint_conflict_rolls_back_new_artifact(self) -> None:
        begin(
            self.registry,
            artifact_id="canonical",
            key="shared-key",
            metadata={"model": "v1"},
        )

        with self.assertRaises(ArtifactRegistryError) as raised:
            begin(
                self.registry,
                artifact_id="conflicting",
                key="shared-key",
                metadata={"model": "v2"},
            )
        self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)
        self.assertIs(ArtifactStatus.PENDING, self.registry.get_metadata("canonical").status)
        with self.assertRaises(ArtifactRegistryError) as absent:
            self.registry.get_metadata("conflicting")
        self.assertEqual("NOT_FOUND", absent.exception.code)

    def test_identical_pending_registration_reuses_canonical_record(self) -> None:
        first = begin(self.registry, artifact_id="first", key="shared-key")
        reused = begin(self.registry, artifact_id="second", key="shared-key")

        self.assertEqual(first, reused)
        self.assertEqual("first", reused.artifact_id)
        with self.assertRaises(ArtifactRegistryError) as absent:
            self.registry.get_metadata("second")
        self.assertEqual("NOT_FOUND", absent.exception.code)

    def test_concurrent_begin_claims_exactly_one_pending_record(self) -> None:
        second_registry = SQLiteArtifactRegistry(self.database_path)
        self.addCleanup(second_registry.close)
        barrier = Barrier(2)
        result_lock = Lock()
        results: list[str] = []
        failures: list[BaseException] = []

        def register(registry: SQLiteArtifactRegistry, artifact_id: str) -> None:
            try:
                barrier.wait(timeout=3)
                view = begin(
                    registry,
                    artifact_id=artifact_id,
                    key="concurrent-begin-key",
                )
                with result_lock:
                    results.append(view.artifact_id)
            except BaseException as exc:  # captured and asserted by the test thread
                with result_lock:
                    failures.append(exc)

        threads = (
            Thread(target=register, args=(self.registry, "candidate-a")),
            Thread(target=register, args=(second_registry, "candidate-b")),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], failures)
        self.assertEqual(2, len(results))
        self.assertEqual(1, len(set(results)))
        winner_id = results[0]
        loser_id = "candidate-b" if winner_id == "candidate-a" else "candidate-a"
        self.assertIs(ArtifactStatus.PENDING, self.registry.get_metadata(winner_id).status)
        with self.assertRaises(ArtifactRegistryError) as absent:
            self.registry.get_metadata(loser_id)
        self.assertEqual("NOT_FOUND", absent.exception.code)

    def test_idempotency_fingerprint_covers_contract_relevant_fields(self) -> None:
        parent_a = begin(self.registry, artifact_id="parent-a", key="parent-a-key")
        parent_b = begin(self.registry, artifact_id="parent-b", key="parent-b-key")
        self.registry.finalize(parent_a.artifact_id, CHECKSUM_A, 10)
        self.registry.finalize(parent_b.artifact_id, CHECKSUM_B, 10)
        changed_geometry = VolumeGeometry(
            size_ijk=geometry().size_ijk,
            spacing_mm=(0.8, 0.8, 3.0),
            origin_mm=geometry().origin_mm,
            direction_cosines=geometry().direction_cosines,
            coordinate_system=CoordinateSystem.LPS,
        )
        cases = (
            (
                "type",
                {},
                {"artifact_type": ArtifactType.NIFTI_LABELMAP},
            ),
            ("producer-name", {}, {"producer_name": "different-tool"}),
            ("producer-version", {}, {"producer_version": "9.9.9"}),
            ("parents", {"parents": ("parent-a",)}, {"parents": ("parent-b",)}),
            ("geometry", {}, {"volume_geometry": changed_geometry}),
            ("metadata", {"metadata": {"model": "v1"}}, {"metadata": {"model": "v2"}}),
        )
        for name, baseline_options, conflicting_options in cases:
            with self.subTest(field=name):
                begin(
                    self.registry,
                    artifact_id=f"baseline-{name}",
                    key=f"fingerprint-{name}",
                    **baseline_options,
                )
                with self.assertRaises(ArtifactRegistryError) as raised:
                    begin(
                        self.registry,
                        artifact_id=f"conflict-{name}",
                        key=f"fingerprint-{name}",
                        **conflicting_options,
                    )
                self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)

    def test_idempotency_key_is_scoped_by_case(self) -> None:
        case_one = begin(
            self.registry,
            artifact_id="case-one-artifact",
            key="shared-across-cases",
            case_id="case-001",
        )
        case_two = begin(
            self.registry,
            artifact_id="case-two-artifact",
            key="shared-across-cases",
            case_id="case-002",
        )
        self.registry.finalize(case_one.artifact_id, CHECKSUM_A, 10)
        self.registry.finalize(case_two.artifact_id, CHECKSUM_B, 10)

        self.assertNotEqual(case_one.artifact_id, case_two.artifact_id)
        with self.assertRaises(ArtifactRegistryError) as ambiguous:
            self.registry.find_available_by_idempotency_key("shared-across-cases")
        self.assertEqual("AMBIGUOUS_IDEMPOTENCY_KEY", ambiguous.exception.code)
        self.assertEqual(
            "case-one-artifact",
            self.registry.find_available_by_idempotency_key(
                "shared-across-cases", case_id="case-001"
            ).artifact_id,
        )
        self.assertEqual(
            "case-two-artifact",
            self.registry.find_ready_by_idempotency_key(
                "shared-across-cases", case_id="case-002"
            ).artifact_id,
        )

    def test_parent_constraints_and_canonical_status(self) -> None:
        with self.assertRaises(ArtifactRegistryError) as missing:
            begin(self.registry, artifact_id="child", key="child-key", parents=("unknown",))
        self.assertEqual("PARENT_NOT_FOUND", missing.exception.code)

        with self.assertRaises(ArtifactRegistryError) as cycle:
            begin(self.registry, artifact_id="child", key="child-key", parents=("child",))
        self.assertEqual("LINEAGE_CYCLE", cycle.exception.code)

        begin(self.registry, artifact_id="other-parent", key="other-parent-key", case_id="case-002")
        self.registry.finalize("other-parent", CHECKSUM_A, 10)
        with self.assertRaises(ArtifactRegistryError) as mismatch:
            begin(
                self.registry,
                artifact_id="cross-case-child",
                key="cross-case-key",
                parents=("other-parent",),
            )
        self.assertEqual("CASE_MISMATCH", mismatch.exception.code)
        self.assertIs(ArtifactStatus.AVAILABLE, self.registry.get_metadata("other-parent").status)

    def test_persisted_multi_node_lineage_cycle_is_detected(self) -> None:
        for index, artifact_id in enumerate(("cycle-a", "cycle-b", "cycle-c")):
            begin(
                self.registry,
                artifact_id=artifact_id,
                key=f"cycle-key-{index}",
            )
            self.registry.finalize(artifact_id, f"{index + 1:x}" * 64, 10)

        # Foreign keys remain enabled: all nodes are valid artifacts, while an
        # out-of-band/legacy writer creates the invalid A <- B <- C <- A cycle.
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executemany(
                """
                INSERT INTO artifact_lineage (child_artifact_id, parent_artifact_id)
                VALUES (?, ?)
                """,
                (
                    ("cycle-b", "cycle-a"),
                    ("cycle-c", "cycle-b"),
                    ("cycle-a", "cycle-c"),
                ),
            )

        for artifact_id in ("cycle-a", "cycle-b", "cycle-c"):
            with self.subTest(artifact_id=artifact_id):
                with self.assertRaises(ArtifactRegistryError) as raised:
                    self.registry.get_lineage(artifact_id)
                self.assertEqual("LINEAGE_CYCLE", raised.exception.code)

    def test_two_registry_instances_concurrently_finalize_same_content_once(self) -> None:
        second_registry = SQLiteArtifactRegistry(self.database_path)
        self.addCleanup(second_registry.close)
        first = begin(self.registry, artifact_id="worker-a", key="concurrent-key")
        second = begin(second_registry, artifact_id="worker-b", key="concurrent-key")
        self.assertEqual(first, second)
        self.assertEqual("worker-a", second.artifact_id)
        barrier = Barrier(2)
        result_lock = Lock()
        results: list[str] = []
        failures: list[BaseException] = []

        def finalize(registry: SQLiteArtifactRegistry, artifact_id: str) -> None:
            try:
                barrier.wait(timeout=3)
                result = registry.finalize(artifact_id, CHECKSUM_A, 100)
                with result_lock:
                    results.append(result.artifact_id)
            except BaseException as exc:  # captured and asserted in the main test thread
                with result_lock:
                    failures.append(exc)

        threads = (
            Thread(target=finalize, args=(self.registry, first.artifact_id)),
            Thread(target=finalize, args=(second_registry, second.artifact_id)),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], failures)
        self.assertEqual(2, len(results))
        self.assertEqual(1, len(set(results)))
        self.assertEqual("worker-a", results[0])
        self.assertIs(ArtifactStatus.AVAILABLE, self.registry.get_metadata("worker-a").status)
        with self.assertRaises(ArtifactRegistryError):
            self.registry.get_metadata("worker-b")

    def test_two_registry_instances_concurrently_reject_different_content(self) -> None:
        second_registry = SQLiteArtifactRegistry(self.database_path)
        self.addCleanup(second_registry.close)
        first = begin(self.registry, artifact_id="worker-a", key="concurrent-key")
        second = begin(second_registry, artifact_id="worker-b", key="concurrent-key")
        self.assertEqual(first.artifact_id, second.artifact_id)
        barrier = Barrier(2)
        result_lock = Lock()
        successes: list[str] = []
        errors: list[tuple[str, str]] = []

        def finalize(
            registry: SQLiteArtifactRegistry, artifact_id: str, checksum: str
        ) -> None:
            try:
                barrier.wait(timeout=3)
                result = registry.finalize(artifact_id, checksum, 100)
                with result_lock:
                    successes.append(result.artifact_id)
            except ArtifactRegistryError as exc:
                with result_lock:
                    errors.append((artifact_id, exc.code))

        threads = (
            Thread(target=finalize, args=(self.registry, first.artifact_id, CHECKSUM_A)),
            Thread(target=finalize, args=(second_registry, second.artifact_id, CHECKSUM_B)),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(1, len(successes))
        self.assertEqual(1, len(errors))
        self.assertEqual("CONFLICT", errors[0][1])
        self.assertIs(ArtifactStatus.AVAILABLE, self.registry.get_metadata(successes[0]).status)
        self.assertEqual(successes[0], errors[0][0])
        self.assertIsNot(ArtifactStatus.PENDING, self.registry.get_metadata(errors[0][0]).status)

    def test_legacy_competing_finalize_conflict_never_leaves_pending_orphan(self) -> None:
        """Exercise the defensive path for a legacy/non-cooperating DB writer."""

        begin(self.registry, artifact_id="canonical", key="legacy-duplicate-key")
        # A normal client cannot create this state. Dropping the active-scope
        # index emulates a pre-constraint database or an out-of-band writer.
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DROP INDEX uq_active_artifact_case_idempotency_key")
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, case_id, artifact_type, status, internal_uri,
                    created_by, idempotency_key, producer_name, producer_version,
                    geometry_json, metadata_json, registration_fingerprint
                )
                SELECT
                    'legacy-loser', case_id, artifact_type, status,
                    'mock://private/legacy-loser', created_by, idempotency_key,
                    producer_name, producer_version, geometry_json, metadata_json,
                    registration_fingerprint
                FROM artifacts WHERE artifact_id = 'canonical'
                """
            )

        self.registry.finalize("canonical", CHECKSUM_A, 10)
        with self.assertRaises(ArtifactRegistryError) as raised:
            self.registry.finalize("legacy-loser", CHECKSUM_B, 20)
        self.assertEqual("IDEMPOTENCY_CONFLICT", raised.exception.code)
        self.assertIs(ArtifactStatus.INVALID, self.registry.get_metadata("legacy-loser").status)
        self.assertIs(ArtifactStatus.AVAILABLE, self.registry.get_metadata("canonical").status)

    def test_locked_database_is_retryable_not_not_found(self) -> None:
        second_registry = SQLiteArtifactRegistry(
            self.database_path,
            busy_timeout_seconds=0.05,
        )
        self.addCleanup(second_registry.close)
        locker = sqlite3.connect(self.database_path, isolation_level=None)
        self.addCleanup(locker.close)
        locker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(ArtifactRegistryError) as raised:
                begin(second_registry, artifact_id="locked", key="locked-key")
        finally:
            locker.rollback()
        self.assertEqual("REGISTRY_BUSY", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    def test_scoped_idempotency_lookup_uses_an_index(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            plan = connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM artifacts
                WHERE case_id = ? AND idempotency_key = ? AND status = ?
                """,
                ("case-001", "key", ArtifactStatus.AVAILABLE.value),
            ).fetchall()
        details = " ".join(str(row[3]).upper() for row in plan)
        self.assertIn("SEARCH ARTIFACTS", details)
        self.assertNotIn("SCAN ARTIFACTS", details)

    def test_database_and_sidecars_are_private_files(self) -> None:
        candidates = (
            self.database_path,
            Path(str(self.database_path) + "-wal"),
            Path(str(self.database_path) + "-shm"),
        )
        existing = [path for path in candidates if path.exists()]
        self.assertTrue(existing)
        for path in existing:
            with self.subTest(path=path.name):
                self.assertEqual(0, path.stat().st_mode & 0o077)

    def test_symlink_database_path_is_rejected(self) -> None:
        target = Path(self.temporary_directory.name) / "outside.sqlite3"
        target.write_bytes(b"do-not-touch")
        symlink = Path(self.temporary_directory.name) / "registry-link.sqlite3"
        symlink.symlink_to(target)
        with self.assertRaises(ArtifactRegistryError) as raised:
            SQLiteArtifactRegistry(symlink)
        self.assertEqual("INVALID_DATABASE_PATH", raised.exception.code)
        self.assertEqual(b"do-not-touch", target.read_bytes())

    def test_permission_hardening_failure_rolls_back_before_commit(self) -> None:
        with patch(
            "puncture_agent.artifacts.sqlite_registry.Path.chmod",
            side_effect=PermissionError("simulated chmod denial"),
        ):
            with self.assertRaises(ArtifactRegistryError) as raised:
                begin(self.registry, artifact_id="must-rollback", key="permission-key")
        self.assertEqual("STORAGE_PERMISSION_ERROR", raised.exception.code)
        with self.assertRaises(ArtifactRegistryError) as missing:
            self.registry.get_metadata("must-rollback")
        self.assertEqual("NOT_FOUND", missing.exception.code)


if __name__ == "__main__":
    unittest.main()
