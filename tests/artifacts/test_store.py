"""Correctness tests for the local immutable artifact object store."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from puncture_agent.artifacts.store import (
    ArtifactStore,
    ArtifactStoreError,
    LocalArtifactStore,
    StagedObject,
)


class LocalArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = LocalArtifactStore(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def assert_no_temporary_objects(self) -> None:
        self.assertEqual([], list((self.root / ".tmp").iterdir()))

    def test_protocol_normal_put_read_and_metadata_redaction(self) -> None:
        self.assertIsInstance(self.store, ArtifactStore)
        payload = b"mock-nifti-volume\x00\x01"

        stored = self.store.put("case-001/ct/volume.nii.gz", payload)

        self.assertEqual(hashlib.sha256(payload).hexdigest(), stored.checksum_sha256)
        self.assertEqual(len(payload), stored.size_bytes)
        self.assertTrue(self.store.exists(stored.object_key))
        self.assertEqual(payload, self.store.read(stored.object_key))
        self.assertEqual(
            {"object_key", "checksum_sha256", "size_bytes"}, set(asdict(stored))
        )
        self.assertNotIn("path", repr(stored).lower())
        self.assertNotIn("payload", repr(stored).lower())
        self.assert_no_temporary_objects()

    def test_checksum_is_deterministic_across_chunk_boundaries(self) -> None:
        payload = b"abcdef" * 4096
        first = self.store.put("case/a.bin", payload)
        second = self.store.put(
            "case/b.bin", (payload[index : index + 37] for index in range(0, len(payload), 37))
        )
        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(expected, first.checksum_sha256)
        self.assertEqual(expected, second.checksum_sha256)
        self.assertEqual(first.size_bytes, second.size_bytes)

    def test_empty_object_is_valid_boundary_case(self) -> None:
        stored = self.store.put("case/empty.bin", [])
        self.assertEqual(hashlib.sha256(b"").hexdigest(), stored.checksum_sha256)
        self.assertEqual(0, stored.size_bytes)
        self.assertEqual(b"", self.store.read(stored.object_key))

    def test_rejects_invalid_and_traversing_keys(self) -> None:
        invalid_keys = (
            "",
            " /trimmed",
            "/absolute",
            "../outside",
            "case/../../outside",
            "case//volume",
            "./case/volume",
            "case/./volume",
            "case\\volume",
            "case/volume\x00.bin",
            "case/" + "x" * 256,
        )
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises(ArtifactStoreError) as raised:
                    self.store.put(key, b"data")
                self.assertEqual("INVALID_KEY", raised.exception.code)
        self.assert_no_temporary_objects()

    def test_rejects_symlink_escape(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "objects" / "escape").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.put("escape/leak.bin", b"secret")

        self.assertEqual("INVALID_KEY", raised.exception.code)
        self.assertFalse((outside / "leak.bin").exists())
        self.assert_no_temporary_objects()

    def test_rejects_preexisting_symlink_storage_namespace(self) -> None:
        root = self.root / "malicious-root"
        outside = self.root / "outside-namespace"
        root.mkdir()
        outside.mkdir()
        (root / "objects").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactStoreError) as raised:
            LocalArtifactStore(root)
        self.assertEqual("INVALID_ROOT", raised.exception.code)
        self.assertEqual([], list(outside.iterdir()))

    def test_direct_commit_cleans_temp_when_parent_becomes_symlink(self) -> None:
        staged = self.store.stage("escape-after-stage/object.bin", b"secret")
        outside = self.root / "outside-after-stage"
        outside.mkdir()
        (self.root / "objects" / "escape-after-stage").symlink_to(
            outside, target_is_directory=True
        )
        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.commit(staged)
        self.assertEqual("INVALID_KEY", raised.exception.code)
        self.assert_no_temporary_objects()
        self.assertEqual([], list(outside.iterdir()))

    def test_partial_stream_failure_removes_temporary_object(self) -> None:
        def broken_stream():
            yield b"partial"
            raise OSError("simulated source failure")

        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.stage("case/partial.bin", broken_stream())

        self.assertEqual("WRITE_FAILED", raised.exception.code)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(self.store.exists("case/partial.bin"))
        self.assert_no_temporary_objects()

    def test_invalid_chunk_removes_partial_temporary_object(self) -> None:
        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.stage("case/invalid.bin", [b"valid", "not-bytes"])  # type: ignore[list-item]
        self.assertEqual("INVALID_CHUNK", raised.exception.code)
        self.assert_no_temporary_objects()

    def test_commit_failure_removes_temporary_object(self) -> None:
        staged = self.store.stage("case/failure.bin", b"complete-temp")

        with patch("puncture_agent.artifacts.store.os.link", side_effect=OSError("disk full")):
            with self.assertRaises(ArtifactStoreError) as raised:
                self.store.commit(staged)

        self.assertEqual("COMMIT_FAILED", raised.exception.code)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(self.store.exists("case/failure.bin"))
        self.assert_no_temporary_objects()

    def test_cross_instance_retry_survives_post_link_durability_failure(self) -> None:
        second_store = LocalArtifactStore(self.root)
        (self.root / "objects" / "case").mkdir()
        first_stage = self.store.stage("case/durable-race.bin", b"same-content")
        second_stage = second_store.stage("case/durable-race.bin", b"same-content")
        linked = Event()
        release_failure = Event()
        first_errors: list[ArtifactStoreError] = []
        second_results = []

        def fail_fsync(_: Path) -> None:
            linked.set()
            release_failure.wait(timeout=3)
            raise OSError("simulated directory fsync failure")

        def first_commit() -> None:
            try:
                with patch.object(self.store, "_fsync_directory", side_effect=fail_fsync):
                    self.store.commit(first_stage)
            except ArtifactStoreError as exc:
                first_errors.append(exc)

        def second_commit() -> None:
            linked.wait(timeout=3)
            second_results.append(second_store.commit(second_stage))

        first_thread = Thread(target=first_commit)
        second_thread = Thread(target=second_commit)
        first_thread.start()
        second_thread.start()
        self.assertTrue(linked.wait(timeout=3))
        release_failure.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

        self.assertEqual(1, len(first_errors))
        self.assertEqual("COMMIT_DURABILITY_UNKNOWN", first_errors[0].code)
        self.assertEqual(1, len(second_results))
        self.assertEqual(b"same-content", second_store.read("case/durable-race.bin"))
        self.assert_no_temporary_objects()

    def test_commit_is_idempotent_but_never_overwrites_different_content(self) -> None:
        original = self.store.put("case/result.bin", b"first")
        repeated = self.store.put("case/result.bin", [b"fi", b"rst"])

        self.assertEqual(original, repeated)
        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.put("case/result.bin", b"second")
        self.assertEqual("OBJECT_EXISTS", raised.exception.code)
        self.assertEqual(b"first", self.store.read("case/result.bin"))
        mode = (self.root / "objects" / "case" / "result.bin").stat().st_mode
        self.assertEqual(0, mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        self.assert_no_temporary_objects()

    def test_new_directory_chain_is_fsynced_from_parent_to_leaf(self) -> None:
        fsynced: list[Path] = []
        original = self.store._fsync_directory

        def record(path: Path) -> None:
            fsynced.append(path)
            original(path)

        with patch.object(self.store, "_fsync_directory", side_effect=record):
            self.store.put("new-case/new-type/result.bin", b"durable")

        objects = self.root / "objects"
        self.assertIn(objects, fsynced)
        self.assertIn(objects / "new-case", fsynced)
        self.assertIn(objects / "new-case" / "new-type", fsynced)

    def test_delete_temp_is_safe_and_idempotent(self) -> None:
        staged = self.store.stage("case/discarded.bin", b"discard me")
        self.assertTrue(self.store.delete_temp(staged))
        self.assertFalse(self.store.delete_temp(staged))
        self.assertFalse(self.store.exists("case/discarded.bin"))
        self.assert_no_temporary_objects()

    def test_startup_sweeps_old_orphans_but_preserves_recent_and_active_temps(self) -> None:
        temporary_namespace = self.root / ".tmp"
        orphan = temporary_namespace / "crashed-upload.part"
        orphan.write_bytes(b"partial")
        two_days_ago = time.time() - (2 * 86_400)
        os.utime(orphan, (two_days_ago, two_days_ago))
        recent = temporary_namespace / "recent-other-process.part"
        recent.write_bytes(b"still-live")

        second_store = LocalArtifactStore(self.root)

        self.assertFalse(orphan.exists())
        self.assertTrue(recent.exists())
        recent.unlink()
        active = second_store.stage("case/active.bin", b"active")
        self.assertEqual(0, second_store.cleanup_orphans(older_than_seconds=0))
        self.assertTrue(second_store.delete_temp(active))

    def test_cleanup_does_not_remove_a_temp_while_stage_is_streaming(self) -> None:
        streaming = Event()
        release = Event()
        staged_results = []
        errors: list[BaseException] = []

        def slow_payload():
            streaming.set()
            release.wait(timeout=3)
            yield b"completed-after-cleanup"

        def stage() -> None:
            try:
                staged_results.append(self.store.stage("case/slow.bin", slow_payload()))
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=stage)
        thread.start()
        self.assertTrue(streaming.wait(timeout=3))
        self.assertEqual(0, self.store.cleanup_orphans(older_than_seconds=0))
        release.set()
        thread.join(timeout=5)
        self.assertEqual([], errors)
        self.assertEqual(1, len(staged_results))
        self.assertTrue(self.store.delete_temp(staged_results[0]))

    def test_orphan_cleanup_rejects_invalid_age(self) -> None:
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ArtifactStoreError) as raised:
                    self.store.cleanup_orphans(older_than_seconds=value)
                self.assertEqual("INVALID_ARGUMENT", raised.exception.code)

    def test_forged_staging_metadata_is_rejected_without_deleting_real_temp(self) -> None:
        staged = self.store.stage("case/real.bin", b"real")
        forged = StagedObject(staged.upload_id, staged.object_key, "0" * 64, staged.size_bytes)

        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.commit(forged)
        self.assertEqual("INVALID_TEMP", raised.exception.code)
        self.assertTrue(self.store.delete_temp(staged))

    def test_missing_read_is_structured_and_exists_is_false(self) -> None:
        self.assertFalse(self.store.exists("case/missing.bin"))
        with self.assertRaises(ArtifactStoreError) as raised:
            self.store.read("case/missing.bin")
        self.assertEqual("NOT_FOUND", raised.exception.code)

    def test_concurrent_same_content_commits_are_idempotent_and_clean(self) -> None:
        payload = os.urandom(64 * 1024)

        def upload(_: int):
            return self.store.put("case/concurrent.bin", payload)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(upload, range(24)))

        self.assertEqual(1, len(set(results)))
        self.assertEqual(payload, self.store.read("case/concurrent.bin"))
        self.assert_no_temporary_objects()

    def test_concurrent_different_content_has_one_winner_and_no_overwrite(self) -> None:
        payloads = (b"writer-a" * 1024, b"writer-b" * 1024)

        def upload(payload: bytes):
            try:
                return ("ok", self.store.put("case/race.bin", payload))
            except ArtifactStoreError as exc:
                return (exc.code, None)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(upload, payloads))

        self.assertEqual(1, sum(status == "ok" for status, _ in outcomes))
        self.assertEqual(1, sum(status == "OBJECT_EXISTS" for status, _ in outcomes))
        self.assertIn(self.store.read("case/race.bin"), payloads)
        self.assert_no_temporary_objects()

    def test_performance_smoke_streams_multi_megabyte_object(self) -> None:
        chunk = b"0123456789abcdef" * 4096  # 64 KiB
        started = time.perf_counter()
        stored = self.store.put("perf/four-megabytes.bin", (chunk for _ in range(64)))
        elapsed = time.perf_counter() - started

        self.assertEqual(4 * 1024 * 1024, stored.size_bytes)
        # Generous smoke bound: catches accidental quadratic behavior without
        # turning shared/CI machine scheduling into a flaky benchmark.
        self.assertLess(elapsed, 10.0)
        self.assert_no_temporary_objects()


if __name__ == "__main__":
    unittest.main()
