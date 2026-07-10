"""Secure local object-storage boundary for immutable artifacts.

This module is deliberately an internal service boundary.  ``StagedObject``
and ``StoredObject`` contain only opaque identifiers and integrity metadata;
they never contain a filesystem path or payload bytes and therefore are safe
to pass between deterministic services.  Payloads returned by ``read`` are
for trusted tool/runtime code only and must not be serialized into an LLM or
public API request.

The local implementation follows the same lifecycle expected from S3/MinIO:

1. stream bytes into a private temporary object;
2. calculate SHA-256 and size while streaming and fsync the object;
3. atomically publish it without ever replacing an existing object;
4. remove the temporary object on success or failure.

Committed objects are content-immutable through this API.  Recommitting the
same key and content is idempotent; recommitting different content raises a
structured conflict and leaves the original bytes untouched.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator, Protocol, runtime_checkable
from uuid import uuid4

try:  # POSIX process-wide lease; the in-process root lock remains the fallback.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]


_ROOT_COMMIT_LOCKS: dict[str, RLock] = {}
_ROOT_COMMIT_LOCKS_GUARD = RLock()


class ArtifactStoreError(RuntimeError):
    """Stable error raised by object-store implementations."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class StagedObject:
    """Opaque handle for a fully written but not yet committed object."""

    upload_id: str
    object_key: str
    checksum_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Integrity metadata for an immutable committed object.

    No local path, URI, or content is included.  A registry may safely retain
    these fields and resolve storage access separately for an authorized tool.
    """

    object_key: str
    checksum_sha256: str
    size_bytes: int


Payload = bytes | bytearray | memoryview | Iterable[bytes]


@runtime_checkable
class ArtifactStore(Protocol):
    """Object-storage operations required by the artifact runtime.

    Implementations may use local disk, S3, MinIO, or another durable store,
    but must preserve staging, immutability, integrity, and cleanup semantics.
    """

    def stage(self, object_key: str, payload: Payload) -> StagedObject:
        """Write and verify a private temporary object."""

    def commit(self, staged: StagedObject) -> StoredObject:
        """Atomically publish a staged object without replacing old content."""

    def put(self, object_key: str, payload: Payload) -> StoredObject:
        """Stage and commit an object, cleaning temporary data on all outcomes."""

    def read(self, object_key: str) -> bytes:
        """Read bytes for trusted deterministic code, never for an LLM API."""

    def exists(self, object_key: str) -> bool:
        """Return whether a regular committed object exists."""

    def delete_temp(self, staged: StagedObject) -> bool:
        """Delete an uncommitted temporary object; return whether it existed."""

    def cleanup_orphans(self, *, older_than_seconds: float = 86_400.0) -> int:
        """Remove stale temporary files and return the number removed."""


@dataclass(frozen=True, slots=True)
class _StagedRecord:
    descriptor: StagedObject
    path: Path


class LocalArtifactStore:
    """Thread-safe local reference implementation of :class:`ArtifactStore`.

    The storage root contains an ``objects`` namespace and a private ``.tmp``
    namespace.  Object keys are canonical POSIX-style relative names.  The
    implementation rejects traversal, absolute paths, non-canonical aliases,
    NUL/control characters, backslashes, and symlink escapes.

    ``os.link`` is used as the commit primitive: creation of the destination is
    atomic and fails if it already exists, so a committed object can never be
    overwritten by a racing writer.  This requires the temporary and committed
    namespaces to be on the same filesystem, which the constructor guarantees
    by placing both below the same root.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_path = Path(root).expanduser()
        if root_path.is_symlink():
            raise ArtifactStoreError("INVALID_ROOT", "storage root must not be a symlink")
        root_path.mkdir(parents=True, exist_ok=True)
        self._root = root_path.resolve(strict=True)
        self._objects_dir = self._prepare_namespace(self._root / "objects", mode=0o750)
        self._temp_dir = self._prepare_namespace(self._root / ".tmp", mode=0o700)
        self._commit_lock_path = self._root / ".commit.lock"
        self._initialize_commit_lock_file()
        with _ROOT_COMMIT_LOCKS_GUARD:
            self._root_commit_lock = _ROOT_COMMIT_LOCKS.setdefault(str(self._root), RLock())
        self._staged: dict[str, _StagedRecord] = {}
        self._active_temp_paths: set[Path] = set()
        self._lock = RLock()
        # A crashed process may leave an untracked ``.part`` file.  Sweep only
        # old entries on startup so a second live store instance sharing this
        # root cannot remove a newly staged upload.
        self.cleanup_orphans()

    def stage(self, object_key: str, payload: Payload) -> StagedObject:
        key = self._validate_key(object_key)
        upload_id = uuid4().hex
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{upload_id}-", suffix=".part", dir=self._temp_dir
        )
        temporary_path = Path(temporary_name)
        with self._lock:
            self._active_temp_paths.add(temporary_path)
        digest = hashlib.sha256()
        size = 0

        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                for chunk in self._iter_chunks(payload):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())

            descriptor = StagedObject(
                upload_id=upload_id,
                object_key=key,
                checksum_sha256=digest.hexdigest(),
                size_bytes=size,
            )
            with self._lock:
                self._staged[upload_id] = _StagedRecord(descriptor, temporary_path)
                self._active_temp_paths.discard(temporary_path)
            return descriptor
        except ArtifactStoreError:
            with self._lock:
                self._active_temp_paths.discard(temporary_path)
            self._unlink_quietly(temporary_path)
            raise
        except Exception as exc:
            with self._lock:
                self._active_temp_paths.discard(temporary_path)
            self._unlink_quietly(temporary_path)
            raise ArtifactStoreError(
                "WRITE_FAILED", "failed to write temporary artifact", retryable=True
            ) from exc

    @property
    def coordination_key(self) -> str:
        return f"local-artifact-store:{self._root}"

    def commit(self, staged: StagedObject) -> StoredObject:
        if not isinstance(staged, StagedObject):
            raise ArtifactStoreError("INVALID_TEMP", "a StagedObject handle is required")

        with self._lock, self._commit_lease():
            record = self._staged.get(staged.upload_id)
            if record is None:
                raise ArtifactStoreError("INVALID_TEMP", "temporary object is unknown or expired")
            if record.descriptor != staged:
                raise ArtifactStoreError("INVALID_TEMP", "temporary object metadata does not match")
            try:
                destination = self._destination(staged.object_key)
                self._create_safe_parent(destination.parent)
                # Make the inode immutable through ordinary file permissions
                # before publishing its destination link, closing the small
                # visibility window that would exist with chmod-after-link.
                record.path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                os.link(record.path, destination)
            except FileExistsError:
                try:
                    existing = self._inspect_committed(destination, staged.object_key)
                    if (
                        existing.checksum_sha256 == staged.checksum_sha256
                        and existing.size_bytes == staged.size_bytes
                    ):
                        try:
                            self._fsync_directory(destination.parent)
                        except OSError as exc:
                            self._remove_stage_locked(staged.upload_id)
                            raise ArtifactStoreError(
                                "COMMIT_DURABILITY_UNKNOWN",
                                "object exists but directory durability could not be confirmed",
                                retryable=True,
                            ) from exc
                        self._remove_stage_locked(staged.upload_id)
                        return existing
                    raise ArtifactStoreError(
                        "OBJECT_EXISTS",
                        "committed object is immutable and contains different content",
                    )
                except ArtifactStoreError:
                    self._remove_stage_locked(staged.upload_id)
                    raise
            except ArtifactStoreError:
                self._remove_stage_locked(staged.upload_id)
                raise
            except Exception as exc:
                self._remove_stage_locked(staged.upload_id)
                raise ArtifactStoreError(
                    "COMMIT_FAILED", "failed to atomically commit artifact", retryable=True
                ) from exc

            try:
                # The mode was set before publication; assert/reapply it in case
                # a platform altered link metadata unexpectedly.
                destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                self._fsync_directory(destination.parent)
            except Exception as exc:
                # The destination is already visible. Removing it would let one
                # writer erase an object another writer has observed. Leave it
                # immutable and let an idempotent retry confirm directory fsync.
                self._remove_stage_locked(staged.upload_id)
                raise ArtifactStoreError(
                    "COMMIT_DURABILITY_UNKNOWN",
                    "artifact is visible but durability confirmation failed",
                    retryable=True,
                ) from exc

            self._remove_stage_locked(staged.upload_id)
            return StoredObject(
                object_key=staged.object_key,
                checksum_sha256=staged.checksum_sha256,
                size_bytes=staged.size_bytes,
            )

    def put(self, object_key: str, payload: Payload) -> StoredObject:
        staged = self.stage(object_key, payload)
        try:
            return self.commit(staged)
        finally:
            # ``commit`` already cleans every terminal outcome.  This protects
            # future implementations and exceptional paths without masking the
            # original error.
            self.delete_temp(staged)

    def read(self, object_key: str) -> bytes:
        key = self._validate_key(object_key)
        path = self._destination(key)
        with self._lock:
            if path.is_symlink() or not path.is_file():
                raise ArtifactStoreError("NOT_FOUND", "committed object was not found")
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
                with os.fdopen(descriptor, "rb") as stream:
                    return stream.read()
            except FileNotFoundError as exc:
                raise ArtifactStoreError("NOT_FOUND", "committed object was not found") from exc
            except OSError as exc:
                raise ArtifactStoreError("READ_FAILED", "failed to read committed object") from exc

    def exists(self, object_key: str) -> bool:
        key = self._validate_key(object_key)
        path = self._destination(key)
        with self._lock:
            return path.is_file() and not path.is_symlink()

    def delete_temp(self, staged: StagedObject) -> bool:
        if not isinstance(staged, StagedObject):
            raise ArtifactStoreError("INVALID_TEMP", "a StagedObject handle is required")
        with self._lock:
            record = self._staged.get(staged.upload_id)
            if record is None:
                return False
            if record.descriptor != staged:
                raise ArtifactStoreError("INVALID_TEMP", "temporary object metadata does not match")
            self._remove_stage_locked(staged.upload_id)
            return True

    def cleanup_orphans(self, *, older_than_seconds: float = 86_400.0) -> int:
        if (
            not isinstance(older_than_seconds, (int, float))
            or not math.isfinite(older_than_seconds)
            or older_than_seconds < 0
        ):
            raise ArtifactStoreError(
                "INVALID_ARGUMENT", "older_than_seconds must be a finite non-negative number"
            )
        cutoff = time.time() - float(older_than_seconds)
        removed = 0
        with self._lock:
            active_paths = {record.path for record in self._staged.values()} | set(
                self._active_temp_paths
            )
            for candidate in self._temp_dir.glob("*.part"):
                if candidate in active_paths:
                    continue
                try:
                    metadata = candidate.lstat()
                    if stat.S_ISDIR(metadata.st_mode) or metadata.st_mtime > cutoff:
                        continue
                    candidate.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
                except OSError:
                    # A production service should emit a cleanup-failure metric;
                    # one inaccessible orphan must not block the rest of a sweep.
                    continue
        return removed

    def _prepare_namespace(self, path: Path, *, mode: int) -> Path:
        if path.is_symlink():
            raise ArtifactStoreError("INVALID_ROOT", "storage namespace must not be a symlink")
        path.mkdir(mode=mode, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ArtifactStoreError("INVALID_ROOT", "storage namespace is not a directory")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self._root):
            raise ArtifactStoreError("INVALID_ROOT", "storage namespace escapes its root")
        path.chmod(mode)
        return resolved

    def _initialize_commit_lock_file(self) -> None:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._commit_lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
        except OSError as exc:
            raise ArtifactStoreError(
                "INVALID_ROOT", "commit lock is not a safe regular file"
            ) from exc
        finally:
            if "descriptor" in locals():
                os.close(descriptor)

    @contextmanager
    def _commit_lease(self) -> Iterator[None]:
        with self._root_commit_lock:
            flags = os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self._commit_lock_path, flags)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            except ArtifactStoreError:
                raise
            except OSError as exc:
                raise ArtifactStoreError(
                    "COMMIT_LOCK_FAILED", "failed to acquire artifact commit lease", retryable=True
                ) from exc
            finally:
                if "descriptor" in locals():
                    if fcntl is not None:
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        except OSError:
                            pass
                    os.close(descriptor)

    @staticmethod
    def _validate_key(object_key: str) -> str:
        if not isinstance(object_key, str):
            raise ArtifactStoreError("INVALID_KEY", "object key must be a string")
        if not object_key or object_key != object_key.strip():
            raise ArtifactStoreError("INVALID_KEY", "object key is empty or non-canonical")
        if object_key.startswith("/") or "\\" in object_key or "//" in object_key:
            raise ArtifactStoreError("INVALID_KEY", "object key must be a relative POSIX path")
        if any(ord(character) < 32 or ord(character) == 127 for character in object_key):
            raise ArtifactStoreError("INVALID_KEY", "object key contains a control character")
        components = object_key.split("/")
        if any(component in {"", ".", ".."} for component in components):
            raise ArtifactStoreError("INVALID_KEY", "object key contains traversal or aliases")
        if any(len(component.encode("utf-8")) > 255 for component in components):
            raise ArtifactStoreError("INVALID_KEY", "object key component is too long")
        return object_key

    @staticmethod
    def _iter_chunks(payload: Payload) -> Iterator[bytes]:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            yield bytes(payload)
            return
        if isinstance(payload, str):
            raise ArtifactStoreError("INVALID_CHUNK", "artifact payload must contain bytes")
        try:
            iterator = iter(payload)
        except TypeError as exc:
            raise ArtifactStoreError("INVALID_CHUNK", "artifact payload is not iterable") from exc
        for chunk in iterator:
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise ArtifactStoreError("INVALID_CHUNK", "artifact chunks must be bytes-like")
            yield bytes(chunk)

    def _destination(self, key: str) -> Path:
        candidate = self._objects_dir.joinpath(*key.split("/"))
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self._objects_dir):
            raise ArtifactStoreError("INVALID_KEY", "object key escapes the storage root")
        return candidate

    def _create_safe_parent(self, parent: Path) -> None:
        relative = parent.relative_to(self._objects_dir)
        current = self._objects_dir
        for component in relative.parts:
            current = current / component
            created = False
            try:
                current.mkdir(mode=0o750)
                created = True
            except FileExistsError:
                pass
            if current.is_symlink() or not current.is_dir():
                raise ArtifactStoreError("INVALID_KEY", "object parent is not a safe directory")
            if not current.resolve(strict=True).is_relative_to(self._objects_dir):
                raise ArtifactStoreError("INVALID_KEY", "object parent escapes the storage root")
            if created:
                # Persist each new directory entry before relying on descendants.
                self._fsync_directory(current.parent)

    def _inspect_committed(self, path: Path, object_key: str) -> StoredObject:
        if path.is_symlink() or not path.is_file():
            raise ArtifactStoreError("OBJECT_EXISTS", "destination is not a regular object")
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise ArtifactStoreError("READ_FAILED", "failed to verify existing object") from exc
        return StoredObject(object_key, digest.hexdigest(), size)

    def _remove_stage_locked(self, upload_id: str) -> None:
        record = self._staged.pop(upload_id, None)
        if record is not None:
            self._unlink_quietly(record.path)

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Cleanup is best effort.  Production implementations should emit a
            # metric and run a TTL sweeper for any abandoned temporary object.
            pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
