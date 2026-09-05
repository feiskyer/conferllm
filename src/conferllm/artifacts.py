"""Session-owned image artifacts and per-turn storage transactions."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .images import MIME_EXTENSIONS

ArtifactDirection = Literal["input", "output"]

ARTIFACT_ID_PATTERN = re.compile(
    r"^t(?P<turn>\d{4})-(?P<direction>input|output)-(?P<number>\d{3})$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SESSION_ID_PATTERN = re.compile(r"^\d{8}-[0-9a-f]{32}$")


class ArtifactError(ValueError):
    """Raised when artifact metadata or storage is invalid."""

    def __init__(self, message: str, *, code: str = "storage_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Artifact:
    """Immutable metadata for one session-owned image."""

    id: str
    direction: ArtifactDirection
    index: int
    mime_type: str
    size_bytes: int
    sha256: str
    relative_path: str
    kind: str = "image"

    def __post_init__(self) -> None:
        """Validate all persisted fields and their cross-field relationships."""
        if not isinstance(self.id, str):
            raise ArtifactError("Artifact ID must be a string.")
        match = ARTIFACT_ID_PATTERN.fullmatch(self.id)
        if match is None:
            raise ArtifactError(f"Invalid artifact ID '{self.id}'.")
        if int(match.group("turn")) < 1:
            raise ArtifactError("Artifact ID turn must be between 1 and 9999.")
        if self.direction not in ("input", "output"):
            raise ArtifactError(f"Invalid artifact direction '{self.direction}'.")
        if match.group("direction") != self.direction:
            raise ArtifactError("Artifact ID direction does not match direction.")
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ArtifactError("Artifact index must be an integer.")
        if self.index < 0 or self.index > 998:
            raise ArtifactError("Artifact index must be between 0 and 998.")
        if int(match.group("number")) != self.index + 1:
            raise ArtifactError("Artifact ID number does not match index.")
        if self.kind != "image":
            raise ArtifactError("Artifact kind must be 'image'.")
        if not isinstance(self.mime_type, str) or self.mime_type not in MIME_EXTENSIONS:
            raise ArtifactError(f"Unsupported artifact MIME type '{self.mime_type}'.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ArtifactError("Artifact size_bytes must be a non-negative integer.")
        if (
            not isinstance(self.sha256, str)
            or SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ArtifactError(
                "Artifact sha256 must be 64 lowercase hexadecimal digits."
            )

        relative = _validate_relative_path(self.relative_path)
        expected = PurePosixPath(
            f"turn-{match.group('turn')}",
            f"{self.direction}-{self.index + 1:03d}{MIME_EXTENSIONS[self.mime_type]}",
        )
        if relative != expected:
            raise ArtifactError(
                f"Artifact relative_path must be '{expected.as_posix()}'."
            )

    @property
    def turn(self) -> int:
        """Return the one-based turn encoded in the artifact ID."""
        match = ARTIFACT_ID_PATTERN.fullmatch(self.id)
        assert match is not None
        return int(match.group("turn"))

    def to_record(self) -> dict[str, Any]:
        """Serialize metadata for a schema-v2 JSONL turn record."""
        return {
            "id": self.id,
            "kind": self.kind,
            "direction": self.direction,
            "index": self.index,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_record(cls, value: Any) -> Artifact:
        """Parse and strictly validate one JSON artifact record."""
        if not isinstance(value, dict):
            raise ArtifactError("Artifact record must be an object.")
        required = {
            "id",
            "kind",
            "direction",
            "index",
            "mime_type",
            "size_bytes",
            "sha256",
            "relative_path",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if extra:
                detail.append(f"unexpected {', '.join(extra)}")
            raise ArtifactError(f"Invalid artifact record: {'; '.join(detail)}.")
        return cls(
            id=value["id"],
            kind=value["kind"],
            direction=value["direction"],
            index=value["index"],
            mime_type=value["mime_type"],
            size_bytes=value["size_bytes"],
            sha256=value["sha256"],
            relative_path=value["relative_path"],
        )

    def to_public(
        self,
        session_id: str,
        asset_root: Path,
        *,
        include_local_path: bool = True,
    ) -> dict[str, Any]:
        """Serialize metadata for callers, including a stable resource URI."""
        _validate_session_id(session_id)
        local_path = resolve_relative_path(asset_root, self.relative_path)
        result = {
            "id": self.id,
            "kind": self.kind,
            "direction": self.direction,
            "index": self.index,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "uri": f"conferllm://sessions/{session_id}/artifacts/{self.id}",
        }
        if include_local_path:
            result["local_path"] = str(local_path)
        return result


class ArtifactTransaction:
    """Stage and atomically install all artifacts belonging to one turn."""

    def __init__(
        self,
        *,
        session_id: str,
        turn: int,
        asset_root: Path,
        staging_parent: Path,
    ) -> None:
        _validate_session_id(session_id)
        _validate_turn(turn)
        if asset_root.is_symlink() or staging_parent.is_symlink():
            raise ArtifactError("Artifact directories must not be a symlink.")
        if asset_root.exists() and not asset_root.is_dir():
            raise ArtifactError("Artifact root must be a directory.")
        self.session_id = session_id
        self.turn = turn
        self.asset_root = asset_root.resolve(strict=False)
        self.staging_parent = staging_parent.resolve(strict=False)
        _ensure_directory(self.staging_parent)
        staging_name = tempfile.mkdtemp(
            prefix=f".{session_id}.turn-{turn:04d}.",
            suffix=".staging",
            dir=self.staging_parent,
        )
        self.staging_path = Path(staging_name)
        os.chmod(self.staging_path, 0o700)
        self.final_path = self.asset_root / f"turn-{turn:04d}"
        self._artifacts: list[Artifact] = []
        self._installed = False
        self._finalized = False

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Return staged artifact metadata in staging order."""
        return tuple(self._artifacts)

    @property
    def installed(self) -> bool:
        """Return whether the turn directory has been installed."""
        return self._installed

    def stage_bytes(
        self,
        data: bytes,
        *,
        direction: ArtifactDirection,
        mime_type: str,
        index: int | None = None,
    ) -> Artifact:
        """Write one immutable artifact into the private staging directory."""
        self._ensure_active()
        if not isinstance(data, bytes):
            raise ArtifactError("Artifact data must be bytes.")
        if direction not in ("input", "output"):
            raise ArtifactError(f"Invalid artifact direction '{direction}'.")
        if mime_type not in MIME_EXTENSIONS:
            raise ArtifactError(f"Unsupported artifact MIME type '{mime_type}'.")

        if index is None:
            index = sum(artifact.direction == direction for artifact in self._artifacts)
        _validate_index(index)
        artifact_id = f"t{self.turn:04d}-{direction}-{index + 1:03d}"
        if any(artifact.id == artifact_id for artifact in self._artifacts):
            raise ArtifactError(f"Artifact '{artifact_id}' is already staged.")

        filename = f"{direction}-{index + 1:03d}{MIME_EXTENSIONS[mime_type]}"
        relative_path = f"turn-{self.turn:04d}/{filename}"
        artifact = Artifact(
            id=artifact_id,
            direction=direction,
            index=index,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            relative_path=relative_path,
        )
        destination = self.staging_path / filename
        descriptor: int | None = None
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            _write_all(descriptor, data)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            descriptor = None
            _fsync_directory(self.staging_path)
        except OSError as exc:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(FileNotFoundError):
                destination.unlink()
            self.rollback()
            raise ArtifactError(
                f"Unable to stage artifact '{artifact_id}': {exc}"
            ) from exc

        self._artifacts.append(artifact)
        return artifact

    def stage_file(
        self,
        source: Path,
        *,
        direction: ArtifactDirection,
        mime_type: str,
        index: int | None = None,
    ) -> Artifact:
        """Read a source once and stage its bytes."""
        try:
            expanded_source = source.expanduser()
            if expanded_source.is_symlink():
                raise OSError("source must not be a symlink")
            source_path = expanded_source.resolve(strict=True)
            if not source_path.is_file():
                raise OSError("source is not a regular file")
            data = source_path.read_bytes()
        except OSError as exc:
            self.rollback()
            raise ArtifactError(
                f"Unable to read artifact source '{source}': {exc}"
            ) from exc
        return self.stage_bytes(
            data,
            direction=direction,
            mime_type=mime_type,
            index=index,
        )

    def install(self) -> Path:
        """Atomically move the complete staged turn into canonical storage."""
        if self._finalized:
            raise ArtifactError("Artifact transaction is already finalized.")
        if self._installed:
            return self.final_path
        self._ensure_active()
        try:
            _ensure_directory(self.asset_root)
            _validate_child(self.asset_root, self.final_path)
            if self.final_path.exists() or self.final_path.is_symlink():
                raise ArtifactError(
                    f"Artifact turn directory already exists: {self.final_path}"
                )
            os.replace(self.staging_path, self.final_path)
            self._installed = True
            _fsync_directory(self.asset_root)
        except ArtifactError:
            self.rollback()
            raise
        except OSError as exc:
            self.rollback()
            raise ArtifactError(
                f"Unable to install artifacts for turn {self.turn}: {exc}"
            ) from exc
        return self.final_path

    def finalize(self) -> tuple[Artifact, ...]:
        """Mark an installed transaction as committed."""
        if not self._installed:
            raise ArtifactError("Artifact transaction must be installed first.")
        self._finalized = True
        return self.artifacts

    def rollback(self) -> None:
        """Remove staged or installed files unless the transaction was finalized."""
        if self._finalized:
            return
        candidates = [(self.staging_path, self.staging_parent)]
        if self._installed:
            candidates.append((self.final_path, self.asset_root))
        for candidate, parent in candidates:
            _validate_child(parent, candidate)
            if candidate.is_symlink():
                candidate.unlink()
            elif candidate.exists():
                shutil.rmtree(candidate)
        self._installed = False

    def __enter__(self) -> ArtifactTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None or not self._finalized:
            self.rollback()

    def _ensure_active(self) -> None:
        if self._finalized:
            raise ArtifactError("Artifact transaction is already finalized.")
        if self._installed:
            raise ArtifactError("Artifact transaction is already installed.")
        if not self.staging_path.is_dir() or self.staging_path.is_symlink():
            raise ArtifactError("Artifact staging directory is unavailable.")


# ArtifactBatch is the shorter public name used by callers that build a turn.
ArtifactBatch = ArtifactTransaction


def resolve_relative_path(asset_root: Path, relative_path: str) -> Path:
    """Resolve a validated artifact path without permitting root escape."""
    relative = _validate_relative_path(relative_path)
    root = asset_root.resolve(strict=False)
    candidate = root.joinpath(*relative.parts)
    _validate_child(root, candidate)
    return candidate


def verify_artifact_file(path: Path, artifact: Artifact) -> bytes:
    """Read an artifact and verify its persisted size and SHA-256."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("artifact is not a regular file")
            if metadata.st_size != artifact.size_bytes:
                raise ArtifactError(
                    f"Artifact '{artifact.id}' size does not match its record.",
                    code="session_corrupt",
                )
            data = stream.read(artifact.size_bytes + 1)
    except OSError as exc:
        raise ArtifactError(
            f"Unable to read artifact '{artifact.id}': {exc}",
            code="artifact_not_found"
            if isinstance(exc, FileNotFoundError)
            else "storage_error",
        ) from exc
    if len(data) != artifact.size_bytes:
        raise ArtifactError(
            f"Artifact '{artifact.id}' size does not match its record.",
            code="session_corrupt",
        )
    if hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ArtifactError(
            f"Artifact '{artifact.id}' hash does not match its record.",
            code="session_corrupt",
        )
    return data


def _validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ArtifactError("Artifact relative_path must be a non-empty string.")
    if "\\" in value:
        raise ArtifactError("Artifact relative_path must use POSIX separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2:
        raise ArtifactError(
            "Artifact relative_path must have exactly two relative parts."
        )
    if any(part in ("", ".", "..") for part in path.parts):
        raise ArtifactError("Artifact relative_path contains an unsafe component.")
    if path.as_posix() != value:
        raise ArtifactError("Artifact relative_path is not canonical.")
    return path


def _validate_session_id(session_id: str) -> None:
    if (
        not isinstance(session_id, str)
        or SESSION_ID_PATTERN.fullmatch(session_id) is None
    ):
        raise ArtifactError(f"Invalid session ID '{session_id}'.")


def _validate_turn(turn: int) -> None:
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= 9999:
        raise ArtifactError("Artifact turn must be between 1 and 9999.")


def _validate_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 998:
        raise ArtifactError("Artifact index must be between 0 and 998.")


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ArtifactError(f"Artifact directory must not be a symlink: {path}")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.is_dir():
            raise OSError("path is not a directory")
        os.chmod(path, 0o700)
    except OSError as exc:
        raise ArtifactError(
            f"Unable to prepare artifact directory '{path}': {exc}"
        ) from exc


def _validate_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve(strict=False)
    child_resolved = child.resolve(strict=False)
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise ArtifactError(
            f"Artifact path resolves outside '{parent_resolved}'."
        ) from exc


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("zero-byte write while writing artifact data")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
