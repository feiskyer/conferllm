"""Persistent JSONL sessions for ConferLLM conversations."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

from .artifacts import (
    ARTIFACT_ID_PATTERN,
    MIME_EXTENSIONS,
    Artifact,
    ArtifactError,
    ArtifactTransaction,
    resolve_relative_path,
    verify_artifact_file,
)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
SESSION_ID_PATTERN = re.compile(r"^(?P<date>\d{8})-(?P<uuid>[0-9a-f]{32})$")

_THREAD_LOCKS: WeakValueDictionary[Path, threading.Lock] = WeakValueDictionary()
_THREAD_LOCKS_GUARD = threading.Lock()


class SessionError(ValueError):
    """Raised when session input or persisted session data is invalid."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionMetadata:
    """Immutable metadata stored in the first line of a session file."""

    session_id: str
    name: str
    model: str
    created_at: datetime

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible session metadata."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class LoadedSession:
    """A validated session and its reconstructed conversation history."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]]
    next_turn: int
    schema_version: int = SCHEMA_VERSION
    turns: list[dict[str, Any]] = field(default_factory=list, compare=False)
    artifacts: list[Artifact] = field(default_factory=list, compare=False)

    def artifact(self, artifact_id: str) -> Artifact | None:
        """Return referenced artifact metadata by ID."""
        return next((item for item in self.artifacts if item.id == artifact_id), None)


@dataclass(frozen=True)
class SessionListWarning:
    """A non-fatal problem encountered while listing session headers."""

    session_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible warning."""
        return {"session_id": self.session_id, "message": self.message}


@dataclass(frozen=True)
class SessionListResult:
    """Valid session headers plus non-fatal listing warnings."""

    sessions: list[SessionMetadata]
    warnings: list[SessionListWarning]


def generate_session_id(now: datetime | None = None) -> str:
    """Generate a session ID using the supplied or current local date."""
    current = now if now is not None else datetime.now().astimezone()
    return f"{current:%Y%m%d}-{uuid.uuid4().hex}"


def derive_session_name(message: str, explicit_name: str | None = None) -> str:
    """Validate an explicit name or derive one from the first text message."""
    if explicit_name is not None:
        collapsed_name = _collapse_whitespace(explicit_name)
        if not collapsed_name:
            raise SessionError("Session name must not be empty.")
        return collapsed_name

    collapsed_message = _collapse_whitespace(message)
    if not collapsed_message:
        return "Untitled session"
    if len(collapsed_message) > 60:
        return f"{collapsed_message[:60]}…"
    return collapsed_message


class SessionStore:
    """Store ConferLLM sessions as permission-restricted JSONL files."""

    generate_session_id = staticmethod(generate_session_id)
    derive_session_name = staticmethod(derive_session_name)

    def __init__(self, root: Path | None = None) -> None:
        """Initialize a session store, creating its root when necessary."""
        conferllm_root = Path.home() / ".conferllm"
        if root is None or root.expanduser() == conferllm_root / "sessions":
            self._ensure_directory(conferllm_root)
            root = conferllm_root / "sessions"

        expanded_root = root.expanduser()
        if expanded_root.is_symlink():
            raise SessionError(
                f"Session directory must not be a symlink: {expanded_root}"
            )
        self.root = expanded_root.resolve(strict=False)
        self._ensure_directory(self.root)

    def session_path(self, session_id: str) -> Path:
        """Return the validated JSONL path for a session."""
        session_date = self._validate_session_id(session_id)
        candidate = (
            self.root
            / f"{session_date.year:04d}"
            / f"{session_date.month:02d}"
            / f"{session_date.day:02d}"
            / f"{session_id}.jsonl"
        )
        self._validate_resolved_path(candidate, session_id)
        return candidate

    def assets_path(self, session_id: str) -> Path:
        """Return the validated persistent-assets path for a session."""
        session_file = self.session_path(session_id)
        candidate = session_file.with_name(f"{session_id}.assets")
        self._validate_resolved_path(candidate, session_id)
        return candidate

    def artifact_transaction(
        self,
        session_id: str,
        turn: int,
    ) -> ArtifactTransaction:
        """Create a private per-turn artifact staging transaction."""
        session_file = self.session_path(session_id)
        self._ensure_date_directories(session_file.parent)
        try:
            return ArtifactTransaction(
                session_id=session_id,
                turn=turn,
                asset_root=self.assets_path(session_id),
                staging_parent=session_file.parent,
            )
        except ArtifactError as exc:
            raise SessionError(str(exc), code="storage_error") from exc

    def discard_uncommitted_artifacts(self, loaded: LoadedSession) -> None:
        """Recover only the next uncommitted turn while its session lock is held.

        A crash between installing artifacts and replacing JSONL can leave this
        directory behind. The validated JSONL is the commit record; never remove
        any earlier directory or attempt to repair a corrupt session.
        """
        session_id = loaded.metadata.session_id
        candidate = self.assets_path(session_id) / f"turn-{loaded.next_turn:04d}"
        self._validate_resolved_path(candidate, session_id)
        if not candidate.exists():
            return
        if self.load_session(session_id).next_turn != loaded.next_turn:
            raise SessionError(
                "Cannot recover artifacts using a stale session snapshot."
            )
        try:
            if not candidate.is_dir():
                raise OSError("uncommitted artifact path is not a directory")
            shutil.rmtree(candidate)
            self._fsync_directory(candidate.parent)
        except OSError as exc:
            raise SessionError(
                f"Unable to recover uncommitted artifacts for '{session_id}': {exc}",
                code="storage_error",
            ) from exc

    def begin_artifact_transaction(
        self,
        session_id: str,
        turn: int,
    ) -> ArtifactTransaction:
        """Alias with an action-oriented name for chat-service integrations."""
        return self.artifact_transaction(session_id, turn)

    @contextmanager
    def lock(self, session_id: str) -> Iterator[None]:
        """Hold an exclusive cross-process lock for one session."""
        session_file = self.session_path(session_id)
        self._ensure_date_directories(session_file.parent)
        lock_path = session_file.with_name(f"{session_id}.lock")
        self._validate_resolved_path(lock_path, session_id)

        thread_lock = self._thread_lock_for(lock_path)
        with thread_lock:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                )
            except OSError as exc:
                raise SessionError(
                    f"Unable to open lock for session '{session_id}': {exc}",
                    code="storage_error",
                ) from exc

            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("session lock must be a regular file")
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                os.close(descriptor)
                raise SessionError(
                    f"Unable to prepare lock for session '{session_id}': {exc}",
                    code="storage_error",
                ) from exc

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                os.close(descriptor)
                raise SessionError(
                    f"Unable to lock session '{session_id}': {exc}",
                    code="storage_error",
                ) from exc

            try:
                yield
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def create_session(
        self,
        session_id: str,
        name: str,
        model: str,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        response_id: str | None = None,
        usage: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        artifacts: list[Artifact] | tuple[Artifact, ...] | None = None,
        artifact_transaction: ArtifactTransaction | None = None,
    ) -> SessionMetadata:
        """Atomically create a session containing its first complete turn."""
        session_file = self.session_path(session_id)
        normalized_name = derive_session_name("", explicit_name=name)
        normalized_model = self._validate_model(model)
        timestamp = self._normalize_datetime(created_at)
        if timestamp.date() != self._validate_session_id(session_id):
            raise SessionError("Session creation date must match the date in its ID.")
        turn_artifacts = self._normalize_artifacts(
            artifacts,
            artifact_transaction,
            session_id=session_id,
            turn=1,
        )
        self._validate_message(user_message, "user", session_id, 2)
        self._validate_message(assistant_message, "assistant", session_id, 2)
        self._validate_image_references(
            user_message,
            assistant_message,
            turn_artifacts,
            session_id,
            2,
        )
        self._validate_optional_turn_fields(
            response_id,
            usage,
            session_id,
            2,
        )

        metadata = SessionMetadata(
            session_id=session_id,
            name=normalized_name,
            model=normalized_model,
            created_at=timestamp,
        )
        header = {
            "type": "session",
            "schema_version": SCHEMA_VERSION,
            **metadata.to_dict(),
        }
        turn = self._turn_record(
            turn=1,
            user_message=user_message,
            assistant_message=assistant_message,
            response_id=response_id,
            usage=usage,
            created_at=timestamp,
            schema_version=SCHEMA_VERSION,
            artifacts=turn_artifacts,
        )
        payload = self._serialize_records(session_id, header, turn)

        self._ensure_date_directories(session_file.parent)
        temporary_path: Path | None = None
        descriptor: int | None = None
        session_linked = False
        try:
            if artifact_transaction is not None:
                artifact_transaction.install()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{session_id}.",
                suffix=".tmp",
                dir=session_file.parent,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(
                descriptor,
                mode="w",
                encoding="utf-8",
                newline="\n",
            )
            descriptor = None  # The stream now owns the descriptor.
            with stream as session_stream:
                session_stream.write(payload)
                session_stream.flush()
                os.fsync(session_stream.fileno())

            try:
                os.link(temporary_path, session_file)
            except FileExistsError as exc:
                raise SessionError(f"Session '{session_id}' already exists.") from exc

            session_linked = True
            self._fsync_directory(session_file.parent)
            if artifact_transaction is not None:
                artifact_transaction.finalize()
        except SessionError:
            if session_linked:
                with suppress(FileNotFoundError):
                    session_file.unlink()
            if artifact_transaction is not None:
                artifact_transaction.rollback()
            raise
        except ArtifactError as exc:
            if session_linked:
                with suppress(FileNotFoundError):
                    session_file.unlink()
            if artifact_transaction is not None:
                artifact_transaction.rollback()
            raise SessionError(str(exc), code="storage_error") from exc
        except OSError as exc:
            if session_linked:
                with suppress(FileNotFoundError):
                    session_file.unlink()
            if artifact_transaction is not None:
                artifact_transaction.rollback()
            raise SessionError(
                f"Unable to create session '{session_id}': {exc}",
                code="storage_error",
            ) from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()

        return metadata

    def load_session(self, session_id: str) -> LoadedSession:
        """Load and validate a session, reconstructing its message history."""
        session_file = self.session_path(session_id)
        try:
            session_stream = session_file.open("rb")
        except FileNotFoundError as exc:
            raise SessionError(
                f"Session '{session_id}' does not exist.", code="session_not_found"
            ) from exc
        except OSError as exc:
            raise SessionError(
                f"Unable to read session '{session_id}': {exc}", code="storage_error"
            ) from exc

        with session_stream:
            first_line = session_stream.readline()
            if not first_line:
                raise self._corruption(session_id, 1, "missing session header")

            header = self._parse_record(first_line, session_id, 1)
            schema_version = self._schema_version_from_record(header, session_id, 1)
            metadata = self._metadata_from_record(header, session_id, 1)
            messages: list[dict[str, Any]] = []
            turns: list[dict[str, Any]] = []
            artifacts: list[Artifact] = []
            expected_turn = 1

            for line_number, raw_line in enumerate(session_stream, start=2):
                record = self._parse_record(raw_line, session_id, line_number)
                self._validate_turn_record(
                    record,
                    session_id,
                    line_number,
                    expected_turn,
                    schema_version,
                )
                turn_artifacts = self._artifacts_from_turn(
                    record,
                    session_id,
                    line_number,
                    schema_version,
                )
                if schema_version == SCHEMA_VERSION:
                    self._validate_image_references(
                        record["user"],
                        record["assistant"],
                        turn_artifacts,
                        session_id,
                        line_number,
                    )
                messages.extend([record["user"], record["assistant"]])
                turns.append(record)
                artifacts.extend(turn_artifacts)
                expected_turn += 1

        if expected_turn == 1:
            raise self._corruption(session_id, 2, "missing first turn")

        return LoadedSession(
            metadata=metadata,
            messages=messages,
            next_turn=expected_turn,
            schema_version=schema_version,
            turns=turns,
            artifacts=artifacts,
        )

    def append_turn(
        self,
        session_id: str,
        turn: int,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        response_id: str | None = None,
        usage: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        artifacts: list[Artifact] | tuple[Artifact, ...] | None = None,
        artifact_transaction: ArtifactTransaction | None = None,
    ) -> None:
        """Atomically add one complete turn; the caller must hold the session lock."""
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
            raise SessionError("Turn number must be a positive integer.")

        loaded = self.load_session(session_id)
        if turn != loaded.next_turn:
            raise SessionError(
                f"Session '{session_id}' expected turn {loaded.next_turn}, "
                f"received {turn}."
            )

        line_number = turn + 1
        turn_artifacts = self._normalize_artifacts(
            artifacts,
            artifact_transaction,
            session_id=session_id,
            turn=turn,
        )
        self._validate_message(user_message, "user", session_id, line_number)
        self._validate_message(
            assistant_message,
            "assistant",
            session_id,
            line_number,
        )
        if loaded.schema_version == SCHEMA_VERSION:
            self._validate_image_references(
                user_message,
                assistant_message,
                turn_artifacts,
                session_id,
                line_number,
            )
        self._validate_optional_turn_fields(
            response_id,
            usage,
            session_id,
            line_number,
        )
        record = self._turn_record(
            turn=turn,
            user_message=user_message,
            assistant_message=assistant_message,
            response_id=response_id,
            usage=usage,
            created_at=self._normalize_datetime(created_at),
            schema_version=loaded.schema_version,
            artifacts=turn_artifacts,
        )
        serialized = self._serialize_records(session_id, record).encode("utf-8")
        session_file = self.session_path(session_id)
        temporary_path: Path | None = None
        descriptor: int | None = None

        try:
            if artifact_transaction is not None:
                artifact_transaction.install()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{session_id}.",
                suffix=".tmp",
                dir=session_file.parent,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)

            last_byte = b""
            with session_file.open("rb") as session_stream:
                while chunk := session_stream.read(1024 * 1024):
                    self._write_all(descriptor, chunk)
                    last_byte = chunk[-1:]
            # JSONL permits a complete final JSON value without a newline.
            if last_byte != b"\n":
                self._write_all(descriptor, b"\n")
            self._write_all(descriptor, serialized)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None

            os.replace(temporary_path, session_file)
            temporary_path = None
            self._fsync_directory(session_file.parent)
            if artifact_transaction is not None:
                artifact_transaction.finalize()
        except ArtifactError as exc:
            if artifact_transaction is not None:
                artifact_transaction.rollback()
            raise SessionError(str(exc), code="storage_error") from exc
        except OSError as exc:
            if artifact_transaction is not None:
                artifact_transaction.rollback()
            raise SessionError(
                f"Unable to append turn to session '{session_id}': {exc}",
                code="storage_error",
            ) from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()

    def list_sessions(
        self,
        query: str | None = None,
        model: str | None = None,
        since: date | None = None,
        until: date | None = None,
        limit: int = 50,
    ) -> list[SessionMetadata]:
        """List matching session headers newest first."""
        return self.list_sessions_detailed(
            query=query,
            model=model,
            since=since,
            until=until,
            limit=limit,
        ).sessions

    def list_sessions_detailed(
        self,
        query: str | None = None,
        model: str | None = None,
        since: date | None = None,
        until: date | None = None,
        limit: int = 50,
    ) -> SessionListResult:
        """List valid headers and report corrupt ones without aborting."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise SessionError("Session limit must be a non-negative integer.")
        if since is not None and until is not None and since > until:
            raise SessionError("since must be on or before until.")

        normalized_query = query.casefold() if query is not None else None
        sessions: list[SessionMetadata] = []
        warnings: list[SessionListWarning] = []

        for session_file in self._iter_session_files(since, until):
            session_id = session_file.stem
            try:
                metadata = self._read_metadata(session_file, session_id)
            except SessionError as exc:
                warnings.append(
                    SessionListWarning(session_id=session_id, message=str(exc))
                )
                continue
            if normalized_query is not None:
                searchable = f"{metadata.session_id}\n{metadata.name}".casefold()
                if normalized_query not in searchable:
                    continue
            if model is not None and metadata.model != model:
                continue
            created_date = metadata.created_at.date()
            if since is not None and created_date < since:
                continue
            if until is not None and created_date > until:
                continue
            sessions.append(metadata)

        sessions.sort(
            key=lambda metadata: (metadata.created_at, metadata.session_id),
            reverse=True,
        )
        if limit:
            sessions = sessions[:limit]
        return SessionListResult(sessions=sessions, warnings=warnings)

    def resolve_artifact(
        self,
        session_id: str,
        artifact_id: str,
        *,
        verify: bool = True,
    ) -> Path:
        """Resolve a referenced artifact, optionally checking size and hash."""
        loaded = self.load_session(session_id)
        artifact = loaded.artifact(artifact_id)
        if artifact is None:
            legacy_path = self._legacy_artifact_path(loaded, artifact_id)
            if legacy_path is not None:
                if verify and (legacy_path.is_symlink() or not legacy_path.is_file()):
                    raise SessionError(
                        f"Unable to read artifact '{artifact_id}': "
                        "artifact is not a regular file"
                    )
                return legacy_path
            raise SessionError(
                f"Artifact '{artifact_id}' is not referenced by session '{session_id}'.",
                code="artifact_not_found",
            )
        asset_root = self.assets_path(session_id)
        if asset_root.is_symlink():
            raise SessionError(
                f"Artifact directory must not be a symlink: {asset_root}"
            )
        try:
            path = resolve_relative_path(asset_root, artifact.relative_path)
            if verify:
                verify_artifact_file(path, artifact)
        except ArtifactError as exc:
            raise SessionError(str(exc), code=exc.code) from exc
        return path

    def read_artifact(self, session_id: str, artifact_id: str) -> bytes:
        """Read and integrity-check a referenced session artifact."""
        return self.read_loaded_artifact(self.load_session(session_id), artifact_id)

    def read_loaded_artifact(self, loaded: LoadedSession, artifact_id: str) -> bytes:
        """Read an artifact using an already validated, immutable session snapshot."""
        session_id = loaded.metadata.session_id
        artifact = loaded.artifact(artifact_id)
        if artifact is None:
            legacy_path = self._legacy_artifact_path(loaded, artifact_id)
            if legacy_path is not None:
                try:
                    if legacy_path.is_symlink() or not legacy_path.is_file():
                        raise OSError("artifact is not a regular file")
                    return legacy_path.read_bytes()
                except OSError as exc:
                    raise SessionError(
                        f"Unable to read artifact '{artifact_id}': {exc}",
                        code="artifact_not_found",
                    ) from exc
            raise SessionError(
                f"Artifact '{artifact_id}' is not referenced by session '{session_id}'.",
                code="artifact_not_found",
            )
        asset_root = self.assets_path(session_id)
        if asset_root.is_symlink():
            raise SessionError(
                f"Artifact directory must not be a symlink: {asset_root}"
            )
        try:
            path = resolve_relative_path(asset_root, artifact.relative_path)
            return verify_artifact_file(path, artifact)
        except ArtifactError as exc:
            raise SessionError(str(exc), code=exc.code) from exc

    def _legacy_artifact_path(
        self,
        loaded: LoadedSession,
        artifact_id: str,
    ) -> Path | None:
        """Resolve a canonical path explicitly referenced by a v1 message."""
        if loaded.schema_version != 1:
            return None
        match = ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
        if match is None or int(match.group("turn")) < 1:
            return None
        relative_stem = (
            f"turn-{match.group('turn')}/"
            f"{match.group('direction')}-{match.group('number')}"
        )
        asset_root = self.assets_path(loaded.metadata.session_id)
        if asset_root.is_symlink():
            raise SessionError(
                f"Artifact directory must not be a symlink: {asset_root}"
            )
        expected_prefix = str(asset_root / relative_stem)
        allowed_suffixes = tuple(MIME_EXTENSIONS.values())
        for value in self._iter_string_values(loaded.messages):
            start = value.find(expected_prefix)
            while start >= 0:
                for suffix in allowed_suffixes:
                    end = start + len(expected_prefix) + len(suffix)
                    candidate_text = value[start:end]
                    if candidate_text == f"{expected_prefix}{suffix}":
                        if end < len(value) and (
                            value[end].isalnum() or value[end] in {".", "/", "_", "-"}
                        ):
                            continue
                        try:
                            candidate = Path(candidate_text)
                            self._validate_resolved_path(
                                candidate,
                                loaded.metadata.session_id,
                            )
                        except (OSError, SessionError):
                            return None
                        return candidate
                start = value.find(expected_prefix, start + 1)
        return None

    @staticmethod
    def _iter_string_values(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from SessionStore._iter_string_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from SessionStore._iter_string_values(item)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        try:
            if path.is_symlink():
                raise SessionError(f"Session directory must not be a symlink: {path}")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not path.is_dir():
                raise SessionError(f"Session directory is not a directory: {path}")
            os.chmod(path, 0o700)
        except SessionError:
            raise
        except OSError as exc:
            raise SessionError(
                f"Unable to prepare session directory '{path}': {exc}",
                code="storage_error",
            ) from exc

    def _ensure_date_directories(self, leaf: Path) -> None:
        current = self.root
        try:
            relative_parts = leaf.relative_to(self.root).parts
        except ValueError as exc:
            raise SessionError(
                f"Session directory resolves outside root '{self.root}'."
            ) from exc

        for part in relative_parts:
            current /= part
            self._ensure_directory(current)

    @staticmethod
    def _thread_lock_for(lock_path: Path) -> threading.Lock:
        key = lock_path.resolve(strict=False)
        with _THREAD_LOCKS_GUARD:
            return _THREAD_LOCKS.setdefault(key, threading.Lock())

    @staticmethod
    def _validate_session_id(session_id: str) -> date:
        if not isinstance(session_id, str):
            raise SessionError("Session ID must be a string.")
        match = SESSION_ID_PATTERN.fullmatch(session_id)
        if match is None:
            raise SessionError(f"Invalid session ID '{session_id}'.")
        try:
            return datetime.strptime(match.group("date"), "%Y%m%d").date()
        except ValueError as exc:
            raise SessionError(f"Invalid session ID '{session_id}'.") from exc

    def _validate_resolved_path(self, path: Path, session_id: str) -> None:
        # Check before resolve(), which would otherwise hide symlinks that point
        # inside the root (including a lock pointing at an unrelated file).
        current = self.root
        for part in path.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                raise SessionError(
                    f"Session '{session_id}' path must not contain a symlink.",
                    code="storage_error",
                )
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SessionError(
                f"Session '{session_id}' resolves outside the session root."
            ) from exc

    @staticmethod
    def _validate_model(model: str) -> str:
        if not isinstance(model, str) or not model.strip():
            raise SessionError("Session model must not be empty.")
        return model

    @staticmethod
    def _normalize_datetime(value: datetime | None) -> datetime:
        timestamp = value if value is not None else datetime.now().astimezone()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.astimezone()
        return timestamp

    @staticmethod
    def _parse_datetime(
        value: Any,
        session_id: str,
        line_number: int,
    ) -> datetime:
        if not isinstance(value, str):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "created_at must be an ISO 8601 string",
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "invalid created_at timestamp",
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "created_at timestamp must include a UTC offset",
            )
        return parsed

    @staticmethod
    def _validate_message(
        message: Any,
        expected_role: str,
        session_id: str,
        line_number: int,
    ) -> None:
        if not isinstance(message, dict):
            raise SessionStore._corruption(
                session_id,
                line_number,
                f"{expected_role} message must be an object",
            )
        if message.get("role") != expected_role:
            raise SessionStore._corruption(
                session_id,
                line_number,
                f"{expected_role} message has an invalid role",
            )
        if "content" not in message:
            raise SessionStore._corruption(
                session_id,
                line_number,
                f"{expected_role} message is missing content",
            )
        content = message["content"]
        if content is None and expected_role == "assistant":
            return  # Legacy responses may have an empty assistant message.
        if isinstance(content, str):
            return
        if not isinstance(content, list):
            raise SessionStore._corruption(
                session_id,
                line_number,
                f"{expected_role} content must be text or an array",
            )
        for item in content:
            valid = False
            if isinstance(item, dict):
                if item.get("type") == "text":
                    valid = isinstance(item.get("text"), str)
                elif item.get("type") == "image_ref":
                    valid = isinstance(item.get("artifact_id"), str)
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    valid = isinstance(image_url, dict) and isinstance(
                        image_url.get("url"), str
                    )
            if not valid:
                raise SessionStore._corruption(
                    session_id, line_number, f"invalid {expected_role} content block"
                )

    @staticmethod
    def _turn_record(
        turn: int,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        response_id: str | None,
        usage: dict[str, Any] | None,
        created_at: datetime,
        schema_version: int,
        artifacts: list[Artifact],
    ) -> dict[str, Any]:
        record = {
            "type": "turn",
            "turn": turn,
            "created_at": created_at.isoformat(),
            "user": user_message,
            "assistant": assistant_message,
            "response_id": response_id,
            "usage": usage,
        }
        if schema_version == SCHEMA_VERSION:
            record["artifacts"] = [artifact.to_record() for artifact in artifacts]
        return record

    @staticmethod
    def _serialize_records(
        session_id: str,
        *records: dict[str, Any],
    ) -> str:
        try:
            return "".join(
                f"{json.dumps(record, ensure_ascii=False, allow_nan=False)}\n"
                for record in records
            )
        except (TypeError, ValueError) as exc:
            raise SessionError(
                f"Session '{session_id}' contains data that is not valid JSON: {exc}"
            ) from exc

    @staticmethod
    def _parse_record(
        raw_line: str | bytes,
        session_id: str,
        line_number: int,
    ) -> dict[str, Any]:
        if isinstance(raw_line, bytes):
            try:
                raw_line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SessionStore._corruption(
                    session_id,
                    line_number,
                    "record is not valid UTF-8",
                ) from exc
        try:
            record = json.loads(raw_line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "malformed JSON",
            ) from exc
        if not isinstance(record, dict):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "record must be a JSON object",
            )
        return record

    @staticmethod
    def _metadata_from_record(
        record: dict[str, Any],
        session_id: str,
        line_number: int,
    ) -> SessionMetadata:
        if record.get("type") != "session":
            raise SessionStore._corruption(
                session_id,
                line_number,
                "first record must be a session header",
            )
        SessionStore._schema_version_from_record(record, session_id, line_number)
        if record.get("session_id") != session_id:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "header session ID does not match its file",
            )

        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SessionStore._corruption(
                session_id,
                line_number,
                "session name must not be empty",
            )
        model = record.get("model")
        if not isinstance(model, str) or not model.strip():
            raise SessionStore._corruption(
                session_id,
                line_number,
                "session model must not be empty",
            )

        created_at = SessionStore._parse_datetime(
            record.get("created_at"), session_id, line_number
        )
        if created_at.date() != SessionStore._validate_session_id(session_id):
            raise SessionStore._corruption(
                session_id, line_number, "creation date does not match the session ID"
            )
        return SessionMetadata(
            session_id=session_id,
            name=name,
            model=model,
            created_at=created_at,
        )

    @staticmethod
    def _schema_version_from_record(
        record: dict[str, Any],
        session_id: str,
        line_number: int,
    ) -> int:
        schema_version = record.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in SUPPORTED_SCHEMA_VERSIONS
        ):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "unsupported schema version",
            )
        return schema_version

    @staticmethod
    def _validate_turn_record(
        record: dict[str, Any],
        session_id: str,
        line_number: int,
        expected_turn: int,
        schema_version: int,
    ) -> None:
        if record.get("type") != "turn":
            raise SessionStore._corruption(
                session_id,
                line_number,
                "record must be a turn",
            )
        turn = record.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn != expected_turn:
            raise SessionStore._corruption(
                session_id,
                line_number,
                f"expected turn {expected_turn}",
            )
        SessionStore._parse_datetime(
            record.get("created_at"),
            session_id,
            line_number,
        )
        SessionStore._validate_message(
            record.get("user"),
            "user",
            session_id,
            line_number,
        )
        SessionStore._validate_message(
            record.get("assistant"),
            "assistant",
            session_id,
            line_number,
        )
        SessionStore._validate_optional_turn_fields(
            record.get("response_id"),
            record.get("usage"),
            session_id,
            line_number,
        )
        if schema_version == 1 and "artifacts" in record:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "schema version 1 turn must not contain artifacts",
            )
        if schema_version == SCHEMA_VERSION and "artifacts" not in record:
            raise SessionStore._corruption(
                session_id,
                line_number,
                "schema version 2 turn is missing artifacts",
            )

    @staticmethod
    def _artifacts_from_turn(
        record: dict[str, Any],
        session_id: str,
        line_number: int,
        schema_version: int,
    ) -> list[Artifact]:
        if schema_version == 1:
            return []
        raw_artifacts = record.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "artifacts must be an array",
            )
        artifacts: list[Artifact] = []
        seen_ids: set[str] = set()
        try:
            for raw_artifact in raw_artifacts:
                artifact = Artifact.from_record(raw_artifact)
                if artifact.turn != record["turn"]:
                    raise ArtifactError(
                        f"Artifact '{artifact.id}' belongs to a different turn."
                    )
                if artifact.id in seen_ids:
                    raise ArtifactError(
                        f"Artifact '{artifact.id}' is duplicated in the turn."
                    )
                seen_ids.add(artifact.id)
                artifacts.append(artifact)
        except ArtifactError as exc:
            raise SessionStore._corruption(
                session_id,
                line_number,
                str(exc),
            ) from exc
        return artifacts

    def _normalize_artifacts(
        self,
        artifacts: list[Artifact] | tuple[Artifact, ...] | None,
        transaction: ArtifactTransaction | None,
        *,
        session_id: str,
        turn: int,
    ) -> list[Artifact]:
        if artifacts is None:
            normalized = list(transaction.artifacts) if transaction is not None else []
        else:
            normalized = list(artifacts)
            if not all(isinstance(artifact, Artifact) for artifact in normalized):
                raise SessionError("Turn artifacts must be Artifact instances.")
        if transaction is not None:
            if transaction.session_id != session_id or transaction.turn != turn:
                raise SessionError(
                    "Artifact transaction does not match the session and turn."
                )
            if transaction.asset_root != self.assets_path(session_id).resolve(
                strict=False
            ):
                raise SessionError(
                    "Artifact transaction does not use this session's asset root."
                )
            if tuple(normalized) != transaction.artifacts:
                raise SessionError(
                    "Turn artifacts do not match the artifact transaction."
                )

        seen_ids: set[str] = set()
        for artifact in normalized:
            if artifact.turn != turn:
                raise SessionError(
                    f"Artifact '{artifact.id}' does not belong to turn {turn}."
                )
            if artifact.id in seen_ids:
                raise SessionError(f"Artifact '{artifact.id}' is duplicated.")
            seen_ids.add(artifact.id)
        if transaction is None and normalized:
            asset_root = self.assets_path(session_id)
            if asset_root.is_symlink():
                raise SessionError(
                    f"Artifact directory must not be a symlink: {asset_root}"
                )
            try:
                for artifact in normalized:
                    path = resolve_relative_path(
                        asset_root,
                        artifact.relative_path,
                    )
                    verify_artifact_file(path, artifact)
            except ArtifactError as exc:
                raise SessionError(str(exc)) from exc
        return normalized

    @staticmethod
    def _validate_image_references(
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        artifacts: list[Artifact],
        session_id: str,
        line_number: int,
    ) -> None:
        available = {artifact.id for artifact in artifacts}
        for message in (user_message, assistant_message):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    raise SessionStore._corruption(
                        session_id,
                        line_number,
                        "schema version 2 images must use image_ref",
                    )
                if not isinstance(item, dict) or item.get("type") != "image_ref":
                    continue
                artifact_id = item.get("artifact_id")
                if not isinstance(artifact_id, str) or artifact_id not in available:
                    raise SessionStore._corruption(
                        session_id,
                        line_number,
                        f"image_ref '{artifact_id}' does not reference a turn artifact",
                    )

    @staticmethod
    def _validate_optional_turn_fields(
        response_id: Any,
        usage: Any,
        session_id: str,
        line_number: int,
    ) -> None:
        if response_id is not None and not isinstance(response_id, str):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "response_id must be a string or null",
            )
        if usage is not None and not isinstance(usage, dict):
            raise SessionStore._corruption(
                session_id,
                line_number,
                "usage must be an object or null",
            )

    def _read_metadata(
        self,
        session_file: Path,
        session_id: str,
    ) -> SessionMetadata:
        try:
            with session_file.open("rb") as session_stream:
                first_line = session_stream.readline()
        except OSError as exc:
            raise SessionError(
                f"Unable to read session '{session_id}': {exc}", code="storage_error"
            ) from exc
        if not first_line:
            raise self._corruption(session_id, 1, "missing session header")
        record = self._parse_record(first_line, session_id, 1)
        return self._metadata_from_record(record, session_id, 1)

    def _iter_session_files(
        self,
        since: date | None,
        until: date | None,
    ) -> Iterator[Path]:
        try:
            year_directories = list(self.root.iterdir())
        except OSError as exc:
            raise SessionError(
                f"Unable to list session root '{self.root}': {exc}",
                code="storage_error",
            ) from exc

        for year_directory in year_directories:
            if not self._is_plain_directory(year_directory):
                continue
            if re.fullmatch(r"\d{4}", year_directory.name) is None:
                continue
            for month_directory in year_directory.iterdir():
                if not self._is_plain_directory(month_directory):
                    continue
                if re.fullmatch(r"\d{2}", month_directory.name) is None:
                    continue
                for day_directory in month_directory.iterdir():
                    if not self._is_plain_directory(day_directory):
                        continue
                    if re.fullmatch(r"\d{2}", day_directory.name) is None:
                        continue
                    try:
                        directory_date = date(
                            int(year_directory.name),
                            int(month_directory.name),
                            int(day_directory.name),
                        )
                    except ValueError:
                        continue
                    if since is not None and directory_date < since:
                        continue
                    if until is not None and directory_date > until:
                        continue
                    for session_file in day_directory.iterdir():
                        if session_file.is_symlink() or not session_file.is_file():
                            continue
                        if session_file.suffix != ".jsonl":
                            continue
                        try:
                            id_date = self._validate_session_id(session_file.stem)
                        except SessionError:
                            continue
                        if id_date != directory_date:
                            continue
                        self._validate_resolved_path(
                            session_file,
                            session_file.stem,
                        )
                        yield session_file

    @staticmethod
    def _is_plain_directory(path: Path) -> bool:
        return not path.is_symlink() and path.is_dir()

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("zero-byte write while writing session data")
            remaining = remaining[written:]

    @staticmethod
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

    @staticmethod
    def _corruption(
        session_id: str,
        line_number: int,
        reason: str,
    ) -> SessionError:
        return SessionError(
            f"Session '{session_id}' is corrupt at line {line_number}: {reason}.",
            code="session_corrupt",
        )


def _collapse_whitespace(value: str) -> str:
    if not isinstance(value, str):
        raise SessionError("Session text must be a string.")
    return " ".join(value.split())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
