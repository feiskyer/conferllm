"""Basic local file tools, without read-state or permission gates."""

from __future__ import annotations

import difflib
import fnmatch
import heapq
import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import Field

from .common import MAX_FILE_BYTES, Arguments, resolve_path, truncate


class ReadFileArguments(Arguments):
    path: str = Field(min_length=1)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=2000, ge=1, le=10000)


class WriteFileArguments(Arguments):
    path: str = Field(min_length=1)
    content: str = Field(max_length=MAX_FILE_BYTES)


class EditFileArguments(Arguments):
    path: str = Field(min_length=1)
    old_string: str = Field(max_length=MAX_FILE_BYTES)
    new_string: str = Field(max_length=MAX_FILE_BYTES)
    replace_all: bool = False


class ListDirectoryArguments(Arguments):
    path: str = "."
    ignore: list[str] = Field(default_factory=list)
    limit: int = Field(default=1000, ge=1, le=10000)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _line_ending(text: str) -> str:
    counts = {
        "\n": text.count("\n") - text.count("\r\n"),
        "\r\n": text.count("\r\n"),
        "\r": text.count("\r") - text.count("\r\n"),
    }
    return max(counts, key=lambda ending: counts[ending])


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 10 MiB text operation limit.")
    return data.decode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Content exceeds the 10 MiB text operation limit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


class FileTools:
    """Resolve paths without changing cwd or imposing a workspace boundary."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def read_file(
        self, path: str, offset: int = 1, limit: int = 2000
    ) -> dict[str, Any]:
        target = resolve_path(path, self.cwd)
        lines = _read_text(target).splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        output = "\n".join(
            f"{number:6d}|{line}" for number, line in enumerate(selected, start=offset)
        )
        displayed = truncate(output)
        return {
            "ok": True,
            "path": str(target),
            "output": displayed,
            "total_lines": len(lines),
            "truncated": (
                displayed != output or offset > 1 or offset - 1 + limit < len(lines)
            ),
        }

    def _write(
        self, target: Path, old: str, new: str, *, existed: bool
    ) -> dict[str, Any]:
        # New files keep supplied endings; existing files keep their own style.
        disk_text = (
            _normalize_newlines(new).replace("\n", _line_ending(old))
            if existed
            else new
        )
        data = disk_text.encode("utf-8")
        _atomic_write(target, data)
        diff = difflib.unified_diff(
            _normalize_newlines(old).splitlines(keepends=True),
            _normalize_newlines(new).splitlines(keepends=True),
            fromfile=str(target) if existed else "/dev/null",
            tofile=str(target),
        )
        return {
            "ok": True,
            "path": str(target),
            "bytes_written": len(data),
            "output": truncate("".join(diff)),
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = resolve_path(path, self.cwd)
        existed = target.exists()
        old = _read_text(target) if existed else ""
        return self._write(target, old, content, existed=existed)

    def append_file(self, path: str, content: str) -> dict[str, Any]:
        target = resolve_path(path, self.cwd)
        existed = target.exists()
        old = _read_text(target) if existed else ""
        return self._write(target, old, old + content, existed=existed)

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        target = resolve_path(path, self.cwd)
        existed = target.exists()
        raw = _read_text(target) if existed else ""
        content = _normalize_newlines(raw)
        old = _normalize_newlines(old_string)
        new = _normalize_newlines(new_string)
        if not old:
            if content:
                raise ValueError("Empty old_string can only create an empty/new file.")
        else:
            count = content.count(old)
            if count == 0:
                raise ValueError("old_string was not found in the file.")
            if count > 1 and not replace_all:
                raise ValueError(
                    f"Found {count} matches; provide unique text or set replace_all."
                )
            new = content.replace(old, new, -1 if replace_all else 1)
        return self._write(target, raw, new, existed=existed)

    def list_directory(
        self,
        path: str = ".",
        ignore: list[str] | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        target = resolve_path(path, self.cwd)
        patterns = ignore or []
        entries = heapq.nsmallest(
            limit + 1,
            (
                entry
                for entry in target.iterdir()
                if not any(fnmatch.fnmatchcase(entry.name, p) for p in patterns)
            ),
            key=lambda entry: entry.name,
        )
        lines = []
        for entry in entries[:limit]:
            if entry.is_symlink():
                lines.append(f"[LINK] {entry.name} -> {entry.readlink()}")
            elif entry.is_dir():
                lines.append(f"[DIR]  {entry.name}/")
            else:
                lines.append(f"[FILE] {entry.name} ({entry.stat().st_size} bytes)")
        output = "\n".join(lines) or "Directory is empty."
        displayed = truncate(output)
        return {
            "ok": True,
            "path": str(target),
            "output": displayed,
            "truncated": len(entries) > limit or displayed != output,
        }
