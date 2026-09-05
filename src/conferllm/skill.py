"""Install the version-matched ConferLLM Agent Skill."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from . import __version__

SkillTarget = Literal["agents", "codex"]

_BUNDLE_FILES = (
    "SKILL.md",
    "agents/openai.yaml",
    "references/errors.md",
    "references/installation.md",
    "references/multimodal.md",
)


def _default_destination(target: SkillTarget) -> Path:
    base = ".agents" if target == "agents" else ".codex"
    return Path.home() / base / "skills" / "conferllm"


def _source_tree_file(relative_path: str) -> Path | None:
    """Return a source-tree bundle file during editable development."""
    repository_root = Path(__file__).resolve().parents[2]
    candidate = repository_root / "skills" / "conferllm" / relative_path
    return candidate if candidate.is_file() else None


def _read_bundle_file(relative_path: str) -> bytes:
    bundled = resources.files("conferllm").joinpath("skill", *relative_path.split("/"))
    if bundled.is_file():
        return bundled.read_bytes()

    source_file = _source_tree_file(relative_path)
    if source_file is not None:
        return source_file.read_bytes()
    raise FileNotFoundError(f"Bundled Skill file is missing: {relative_path}")


def _bundle_payload() -> dict[str, bytes]:
    return {path: _read_bundle_file(path) for path in _BUNDLE_FILES}


def _payload_digest(payload: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(payload):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload[relative_path])
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_payload(destination: Path) -> dict[str, bytes] | None:
    if destination.is_symlink() or not destination.is_dir():
        return None

    payload: dict[str, bytes] = {}
    for relative_path in _BUNDLE_FILES:
        current = destination
        for part in relative_path.split("/"):
            current /= part
            if current.is_symlink():
                return None
        candidate = destination.joinpath(*relative_path.split("/"))
        if not candidate.is_file():
            return None
        payload[relative_path] = candidate.read_bytes()
    return payload


def _validate_destination(destination: Path) -> None:
    """Keep --force confined to a dedicated installation directory."""
    resolved = destination.resolve()
    protected = (Path.home().resolve(), Path.cwd().resolve(), Path(__file__).resolve())
    if any(resolved == path or resolved in path.parents for path in protected):
        raise ValueError(
            "Skill destination must be a dedicated directory, not a home, "
            "workspace, package directory, or their ancestor."
        )


def _write_staged_bundle(stage: Path, payload: dict[str, bytes]) -> None:
    for relative_path, content in payload.items():
        output = stage.joinpath(*relative_path.split("/"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        output.chmod(0o644)
    for directory in sorted(
        (path for path in stage.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o755)
    stage.chmod(0o755)


def install_skill(
    target: SkillTarget = "agents",
    destination: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically install the Skill bundled with this ConferLLM package.

    The result contains paths, package metadata, and copied filenames only. It
    never contains configuration contents or credential values.
    """
    if target not in ("agents", "codex"):
        raise ValueError("target must be 'agents' or 'codex'")
    validated_target = target

    resolved_destination = (
        _default_destination(validated_target)
        if destination is None
        else Path(destination).expanduser()
    )
    _validate_destination(resolved_destination)
    payload = _bundle_payload()
    bundle_digest = _payload_digest(payload)

    if resolved_destination.exists() or resolved_destination.is_symlink():
        installed = _installed_payload(resolved_destination)
        if installed is not None and _payload_digest(installed) == bundle_digest:
            return {
                "schema_version": "conferllm.skill.install.v1",
                "status": "already_installed",
                "message": (
                    f"ConferLLM Skill is already installed at {resolved_destination}."
                ),
                "target": validated_target,
                "destination": str(resolved_destination),
                "package_version": __version__,
                "files": list(_BUNDLE_FILES),
            }
        if not force:
            raise FileExistsError(
                f"Skill destination already exists: {resolved_destination}. "
                "Use force=True to replace it."
            )

    parent = resolved_destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{resolved_destination.name}.stage-", dir=parent)
    )
    backup: Path | None = None
    replaced = resolved_destination.exists() or resolved_destination.is_symlink()

    try:
        _write_staged_bundle(stage, payload)
        if replaced:
            backup = parent / f".{resolved_destination.name}.backup-{uuid.uuid4().hex}"
            resolved_destination.rename(backup)
        try:
            stage.rename(resolved_destination)
        except Exception:
            if backup is not None and not resolved_destination.exists():
                backup.rename(resolved_destination)
            raise
        if backup is not None:
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "schema_version": "conferllm.skill.install.v1",
        "status": "replaced" if replaced else "installed",
        "message": (
            f"ConferLLM Skill {'replaced' if replaced else 'installed'} at "
            f"{resolved_destination}."
        ),
        "target": validated_target,
        "destination": str(resolved_destination),
        "package_version": __version__,
        "files": list(_BUNDLE_FILES),
    }


__all__ = ["SkillTarget", "install_skill"]
