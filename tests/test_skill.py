from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from conferllm.skill import _BUNDLE_FILES, _read_bundle_file, install_skill


def test_editable_bundle_uses_canonical_skills_directory(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with (
        patch(
            "conferllm.skill.resources.files", return_value=tmp_path / "missing-bundle"
        ),
        patch(
            "conferllm.skill.__file__", str(repository_root / "src/conferllm/skill.py")
        ),
    ):
        for relative_path in _BUNDLE_FILES:
            assert (
                _read_bundle_file(relative_path)
                == (
                    repository_root / "skills" / "conferllm" / relative_path
                ).read_bytes()
            )
    assert not (repository_root / "SKILL.md").exists()


def test_install_skill_uses_default_agents_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = install_skill()

    destination = tmp_path / ".agents" / "skills" / "conferllm"
    assert result["status"] == "installed"
    assert result["destination"] == str(destination)
    assert (destination / "SKILL.md").is_file()
    assert (destination / "agents" / "openai.yaml").is_file()
    assert (destination / "references" / "installation.md").is_file()
    assert (destination / "references" / "multimodal.md").is_file()
    assert (destination / "references" / "errors.md").is_file()


def test_install_skill_is_idempotent_for_same_bundle(tmp_path: Path) -> None:
    destination = tmp_path / "custom" / "conferllm"

    first = install_skill(destination=destination)
    second = install_skill(destination=destination)

    assert first["status"] == "installed"
    assert second["status"] == "already_installed"


def test_install_skill_refuses_different_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "conferllm"
    destination.mkdir()
    (destination / "user-file.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        install_skill(destination=destination)

    assert (destination / "user-file.txt").read_text(encoding="utf-8") == "keep me"


def test_install_skill_force_replaces_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "conferllm"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    result = install_skill(target="codex", destination=destination, force=True)

    assert result["status"] == "replaced"
    assert not (destination / "old.txt").exists()
    assert (destination / "SKILL.md").is_file()


def test_install_skill_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        install_skill(
            target="other",  # type: ignore[arg-type]
            destination=tmp_path / "conferllm",
        )
