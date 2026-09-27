from __future__ import annotations

import re
import shlex
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import yaml

from conferllm.cli import create_parser
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


def test_installed_skill_references_resolve_within_the_bundle(tmp_path: Path) -> None:
    """Conditional guidance must remain accessible outside the source checkout."""
    destination = tmp_path / "conferllm"
    install_skill(destination=destination)

    for relative_path in _BUNDLE_FILES:
        installed_file = destination / relative_path
        assert installed_file.read_bytes() == _read_bundle_file(relative_path)
        if installed_file.suffix != ".md":
            continue
        for link in re.findall(
            r"\[[^\]]+\]\(([^)\s]+)\)", installed_file.read_text(encoding="utf-8")
        ):
            target = urlsplit(link)
            if target.scheme or not target.path:
                continue
            resolved = (installed_file.parent / target.path).resolve()
            assert resolved.is_relative_to(destination.resolve()), link
            assert resolved.is_file(), f"{relative_path}: {link}"


def test_skill_ui_metadata_matches_its_invocation() -> None:
    """Validate the UI field contract, not the wording of its starting prompt."""
    entrypoint = _read_bundle_file("SKILL.md").decode("utf-8")
    frontmatter = yaml.safe_load(entrypoint.split("---", 2)[1])
    metadata = yaml.safe_load(_read_bundle_file("agents/openai.yaml"))
    interface = metadata["interface"]

    assert interface["display_name"]
    assert 25 <= len(interface["short_description"]) <= 64
    assert f"${frontmatter['name']}" in interface["default_prompt"]
    assert metadata["policy"]["allow_implicit_invocation"] is True


def test_documented_skill_commands_match_the_cli_parser() -> None:
    """Parse published examples without executing tools or loading real config."""
    parser = create_parser()
    parsed_commands = 0
    for relative_path in _BUNDLE_FILES:
        if not relative_path.endswith(".md"):
            continue
        text = _read_bundle_file(relative_path).decode("utf-8")
        for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
            for line in block.replace("\\\n", " ").splitlines():
                command = shlex.split(line, comments=True)
                if not command or command[0] != "conferllm":
                    continue
                try:
                    args = parser.parse_args(command[1:])
                except SystemExit as error:
                    assert error.code == 0, f"{relative_path}: {line}"
                else:
                    assert args.command is not None, f"{relative_path}: {line}"
                parsed_commands += 1

    assert parsed_commands > 0


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
