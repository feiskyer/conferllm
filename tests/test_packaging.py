from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import zipfile
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path

import pytest


def test_wheel_contains_complete_version_matched_skill(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the wheel-content test")

    repository_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dist"
    subprocess.run(
        [uv, "build", "--out-dir", str(output_dir)],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(output_dir.glob("conferllm-*.whl"))
    init_source = (repository_root / "src" / "conferllm" / "__init__.py").read_text(
        encoding="utf-8"
    )
    version_match = re.search(r'^__version__ = "([^"]+)"$', init_source, re.MULTILINE)
    assert version_match is not None
    package_version = version_match.group(1)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_directory = f"conferllm-{package_version}.dist-info"
        assert {name.split("/", 1)[0] for name in names} == {
            "conferllm",
            metadata_directory,
        }
        assert {
            "conferllm/__init__.py",
            "conferllm/__main__.py",
            "conferllm/py.typed",
            "conferllm/skill/SKILL.md",
            "conferllm/skill/agents/openai.yaml",
            "conferllm/skill/references/installation.md",
            "conferllm/skill/references/multimodal.md",
            "conferllm/skill/references/errors.md",
        } <= names
        for source, bundled in [
            ("skills/conferllm/SKILL.md", "conferllm/skill/SKILL.md"),
            (
                "skills/conferllm/agents/openai.yaml",
                "conferllm/skill/agents/openai.yaml",
            ),
            *[
                (
                    f"skills/conferllm/references/{name}.md",
                    f"conferllm/skill/references/{name}.md",
                )
                for name in ("installation", "multimodal", "errors")
            ],
        ]:
            assert archive.read(bundled) == (repository_root / source).read_bytes()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        assert metadata["Name"] == "conferllm"
        assert metadata["Version"] == package_version

        entry_points = ConfigParser()
        entry_points.read_string(
            archive.read(f"{metadata_directory}/entry_points.txt").decode("utf-8")
        )
        assert dict(entry_points["console_scripts"]) == {
            "conferllm": "conferllm.cli:main",
        }

    with tarfile.open(next(output_dir.glob("conferllm-*.tar.gz"))) as archive:
        names = set(archive.getnames())
        prefix = f"conferllm-{package_version}/"
        assert prefix + "skills/conferllm/SKILL.md" in names
        assert prefix + "skills/conferllm/agents/openai.yaml" in names
        assert all(
            prefix + f"skills/conferllm/references/{name}.md" in names
            for name in ("installation", "multimodal", "errors")
        )
        source_prefix = prefix + "src/"
        assert {
            name.removeprefix(source_prefix).split("/", 1)[0]
            for name in names
            if name.startswith(source_prefix)
        } == {"conferllm"}
        assert prefix + "SKILL.md" not in names
        assert not any(name.startswith(prefix + "skill_references/") for name in names)
