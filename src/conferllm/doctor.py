"""Safe diagnostics for ConferLLM installations."""

from __future__ import annotations

import shlex
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConferLLMConfig, existing_legacy_configs, get_default_app_dir


def _permission_metadata(path: Path, *, expected_mode: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "expected_mode": f"{expected_mode:04o}",
        "mode": None,
        "secure": None,
    }
    if not path.exists():
        return metadata

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        metadata["readable"] = False
        return metadata

    metadata["mode"] = f"{mode:04o}"
    metadata["secure"] = mode == expected_mode
    return metadata


def _capability_summary(model: Any) -> dict[str, Any]:
    alias = str(getattr(model, "model_name", ""))
    capabilities = getattr(model, "capabilities", None)
    if capabilities is None:
        return {
            "alias": alias,
            "capabilities": {
                "state": "unknown",
                "input_modalities": None,
                "output_modalities": None,
                "max_input_images": None,
            },
        }

    if isinstance(capabilities, dict):
        input_modalities = capabilities.get("input_modalities")
        output_modalities = capabilities.get("output_modalities")
        max_input_images = capabilities.get("max_input_images")
    else:
        input_modalities = getattr(capabilities, "input_modalities", None)
        output_modalities = getattr(capabilities, "output_modalities", None)
        max_input_images = getattr(capabilities, "max_input_images", None)

    return {
        "alias": alias,
        "capabilities": {
            "state": "declared",
            "input_modalities": (
                list(input_modalities) if input_modalities is not None else None
            ),
            "output_modalities": (
                list(output_modalities) if output_modalities is not None else None
            ),
            "max_input_images": max_input_images,
        },
    }


def _skill_install_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "installed": (
            path.joinpath("SKILL.md").is_file()
            and path.joinpath("agents", "openai.yaml").is_file()
            and path.joinpath("references", "installation.md").is_file()
            and path.joinpath("references", "multimodal.md").is_file()
            and path.joinpath("references", "errors.md").is_file()
        ),
    }


def _find_conferllm_executable() -> str | None:
    executable = shutil.which("conferllm")
    if executable is not None:
        return executable

    invoked_path = Path(sys.argv[0])
    if invoked_path.name == "conferllm" and invoked_path.is_file():
        return str(invoked_path.resolve())
    return None


def run_doctor(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return JSON-compatible diagnostics containing safe metadata only."""
    selected_config_path = (
        ConferLLMConfig.get_default_config_path()
        if config_path is None
        else Path(config_path).expanduser()
    )
    app_dir = get_default_app_dir()
    config_state = "missing"
    config: ConferLLMConfig | None = None

    if selected_config_path.exists():
        try:
            config = ConferLLMConfig.load_config(selected_config_path)
            config_state = "valid"
        except Exception:
            config_state = "invalid"
    elif config_path is None and existing_legacy_configs():
        config_state = "legacy_migration_required"

    models = (
        [_capability_summary(model) for model in config.model_list]
        if config is not None
        else []
    )
    sessions_path = (
        config.get_session_root()
        if config is not None
        else get_default_app_dir() / "sessions"
    )

    permissions = {
        "app_directory": _permission_metadata(app_dir, expected_mode=0o700),
        "config_file": _permission_metadata(selected_config_path, expected_mode=0o600),
        "sessions_directory": _permission_metadata(sessions_path, expected_mode=0o700),
    }

    next_steps: list[dict[str, str]] = []
    conferllm_executable = _find_conferllm_executable()
    if conferllm_executable is None:
        install_command = (
            "uv tool install conferllm"
            if shutil.which("uv") is not None
            else "python -m pip install --user conferllm"
        )
        next_steps.append(
            {
                "code": "install_package",
                "message": "Install ConferLLM and verify the executable.",
                "command": install_command,
            }
        )
    if config_state == "missing":
        next_steps.append(
            {
                "code": "create_config",
                "message": (
                    "Create ~/.conferllm/config.yaml from the documented example, "
                    "add provider credentials yourself, then run "
                    "chmod 600 ~/.conferllm/config.yaml."
                ),
                "command": "mkdir -p ~/.conferllm && chmod 700 ~/.conferllm",
            }
        )
    elif config_state == "legacy_migration_required":
        next_steps.append(
            {
                "code": "migrate_config",
                "message": (
                    "Move the legacy configuration to ~/.conferllm/config.yaml "
                    "manually and restrict its permissions."
                ),
                "command": "conferllm doctor --json",
            }
        )
    elif config_state == "invalid":
        next_steps.append(
            {
                "code": "fix_config",
                "message": (
                    "Fix the configuration syntax or schema without sharing "
                    "credential values."
                ),
                "command": "conferllm doctor --json",
            }
        )
    elif not models:
        next_steps.append(
            {
                "code": "configure_model",
                "message": "Add at least one model alias to the configuration.",
                "command": "conferllm models --json",
            }
        )

    for key, metadata in permissions.items():
        if metadata["exists"] and metadata["secure"] is False:
            next_steps.append(
                {
                    "code": f"restrict_{key}",
                    "message": f"Restrict permissions for {metadata['path']}.",
                    "command": (
                        f"chmod {metadata['expected_mode']} "
                        f"{shlex.quote(str(metadata['path']))}"
                    ),
                }
            )

    agents_skill = Path.home() / ".agents" / "skills" / "conferllm"
    codex_skill = Path.home() / ".codex" / "skills" / "conferllm"
    skill_installs = {
        "agents": _skill_install_metadata(agents_skill),
        "codex": _skill_install_metadata(codex_skill),
    }
    if not any(item["installed"] for item in skill_installs.values()):
        next_steps.append(
            {
                "code": "install_skill",
                "message": "Install the bundled ConferLLM Agent Skill.",
                "command": "conferllm skill install",
            }
        )

    return {
        "schema_version": "conferllm.doctor.v1",
        "ok": config_state == "valid" and bool(models),
        "package": {"name": "conferllm", "version": __version__},
        "runtime": {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "python_executable": sys.executable,
            "conferllm_executable": conferllm_executable,
            "platform": sys.platform,
        },
        "config": {
            "path": str(selected_config_path),
            "exists": selected_config_path.exists(),
            "parse_state": config_state,
            "model_aliases": [model["alias"] for model in models],
            "models": models,
        },
        "paths": {
            "app_directory": str(app_dir),
            "sessions_directory": str(sessions_path),
        },
        "permissions": permissions,
        "skill_installs": skill_installs,
        "next_steps": next_steps,
    }


__all__ = ["run_doctor"]
