from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conferllm.doctor import run_doctor
from conferllm.skill import install_skill


def test_doctor_reports_safe_model_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app_dir = tmp_path / ".conferllm"
    app_dir.mkdir(mode=0o700)
    config_path = app_dir / "config.yaml"
    config_path.write_text(
        """
global_system_prompt: "private prompt"
model_list:
  - model_name: vision
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text, image]
      max_input_images: 4
    litellm_params:
      model: openai/example
      api_key: sk-test-secret
      api_base: https://private.example.invalid
""",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    result = run_doctor()
    serialized = json.dumps(result)

    assert result["ok"] is True
    assert result["config"]["parse_state"] == "valid"
    assert result["config"]["model_aliases"] == ["vision"]
    assert result["config"]["models"][0]["capabilities"] == {
        "state": "declared",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text", "image"],
        "max_input_images": 4,
    }
    assert "sk-test-secret" not in serialized
    assert "private prompt" not in serialized
    assert "private.example.invalid" not in serialized
    assert "litellm_params" not in serialized


def test_doctor_does_not_echo_invalid_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app_dir = tmp_path / ".conferllm"
    app_dir.mkdir(mode=0o700)
    config_path = app_dir / "config.yaml"
    config_path.write_text(
        "model_list: [this-is-invalid: sk-do-not-echo\n", encoding="utf-8"
    )
    config_path.chmod(0o600)

    result = run_doctor()
    serialized = json.dumps(result)

    assert result["ok"] is False
    assert result["config"]["parse_state"] == "invalid"
    assert "sk-do-not-echo" not in serialized


def test_doctor_reports_missing_config_and_skill_next_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = run_doctor()
    codes = {step["code"] for step in result["next_steps"]}

    assert result["config"]["parse_state"] == "missing"
    assert "create_config" in codes
    assert "install_skill" in codes


def test_doctor_recognizes_original_project_config_without_reading_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".ai_hub.yaml").write_text("do not read", encoding="utf-8")
    with monkeypatch.context() as guarded:
        guarded.setattr(
            Path, "open", lambda *args, **kwargs: pytest.fail("legacy file read")
        )
        result = run_doctor()
    assert result["config"]["parse_state"] == "legacy_migration_required"


def test_doctor_uses_pip_fallback_when_uv_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("conferllm.doctor._find_conferllm_executable", lambda: None)
    monkeypatch.setattr("conferllm.doctor.shutil.which", lambda command: None)

    result = run_doctor()
    install_step = next(
        step for step in result["next_steps"] if step["code"] == "install_package"
    )

    assert install_step["command"] == "python -m pip install --user conferllm"


def test_doctor_reports_unknown_capabilities_for_legacy_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_list:
  - model_name: text
    litellm_params:
      model: provider/example
      api_key: secret-not-in-report
""",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    result = run_doctor(config_path)

    assert result["config"]["models"][0]["capabilities"]["state"] == "unknown"
    assert "secret-not-in-report" not in json.dumps(result)


def test_doctor_detects_installed_skill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install_skill(target="codex")

    result = run_doctor()

    assert result["skill_installs"]["codex"]["installed"] is True
    assert all(step["code"] != "install_skill" for step in result["next_steps"])


def test_doctor_detects_current_absolute_console_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "bin" / "conferllm"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(executable), "doctor", "--json"])
    monkeypatch.setattr("conferllm.doctor.shutil.which", lambda command: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    result = run_doctor()

    assert result["runtime"]["conferllm_executable"] == str(executable.resolve())
    assert all(step["code"] != "install_package" for step in result["next_steps"])
