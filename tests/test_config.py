"""Unit tests for configuration management."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from conferllm.config import (
    ConferLLMConfig,
    ConfigMigrationError,
    ConfigurationError,
    ImageLimits,
    ModelCapabilities,
    ModelConfig,
    get_default_app_dir,
)


def write_config(path: Path, config_data: dict[str, Any]) -> None:
    """Write YAML configuration data to a test path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")


class TestModelConfig:
    """Test ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            model_name="gpt-4",
            litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
        )
        assert config.model_name == "gpt-4"
        assert config.litellm_params["model"] == "openai/gpt-4"
        assert config.litellm_params["api_key"] == "test-key"
        assert config.system_prompt is None

    def test_model_config_with_system_prompt(self):
        """Test creating a ModelConfig with system prompt."""
        config = ModelConfig(
            model_name="gpt-4",
            litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
            system_prompt="You are a helpful assistant.",
        )
        assert config.model_name == "gpt-4"
        assert config.system_prompt == "You are a helpful assistant."

    def test_model_config_with_declared_capabilities(self):
        """Store explicit multimodal capability metadata."""
        config = ModelConfig(
            model_name="vision",
            litellm_params={"model": "openai/vision"},
            capabilities=ModelCapabilities(
                input_modalities=["text", "image"],
                output_modalities=["text", "image"],
                max_input_images=4,
            ),
        )

        assert config.capabilities is not None
        assert config.capabilities.input_modalities == ["text", "image"]
        assert config.capabilities.max_input_images == 4

    @pytest.mark.parametrize(
        "field",
        ["input_modalities", "output_modalities"],
    )
    def test_model_capabilities_reject_empty_or_duplicate_modalities(
        self,
        field: str,
    ):
        """Require a meaningful, unambiguous modality declaration."""
        values = {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        }
        values[field] = []
        with pytest.raises(ValueError, match="must not be empty"):
            ModelCapabilities(**values)

        values[field] = ["text", "text"]
        with pytest.raises(ValueError, match="must not contain duplicates"):
            ModelCapabilities(**values)

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Missing required field
        with pytest.raises(ValueError):
            ModelConfig(litellm_params={"model": "openai/gpt-4"})  # type: ignore

        with pytest.raises(ValueError):
            ModelConfig(model_name="gpt-4")  # type: ignore


class TestConferLLMConfig:
    """Test ConferLLMConfig class."""

    def test_default_config_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test getting default config path."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert get_default_app_dir() == tmp_path / ".conferllm"
        path = ConferLLMConfig.get_default_config_path()
        assert path == tmp_path / ".conferllm" / "config.yaml"

    def test_empty_config(self):
        """Test creating empty config."""
        config = ConferLLMConfig()
        assert config.model_list == []
        assert config.global_system_prompt is None
        assert config.image_limits == ImageLimits()

    def test_config_with_global_system_prompt(self):
        """Test creating config with global system prompt."""
        config = ConferLLMConfig(global_system_prompt="Global system prompt")
        assert config.global_system_prompt == "Global system prompt"

    def test_config_with_models(self):
        """Test creating config with models."""
        model_configs = [
            ModelConfig(
                model_name="gpt-4",
                litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
            ),
            ModelConfig(
                model_name="claude-sonnet",
                litellm_params={
                    "model": "anthropic/claude-3-sonnet",
                    "api_key": "test-key",
                },
            ),
        ]
        config = ConferLLMConfig(model_list=model_configs)
        assert len(config.model_list) == 2
        assert config.model_list[0].model_name == "gpt-4"
        assert config.model_list[1].model_name == "claude-sonnet"

    def test_get_model_config_existing(self):
        """Test getting model config for existing model."""
        model_configs = [
            ModelConfig(
                model_name="gpt-4",
                litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
            )
        ]
        config = ConferLLMConfig(model_list=model_configs)

        model_config = config.get_model_config("gpt-4")
        assert model_config is not None
        assert model_config.model_name == "gpt-4"
        assert model_config.litellm_params["model"] == "openai/gpt-4"

    def test_get_model_config_non_existing(self):
        """Test getting model config for non-existing model."""
        config = ConferLLMConfig()
        model_config = config.get_model_config("non-existing")
        assert model_config is None

    def test_list_available_models(self):
        """Test listing available models."""
        model_configs = [
            ModelConfig(
                model_name="gpt-4",
                litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
            ),
            ModelConfig(
                model_name="claude-sonnet",
                litellm_params={
                    "model": "anthropic/claude-3-sonnet",
                    "api_key": "test-key",
                },
            ),
        ]
        config = ConferLLMConfig(model_list=model_configs)

        models = config.list_available_models()
        assert len(models) == 2
        assert "gpt-4" in models
        assert "claude-sonnet" in models

    def test_configured_session_root_and_image_limits(self, tmp_path: Path):
        """Expose configurable storage and deterministic image safety limits."""
        config = ConferLLMConfig(
            sessions_dir=tmp_path / "custom-sessions",
            image_limits=ImageLimits(
                max_count=3,
                max_bytes_per_image=100,
                max_total_bytes=250,
            ),
        )

        assert config.get_session_root() == tmp_path / "custom-sessions"
        assert config.image_limits.max_count == 3
        assert config.image_limits.max_total_bytes == 250


class TestConferLLMConfigLoading:
    """Test configuration loading functionality."""

    def test_load_config_explicit_path_is_used_exactly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test loading an explicit path without consulting default locations."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.chdir(tmp_path)
        legacy_path = Path.home() / ".conferllm.yaml"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text("invalid: yaml: [", encoding="utf-8")
        explicit_path = Path("custom.yaml")
        write_config(
            explicit_path,
            {
                "model_list": [
                    {
                        "model_name": "explicit",
                        "litellm_params": {"model": "openai/gpt-4"},
                    }
                ]
            },
        )

        config = ConferLLMConfig.load_config(explicit_path)

        assert config.list_available_models() == ["explicit"]

    def test_load_config_non_existing_explicit_file(self, tmp_path: Path):
        """Do not silently ignore an explicitly selected missing configuration."""
        config_path = tmp_path / "non-existing.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            ConferLLMConfig.load_config(config_path)

    def test_load_config_from_new_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test loading configuration from the new default path."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_data = {
            "model_list": [
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
                }
            ]
        }
        write_config(tmp_path / ".conferllm" / "config.yaml", config_data)

        config = ConferLLMConfig.load_config()

        assert len(config.model_list) == 1
        assert config.model_list[0].model_name == "gpt-4"
        assert config.model_list[0].litellm_params["model"] == "openai/gpt-4"

    def test_relative_session_root_is_relative_to_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "settings" / "models.yaml"
        write_config(config_path, {"sessions_dir": "history"})
        other_cwd = tmp_path / "caller"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        config = ConferLLMConfig.load_config(config_path)

        assert config.get_session_root() == config_path.parent / "history"

    def test_new_default_takes_precedence_over_legacy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that an existing new default prevents any legacy-file read."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        new_path = tmp_path / ".conferllm" / "config.yaml"
        legacy_path = tmp_path / ".conferllm.yaml"
        write_config(
            new_path,
            {
                "model_list": [
                    {
                        "model_name": "new-default",
                        "litellm_params": {"model": "openai/gpt-4"},
                    }
                ]
            },
        )
        legacy_path.write_text("invalid: yaml: [", encoding="utf-8")
        original_open = Path.open

        def guarded_open(path: Path, *args: Any, **kwargs: Any):
            if path == legacy_path:
                raise AssertionError("legacy configuration must not be read")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            config = ConferLLMConfig.load_config()

        assert config.list_available_models() == ["new-default"]

    def test_legacy_default_requires_manual_migration_without_reading_or_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test migration guidance without accessing legacy contents."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        legacy_path = tmp_path / ".conferllm.yaml"
        original_contents = b"invalid: yaml: [\nsecret: do-not-read\n"
        legacy_path.write_bytes(original_contents)
        legacy_path.chmod(0o640)
        original_stat = legacy_path.stat()

        with (
            patch.object(
                Path,
                "open",
                side_effect=AssertionError("legacy configuration must not be read"),
            ),
            pytest.raises(ConfigMigrationError) as exc_info,
        ):
            ConferLLMConfig.load_config()

        message = str(exc_info.value)
        assert "mkdir -p ~/.conferllm" in message
        assert "mv ~/.conferllm.yaml ~/.conferllm/config.yaml" in message
        assert "chmod 700 ~/.conferllm" in message
        assert "chmod 600 ~/.conferllm/config.yaml" in message
        assert legacy_path.read_bytes() == original_contents
        current_stat = legacy_path.stat()
        assert current_stat.st_mode == original_stat.st_mode
        assert current_stat.st_mtime_ns == original_stat.st_mtime_ns

    def test_both_default_paths_missing_returns_empty_config_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test the existing empty-config behavior when no default exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        expected_path = tmp_path / ".conferllm" / "config.yaml"

        with patch("conferllm.config.logger") as mock_logger:
            config = ConferLLMConfig.load_config()

        assert config.model_list == []
        mock_logger.warning.assert_called_once_with(
            "Configuration file not found at %s, using empty config", expected_path
        )

    def test_load_config_with_system_prompts(self, tmp_path: Path):
        """Test loading config with system prompts."""
        config_data = {
            "global_system_prompt": "Global system prompt",
            "model_list": [
                {
                    "model_name": "gpt-4",
                    "system_prompt": "Model-specific system prompt",
                    "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
                },
                {
                    "model_name": "claude-sonnet",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet",
                        "api_key": "test-key",
                    },
                },
            ],
        }
        config_path = tmp_path / "config.yaml"
        write_config(config_path, config_data)

        config = ConferLLMConfig.load_config(config_path)
        assert config.global_system_prompt == "Global system prompt"
        assert len(config.model_list) == 2
        assert config.model_list[0].model_name == "gpt-4"
        assert config.model_list[0].system_prompt == "Model-specific system prompt"
        assert config.model_list[1].model_name == "claude-sonnet"
        assert config.model_list[1].system_prompt is None

    def test_load_config_invalid_yaml(self, tmp_path: Path):
        """Test loading config from invalid YAML file."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="YAML syntax"):
            ConferLLMConfig.load_config(config_path)

    @pytest.mark.parametrize("filename", [".ai_hub.yaml", ".conferllm.yaml"])
    def test_legacy_presence_is_detected_without_opening(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        legacy = tmp_path / filename
        legacy.write_text("never read this fixture", encoding="utf-8")
        with (
            patch.object(Path, "open", side_effect=AssertionError("legacy read")),
            pytest.raises(ConfigMigrationError),
        ):
            ConferLLMConfig.load_config()

    def test_multiple_legacy_files_are_not_selected_implicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for name in (".ai_hub.yaml", ".conferllm.yaml"):
            (tmp_path / name).write_text("never read this fixture", encoding="utf-8")
        with (
            patch.object(Path, "open", side_effect=AssertionError("legacy read")),
            pytest.raises(ConfigMigrationError, match="Multiple legacy"),
        ):
            ConferLLMConfig.load_config()
