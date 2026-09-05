"""Configuration management for ConferLLM."""

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_IMAGES = 16
DEFAULT_MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TOTAL_INPUT_IMAGE_BYTES = 50 * 1024 * 1024


def _default_text_modalities() -> list[Literal["text", "image"]]:
    return ["text"]


class ConfigurationError(ValueError):
    """A configuration failure whose message never includes supplied values."""

    code = "configuration_error"


class ConfigMigrationError(ConfigurationError):
    """Raised when the legacy default configuration requires manual migration."""


def get_default_app_dir() -> Path:
    """Get the default directory for ConferLLM application data."""
    return Path.home() / ".conferllm"


def existing_legacy_configs() -> list[Path]:
    """Inspect only legacy path presence, never their contents."""
    return [
        path
        for name in (".ai_hub.yaml", ".conferllm.yaml")
        if (path := Path.home() / name).exists()
    ]


class ImageLimits(BaseModel):
    """Application-level safety limits for image inputs."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    max_count: PositiveInt = DEFAULT_MAX_INPUT_IMAGES
    max_bytes_per_image: PositiveInt = DEFAULT_MAX_INPUT_IMAGE_BYTES
    max_total_bytes: PositiveInt = DEFAULT_MAX_TOTAL_INPUT_IMAGE_BYTES


class ModelCapabilities(BaseModel):
    """Optional non-secret capability metadata for a configured model."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    input_modalities: list[Literal["text", "image"]] = Field(
        default_factory=_default_text_modalities
    )
    output_modalities: list[Literal["text", "image"]] = Field(
        default_factory=_default_text_modalities
    )
    max_input_images: PositiveInt | None = None

    @field_validator("input_modalities", "output_modalities")
    @classmethod
    def validate_modalities(
        cls,
        value: list[Literal["text", "image"]],
    ) -> list[Literal["text", "image"]]:
        """Reject empty or duplicate modality declarations."""
        if not value:
            raise ValueError("Model modalities must not be empty.")
        if len(value) != len(set(value)):
            raise ValueError("Model modalities must not contain duplicates.")
        return value


class ModelConfig(BaseModel):
    """Configuration for a single AI model."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model_name: str
    litellm_params: dict[str, Any]
    system_prompt: str | None = None  # Optional system prompt for this model
    capabilities: ModelCapabilities | None = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Use non-empty, consistently normalized local aliases."""
        if not value.strip():
            raise ValueError("Model name must not be empty.")
        return value.strip()

    @field_validator("litellm_params")
    @classmethod
    def validate_provider_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject invalid routing and application-owned message overrides early."""
        model = value.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("litellm_params.model must be a non-empty string.")
        if "messages" in value:
            raise ValueError("litellm_params.messages is managed by ConferLLM.")
        return value


class ConferLLMConfig(BaseModel):
    """Main configuration for ConferLLM."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model_list: list[ModelConfig] = Field(default_factory=list)
    global_system_prompt: str | None = (
        None  # Optional global system prompt for all models
    )
    image_limits: ImageLimits = Field(default_factory=ImageLimits)
    sessions_dir: Path | None = None

    @field_validator("model_list")
    @classmethod
    def validate_unique_aliases(cls, value: list[ModelConfig]) -> list[ModelConfig]:
        """Never silently route a duplicated alias to the first definition."""
        names = [model.model_name for model in value]
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique.")
        return value

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        return get_default_app_dir() / "config.yaml"

    @classmethod
    def load_config(cls, config_path: Path | None = None) -> "ConferLLMConfig":
        """Load configuration from file."""
        using_default_path = config_path is None
        resolved_config_path = (
            cls.get_default_config_path()
            if config_path is None
            else config_path.expanduser()
        )

        if not resolved_config_path.exists():
            if not using_default_path:
                raise ConfigurationError(
                    f"Configuration file not found: {resolved_config_path}"
                )
            legacy_paths = existing_legacy_configs()
            if len(legacy_paths) > 1:
                raise ConfigMigrationError(
                    "Multiple legacy configuration files exist. Choose the intended "
                    "file locally and migrate it to ~/.conferllm/config.yaml; "
                    "ConferLLM will not read or select a legacy file automatically."
                )
            if legacy_paths:
                legacy_name = legacy_paths[0].name
                raise ConfigMigrationError(
                    f"Legacy configuration found at ~/{legacy_name}. Migrate it with:\n"
                    "mkdir -p ~/.conferllm\n"
                    f"test ! -e ~/.conferllm/config.yaml && "
                    f"mv ~/{legacy_name} ~/.conferllm/config.yaml\n"
                    "chmod 700 ~/.conferllm\n"
                    "chmod 600 ~/.conferllm/config.yaml"
                )

            logger.warning(
                "Configuration file not found at %s, using empty config",
                resolved_config_path,
            )
            return cls()

        try:
            with resolved_config_path.open(encoding="utf-8") as config_file:
                config_data = yaml.safe_load(config_file)

            config = cls.model_validate({} if config_data is None else config_data)
            if config.sessions_dir is not None:
                session_root = config.sessions_dir.expanduser()
                if not session_root.is_absolute():
                    session_root = resolved_config_path.resolve().parent / session_root
                config.sessions_dir = session_root
            return config
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
            # YAML and Pydantic exceptions can embed credentials or complete input
            # objects. Expose only the path and failure category, including in logs.
            logger.error(
                "Failed to load configuration from %s (%s)",
                resolved_config_path,
                type(error).__name__,
            )
            raise ConfigurationError(
                f"Unable to load configuration from {resolved_config_path}. "
                "Check file access, YAML syntax, and the configuration schema."
            ) from None

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        for model in self.model_list:
            if model.model_name == model_name:
                return model
        return None

    def list_available_models(self) -> list[str]:
        """List all available model names."""
        return [model.model_name for model in self.model_list]

    def get_session_root(self) -> Path:
        """Return the session root; YAML-relative paths are resolved during loading."""
        if self.sessions_dir is None:
            return get_default_app_dir() / "sessions"
        return self.sessions_dir.expanduser()
