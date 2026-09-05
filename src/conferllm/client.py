"""LiteLLM integration for ConferLLM."""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import litellm
from litellm.types.utils import ModelResponse

from .config import ConferLLMConfig, ModelConfig
from .errors import ConferLLMError
from .images import image_bytes_to_data_url, image_payload_from_bytes

logger = logging.getLogger(__name__)


class ModelNotFoundError(ValueError):
    """A caller selected an alias that is not configured."""

    code = "model_not_found"


class ModelCapabilityError(ValueError):
    """Raised when a request contradicts declared model capabilities."""

    code = "model_capability_mismatch"

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class ImageLimitError(ValueError):
    """Raised when a request exceeds an effective model image limit."""

    code = "image_limit_exceeded"

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class EffectiveImageLimits:
    """Resolved image limits for one configured model."""

    max_count: int
    max_bytes_per_image: int
    max_total_bytes: int
    capabilities_declared: bool


class LLMClient:
    """Wrapper around LiteLM for unified AI provider access."""

    def __init__(self, config: ConferLLMConfig):
        """Initialize AI client with configuration."""
        self.config = config
        # Set LiteLM to suppress output
        litellm.suppress_debug_info = True

    def chat(self, model_name: str, messages: list[dict[str, Any]]) -> ModelResponse:
        """Chat with specified AI model.

        Args:
            model_name: Name of the model to use. Call list_models() tool to see available models.
            messages: List of messages in OpenAI format. Each message should have 'role' and 'content' keys.
                     Content can be:
                     - String for text messages
                     - List of content objects for multimodal messages (text, image_url, etc.)

                     Image formats supported:
                     - Remote URL: {"url": "https://example.com/image.jpg"}
                     - Local file path: {"url": "/path/to/local/image.jpg"}
                     - Base64: {"url": "data:image/jpeg;base64,<base64_string>"}

                     Example formats:
                     - Text only: [{"role": "user", "content": "Hello!"}]
                     - With remote image URL: [{"role": "user", "content": [
                         {"type": "text", "text": "What's in this image?"},
                         {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                       ]}]
                     - With local image path: [{"role": "user", "content": [
                         {"type": "text", "text": "What's in this image?"},
                         {"type": "image_url", "image_url": {"url": "/Users/john/Desktop/photo.jpg"}}
                       ]}]

        Returns:
            Raw LiteLM ModelResponse object containing all response data

        Raises:
            ValueError: If model is not configured or messages format is invalid
            Exception: If API call fails
        """
        model_config = self.require_model_config(model_name)

        # Validate messages format
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of message dictionaries.")

        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must be a dictionary with 'role' and 'content' keys."
                )

        # Process messages to convert local image paths to base64
        processed_messages = self._process_messages_for_local_images(messages)

        # Apply system prompt if configured
        prepared_messages = self._prepare_messages_with_system_prompt(
            processed_messages, model_config
        )

        try:
            # Get the model parameter and validate it
            litellm_model = model_config.litellm_params.get("model")
            if not litellm_model:
                raise ValueError(
                    f"Model configuration for '{model_name}' missing 'model' parameter"
                )

            # Make the API call using LiteLM (ensure non-streaming)
            litellm_params = {
                k: v for k, v in model_config.litellm_params.items() if k != "model"
            }
            litellm_params["stream"] = False  # Explicitly disable streaming

            response = litellm.completion(
                model=litellm_model, messages=prepared_messages, **litellm_params
            )

            # Return the raw ModelResponse object
            # Cast to ModelResponse since LiteLLM can return a union type but we disable streaming
            return cast(ModelResponse, response)

        except Exception as error:
            # Provider errors may contain request bodies, credentials, or signed
            # URLs. Preserve the failure category, never the provider's text.
            error_type = type(error).__name__
            logger.error("Error calling model %s (%s)", model_name, error_type)
            raise ConferLLMError(
                "provider_error",
                f"Failed to get response from {model_name}. "
                "Check provider availability, credentials, and model parameters.",
                details={"exception_type": error_type},
            ) from None

    def _is_local_path(self, url: str) -> bool:
        """Check if a URL is a local file path.

        Args:
            url: The URL to check

        Returns:
            True if the URL is a local file path, False otherwise
        """
        # Skip data URLs (base64) and HTTP(S) URLs
        if url.startswith(("data:", "http://", "https://")):
            return False

        # Check for absolute paths
        # Unix/Mac: starts with /
        # Windows: starts with drive letter (C:, D:, etc.)
        return url.startswith("/") or (len(url) > 1 and url[1] == ":")

    def _read_and_encode_image(self, file_path: str) -> str:
        """Read a local image file and convert it to base64 data URL.

        Args:
            file_path: Path to the local image file

        Returns:
            Base64 data URL string.
        """
        try:
            path = Path(file_path)
            image_data = path.read_bytes()
            payload = image_payload_from_bytes(
                image_data,
                source_name=path.name,
            )
            data_url = image_bytes_to_data_url(payload.data, payload.mime_type)
            logger.info("Converted local image to base64: %s", file_path)
            return data_url
        except Exception as error:
            logger.error("Failed to read and encode image %s: %s", file_path, error)
            raise ValueError(
                f"Failed to read local image '{file_path}': {error}"
            ) from error

    def _process_content_item(self, item: Any) -> Any:
        """Process a single content item to convert local image paths.

        Args:
            item: A content item (dict or other type)

        Returns:
            Processed content item with local paths converted to base64
        """
        # If not a dict, return as-is
        if not isinstance(item, dict):
            return item

        # Check if this is an image_url type
        if item.get("type") == "image_url" and "image_url" in item:
            image_url_obj = item["image_url"]
            if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                url = image_url_obj["url"]

                # Check if it's a local path
                if isinstance(url, str) and self._is_local_path(url):
                    # Convert to base64
                    base64_url = self._read_and_encode_image(url)
                    # Create a new dict to avoid modifying the original
                    new_item = copy.deepcopy(item)
                    new_item["image_url"]["url"] = base64_url
                    return new_item

        return item

    def _process_messages_for_local_images(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process messages to convert local image paths to base64.

        Args:
            messages: List of messages to process

        Returns:
            New list of messages with local images converted to base64
        """
        processed_messages = []

        for message in messages:
            # Create a copy to avoid modifying the original
            new_message = copy.deepcopy(message)

            # Process content if it's a list (multimodal)
            if isinstance(new_message.get("content"), list):
                new_content = []
                for item in new_message["content"]:
                    new_content.append(self._process_content_item(item))
                new_message["content"] = new_content

            processed_messages.append(new_message)

        return processed_messages

    def _prepare_messages_with_system_prompt(
        self,
        messages: list[dict[str, Any]],
        model_config: ModelConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Add system prompt to messages if configured."""
        result_messages: list[dict[str, Any]] = []

        # Determine system prompt with precedence: model-specific > global
        system_prompt: str | None
        if model_config is not None and hasattr(model_config, "system_prompt"):
            # Use model-specific value if explicitly set (including empty string
            # to intentionally disable/override any global prompt). Only fall back
            # to global when the model-level value is None (unset).
            if model_config.system_prompt is not None:
                system_prompt = model_config.system_prompt
            else:
                system_prompt = self.config.global_system_prompt
        else:
            system_prompt = self.config.global_system_prompt

        # Add system prompt if configured
        if system_prompt:
            result_messages.append({"role": "system", "content": system_prompt})

        # Add the original messages
        result_messages.extend(messages)

        return result_messages

    def list_models(self) -> list[str]:
        """List all available models."""
        return self.config.list_available_models()

    def require_model_config(self, model_name: str) -> ModelConfig:
        """Return a configured model or raise a stable validation error."""
        model_config = self.config.get_model_config(model_name)
        if model_config is not None:
            return model_config
        available_models = self.config.list_available_models()
        raise ModelNotFoundError(
            f"Model '{model_name}' not found in configuration. "
            f"Available models: {', '.join(available_models)}"
        )

    def validate_chat_request(
        self,
        model_name: str,
        *,
        image_count: int = 0,
        history_image_count: int = 0,
        require_text: bool = True,
    ) -> tuple[EffectiveImageLimits, list[str]]:
        """Validate declared modalities before any image source is read."""
        model_config = self.require_model_config(model_name)
        limits = self.config.image_limits
        capabilities = model_config.capabilities
        effective_max_count = limits.max_count
        warnings: list[str] = []

        if capabilities is not None:
            if require_text and "text" not in capabilities.input_modalities:
                raise ModelCapabilityError(
                    f"Model '{model_name}' is not configured to accept text prompts.",
                    {
                        "model": model_name,
                        "requested_modality": "text",
                        "input_modalities": capabilities.input_modalities,
                    },
                )
            if (
                image_count or history_image_count
            ) and "image" not in capabilities.input_modalities:
                raise ModelCapabilityError(
                    f"Model '{model_name}' is configured as text-only.",
                    {
                        "model": model_name,
                        "requested_modality": "image",
                        "input_modalities": capabilities.input_modalities,
                    },
                )
            if capabilities.max_input_images is not None:
                effective_max_count = min(
                    effective_max_count,
                    max(0, capabilities.max_input_images - history_image_count),
                )
                if history_image_count > capabilities.max_input_images:
                    raise ImageLimitError(
                        f"Stored history exceeds the image limit for model '{model_name}'. "
                        "Start a new session or revise the declared model limit.",
                        {
                            "model": model_name,
                            "history_count": history_image_count,
                            "max_count": capabilities.max_input_images,
                        },
                    )
        elif image_count or history_image_count:
            warnings.append(
                f"Model '{model_name}' has no declared image capabilities; "
                "provider compatibility was not prevalidated."
            )

        if image_count > effective_max_count:
            raise ImageLimitError(
                (
                    f"At most {effective_max_count} input images are allowed for "
                    f"model '{model_name}'."
                ),
                {
                    "model": model_name,
                    "count": image_count,
                    "history_count": history_image_count,
                    "max_count": effective_max_count,
                },
            )

        return (
            EffectiveImageLimits(
                max_count=effective_max_count,
                max_bytes_per_image=limits.max_bytes_per_image,
                max_total_bytes=limits.max_total_bytes,
                capabilities_declared=capabilities is not None,
            ),
            warnings,
        )

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific model."""
        model_config = self.require_model_config(model_name)
        limits, _ = self.validate_chat_request(model_name, require_text=False)
        capabilities = model_config.capabilities

        return {
            "model_name": model_config.model_name,
            "provider_model": model_config.litellm_params.get("model"),
            "configured_params": list(model_config.litellm_params.keys()),
            "has_model_system_prompt": model_config.system_prompt is not None,
            "uses_global_system_prompt": (
                model_config.system_prompt is None
                and self.config.global_system_prompt is not None
            ),
            "capabilities": {
                "declared": capabilities is not None,
                "input_modalities": (
                    capabilities.input_modalities if capabilities is not None else None
                ),
                "output_modalities": (
                    capabilities.output_modalities if capabilities is not None else None
                ),
                "max_input_images": (
                    capabilities.max_input_images if capabilities is not None else None
                ),
            },
            "effective_image_limits": {
                "max_count": limits.max_count,
                "max_bytes_per_image": limits.max_bytes_per_image,
                "max_total_bytes": limits.max_total_bytes,
            },
        }
