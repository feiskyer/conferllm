"""LiteLM integration wrapper for AI providers."""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, cast

import litellm
from litellm.types.utils import ModelResponse

from .config import AIHubConfig, ModelConfig

logger = logging.getLogger(__name__)


class AIClient:
    """Wrapper around LiteLM for unified AI provider access."""

    def __init__(self, config: AIHubConfig):
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
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            available_models = self.config.list_available_models()
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {', '.join(available_models)}"
            )

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

        except Exception as e:
            logger.error("Error calling model %s: %s", model_name, e)
            raise RuntimeError(
                f"Failed to get response from {model_name}: {str(e)}"
            ) from e

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

    def _read_and_encode_image(self, file_path: str) -> str | None:
        """Read a local image file and convert it to base64 data URL.

        Args:
            file_path: Path to the local image file

        Returns:
            Base64 data URL string, or None if file cannot be read
        """
        try:
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Image file not found: {file_path}")
                return None

            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                # Default to common image types based on extension
                ext = path.suffix.lower()
                mime_types_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                }
                mime_type = mime_types_map.get(ext, "image/jpeg")

            # Read and encode the image
            with open(file_path, "rb") as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode("utf-8")

            # Create data URL
            data_url = f"data:{mime_type};base64,{base64_str}"
            logger.info(f"Converted local image to base64: {file_path}")
            return data_url

        except Exception as e:
            logger.error(f"Failed to read and encode image {file_path}: {e}")
            return None

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
                if self._is_local_path(url):
                    # Convert to base64
                    base64_url = self._read_and_encode_image(url)
                    if base64_url:
                        # Create a new dict to avoid modifying the original
                        import copy

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
        import copy

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

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific model."""
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration.")

        return {
            "model_name": model_config.model_name,
            "provider_model": model_config.litellm_params.get("model"),
            "configured_params": list(model_config.litellm_params.keys()),
            "system_prompt": model_config.system_prompt,
            "global_system_prompt": self.config.global_system_prompt,
        }
