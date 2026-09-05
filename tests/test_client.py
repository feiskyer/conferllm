"""Unit tests for AI client (LiteLM integration)."""

from unittest.mock import MagicMock, patch

import pytest

from conferllm.client import ImageLimitError, LLMClient, ModelCapabilityError
from conferllm.config import (
    ConferLLMConfig,
    ImageLimits,
    ModelCapabilities,
    ModelConfig,
)


class TestLLMClient:
    """Test LLMClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    litellm_params={
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                ),
                ModelConfig(
                    model_name="claude-sonnet",
                    litellm_params={
                        "model": "anthropic/claude-3-5-sonnet-20241022",
                        "api_key": "test-key",
                    },
                ),
            ]
        )
        self.client = LLMClient(self.config)

    def test_init(self):
        """Test LLMClient initialization."""
        assert self.client.config == self.config

    def test_chat_with_string_input(self):
        """Test chat with message list input."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ) as mock_completion:
            messages = [{"role": "user", "content": "Hello, world!"}]
            response = self.client.chat("gpt-4", messages)

            assert response == mock_response
            mock_completion.assert_called_once_with(
                model="openai/gpt-4",
                messages=messages,
                api_key="test-key",
                max_tokens=2048,
                temperature=0.7,
                stream=False,
            )

    def test_chat_with_global_system_prompt(self):
        """Test chat with global system prompt."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                )
            ],
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ) as mock_completion:
            messages = [{"role": "user", "content": "Hello, world!"}]
            response = client.chat("gpt-4", messages)

            assert response == mock_response
            mock_completion.assert_called_once_with(
                model="openai/gpt-4",
                messages=[
                    {"role": "system", "content": "Global system prompt"},
                    {"role": "user", "content": "Hello, world!"},
                ],
                api_key="test-key",
                stream=False,
            )

    def test_chat_with_model_specific_system_prompt(self):
        """Test chat with model-specific system prompt."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    system_prompt="Model-specific system prompt",
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                )
            ],
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ) as mock_completion:
            messages = [{"role": "user", "content": "Hello, world!"}]
            response = client.chat("gpt-4", messages)

            assert response == mock_response
            mock_completion.assert_called_once_with(
                model="openai/gpt-4",
                messages=[
                    {"role": "system", "content": "Model-specific system prompt"},
                    {"role": "user", "content": "Hello, world!"},
                ],
                api_key="test-key",
                stream=False,
            )

    def test_chat_with_messages_input(self):
        """Test chat with messages input."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ) as mock_completion:
            response = self.client.chat("gpt-4", messages)

            assert response == mock_response
            mock_completion.assert_called_once_with(
                model="openai/gpt-4",
                messages=messages,
                api_key="test-key",
                max_tokens=2048,
                temperature=0.7,
                stream=False,
            )

    def test_chat_with_non_existing_model(self):
        """Test chat with non-existing model."""
        with pytest.raises(
            ValueError, match="Model 'non-existing' not found in configuration"
        ):
            self.client.chat("non-existing", [{"role": "user", "content": "Hello!"}])

    def test_chat_missing_model_parameter(self):
        """Reject a missing provider model while validating configuration."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ModelConfig(
                model_name="bad-model",
                litellm_params={"api_key": "test-key"},
            )

    def test_chat_api_error(self):
        """Test chat when API call fails."""
        with (
            patch(
                "conferllm.client.litellm.completion",
                side_effect=Exception("API Error"),
            ),
            pytest.raises(RuntimeError, match="Failed to get response from gpt-4"),
        ):
            self.client.chat("gpt-4", [{"role": "user", "content": "Hello!"}])

    def test_chat_empty_response(self):
        """Test chat with empty response."""
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ):
            response = self.client.chat(
                "gpt-4", [{"role": "user", "content": "Hello!"}]
            )
            assert response == mock_response

    def test_chat_missing_content(self):
        """Test chat with response missing content."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=None))  # No content
        ]

        with patch(
            "conferllm.client.litellm.completion",
            return_value=mock_response,
        ):
            response = self.client.chat(
                "gpt-4", [{"role": "user", "content": "Hello!"}]
            )
            assert response == mock_response

    def test_prepare_messages_with_system_prompt(self):
        """Test preparing messages with system prompt."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                )
            ],
        )
        client = LLMClient(config)
        model_config = config.get_model_config("gpt-4")
        messages = [{"role": "user", "content": "Hello, world!"}]

        prepared = client._prepare_messages_with_system_prompt(messages, model_config)
        assert prepared == [
            {"role": "system", "content": "Global system prompt"},
            {"role": "user", "content": "Hello, world!"},
        ]

    def test_prepare_messages_model_specific_overrides_global(self):
        """Test that model-specific system prompt overrides global system prompt."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    system_prompt="Model-specific system prompt",
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                ),
                ModelConfig(
                    model_name="claude-sonnet",
                    litellm_params={
                        "model": "anthropic/claude-sonnet",
                        "api_key": "test-key",
                    },
                ),
            ],
        )
        client = LLMClient(config)
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Test model with specific system prompt
        gpt4_config = config.get_model_config("gpt-4")
        prepared = client._prepare_messages_with_system_prompt(messages, gpt4_config)
        assert prepared == [
            {"role": "system", "content": "Model-specific system prompt"},
            {"role": "user", "content": "Hello, world!"},
        ]

        # Test model without specific system prompt (should use global)
        claude_config = config.get_model_config("claude-sonnet")
        prepared = client._prepare_messages_with_system_prompt(messages, claude_config)
        assert prepared == [
            {"role": "system", "content": "Global system prompt"},
            {"role": "user", "content": "Hello, world!"},
        ]

    def test_messages_validation_invalid_format(self):
        """Test messages validation with invalid format."""
        with pytest.raises(
            ValueError, match="Messages must be a list of message dictionaries"
        ):
            self.client.chat("gpt-4", "string_instead_of_list")  # type: ignore

    def test_messages_validation_missing_keys(self):
        """Test messages validation with missing keys."""
        invalid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"invalid": "message"},  # Missing role and content
        ]

        with pytest.raises(
            ValueError,
            match="Each message must be a dictionary with 'role' and 'content' keys",
        ):
            self.client.chat("gpt-4", invalid_messages)

    def test_list_models(self):
        """Test listing available models."""
        models = self.client.list_models()
        assert len(models) == 2
        assert "gpt-4" in models
        assert "claude-sonnet" in models

    def test_get_model_info_existing(self):
        """Test getting model info for existing model."""
        info = self.client.get_model_info("gpt-4")
        assert info["model_name"] == "gpt-4"
        assert info["provider_model"] == "openai/gpt-4"
        assert "api_key" in info["configured_params"]
        assert "max_tokens" in info["configured_params"]
        assert "temperature" in info["configured_params"]
        assert info["has_model_system_prompt"] is False
        assert info["uses_global_system_prompt"] is False
        assert info["capabilities"]["declared"] is False
        assert info["effective_image_limits"]["max_count"] == 16
        assert "test-key" not in str(info)

    def test_get_model_info_with_system_prompts(self):
        """Test getting model info with system prompts."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    system_prompt="Model-specific system prompt",
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                )
            ],
        )
        client = LLMClient(config)
        info = client.get_model_info("gpt-4")
        assert info["has_model_system_prompt"] is True
        assert info["uses_global_system_prompt"] is False
        assert "Model-specific system prompt" not in str(info)
        assert "Global system prompt" not in str(info)

    def test_get_model_info_non_existing(self):
        """Test getting model info for non-existing model."""
        with pytest.raises(
            ValueError, match="Model 'non-existing' not found in configuration"
        ):
            self.client.get_model_info("non-existing")

    def test_declared_text_only_model_rejects_images(self):
        """Fail capability preflight before an image source must be read."""
        config = ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="text-only",
                    litellm_params={"model": "openai/text"},
                    capabilities=ModelCapabilities(
                        input_modalities=["text"],
                        output_modalities=["text"],
                    ),
                )
            ]
        )
        client = LLMClient(config)

        with pytest.raises(ModelCapabilityError) as exc_info:
            client.validate_chat_request("text-only", image_count=1)

        assert exc_info.value.code == "model_capability_mismatch"
        assert exc_info.value.details["requested_modality"] == "image"

    def test_unknown_capabilities_warn_but_apply_global_limits(self):
        """Keep old configurations working without claiming image support."""
        limits, warnings = self.client.validate_chat_request(
            "gpt-4",
            image_count=1,
        )

        assert limits.max_count == 16
        assert limits.capabilities_declared is False
        assert warnings == [
            "Model 'gpt-4' has no declared image capabilities; "
            "provider compatibility was not prevalidated."
        ]

    def test_model_specific_image_count_lowers_global_limit(self):
        """Use the stricter of global and declared model image limits."""
        config = ConferLLMConfig(
            image_limits=ImageLimits(
                max_count=8,
                max_bytes_per_image=100,
                max_total_bytes=200,
            ),
            model_list=[
                ModelConfig(
                    model_name="vision",
                    litellm_params={"model": "openai/vision"},
                    capabilities=ModelCapabilities(
                        input_modalities=["text", "image"],
                        output_modalities=["text"],
                        max_input_images=2,
                    ),
                )
            ],
        )
        client = LLMClient(config)

        with pytest.raises(ImageLimitError) as exc_info:
            client.validate_chat_request("vision", image_count=3)

        assert exc_info.value.details["max_count"] == 2

    def test_litellm_suppress_debug_info(self):
        """Test that LiteLM debug info is suppressed."""
        with patch("conferllm.client.litellm") as mock_litellm:
            LLMClient(self.config)
            assert mock_litellm.suppress_debug_info is True

    def test_model_specific_empty_system_prompt_disables_global(self):
        """Model-level empty prompt should override and disable global prompt."""
        config = ConferLLMConfig(
            global_system_prompt="Global system prompt",
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    system_prompt="",  # explicit empty to disable
                    litellm_params={"model": "openai/gpt-4", "api_key": "test-key"},
                )
            ],
        )
        client = LLMClient(config)
        model_cfg = config.get_model_config("gpt-4")
        assert model_cfg is not None

        # _prepare_messages_with_system_prompt should NOT include a system message
        messages = [{"role": "user", "content": "Hello!"}]
        prepared = client._prepare_messages_with_system_prompt(messages, model_cfg)
        assert prepared == [{"role": "user", "content": "Hello!"}]
