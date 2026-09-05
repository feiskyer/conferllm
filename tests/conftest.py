"""Test configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Any, NoReturn

import litellm
import pytest
import yaml

from conferllm.config import ConferLLMConfig, ModelConfig


@pytest.fixture(autouse=True)
def isolate_user_data_and_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never consult real user data or accidentally invoke a provider."""
    isolated_home = tmp_path / "isolated-home"
    isolated_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: isolated_home)

    def unmocked_provider(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("Provider calls must be explicitly mocked in tests.")

    monkeypatch.setattr(litellm, "completion", unmocked_provider)
    monkeypatch.setattr(litellm, "acompletion", unmocked_provider)


def create_test_config(models: list[dict[str, Any]]) -> ConferLLMConfig:
    """Create a test configuration with the specified models."""
    model_configs = []
    for model_data in models:
        model_config = ModelConfig(
            model_name=model_data["model_name"],
            litellm_params=model_data["litellm_params"],
        )
        model_configs.append(model_config)

    return ConferLLMConfig(model_list=model_configs)


def create_temp_config_file(config_data: dict[str, Any]) -> Path:
    """Create a temporary YAML config file with the given data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        return Path(f.name)


@pytest.fixture
def sample_config_data() -> dict[str, Any]:
    """Sample configuration data for testing."""
    return {
        "model_list": [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "${OPENAI_API_KEY}",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            },
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20241022",
                    "api_key": "${ANTHROPIC_API_KEY}",
                },
            },
        ]
    }


@pytest.fixture
def sample_config(sample_config_data: dict[str, Any]) -> ConferLLMConfig:
    """Sample configuration for testing."""
    return create_test_config(sample_config_data["model_list"])


@pytest.fixture
def temp_config_file(sample_config_data: dict[str, Any]) -> Path:
    """Temporary config file fixture."""
    temp_path = create_temp_config_file(sample_config_data)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLM response object."""

    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockChoice:
        def __init__(self, message: MockMessage):
            self.message = message

    class MockResponse:
        def __init__(self, content: str):
            self.choices = [MockChoice(MockMessage(content))]

    return MockResponse


@pytest.fixture
def mock_litellm_empty_response():
    """Mock LiteLM empty response object."""

    class MockResponse:
        def __init__(self):
            self.choices = []

    return MockResponse


@pytest.fixture
def mock_litellm_no_content_response():
    """Mock LiteLM response with no content."""

    class MockMessage:
        def __init__(self):
            self.content = None

    class MockChoice:
        def __init__(self, message: MockMessage):
            self.message = message

    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice(MockMessage())]

    return MockResponse
