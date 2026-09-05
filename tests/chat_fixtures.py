"""Typed provider fixtures shared by cross-layer tests; no real API calls."""

from pathlib import Path
from unittest.mock import MagicMock

from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelCapabilities, ModelConfig

PNG = b"\x89PNG\r\n\x1a\nreview-fixture"


def response(content: object = "answer", **message_fields: object) -> MagicMock:
    result = MagicMock()
    result.model_dump.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    **message_fields,
                }
            }
        ]
    }
    return result


def configured_client(root: Path) -> LLMClient:
    return LLMClient(
        ConferLLMConfig(
            sessions_dir=root,
            model_list=[
                ModelConfig(
                    model_name="vision",
                    litellm_params={"model": "openai/review-fixture"},
                    capabilities=ModelCapabilities(
                        input_modalities=["text", "image"],
                        output_modalities=["text", "image"],
                        max_input_images=4,
                    ),
                )
            ],
        )
    )
