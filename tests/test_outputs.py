"""All choices/blocks are handled; generated images are returned as saved paths."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.images import ImageProcessingError
from conferllm.outputs import image_part
from conferllm.server import _chat_tool_result
from tests.chat_fixtures import PNG
from tests.test_responses import client, function, message, native

DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [
                {"type": "image_url", "image_url": {"url": "/tmp/one.png"}},
                {"type": "image_url", "image_url": {"url": "/tmp/two.png"}},
            ],
            "/tmp/one.png\n/tmp/two.png",
        ),
        (
            [
                {"type": "text", "text": "Before"},
                {"type": "image_url", "image_url": {"url": "/tmp/one.png"}},
                {"type": "text", "text": "After"},
            ],
            "Before\n/tmp/one.png\nAfter",
        ),
        (
            [
                {"type": "text", "text": "Image: "},
                {"type": "image_url", "image_url": {"url": "/tmp/one.png"}},
                {"type": "text", "text": "\nDone"},
                {"type": "text", "text": "."},
            ],
            "Image: /tmp/one.png\nDone.",
        ),
    ],
)
def test_public_image_paths_have_readable_boundaries(
    content: list[dict], expected: str
) -> None:
    assert ChatService._public_text(content, []) == expected


def response(*messages: dict) -> MagicMock:
    result = MagicMock()
    result.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", **item}} for item in messages]
    }
    return result


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "gemini", "deepseek", "ollama_chat"]
)
def test_all_candidates_and_native_content_blocks_execute_and_save(
    tmp_path: Path, provider: str
) -> None:
    target = tmp_path / "written.txt"
    active = LLMClient(
        ConferLLMConfig(
            sessions_dir=tmp_path / "sessions",
            model_list=[
                ModelConfig(
                    model_name="model",
                    api_format="chat_completion",
                    litellm_params={"model": f"{provider}/fixture"},
                )
            ],
        )
    )
    first = response(
        {
            "content": [
                {"type": "text", "text": "First image:"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
            ]
        },
        {
            "content": [
                {"type": "text", "text": "Second candidate"},
                {
                    "type": "tool_use",
                    "id": "call_write",
                    "name": "write_file",
                    "input": {"path": str(target), "content": "one"},
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(PNG).decode(),
                    },
                },
                {
                    "type": "tool_use",
                    "id": "call_append",
                    "name": "append_file",
                    "input": {"path": str(target), "content": "-two"},
                },
            ]
        },
        {
            "content": "Third candidate",
            "images": [{"b64_json": base64.b64encode(PNG).decode()}],
        },
    )
    final = response(
        {"content": "Final one."},
        {"content": None, "images": [{"image_url": {"url": DATA_URL}}]},
        {"content": "Final three."},
    )
    service = ChatService(active)
    with patch(
        "conferllm.client.litellm.completion", side_effect=[first, final]
    ) as endpoint:
        result = service.chat("perform all", model="model", include_raw_response=True)
    assert target.read_text() == "one-two"
    assert len(result.artifacts) == 4
    paths = [artifact["exported_path"] for artifact in result.artifacts]
    assert all(Path(path).read_bytes() == PNG for path in paths)
    assert all(Path(path).parent == tmp_path / "image-exports" for path in paths)
    assert "Final one." in result.text and "Final three." in result.text
    assert all(path in result.text for path in paths)
    assert len(result.response["choices"]) == 3
    assert (
        result.response["choices"][1]["message"]["images"][0]["image_url"]["url"]
        == paths[-1]
    )
    assert "data:image" not in json.dumps(result.to_dict())
    history = endpoint.call_args_list[1].kwargs["messages"]
    assert [call["id"] for call in history[1]["tool_calls"]] == [
        "call_write",
        "call_append",
    ]
    assert len([item for item in history if item["role"] == "tool"]) == 2
    saved = service.session_store.load_session(result.session_id)
    assert saved.messages[1]["content"][1]["type"] == "image_ref"
    assert all(path not in json.dumps(saved.messages) for path in paths)


def test_responses_text_tools_and_native_image_results_are_all_handled(
    tmp_path: Path,
) -> None:
    service = ChatService(client(tmp_path / "sessions"))
    image = {
        "type": "image_generation_call",
        "id": "ig_123",
        "status": "completed",
        "result": base64.b64encode(PNG).decode(),
        "output_format": "png",
    }
    with patch(
        "conferllm.client.litellm.responses",
        side_effect=[
            native(
                [
                    message("Image and tools:"),
                    image,
                    function("list_directory", {"path": str(tmp_path)}, "call_list"),
                    function(
                        "read_file", {"path": str(tmp_path / "missing")}, "call_read"
                    ),
                ]
            ),
            native([message("Completed."), {**image, "id": "ig_456"}]),
        ],
    ) as endpoint:
        result = service.chat(
            "make images", model="test", image_output_dir=tmp_path / "chosen"
        )
    assert len(result.artifacts) == 2
    assert all(
        Path(item["exported_path"]).parent == tmp_path / "chosen"
        for item in result.artifacts
    )
    replay = endpoint.call_args_list[1].kwargs["input"]
    assert len([item for item in replay if item["type"] == "function_call_output"]) == 2
    assert any(
        part["type"] == "input_image" and part["image_url"] == DATA_URL
        for item in replay
        if item["type"] == "message"
        for part in item["content"]
    )
    assert all(
        item["role"] == "user"
        for item in replay
        if item["type"] == "message"
        and any(part["type"] == "input_image" for part in item["content"])
    )
    assert all("conferllm_image_generation" not in item for item in replay)
    assert "data:image" not in json.dumps(result.to_dict())
    envelope = _chat_tool_result(result, service.session_store)
    assert all(item.type != "image" for item in envelope.content)
    assert len([item for item in envelope.content if item.type == "resource_link"]) == 2
    assert all(
        artifact["saved_path"] for artifact in envelope.structured_content["artifacts"]
    )


def test_remote_images_are_downloaded_without_model_credentials(tmp_path: Path) -> None:
    active = client(tmp_path / "sessions")
    service = ChatService(active)
    requests: list[httpx.Request] = []

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: object
    ) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=PNG, request=request)

    with (
        patch(
            "conferllm.client.litellm.responses",
            return_value=native(
                [
                    message("Image"),
                ]
            ),
        ),
        patch.object(
            active,
            "chat",
            return_value=response(
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://images.invalid/a.png"},
                        },
                        {"type": "image_url", "image_url": {"url": DATA_URL}},
                    ]
                }
            ),
        ),
        patch.object(httpx.Client, "send", send),
    ):
        result = service.chat("images", model="test")
    assert len(result.artifacts) == 2
    assert len(requests) == 1
    assert "authorization" not in requests[0].headers
    assert all(
        Path(item["exported_path"]).read_bytes() == PNG for item in result.artifacts
    )


def test_remote_image_errors_are_redacted(tmp_path: Path) -> None:
    service = ChatService(client(tmp_path / "sessions"))
    with (
        patch.object(
            service.client,
            "chat",
            return_value=response(
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://images.invalid/a.png?token=not-for-public-errors"
                            },
                        }
                    ]
                }
            ),
        ),
        patch.object(
            httpx.Client, "send", side_effect=OSError("private-server-detail")
        ),
        pytest.raises(ImageProcessingError) as failure,
    ):
        service.chat("images", model="test")
    assert "not-for-public-errors" not in str(failure.value)
    assert "private-server-detail" not in str(failure.value)


def test_duplicate_ids_across_candidates_fail_before_any_execution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "untouched"
    tool = {
        "id": "same",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": str(target), "content": "x"}),
        },
    }
    service = ChatService(client(tmp_path / "sessions", "chat_completion"))
    with (
        patch.object(
            service.client,
            "chat",
            return_value=response(
                {"content": None, "tool_calls": [tool]},
                {"content": None, "tool_calls": [tool]},
            ),
        ),
        pytest.raises(ImageProcessingError, match="unique"),
    ):
        service.chat("write", model="test")
    assert not target.exists()


def test_mixed_multiple_image_inputs_are_all_sent_to_responses(tmp_path: Path) -> None:
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(PNG)
    second.write_bytes(PNG)
    service = ChatService(client(tmp_path / "sessions"))
    with patch(
        "conferllm.client.litellm.responses", return_value=native([message("both")])
    ) as endpoint:
        service.chat("compare", model="test", images=[first, second])
    images = [
        part
        for item in endpoint.call_args.kwargs["input"]
        for part in item.get("content", [])
        if part["type"] == "input_image"
    ]
    assert len(images) == 2
    assert all(item["image_url"] == DATA_URL for item in images)


@pytest.mark.parametrize(
    "value",
    [
        {"type": "image", "source": {"type": "url", "url": DATA_URL}},
        {"type": "image_url", "image_url": DATA_URL},
        {"type": "output_image", "b64_json": base64.b64encode(PNG).decode()},
    ],
)
def test_sdk_image_shapes_normalize_without_losing_data(value: dict) -> None:
    assert image_part(value)["image_url"]["url"] == DATA_URL
