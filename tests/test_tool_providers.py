"""Exercise real LiteLLM native adapters against an in-memory HTTP transport."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest

from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.errors import ConferLLMError
from conferllm.tools import TOOLS
from tests.chat_fixtures import PNG

# Captured during collection, before the autouse fixture blocks real providers.
# Each use below replaces HTTP send itself; no request can leave the process.
_COMPLETION = litellm.completion
_RESPONSES = litellm.responses


def _responses(provider: str, arguments: dict[str, str]) -> list[dict[str, Any]]:
    if provider == "openai_responses":
        return [
            {
                "id": f"resp_fixture_{index}",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
            for index, output in enumerate(
                [
                    [
                        {
                            "type": "reasoning",
                            "id": "rs_fixture",
                            "summary": [],
                            "encrypted_content": "opaque-reasoning",
                        },
                        {
                            "type": "message",
                            "id": "msg_commentary",
                            "role": "assistant",
                            "status": "completed",
                            "phase": "commentary",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "I will write.",
                                    "annotations": [],
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_item_id",
                            "call_id": "call_0",
                            "status": "completed",
                            "name": "write_file",
                            "arguments": json.dumps(arguments),
                        },
                    ],
                    [
                        {
                            "type": "message",
                            "id": "msg_final",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "native done",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    [
                        {
                            "type": "message",
                            "id": "msg_followup",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "remembered",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                ],
                start=1,
            )
        ]
    if provider == "anthropic":
        envelope = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return [
            {
                **envelope,
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_test",
                        "name": "write_file",
                        "input": arguments,
                    }
                ],
            },
            {
                **envelope,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "native done"}],
            },
        ]
    if provider == "gemini":
        return [
            {
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": parts},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
            for parts in [
                [{"functionCall": {"name": "write_file", "args": arguments}}],
                [{"text": "native done"}],
            ]
        ]
    if provider == "ollama":
        return [
            {
                "model": "qwen2.5",
                "created_at": "2026-09-20T00:00:00Z",
                "done": True,
                "done_reason": "stop",
                "message": message,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
            for message in [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {"name": "write_file", "arguments": arguments},
                        }
                    ],
                },
                {"role": "assistant", "content": "native done"},
            ]
        ]
    return [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls"
                    if "tool_calls" in message
                    else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        for message in [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "native done"},
        ]
    ]


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "openai/gpt-4o"),
        ("openai_responses", "openai/gpt-6-astra"),
        ("anthropic", "anthropic/claude-3-5-sonnet-20241022"),
        ("gemini", "gemini/gemini-2.0-flash"),
        ("ollama", "ollama/qwen2.5"),
    ],
)
def test_native_provider_wire_round_trip(
    tmp_path: Path, provider: str, model: str
) -> None:
    target = tmp_path / "native.txt"
    replies = _responses(provider, {"path": str(target), "content": "native"})
    requests: list[dict[str, Any]] = []
    failures: list[Exception] = []
    client = LLMClient(
        ConferLLMConfig(
            sessions_dir=tmp_path / "sessions",
            model_list=[
                ModelConfig(
                    model_name="test",
                    api_format="chat_completion"
                    if provider == "openai"
                    else "responses",
                    litellm_params={
                        "model": model,
                        "api_base": "https://provider.invalid",
                        "api_key": "synthetic-test-key",
                        "max_retries": 0,
                        "num_retries": 0,
                    },
                )
            ],
        )
    )

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: Any
    ) -> httpx.Response:
        assert request.url.host == "provider.invalid", "Unexpected HTTP request"
        if provider == "ollama" and request.url.path == "/api/show":
            return httpx.Response(
                200,
                request=request,
                json={
                    "template": "{{ .Tools }}",
                    "model_info": {"qwen2.context_length": 32768},
                },
            )
        requests.append(json.loads(request.content))
        assert replies, "Unexpected provider retry"
        return httpx.Response(
            200,
            json=copy.deepcopy(replies.pop(0)),
            headers={"content-type": "application/json"},
            request=request,
        )

    def completion(**parameters: Any) -> Any:
        try:
            return _COMPLETION(**parameters)
        except Exception as error:
            # All request data is synthetic; retain adapter failures in this
            # test while keeping production provider-error redaction intact.
            failures.append(error)
            raise

    def responses(**parameters: Any) -> Any:
        try:
            return _RESPONSES(**parameters)
        except Exception as error:
            failures.append(error)
            raise

    with (
        patch("conferllm.client.litellm.completion", completion),
        patch("conferllm.client.litellm.responses", responses),
        patch.object(httpx.Client, "send", send),
    ):
        try:
            service = ChatService(client)
            result = service.chat("Write the requested file", model="test")
            if provider == "openai_responses":
                followup = service.chat("Remember it", session_id=result.session_id)
                assert followup.text == "remembered"
        except ConferLLMError:
            if failures:
                raise failures[0] from None
            raise

    assert result.text == "native done"
    assert target.read_text() == "native"
    assert len(requests) == (3 if provider == "openai_responses" else 2) and not replies
    expected_names = {tool.name for tool in TOOLS}
    for request in requests:
        if provider in {"anthropic", "openai_responses"}:
            assert {tool["name"] for tool in request["tools"]} == expected_names
            if provider == "openai_responses":
                assert all("function" not in tool for tool in request["tools"])
                assert all(tool["strict"] is False for tool in request["tools"])
                assert request["store"] is False
                assert "messages" not in request
                assert "previous_response_id" not in request
        elif provider == "gemini":
            functions = request["tools"][0]["function_declarations"]
            assert {tool["name"] for tool in functions} == expected_names
        else:
            assert {
                tool["function"]["name"] for tool in request["tools"]
            } == expected_names
    follow_up = requests[1]
    if provider == "openai_responses":
        assert [item["type"] for item in follow_up["input"]] == [
            "message",
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
        ]
        native_call = follow_up["input"][-2]
        assert "id" not in native_call
        assert native_call["call_id"] == "call_0"
        assert follow_up["input"][-1]["call_id"] == "call_0"
        assert isinstance(follow_up["input"][-1]["output"], str)
        assert follow_up["input"][1]["encrypted_content"] == "opaque-reasoning"
        assert follow_up["input"][2]["phase"] == "commentary"
        assert follow_up["input"][2]["content"][0]["type"] == "output_text"
        assert requests[2]["input"][-2]["content"][0]["type"] == "output_text"
        assert requests[2]["input"][: len(follow_up["input"])] == follow_up["input"]
    elif provider == "anthropic":
        assert any(
            block.get("type") == "tool_result"
            for message in follow_up["messages"]
            for block in message["content"]
        )
    elif provider == "gemini":
        assert any(
            "function_response" in part
            for content in follow_up["contents"]
            for part in content["parts"]
        )
    else:
        assert any(message["role"] == "tool" for message in follow_up["messages"])


def test_gemini_multiple_candidates_do_not_inherit_previous_tools_or_images(
    tmp_path: Path,
) -> None:
    target = tmp_path / "native.txt"
    replies = _responses("gemini", {"path": str(target), "content": "native"})
    replies[0]["candidates"][0]["content"]["parts"].insert(
        0,
        {
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(PNG).decode(),
            }
        },
    )
    replies[0]["candidates"].append(
        {
            "index": 1,
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": "Another candidate."}]},
        }
    )
    client = LLMClient(
        ConferLLMConfig(
            sessions_dir=tmp_path / "sessions",
            model_list=[
                ModelConfig(
                    model_name="test",
                    litellm_params={
                        "model": "gemini/gemini-2.0-flash",
                        "api_base": "https://provider.invalid",
                        "api_key": "synthetic",
                        "max_retries": 0,
                        "num_retries": 0,
                    },
                )
            ],
        )
    )

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: Any
    ) -> httpx.Response:
        assert request.url.host == "provider.invalid"
        assert replies
        return httpx.Response(200, request=request, json=replies.pop(0))

    with (
        patch("conferllm.client.litellm.completion", _COMPLETION),
        patch.object(httpx.Client, "send", send),
    ):
        result = ChatService(client).chat("Write the fixture file.", model="test")

    assert target.read_text() == "native"
    assert len(result.artifacts) == 1
    assert result.text.startswith("native done")
    assert not replies


def test_openai_chat_native_content_arrays_survive_sdk_conversion(
    tmp_path: Path,
) -> None:
    target = tmp_path / "native.txt"
    replies = _responses("openai", {"path": str(target), "content": "native"})
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    replies[0]["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": "after"},
    ]
    replies[0]["choices"].append(
        {
            "index": 1,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "second candidate"},
                ],
            },
        }
    )
    client = LLMClient(
        ConferLLMConfig(
            sessions_dir=tmp_path / "sessions",
            model_list=[
                ModelConfig(
                    model_name="test",
                    api_format="chat_completion",
                    litellm_params={
                        "model": "openai/gpt-4o",
                        "api_base": "https://provider.invalid",
                        "api_key": "synthetic",
                        "max_retries": 0,
                        "num_retries": 0,
                    },
                )
            ],
        )
    )

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: Any
    ) -> httpx.Response:
        assert request.url.host == "provider.invalid"
        assert replies
        return httpx.Response(200, request=request, json=replies.pop(0))

    with (
        patch("conferllm.client.litellm.completion", _COMPLETION),
        patch.object(httpx.Client, "send", send),
    ):
        service = ChatService(client)
        result = service.chat("Write the fixture file.", model="test")
    assert target.read_text() == "native"
    assert len(result.artifacts) == 2
    history = service.session_store.load_session(result.session_id).messages
    assert [part["type"] for part in history[1]["content"]][:3] == [
        "text",
        "image_ref",
        "text",
    ]
