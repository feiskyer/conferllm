"""Successive provider requests must extend the previous request byte for byte.

Providers cache the longest prefix they have already processed. Any rewrite of
the system prompt, tools, or replayed history turns into a full-price miss, and
some providers also reject replayed reasoning bound to a changed prefix. These
tests run the real LiteLLM adapters against an in-memory HTTP transport.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import litellm

from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.errors import ConferLLMError

# Captured during collection, before the autouse fixture blocks real providers.
# Each use replaces HTTP send itself; no request can leave the process.
_COMPLETION = litellm.completion
_RESPONSES = litellm.responses


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def assert_strict_prefixes(requests: list[list[Any]]) -> None:
    for previous, current in pairwise(requests):
        assert len(current) > len(previous)
        assert dumps(current[: len(previous)]) == dumps(previous)


def without_cache_markers(value: Any) -> Any:
    """Breakpoints move every request; the content they mark must not."""
    if isinstance(value, dict):
        return {
            key: without_cache_markers(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [without_cache_markers(item) for item in value]
    return value


@contextmanager
def in_memory_provider(host: str, replies: list[dict[str, Any]]) -> Iterator[list]:
    """Serve scripted JSON replies to the real adapters and record each body."""
    requests: list[dict[str, Any]] = []
    failures: list[Exception] = []

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: Any
    ) -> httpx.Response:
        assert request.url.host == host, "Unexpected HTTP request"
        requests.append(json.loads(request.content))
        assert replies, "Unexpected provider retry"
        return httpx.Response(
            200,
            json=copy.deepcopy(replies.pop(0)),
            headers={"content-type": "application/json"},
            request=request,
        )

    def recorded(endpoint: Any) -> Any:
        def call(**parameters: Any) -> Any:
            try:
                return endpoint(**parameters)
            except Exception as error:
                # Synthetic data only; keep adapter failures visible despite
                # production provider-error redaction.
                failures.append(error)
                raise

        return call

    with (
        patch("conferllm.client.litellm.completion", recorded(_COMPLETION)),
        patch("conferllm.client.litellm.responses", recorded(_RESPONSES)),
        patch.object(httpx.Client, "send", send),
    ):
        try:
            yield requests
        except ConferLLMError:
            if failures:
                raise failures[0] from None
            raise


def service_for(tmp_path: Path, api_format: str, **litellm_params: Any) -> ChatService:
    params = {"api_key": "synthetic-test-key", "max_retries": 0, "num_retries": 0}
    return ChatService(
        LLMClient(
            ConferLLMConfig(
                sessions_dir=tmp_path / "sessions",
                model_list=[
                    ModelConfig(
                        model_name="test",
                        api_format=api_format,
                        litellm_params={**params, **litellm_params},
                    )
                ],
            )
        )
    )


def anthropic_reply(
    content: list[dict[str, Any]], stop_reason: str, *, read: int, write: int
) -> dict[str, Any]:
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "content": content,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write,
        },
    }


def responses_reply(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.5",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 1500,
            "output_tokens": 5,
            "total_tokens": 1505,
            "input_tokens_details": {"cached_tokens": 1280},
        },
    }


def output_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_fixture",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def test_anthropic_requests_keep_a_stable_prefix_and_move_one_breakpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    replies = [
        anthropic_reply(
            [
                {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": str(target)},
                },
            ],
            "tool_use",
            read=0,
            write=1000,
        ),
        anthropic_reply(
            [
                {"type": "thinking", "thinking": "done", "signature": "sig-2"},
                {"type": "text", "text": "one"},
            ],
            "end_turn",
            read=1000,
            write=200,
        ),
        anthropic_reply(
            [{"type": "text", "text": "two"}], "end_turn", read=1200, write=100
        ),
    ]
    service = service_for(
        tmp_path,
        "chat_completion",
        model="anthropic/claude-sonnet-4-5",
        api_base="https://provider.invalid",
        max_tokens=4096,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    with in_memory_provider("provider.invalid", replies) as requests:
        first = service.chat("First question", model="test")
        second = service.chat("Second question", session_id=first.session_id)

    assert (first.text, second.text) == ("one", "two")
    assert len(requests) == 3
    assert all(body["system"] == requests[0]["system"] for body in requests)
    assert all(body["tools"] == requests[0]["tools"] for body in requests)
    assert_strict_prefixes(
        [without_cache_markers(body["messages"]) for body in requests]
    )
    # The system prompt and the latest message carry the only breakpoints.
    assert requests[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}
    for body in requests:
        marked = [
            index
            for index, message in enumerate(body["messages"])
            if "cache_control" in dumps(message)
        ]
        assert marked == [len(body["messages"]) - 1]
        assert body["messages"][-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral"
        }
    # Replayed reasoning keeps its signature, so the provider can validate it.
    assert requests[2]["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig-1",
    }
    assert first.to_dict()["usage"] == {
        "input_tokens": 2220,
        "cached_input_tokens": 1000,
        "cache_write_tokens": 1200,
        "output_tokens": 10,
        "cache_hit_ratio": 0.4505,
    }
    assert second.to_dict()["usage"]["cached_input_tokens"] == 1200


def test_responses_requests_extend_the_previous_input_with_one_session_key(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    replies = [
        responses_reply(
            [
                {
                    "type": "reasoning",
                    "id": "rs_fixture",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "function_call",
                    "id": "fc_fixture",
                    "call_id": "call_1",
                    "status": "completed",
                    "name": "read_file",
                    "arguments": json.dumps({"path": str(target)}),
                },
            ]
        ),
        responses_reply([output_message("one")]),
        responses_reply([output_message("two")]),
    ]
    # An official endpoint lets the routing key through; send() stays in memory.
    service = service_for(
        tmp_path,
        "responses",
        model="openai/gpt-5.5",
        api_base="https://api.openai.com/v1",
    )

    with in_memory_provider("api.openai.com", replies) as requests:
        first = service.chat("First question", model="test")
        second = service.chat("Second question", session_id=first.session_id)

    assert (first.text, second.text) == ("one", "two")
    assert_strict_prefixes([body["input"] for body in requests])
    assert all(body["tools"] == requests[0]["tools"] for body in requests)
    assert {body["prompt_cache_key"] for body in requests} == {first.session_id}
    assert second.to_dict()["usage"]["cache_hit_ratio"] == 0.8533


def test_chat_completion_requests_extend_the_previous_messages(tmp_path: Path) -> None:
    """Native assistant fields such as reasoning_content replay unchanged."""
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    def reply(content: str | None, **fields: Any) -> MagicMock:
        result = MagicMock()
        result.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": f"reasoning for {content}",
                        **fields,
                    }
                }
            ]
        }
        return result

    call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": str(target)}),
        },
    }
    service = service_for(tmp_path, "chat_completion", model="deepseek/deepseek-flash")
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[reply(None, tool_calls=[call]), reply("one"), reply("two")],
    ) as completion:
        first = service.chat("First question", model="test")
        service.chat("Second question", session_id=first.session_id)

    requests = [item.kwargs for item in completion.call_args_list]
    assert_strict_prefixes([request["messages"] for request in requests])
    assert all(request["tools"] == requests[0]["tools"] for request in requests)
    assert requests[2]["messages"][2]["reasoning_content"] == "reasoning for None"
