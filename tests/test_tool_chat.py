"""Native tool loops through real service/CLI/MCP code and mocked providers."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from conferllm.chat import ChatService
from conferllm.cli import run_cli
from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.errors import ConferLLMError
from conferllm.images import ImageProcessingError
from conferllm.server import create_mcp_server
from conferllm.session import SessionError
from conferllm.tools import ToolRuntime, tool_definitions
from tests.chat_fixtures import PNG, configured_client, response

DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def call(name: str, arguments: dict | str, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments
            if isinstance(arguments, str)
            else json.dumps(arguments),
        },
    }


def tool_response(*calls: dict, content: object = None, **fields: object) -> MagicMock:
    return response(content, tool_calls=list(calls), **fields)


@pytest.mark.parametrize(
    "model",
    [
        "openai/custom-model",
        "anthropic/custom-model",
        "gemini/custom-model",
        "vertex_ai/custom-model",
        "bedrock/custom-model",
        "azure/deployment",
        "ollama/model",
        "ollama_chat/model",
        "openrouter/custom-model",
        "deepseek/custom-model",
    ],
)
def test_every_configured_model_receives_the_same_native_tools(
    tmp_path: Path, model: str
) -> None:
    client = LLMClient(
        ConferLLMConfig(
            sessions_dir=tmp_path / "sessions",
            model_list=[
                ModelConfig(
                    model_name="arbitrary-alias",
                    litellm_params={"model": model},
                    api_format="chat_completion",
                )
            ],
        )
    )
    path = tmp_path / "output.txt"
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(call("write_file", {"path": str(path), "content": "done"})),
            response("finished"),
        ],
    ) as provider:
        result = ChatService(client).chat("make the file", model="arbitrary-alias")
    assert result.text == "finished"
    assert path.read_text() == "done"
    expected_model = (
        model.replace("ollama/", "ollama_chat/", 1)
        if model.startswith("ollama/")
        else model
    )
    for request in provider.call_args_list:
        assert request.kwargs["model"] == expected_model
        assert request.kwargs["tools"] == tool_definitions()
        assert request.kwargs["tool_choice"] == "auto"
        assert request.kwargs["drop_params"] is False
        assert request.kwargs["stream"] is False


def test_provider_parameters_cannot_disable_or_replace_builtin_tools() -> None:
    params = {
        "model": "openai/custom",
        "tools": [],
        "tool_choice": "none",
        "stream": True,
        "drop_params": True,
        "functions": [{"name": "other"}],
        "function_call": "none",
        "additional_drop_params": ["tools", "temperature", "tool_choice"],
    }
    client = LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="test",
                    litellm_params=params,
                    api_format="chat_completion",
                )
            ]
        )
    )
    with patch(
        "conferllm.client.litellm.completion", return_value=response()
    ) as provider:
        client.chat("test", [{"role": "user", "content": "hello"}])
    sent = provider.call_args.kwargs
    assert sent["tools"] == tool_definitions()
    assert sent["tool_choice"] == "auto"
    assert not sent["stream"] and not sent["drop_params"]
    assert "functions" not in sent and "function_call" not in sent
    assert sent["additional_drop_params"] == ["temperature"]
    assert client.config.model_list[0].litellm_params == params


def test_custom_ollama_provider_uses_native_chat_route() -> None:
    client = LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="test",
                    litellm_params={"model": "custom", "custom_llm_provider": "ollama"},
                ),
            ]
        )
    )
    with patch(
        "conferllm.client.litellm.completion", return_value=response()
    ) as provider:
        client.chat("test", [{"role": "user", "content": "hello"}])
    assert provider.call_args.kwargs["custom_llm_provider"] == "ollama_chat"


def test_multiple_rounds_sequential_calls_history_replay_and_usage(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    path = tmp_path / "output"
    first = tool_response(
        call("write_file", {"path": str(path), "content": "one"}, "write"),
        content="I will create it.",
        reasoning_content="provider reasoning",
    )
    second = tool_response(
        call(
            "edit_file",
            {"path": str(path), "old_string": "one", "new_string": "two"},
            "edit",
        ),
        call("read_file", {"path": str(path)}, "read"),
    )
    final = response("It now says two.")
    for index, item in enumerate([first, second, final], start=1):
        item.model_dump.return_value["usage"] = {
            "total_tokens": index * 10,
            "prompt_tokens_details": {"cached_tokens": index},
        }
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[first, second, final, response("Follow-up")],
    ) as provider:
        result = service.chat(
            "Create and edit", model="vision", include_raw_response=True
        )
        assert path.read_text() == "two"
        loaded = service.session_store.load_session(result.session_id)
        assert [message["role"] for message in loaded.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "tool",
            "assistant",
        ]
        assert loaded.turns[0]["usage"] == {
            "total_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 6},
        }
        assert loaded.messages[1]["reasoning_content"] == "provider reasoning"
        assert json.loads(loaded.messages[5]["content"])["output"].endswith("two")
        assert result.response["usage"]["total_tokens"] == 30
        assert "tool_calls" not in result.content[0]
        path.write_text("external edit", encoding="utf-8")
        with patch.object(
            ToolRuntime, "execute", side_effect=AssertionError("no replay execution")
        ):
            follow_up = service.chat("Continue", session_id=result.session_id)
    assert follow_up.turn == 2
    assert path.read_text() == "external edit"
    replay = provider.call_args_list[3].kwargs["messages"]
    assert replay[:-1] == loaded.messages
    assert len(provider.call_args_list[0].kwargs["messages"]) == 1
    assert len(provider.call_args_list[1].kwargs["messages"]) == 3


def test_tool_failures_allow_model_to_correct_arguments(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    path = tmp_path / "result"
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(
                call("missing_tool", {}, "unknown"),
                call("write_file", "{invalid", "invalid"),
                call("read_file", {"path": str(path)}, "missing-file"),
            ),
            tool_response(
                call("write_file", {"path": str(path), "content": "fixed"}, "fixed")
            ),
            response("Recovered"),
        ],
    ) as provider:
        result = ChatService(client).chat("Create the file", model="vision")
    assert result.text == "Recovered" and path.read_text() == "fixed"
    errors = provider.call_args_list[1].kwargs["messages"][2:]
    assert len(errors) == 3
    assert all(not json.loads(message["content"])["ok"] for message in errors)
    assert [message["tool_call_id"] for message in errors] == [
        "unknown",
        "invalid",
        "missing-file",
    ]


@pytest.mark.parametrize(
    "calls",
    [
        "not-an-array",
        [{}],
        [
            {
                "id": "x",
                "type": "custom",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
        [call("read_file", {}, "")],
        [call("", {})],
        [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "read_file", "arguments": {}},
            }
        ],
        [call("read_file", {}), call("read_file", {})],
    ],
)
def test_malformed_protocol_rejects_whole_batch_before_execution(
    tmp_path: Path, calls: object
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    with (
        patch(
            "conferllm.client.litellm.completion",
            return_value=response(None, tool_calls=calls),
        ),
        patch.object(ToolRuntime, "execute") as execute,
        pytest.raises(ImageProcessingError),
    ):
        service.chat("request", model="vision")
    execute.assert_not_called()
    assert service.session_store.list_sessions() == []


def test_duplicate_id_in_later_round_does_not_execute_twice(tmp_path: Path) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    path = tmp_path / "output"
    requested = call("append_file", {"path": str(path), "content": "once"})
    with (
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[tool_response(requested), tool_response(requested)],
        ),
        pytest.raises(ConferLLMError, match="repeated a tool call ID") as failure,
    ):
        service.chat("append", model="vision")
    assert path.read_text() == "once"
    assert failure.value.details["tool_calls_attempted"] == 1
    assert failure.value.details["side_effects_may_remain"]
    assert service.session_store.list_sessions() == []


@pytest.mark.parametrize("final_reply", [False, True])
def test_tool_round_limit_allows_final_reply_but_no_more_execution(
    tmp_path: Path, final_reply: bool
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    path = tmp_path / "out"
    responses = [
        tool_response(
            call("append_file", {"path": str(path), "content": "x"}, str(index))
        )
        for index in range(3)
    ]
    if final_reply:
        responses[-1] = response("done")
    with (
        patch("conferllm.chat.MAX_TOOL_ROUNDS", 2),
        patch("conferllm.client.litellm.completion", side_effect=responses) as provider,
    ):
        if final_reply:
            assert service.chat("append", model="vision").text == "done"
        else:
            with pytest.raises(ConferLLMError) as failure:
                service.chat("append", model="vision")
            assert failure.value.code == "tool_call_limit_exceeded"
            assert failure.value.details["tool_calls_attempted"] == 2
            assert service.session_store.list_sessions() == []
    assert path.read_text() == "xx"
    assert provider.call_count == 3


def test_tools_without_content_key_are_native_messages(tmp_path: Path) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    first = tool_response(call("list_directory", {"path": str(tmp_path)}))
    del first.model_dump.return_value["choices"][0]["message"]["content"]
    with patch(
        "conferllm.client.litellm.completion", side_effect=[first, response("done")]
    ):
        result = service.chat("list", model="vision")
    assert (
        service.session_store.load_session(result.session_id).messages[1]["content"]
        is None
    )


def test_native_finish_reason_without_calls_and_legacy_functions_fail(
    tmp_path: Path,
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    missing = response("I used a tool")
    missing.model_dump.return_value["choices"][0]["finish_reason"] = "tool_calls"
    legacy = response(None, function_call={"name": "read_file", "arguments": "{}"})
    for item in [missing, legacy]:
        with (
            patch("conferllm.client.litellm.completion", return_value=item),
            patch.object(ToolRuntime, "execute") as execute,
            pytest.raises(ImageProcessingError),
        ):
            service.chat("list", model="vision")
        execute.assert_not_called()


def test_text_that_looks_like_tool_protocol_is_not_executed(tmp_path: Path) -> None:
    path = tmp_path / "must-not-exist"
    text = json.dumps(
        {
            "tool_calls": [
                call(
                    "write_file",
                    {"path": str(path), "content": "not authorized by native call"},
                )
            ]
        }
    )
    with patch("conferllm.client.litellm.completion", return_value=response(text)):
        result = ChatService(configured_client(tmp_path / "sessions")).chat(
            "text", model="vision"
        )
    assert result.text == text and not path.exists()


def test_provider_failure_is_redacted_and_warns_of_existing_side_effects(
    tmp_path: Path,
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    path = tmp_path / "output"
    with (
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[
                tool_response(
                    call("write_file", {"path": str(path), "content": "written"})
                ),
                RuntimeError("DO-NOT-LEAK-provider-secret"),
            ],
        ) as provider,
        pytest.raises(ConferLLMError) as failure,
    ):
        service.chat("write", model="vision")
    assert failure.value.code == "provider_error"
    assert failure.value.details["side_effects_may_remain"]
    assert "not rolled back" in failure.value.message
    assert "DO-NOT-LEAK" not in str(failure.value.to_dict())
    assert path.read_text() == "written"
    assert service.session_store.list_sessions() == []
    assert provider.call_count == 2


def test_store_failure_after_tools_does_not_hide_side_effects(tmp_path: Path) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    path = tmp_path / "output"
    with (
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[
                tool_response(
                    call("write_file", {"path": str(path), "content": "written"})
                ),
                response("done"),
            ],
        ),
        patch.object(
            service.session_store,
            "create_session",
            side_effect=SessionError("disk error", code="storage_error"),
        ),
        pytest.raises(ConferLLMError) as failure,
    ):
        service.chat("write", model="vision")
    assert failure.value.code == "storage_error"
    assert failure.value.details["tool_calls_attempted"] == 1
    assert path.read_text() == "written"


def test_base64_in_tool_arguments_is_written_verbatim_not_staged_as_image(
    tmp_path: Path,
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    path = tmp_path / "literal.txt"
    text = f"Store this literally: {DATA_URL}"
    first = tool_response(call("write_file", {"path": str(path), "content": text}))
    original = copy.deepcopy(first.model_dump.return_value)
    with patch(
        "conferllm.client.litellm.completion", side_effect=[first, response("done")]
    ):
        result = service.chat("write", model="vision")
    assert path.read_text() == text
    assert result.artifacts == []
    assert first.model_dump.return_value == original
    loaded = service.session_store.load_session(result.session_id)
    assert (
        loaded.messages[1]["tool_calls"][0]["function"]["arguments"]
        == (original["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_images_in_multiple_tool_rounds_are_unique_and_replayable(
    tmp_path: Path, legacy: bool
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    with patch("conferllm.client.litellm.completion", return_value=response("start")):
        first = service.chat("start", model="vision")
    store = service.session_store
    session_path = store.session_path(first.session_id)
    if legacy:
        records = [json.loads(line) for line in session_path.read_text().splitlines()]
        records[0]["schema_version"] = 1
        records[1].pop("artifacts")
        session_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    requested = call("list_directory", {"path": str(tmp_path)})
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(requested, images=[{"image_url": {"url": DATA_URL}}]),
            response(DATA_URL),
            response("follow-up"),
        ],
    ) as provider:
        result = service.chat("use tools and draw", session_id=first.session_id)
        service.chat("remember both images", session_id=first.session_id)
    assert [artifact["id"] for artifact in result.artifacts] == [
        "t0002-output-001",
        "t0002-output-002",
    ]
    assert all(
        store.read_artifact(first.session_id, item["id"]) == PNG
        for item in result.artifacts
    )
    intermediate_replay = provider.call_args_list[1].kwargs["messages"]
    assert intermediate_replay[-2]["content"][0]["image_url"]["url"] == DATA_URL
    continued = provider.call_args_list[2].kwargs["messages"]
    urls = [
        item["image_url"]["url"]
        for message in continued
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item["type"] == "image_url"
    ]
    assert urls == [DATA_URL, DATA_URL]
    assert store.load_session(first.session_id).schema_version == (1 if legacy else 2)


@pytest.mark.parametrize(
    "corruption",
    [
        "null",
        "missing-result",
        "wrong-id",
        "wrong-name",
        "duplicate-result",
        "wrong-role",
        "nontext",
        "no-calls",
        "final-calls",
        "out-of-order",
    ],
)
def test_corrupt_tool_history_fails_before_provider(
    tmp_path: Path, corruption: str
) -> None:
    service = ChatService(configured_client(tmp_path / "sessions"))
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(call("list_directory", {"path": str(tmp_path)})),
            response("done"),
        ],
    ):
        first = service.chat("list", model="vision")
    path = service.session_store.session_path(first.session_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    turn = records[1]
    messages = turn["tool_messages"]
    if corruption == "null":
        turn["tool_messages"] = None
    elif corruption == "missing-result":
        messages.pop()
    elif corruption == "wrong-id":
        messages[1]["tool_call_id"] = "wrong"
    elif corruption == "wrong-name":
        messages[1]["name"] = "wrong"
    elif corruption == "duplicate-result":
        messages.append(copy.deepcopy(messages[1]))
    elif corruption == "wrong-role":
        messages[0]["role"] = "user"
    elif corruption == "nontext":
        messages[1]["content"] = {}
    elif corruption == "no-calls":
        messages[0]["tool_calls"] = []
    elif corruption == "final-calls":
        turn["assistant"]["tool_calls"] = messages[0]["tool_calls"]
    else:
        messages.reverse()
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with (
        patch("conferllm.client.litellm.completion") as provider,
        pytest.raises(SessionError),
    ):
        service.chat("continue", session_id=first.session_id)
    provider.assert_not_called()


def test_cli_runs_tools_and_emits_only_final_json(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    path = tmp_path / "out"
    output, errors = StringIO(), StringIO()
    with (
        patch("conferllm.cli._load_client", return_value=client),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[
                tool_response(
                    call("write_file", {"path": str(path), "content": "CLI"})
                ),
                tool_response(
                    call(
                        "run_shell",
                        {
                            "command": "test -f out && printf verified",
                            "cwd": str(tmp_path),
                        },
                        "verify",
                    )
                ),
                response("Verified"),
            ],
        ),
    ):
        status = run_cli(
            ["chat", "--model", "vision", "--prompt", "write and check", "--json"],
            output=output,
            error_output=errors,
        )
    assert status == 0 and errors.getvalue() == ""
    assert json.loads(output.getvalue())["message"]["text"] == "Verified"
    assert path.read_text() == "CLI"


async def test_mcp_wire_chat_runs_tools_and_replays_results(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client = configured_client(tmp_path / "sessions")
    path = tmp_path / "mcp-out"
    server = create_mcp_server(client)
    with (
        anyio.fail_after(15),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[
                tool_response(
                    call("write_file", {"path": str(path), "content": "MCP"})
                ),
                response("Written"),
                response("Remembered"),
            ],
        ) as provider,
    ):
        async with (
            create_client_server_memory_streams() as (client_streams, server_streams),
            anyio.create_task_group() as tasks,
        ):
            tasks.start_soon(
                server._lowlevel_server.run,
                *server_streams,
                server._lowlevel_server.create_initialization_options(),
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                created = await session.call_tool(
                    "create_chat", {"model": "vision", "message": "write"}
                )
                assert not created.is_error
                assert created.structured_content["message"]["text"] == "Written"
                continued = await session.call_tool(
                    "continue_chat",
                    {
                        "session_id": created.structured_content["session"]["id"],
                        "message": "remember",
                    },
                )
                assert not continued.is_error
                assert continued.structured_content["session"]["turn"] == 2
            tasks.cancel_scope.cancel()
    assert path.read_text() == "MCP"
    progress = [
        record.progress
        for record in caplog.records
        if record.name == "conferllm.progress"
    ]
    assert [entry["event"] for entry in progress] == [
        "model.request",
        "model.response",
        "tool.start",
        "tool.result",
        "model.request",
        "model.response",
        "model.request",
        "model.response",
    ]
    assert progress[2]["name"] == "write_file"
    assert progress[3]["output"]["ok"] is True
    assert progress[-1]["turn"] == 2
    assert all(
        entry["session"] == created.structured_content["session"]["id"]
        for entry in progress
    )
    assert any(
        message["role"] == "tool"
        for message in provider.call_args_list[2].kwargs["messages"]
    )


async def test_cancelled_mcp_request_does_not_execute_a_late_tool_batch(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    server = create_mcp_server(client)
    target = tmp_path / "must-not-be-written"
    entered, release, finished = Event(), Event(), Event()
    original_chat = ChatService.chat
    attempts = 0

    def provider(**_kwargs: object) -> MagicMock:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            assert release.wait(timeout=5)
            return tool_response(
                call("write_file", {"path": str(target), "content": "too late"})
            )
        return response("done")

    def chat(service: ChatService, *args: object, **kwargs: object) -> object:
        try:
            return original_chat(service, *args, **kwargs)
        finally:
            finished.set()

    with (
        patch("conferllm.client.litellm.completion", side_effect=provider),
        patch.object(ChatService, "chat", chat),
    ):
        task = asyncio.create_task(
            server.call_tool("create_chat", {"model": "vision", "message": "write"})
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 3)
    assert attempts == 1
    assert not target.exists()


def test_cancel_between_tool_calls_reports_only_attempted_calls(tmp_path: Path) -> None:
    from conferllm.tools.common import cancellation_scope

    client = configured_client(tmp_path / "sessions")
    target = tmp_path / "output"
    cancelled = Event()
    original_execute = ToolRuntime.execute

    def execute(runtime: ToolRuntime, name: str, arguments: str) -> str:
        result = original_execute(runtime, name, arguments)
        cancelled.set()
        return result

    with (
        cancellation_scope(cancelled),
        patch.object(ToolRuntime, "execute", execute),
        patch(
            "conferllm.client.litellm.completion",
            return_value=tool_response(
                call("append_file", {"path": str(target), "content": "once"}, "first"),
                call(
                    "append_file", {"path": str(target), "content": "twice"}, "second"
                ),
            ),
        ),
        pytest.raises(ConferLLMError) as failure,
    ):
        ChatService(client).chat("append", model="vision")
    assert target.read_text() == "once"
    assert failure.value.code == "chat_cancelled"
    assert failure.value.details["tool_calls_attempted"] == 1
