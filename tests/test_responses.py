"""Responses format selection, conversion, and state/persistence regressions."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.errors import ConferLLMError
from conferllm.responses import (
    STATE_KEY,
    messages_to_input,
    prepare_params,
    response_items,
    strip_response_state,
    to_model_response,
    validate_response_state,
)
from conferllm.tools import tool_definitions
from tests.chat_fixtures import PNG


def native(output: list[dict], **fields: object) -> MagicMock:
    result = MagicMock()
    result.model_dump.return_value = {
        "id": "resp_test",
        "model": "gpt-6-astra",
        "created_at": 1,
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        **fields,
    }
    return result


def message(text: str, item_id: str = "msg_1") -> dict:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function(name: str, arguments: dict, call_id: str = "call_123") -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def client(root: Path, api_format: str = "responses") -> LLMClient:
    return LLMClient(
        ConferLLMConfig(
            sessions_dir=root,
            model_list=[
                ModelConfig(
                    model_name="test",
                    api_format=api_format,
                    litellm_params={
                        "model": "openai/gpt-6-astra",
                        "api_key": "synthetic",
                    },
                )
            ],
        )
    )


def test_format_is_model_level_and_defaults_to_responses() -> None:
    model = ModelConfig(model_name="test", litellm_params={"model": "openai/gpt-4o"})
    assert model.api_format == "responses"
    assert (
        ModelConfig(
            **{**model.model_dump(), "api_format": "chat_completion"}
        ).api_format
        == "chat_completion"
    )
    for value in ("auto", "chat_completions", "", None, 1):
        with pytest.raises(ValueError):
            ModelConfig(**{**model.model_dump(), "api_format": value})
    with pytest.raises(ValueError, match="not inside litellm_params"):
        ModelConfig(
            model_name="test",
            litellm_params={"model": "openai/gpt-4o", "api_format": "responses"},
        )


@pytest.mark.parametrize("provider_model", ["openai/custom", "gpt-4o"])
def test_default_openai_requests_use_responses(
    tmp_path: Path, provider_model: str
) -> None:
    active = client(tmp_path)
    active.config.model_list[0].litellm_params["model"] = provider_model
    with (
        patch(
            "conferllm.client.litellm.responses", return_value=native([message("done")])
        ) as endpoint,
        patch("conferllm.client.litellm.completion") as legacy,
    ):
        assert (
            active.chat("test", [{"role": "user", "content": "hello"}])
            .choices[0]
            .message.content
            == "done"
        )
    legacy.assert_not_called()
    assert endpoint.call_args.kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    assert endpoint.call_args.kwargs["tools"][0]["name"] == "run_shell"
    assert active.get_model_info("test")["uses_responses_api"]


@pytest.mark.parametrize("api_format", ["responses", "chat_completion"])
def test_non_openai_providers_keep_their_existing_path(
    tmp_path: Path, api_format: str
) -> None:
    active = client(tmp_path, api_format)
    active.config.model_list[0].litellm_params["model"] = "anthropic/claude-test"
    with patch("conferllm.client.litellm.completion") as completion:
        active.chat("test", [{"role": "user", "content": "hello"}])
    assert completion.call_args.kwargs["model"] == "anthropic/claude-test"
    assert not active.get_model_info("test")["uses_responses_api"]


def test_custom_provider_takes_precedence_over_model_name(tmp_path: Path) -> None:
    active = client(tmp_path)
    active.config.model_list[0].litellm_params.update(
        {"model": "private-deployment", "custom_llm_provider": "openai"}
    )
    with patch(
        "conferllm.client.litellm.responses", return_value=native([message("done")])
    ) as endpoint:
        active.chat("test", [{"role": "user", "content": "hello"}])
    assert endpoint.call_args.kwargs["custom_llm_provider"] == "openai"


def test_response_parameter_mapping_preserves_existing_options() -> None:
    params = {
        "max_tokens": 100,
        "max_completion_tokens": 200,
        "reasoning_effort": "high",
        "api_base": "https://example.invalid/v1",
        "api_key": "synthetic",
        "temperature": 0.5,
        "include": [],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "strict": True,
                "schema": {"type": "object"},
            },
        },
    }
    original = copy.deepcopy(params)
    result = prepare_params(params, tool_definitions())
    assert result["max_output_tokens"] == 200
    assert "max_tokens" not in result and "max_completion_tokens" not in result
    assert result["reasoning"] == {"effort": "high"}
    assert result["text"]["format"]["name"] == "result"
    assert "response_format" not in result and "reasoning_effort" not in result
    assert result["api_base"] == params["api_base"]
    assert result["api_key"] == params["api_key"]
    assert result["include"] == ["reasoning.encrypted_content"]
    assert not result["store"] and not result["background"] and not result["stream"]
    assert params == original


@pytest.mark.parametrize(
    "params",
    [
        {"input": []},
        {"previous_response_id": "other"},
        {"conversation": "other"},
        {"instructions": "override"},
        {"n": 2},
        {"stop": ["end"]},
        {"include": "invalid"},
    ],
)
def test_incompatible_or_history_overriding_parameters_are_explicit_errors(
    params: dict,
) -> None:
    with pytest.raises(ConferLLMError, match="configuration_error"):
        prepare_params(params, tool_definitions())


def test_all_output_items_are_retained_and_call_id_is_not_replaced() -> None:
    items = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "opaque-1",
        },
        message("Before."),
        function("read_file", {"path": "one"}, "call_0"),
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [],
            "encrypted_content": "opaque-2",
        },
        function("read_file", {"path": "two"}, "call_1"),
        message("After.", "msg_2"),
    ]
    result = to_model_response(native(items).model_dump()).model_dump()
    assert len(result["choices"]) == 1
    assistant = result["choices"][0]["message"]
    assert assistant["content"] == "Before.After."
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_0", "call_1"]
    replay = messages_to_input(
        [
            assistant,
            {"role": "tool", "tool_call_id": "call_0", "content": "first"},
            {"role": "tool", "tool_call_id": "call_1", "content": "second"},
        ]
    )
    assert [item["type"] for item in replay] == [
        "reasoning",
        "message",
        "function_call",
        "reasoning",
        "function_call",
        "message",
        "function_call_output",
        "function_call_output",
    ]
    assert replay[0]["encrypted_content"] == "opaque-1"
    assert replay[3]["encrypted_content"] == "opaque-2"
    assert "id" not in replay[2] and replay[2]["call_id"] == "call_0"
    assert replay[-2]["call_id"] == "call_0" and replay[-2]["output"] == "first"


@pytest.mark.parametrize(
    "status", ["incomplete", "failed", "cancelled", "in_progress", None]
)
def test_noncompleted_responses_never_execute_tools(
    tmp_path: Path, status: str | None
) -> None:
    path = tmp_path / "must-not-exist"
    with (
        patch(
            "conferllm.client.litellm.responses",
            return_value=native(
                [function("write_file", {"path": str(path), "content": "bad"})],
                status=status,
            ),
        ),
        pytest.raises(ConferLLMError, match="completed response"),
    ):
        ChatService(client(tmp_path / "sessions")).chat("write", model="test")
    assert not path.exists()


def test_malformed_native_call_does_not_fall_back_to_its_item_id() -> None:
    item = function("read_file", {"path": "test"})
    del item["call_id"]
    with pytest.raises(ValueError, match="function call"):
        to_model_response(native([item]).model_dump())


def test_changed_native_state_is_detected() -> None:
    assistant = to_model_response(
        native([function("read_file", {"path": "test"})]).model_dump()
    ).model_dump()["choices"][0]["message"]
    response_items(assistant)[0]["call_id"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        validate_response_state(assistant)


def test_oversized_optional_output_ids_are_not_echoed_as_input_ids() -> None:
    call = function("read_file", {"path": "test"})
    call["id"] = "fc_" + "x" * 4096
    text = message("commentary", "msg_" + "x" * 4096)
    reasoning = {
        "type": "reasoning",
        "id": "rs_original",
        "summary": [],
        "encrypted_content": "opaque",
    }
    assistant = to_model_response(
        native([reasoning, text, call]).model_dump()
    ).model_dump()["choices"][0]["message"]
    replay = messages_to_input(
        [assistant, {"role": "tool", "tool_call_id": "call_123", "content": "done"}]
    )
    assert replay[0]["id"] == "rs_original"
    assert "id" not in replay[1] and "status" not in replay[1]
    assert replay[1]["content"][0]["type"] == "output_text"
    assert "id" not in replay[2]
    assert replay[2]["call_id"] == replay[3]["call_id"] == "call_123"
    assert response_items(assistant)[2]["id"] == call["id"]


@pytest.mark.parametrize(
    ("role", "content_type"),
    [
        ("user", "input_text"),
        ("system", "input_text"),
        ("developer", "input_text"),
        ("assistant", "output_text"),
    ],
)
def test_replayed_text_content_type_matches_role(role: str, content_type: str) -> None:
    replay = messages_to_input([{"role": role, "content": "remembered"}])
    assert replay[0]["content"][0]["type"] == content_type
    assert replay[0]["content"][0]["text"] == "remembered"


def test_native_assistant_phase_survives_text_replay() -> None:
    item = {**message("remembered"), "phase": "final_answer"}
    assistant = to_model_response(native([item]).model_dump()).model_dump()["choices"][
        0
    ]["message"]
    replay = messages_to_input([assistant])
    assert replay[0]["phase"] == "final_answer"
    assert replay[0]["content"][0]["type"] == "output_text"


def test_assistant_images_replay_as_attributed_inputs_in_original_order() -> None:
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    original = {
        "role": "assistant",
        "phase": "final_answer",
        "content": [
            {"type": "text", "text": "Before."},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "After."},
        ],
    }
    snapshot = copy.deepcopy(original)
    replay = messages_to_input([original])
    assert [item["role"] for item in replay] == ["assistant", "user", "assistant"]
    assert replay[0]["content"] == [{"type": "output_text", "text": "Before."}]
    assert replay[2]["content"] == [{"type": "output_text", "text": "After."}]
    assert replay[0]["phase"] == replay[2]["phase"] == "final_answer"
    assert "phase" not in replay[1]
    assert replay[1]["content"][0]["type"] == "input_text"
    assert "generated by the assistant" in replay[1]["content"][0]["text"]
    assert [part["type"] for part in replay[1]["content"][1:]] == [
        "input_image",
        "input_image",
    ]
    assert all(part["image_url"] == data_url for part in replay[1]["content"][1:])
    assert original == snapshot


def test_legacy_format_can_replay_canonical_tools_without_native_metadata() -> None:
    assistant = to_model_response(
        native([function("read_file", {"path": "test"})]).model_dump()
    ).model_dump()["choices"][0]["message"]
    stripped = strip_response_state([assistant])[0]
    assert "provider_specific_fields" not in stripped
    assert stripped["tool_calls"][0]["id"] == "call_123"
    assert STATE_KEY in assistant["provider_specific_fields"]


@pytest.mark.parametrize("legacy", [False, True])
def test_responses_images_and_native_state_survive_session_replay(
    tmp_path: Path, legacy: bool
) -> None:
    service = ChatService(client(tmp_path / "sessions"))
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    with patch(
        "conferllm.client.litellm.responses", return_value=native([message("start")])
    ):
        first = service.chat("start", model="test")
    session_path = service.session_store.session_path(first.session_id)
    if legacy:
        records = [json.loads(line) for line in session_path.read_text().splitlines()]
        records[0]["schema_version"] = 1
        records[1].pop("artifacts")
        session_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    with patch(
        "conferllm.client.litellm.responses",
        side_effect=[
            native(
                [message(data_url), function("list_directory", {"path": str(tmp_path)})]
            ),
            native([message(data_url)]),
            native([message("remembered")]),
        ],
    ) as endpoint:
        result = service.chat("draw and list", session_id=first.session_id)
        service.chat("recall", session_id=first.session_id)
    assert len(result.artifacts) == 2
    assert data_url not in session_path.read_text()
    for request in endpoint.call_args_list[1:]:
        images = [
            part["image_url"]
            for item in request.kwargs["input"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "input_image"
        ]
        assert images and all(url == data_url for url in images)
    assert service.session_store.load_session(first.session_id).schema_version == (
        1 if legacy else 2
    )
