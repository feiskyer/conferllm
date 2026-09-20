"""Per-call native response preservation, without retaining request/credential data."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from litellm.types.utils import ModelResponse

from conferllm.completions import CompletionCapture, _gemini_content
from conferllm.outputs import assistant_choices
from tests.chat_fixtures import PNG
from tests.test_outputs import DATA_URL


def test_capture_keeps_only_response_and_preserves_existing_callback() -> None:
    callback = MagicMock()
    capture = CompletionCapture(callback)
    payload = {"choices": [{"message": {"role": "assistant", "content": "original"}}]}
    event = {
        "log_event_type": "post_api_call",
        "api_key": "synthetic-not-for-storage",
        "input": ["private-input"],
        "original_response": payload,
    }
    capture(event)
    callback.assert_called_once_with(event)
    assert capture.payload == payload
    assert "synthetic-not-for-storage" not in json.dumps(capture.payload)
    assert "private-input" not in json.dumps(capture.payload)
    payload["choices"].clear()
    assert len(capture.payload["choices"]) == 1


@pytest.mark.parametrize("raw", [None, "not-json", "[]", 123])
def test_non_json_or_non_object_logging_events_are_not_output(raw: object) -> None:
    capture = CompletionCapture()
    capture({"log_event_type": "post_api_call", "original_response": raw})
    assert capture.payload is None


def test_a_new_request_cannot_recover_a_previous_attempt() -> None:
    capture = CompletionCapture()
    capture({"log_event_type": "post_api_call", "original_response": {"choices": []}})
    capture({"log_event_type": "pre_api_call", "api_key": "synthetic"})
    assert capture.payload is None
    assert capture.recover_content_arrays() is None


@pytest.mark.parametrize(
    "update",
    [
        {"error": {"message": "upstream failure"}},
        {"object": "error"},
        {"choices": []},
        {"choices": [{"message": {"content": []}, "finish_reason": None}]},
        {"choices": [{"message": {"content": "plain text"}, "finish_reason": "stop"}]},
    ],
)
def test_error_incomplete_and_plain_text_responses_are_not_recoverable(
    update: dict,
) -> None:
    capture = CompletionCapture()
    capture.payload = {
        "object": "chat.completion",
        "choices": [{"message": {"content": []}, "finish_reason": "stop"}],
        **update,
    }
    assert capture.recover_content_arrays() is None


def test_native_text_wins_over_sdk_inferred_tool_call() -> None:
    capture = CompletionCapture()
    text = '{"type":"function","name":"run_shell","arguments":{"command":"exit"}}'
    capture(
        {
            "log_event_type": "post_api_call",
            "original_response": json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text},
                        }
                    ]
                }
            ),
        }
    )
    inferred = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "sdk-inferred",
                            "type": "function",
                            "function": {"name": "run_shell", "arguments": "{}"},
                        }
                    ],
                },
            }
        ]
    )
    restored = capture.restore(inferred, {}).model_dump()
    assert restored["choices"][0]["finish_reason"] == "stop"
    message = assistant_choices(restored)[0]
    assert message["content"] == text
    assert not message.get("tool_calls")


def test_native_gemini_parts_keep_interleaved_visible_order() -> None:
    import base64

    result = _gemini_content(
        [
            {"text": "before"},
            {"text": "private reasoning", "thought": True},
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(PNG).decode(),
                }
            },
            {"text": "after"},
            {"functionCall": {"name": "read_file", "args": {}}},
            {"thoughtSignature": "opaque"},
        ]
    )
    assert [part["type"] for part in result] == ["text", "image_url", "text"]
    assert result[1]["image_url"]["url"] == DATA_URL
    assert "private reasoning" not in json.dumps(result)


def test_unhandled_gemini_parts_do_not_disappear_silently() -> None:
    assert _gemini_content(
        [{"inlineData": {"mimeType": "audio/wav", "data": "AAAA"}}]
    ) == [{"type": "unsupported_gemini_part"}]
