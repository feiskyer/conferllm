"""Provider prompt-cache hints, request profiles, and their client wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig, PromptCacheSettings
from conferllm.prompt_cache import profile_changes, request_hints, request_profile
from tests.chat_fixtures import response

MARKERS = [
    {"location": "message", "role": "system", "control": {"type": "ephemeral"}},
    {"location": "message", "index": -1, "control": {"type": "ephemeral"}},
]


def hints(
    litellm_params: dict[str, Any],
    *,
    provider: str | None,
    cache_key: str | None = "session-1",
    prompt_cache: PromptCacheSettings | None = None,
) -> dict[str, Any]:
    model = ModelConfig(
        model_name="test",
        litellm_params=litellm_params,
        prompt_cache=prompt_cache or PromptCacheSettings(),
    )
    return request_hints(model, provider=provider, cache_key=cache_key)


def client_for(litellm_params: dict[str, Any], **fields: Any) -> LLMClient:
    return LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(model_name="test", litellm_params=litellm_params, **fields)
            ]
        )
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "anthropic/claude-opus-5"),
        ("vertex_ai", "vertex_ai/claude-sonnet-5@20260101"),
        ("bedrock", "bedrock/us.anthropic.claude-opus-5-v1:0"),
        ("openrouter", "openrouter/anthropic/claude-sonnet-5"),
        ("azure_ai", "azure_ai/claude-opus-5"),
    ],
)
def test_claude_routes_mark_the_system_prompt_and_latest_message(
    provider: str, model: str
) -> None:
    assert hints({"model": model}, provider=provider, cache_key=None) == {
        "cache_control_injection_points": MARKERS
    }


def test_one_hour_ttl_applies_to_every_breakpoint() -> None:
    points = hints(
        {"model": "anthropic/claude-opus-5"},
        provider="anthropic",
        prompt_cache=PromptCacheSettings(ttl="1h"),
    )["cache_control_injection_points"]

    assert [point["control"] for point in points] == [
        {"type": "ephemeral", "ttl": "1h"},
        {"type": "ephemeral", "ttl": "1h"},
    ]


@pytest.mark.parametrize(
    "configured",
    [
        {"cache_control": {"type": "ephemeral"}},
        {"cache_control_injection_points": [{"location": "message", "index": -1}]},
    ],
)
def test_configured_claude_breakpoints_take_precedence(
    configured: dict[str, Any],
) -> None:
    params = {"model": "anthropic/claude-opus-5", **configured}

    assert hints(params, provider="anthropic") == {}


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("vertex_ai", "vertex_ai/gemini-3.8-flash"),
        ("gemini", "gemini/gemini-3.8-flash"),
        ("openrouter", "openrouter/openai/gpt-6-astra"),
        ("bedrock", "bedrock/amazon.nova-pro-v1:0"),
        ("deepseek", "deepseek/deepseek-flash"),
        ("ollama", "ollama/llama4"),
    ],
)
def test_other_models_receive_no_breakpoints(provider: str, model: str) -> None:
    # Gemini would turn cache_control into a billed explicit cache.
    assert "cache_control_injection_points" not in hints(
        {"model": model}, provider=provider
    )


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.5",
        "openai/gpt-4.1",
        "openai/o3",
        "openai/ft:gpt-4o-mini:org::abc123",
    ],
)
def test_openai_models_before_5_6_route_by_session_key(model: str) -> None:
    assert hints({"model": model}, provider="openai") == {
        "prompt_cache_key": "session-1"
    }


@pytest.mark.parametrize("model", ["openai/gpt-5.6-sol", "openai/gpt-6-astra"])
def test_openai_5_6_and_later_keep_shared_cache_accounting(model: str) -> None:
    assert hints({"model": model}, provider="openai") == {}


def test_session_keys_are_sent_only_to_official_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "openai/gpt-5.5"
    gateway = {"model": model, "api_base": "https://gateway.example/v1"}
    regional = {"model": model, "api_base": "https://eu.api.openai.com/v1"}

    assert hints(gateway, provider="openai") == {}
    assert hints(regional, provider="openai") == {"prompt_cache_key": "session-1"}
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert hints({"model": model}, provider="openai") == {}


def test_provider_specific_session_affinity() -> None:
    assert hints({"model": "moonshot/kimi-k3"}, provider="moonshot") == {
        "prompt_cache_key": "session-1"
    }
    assert hints({"model": "openrouter/google/gemini-3"}, provider="openrouter") == {
        "session_id": "session-1"
    }
    assert hints(
        {"model": "xai/grok-5", "extra_headers": {"X-Trace": "1"}}, provider="xai"
    ) == {"extra_headers": {"X-Trace": "1", "x-grok-conv-id": "session-1"}}
    assert hints(
        {"model": "mistral/mistral-large", "extra_body": {"safe_prompt": True}},
        provider="mistral",
    ) == {"extra_body": {"safe_prompt": True, "prompt_cache_key": "session-1"}}


def test_openrouter_claude_gets_breakpoints_and_session_affinity() -> None:
    assert hints(
        {"model": "openrouter/anthropic/claude-sonnet-5"}, provider="openrouter"
    ) == {"cache_control_injection_points": MARKERS, "session_id": "session-1"}


@pytest.mark.parametrize(
    ("params", "provider"),
    [
        ({"model": "openai/gpt-5.5", "prompt_cache_key": "tenant-a"}, "openai"),
        ({"model": "moonshot/kimi-k3", "prompt_cache_key": "tenant-a"}, "moonshot"),
        ({"model": "openrouter/x/y", "session_id": "mine"}, "openrouter"),
        (
            {"model": "xai/grok-5", "extra_headers": {"X-Grok-Conv-Id": "mine"}},
            "xai",
        ),
        (
            {"model": "mistral/large", "extra_body": {"prompt_cache_key": "mine"}},
            "mistral",
        ),
    ],
)
def test_configured_session_keys_take_precedence(
    params: dict[str, Any], provider: str
) -> None:
    assert hints(params, provider=provider) == {}


def test_disabled_settings_or_unknown_providers_add_nothing() -> None:
    disabled = PromptCacheSettings(enabled=False)

    assert (
        hints(
            {"model": "anthropic/claude-opus-5"},
            provider="anthropic",
            prompt_cache=disabled,
        )
        == {}
    )
    assert hints({"model": "openai/gpt-5.5"}, provider="openai", cache_key=None) == {}
    assert hints({"model": "custom-model"}, provider=None) == {}


def test_prompt_cache_settings_belong_on_the_model() -> None:
    with pytest.raises(ValueError, match="not inside litellm_params"):
        ModelConfig(
            model_name="test",
            litellm_params={"model": "anthropic/claude", "prompt_cache": {}},
        )
    with pytest.raises(ValueError):
        PromptCacheSettings(ttl="10m")  # type: ignore[arg-type]


def test_request_profile_keeps_only_non_secret_prefix_settings() -> None:
    model = ModelConfig(
        model_name="test",
        litellm_params={
            "model": "anthropic/claude-opus-5",
            "api_key": "synthetic-secret",
            "api_base": "https://provider.invalid",
            "extra_headers": {"Authorization": "Bearer synthetic"},
            "thinking": {"type": "adaptive"},
            "max_tokens": 1000,
        },
    )

    assert request_profile(model, transport="chat_completion") == {
        "transport": "chat_completion",
        "model": "anthropic/claude-opus-5",
        "thinking": {"type": "adaptive"},
    }


def test_profile_changes_name_each_differing_setting() -> None:
    pinned = {"transport": "responses", "model": "a", "reasoning_effort": "low"}
    current = {"transport": "responses", "model": "b", "verbosity": "low"}

    assert profile_changes(pinned, current) == [
        "model",
        "reasoning_effort",
        "verbosity",
    ]
    assert profile_changes(pinned, dict(pinned)) == []


def test_client_adds_claude_breakpoints_to_completion_calls() -> None:
    client = client_for(
        {"model": "anthropic/claude-opus-5"}, api_format="chat_completion"
    )
    with patch(
        "conferllm.client.litellm.completion", return_value=response("done")
    ) as completion:
        client.chat(
            "test",
            [{"role": "user", "content": "hi"}],
            system_prompt="Pinned.",
            cache_key="session-1",
        )

    assert completion.call_args.kwargs["cache_control_injection_points"] == MARKERS
    assert "prompt_cache_key" not in completion.call_args.kwargs


@pytest.mark.parametrize("api_format", ["responses", "chat_completion"])
def test_client_sends_the_session_key_on_both_openai_transports(
    api_format: str,
) -> None:
    client = client_for({"model": "openai/gpt-5.5"}, api_format=api_format)
    native = MagicMock()
    native.model_dump.return_value = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    }
    with (
        patch("conferllm.client.litellm.responses", return_value=native) as responses,
        patch(
            "conferllm.client.litellm.completion", return_value=response("done")
        ) as completion,
    ):
        client.chat(
            "test",
            [{"role": "user", "content": "hi"}],
            system_prompt=None,
            cache_key="session-1",
        )

    endpoint = responses if api_format == "responses" else completion
    assert endpoint.call_args.kwargs["prompt_cache_key"] == "session-1"


def test_client_uses_pinned_tool_definitions_when_supplied() -> None:
    client = client_for({"model": "openai/gpt-5.5"}, api_format="chat_completion")
    pinned = [
        {
            "type": "function",
            "function": {"name": "legacy_tool", "parameters": {"type": "object"}},
        }
    ]
    with patch(
        "conferllm.client.litellm.completion", return_value=response("done")
    ) as completion:
        client.chat(
            "test",
            [{"role": "user", "content": "hi"}],
            system_prompt=None,
            tools=pinned,
        )

    assert completion.call_args.kwargs["tools"] == pinned


def test_model_info_reports_prompt_cache_settings_and_hints() -> None:
    client = LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="claude",
                    litellm_params={"model": "anthropic/claude-opus-5"},
                    prompt_cache=PromptCacheSettings(ttl="1h"),
                ),
                ModelConfig(
                    model_name="gpt", litellm_params={"model": "openai/gpt-5.5"}
                ),
                ModelConfig(
                    model_name="off",
                    litellm_params={"model": "anthropic/claude-opus-5"},
                    prompt_cache=PromptCacheSettings(enabled=False),
                ),
            ]
        )
    )

    assert client.get_model_info("claude")["prompt_cache"] == {
        "enabled": True,
        "ttl": "1h",
        "request_hints": ["cache_control_injection_points"],
    }
    assert client.get_model_info("gpt")["prompt_cache"]["request_hints"] == [
        "prompt_cache_key"
    ]
    assert client.get_model_info("off")["prompt_cache"]["request_hints"] == []


def test_request_profile_records_the_actual_transport() -> None:
    client = LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="gpt",
                    litellm_params={"model": "openai/gpt-5.5", "api_key": "s"},
                ),
                ModelConfig(
                    model_name="claude",
                    litellm_params={"model": "anthropic/claude-opus-5"},
                ),
            ]
        )
    )

    assert client.request_profile("gpt") == {
        "transport": "responses",
        "model": "openai/gpt-5.5",
    }
    # api_format only selects a transport for OpenAI routes.
    assert client.request_profile("claude")["transport"] == "chat_completion"
