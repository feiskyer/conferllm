"""Provider prompt-cache hints and the settings that shape a cached prefix.

Hints are added to each provider request and never persisted, so replayed
history stays byte-identical across turns. Providers that cache automatically
need only that stable prefix; the hints enable opt-in caching or keep one
session's requests on the same cache.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .config import ModelConfig

# Anthropic-format caching is opt-in on every Claude route. Marking the system
# prompt caches the shared tools+system prefix; marking the latest message
# advances the breakpoint as the conversation grows.
_CLAUDE_ROUTES = frozenset(
    {"anthropic", "azure_ai", "bedrock", "openrouter", "vertex_ai"}
)
_BREAKPOINT_PARAMS = ("cache_control", "cache_control_injection_points")

# Before GPT-5.6, OpenAI routes requests by prompt_cache_key. Later models use
# the key only to separate cache accounting, so a per-session key would stop
# sessions from sharing a prefix.
_GPT_VERSION = re.compile(r"gpt-(\d+)(?:\.(\d+))?")
_KEY_ROUTING_BEFORE = (5, 6)


@dataclass(frozen=True)
class _Affinity:
    """Where a provider's official endpoint accepts a session cache key."""

    hosts: tuple[str, ...]
    parameter: str
    entry: str | None = None  # key inside a mapping parameter


# Session keys go only to each provider's own endpoint, because compatible
# gateways may reject parameters they do not know.
_SESSION_AFFINITY = {
    "openai": _Affinity(("api.openai.com",), "prompt_cache_key"),
    "moonshot": _Affinity(("api.moonshot.ai", "api.moonshot.cn"), "prompt_cache_key"),
    "openrouter": _Affinity(("openrouter.ai",), "session_id"),
    "xai": _Affinity(("api.x.ai",), "extra_headers", "x-grok-conv-id"),
    # Documented by Mistral, but not a LiteLLM-supported Mistral parameter.
    "mistral": _Affinity(("api.mistral.ai",), "extra_body", "prompt_cache_key"),
}

# Settings whose change makes providers reprocess replayed history. Values are
# provider options, never credentials, endpoints, or headers.
_PROFILE_PARAMS = (
    "model",
    "custom_llm_provider",
    "reasoning_effort",
    "reasoning",
    "thinking",
    "verbosity",
    "text",
    "response_format",
    "service_tier",
    "parallel_tool_calls",
    "speed",
    "output_config",
    "web_search_options",
    "context_management",
    "prompt_cache_key",
)


def request_hints(
    model_config: ModelConfig, *, provider: str | None, cache_key: str | None
) -> dict[str, Any]:
    """Return LiteLLM arguments that enable or steer provider prompt caching.

    Explicitly configured cache parameters always win over these hints.
    """
    if not model_config.prompt_cache.enabled or provider is None:
        return {}
    params = model_config.litellm_params
    model = str(params["model"])
    hints: dict[str, Any] = {}
    if _is_claude(provider, model) and not any(
        name in params for name in _BREAKPOINT_PARAMS
    ):
        control = {"type": "ephemeral"}
        if model_config.prompt_cache.ttl == "1h":
            control["ttl"] = "1h"
        hints["cache_control_injection_points"] = [
            {"location": "message", "role": "system", "control": dict(control)},
            {"location": "message", "index": -1, "control": dict(control)},
        ]
    if cache_key is not None:
        hints.update(_session_affinity(provider, model, params, cache_key))
    return hints


def request_profile(model_config: ModelConfig, *, transport: str) -> dict[str, Any]:
    """Return the non-secret settings that shape a session's cached prefix."""
    params = model_config.litellm_params
    return {
        "transport": transport,
        **{
            name: copy.deepcopy(params[name])
            for name in _PROFILE_PARAMS
            if name in params
        },
    }


def profile_changes(pinned: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Name the settings that differ between a session's creation and now."""
    return sorted(
        name
        for name in pinned.keys() | current.keys()
        if pinned.get(name) != current.get(name)
    )


def _is_claude(provider: str, model: str) -> bool:
    return provider == "anthropic" or (
        provider in _CLAUDE_ROUTES and "claude" in model.lower()
    )


def _session_affinity(
    provider: str, model: str, params: Mapping[str, Any], cache_key: str
) -> dict[str, Any]:
    affinity = _SESSION_AFFINITY.get(provider)
    if affinity is None or not _is_official_endpoint(provider, affinity, params):
        return {}
    if provider == "openai" and not _routes_by_cache_key(model):
        return {}
    if affinity.entry is not None:
        return _with_entry(params, affinity.parameter, affinity.entry, cache_key)
    if affinity.parameter in params:
        return {}
    return {affinity.parameter: cache_key}


def _is_official_endpoint(
    provider: str, affinity: _Affinity, params: Mapping[str, Any]
) -> bool:
    base = params.get("api_base") or params.get("base_url")
    if not base:
        names = [f"{provider.upper()}_API_BASE"]
        if provider == "openai":
            names.append("OPENAI_BASE_URL")
        base = next((os.environ[name] for name in names if os.environ.get(name)), "")
    if not base:
        return True
    host = urlparse(str(base)).hostname or ""
    return any(host == name or host.endswith(f".{name}") for name in affinity.hosts)


def _routes_by_cache_key(model: str) -> bool:
    version = _GPT_VERSION.match(model.rsplit("/", 1)[-1].lower())
    if version is None:
        return True  # o-series, fine-tunes, and other earlier model families
    return (int(version[1]), int(version[2] or 0)) < _KEY_ROUTING_BEFORE


def _with_entry(
    params: Mapping[str, Any], name: str, key: str, value: str
) -> dict[str, Any]:
    """Add one entry to a configured mapping parameter without overriding it."""
    existing = params.get(name) or {}
    if not isinstance(existing, Mapping) or any(
        str(item).lower() == key for item in existing
    ):
        return {}
    return {name: {**copy.deepcopy(dict(existing)), key: value}}
