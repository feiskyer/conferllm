"""OpenAI Responses wire format with lossless native tool/reasoning replay.

The rest of ConferLLM continues to use chat-shaped messages. Native output items
are retained as private message metadata, with visible content in ConferLLM's
normal text/image representation. Importing this module does not load an SDK.
"""

from __future__ import annotations

import base64
import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .errors import ConferLLMError
from .images import (
    image_bytes_to_data_url,
    image_payload_from_bytes,
    normalize_assistant_content,
    replace_image_refs,
)
from .tool_protocol import tool_calls

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse

STATE_KEY = "conferllm_responses_output"
IMAGE_METADATA_KEY = "conferllm_image_generation"


def response_items(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return this adapter's state, without touching other providers' metadata."""
    fields = message.get("provider_specific_fields")
    if not isinstance(fields, dict) or STATE_KEY not in fields:
        return None
    items = fields[STATE_KEY]
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("Responses output state must be an array of objects.")
    return items


def validate_response_state(message: dict[str, Any]) -> None:
    """Do not replay native state that contradicts the stored tool transcript."""
    items = response_items(message)
    if items is None:
        return
    if message.get("role") != "assistant":
        raise ValueError("Responses output state belongs to assistant messages.")
    native_calls = []
    for item in items:
        kind = item.get("type")
        if kind == "function_call":
            native_calls.append(
                (item.get("call_id"), item.get("name"), item.get("arguments"))
            )
        elif kind == "message":
            if item.get("role") != "assistant" or not isinstance(
                item.get("content"), list
            ):
                raise ValueError("Invalid Responses assistant message state.")
        elif kind == "reasoning":
            if not isinstance(item.get("summary"), list):
                raise ValueError("Responses reasoning state requires a summary array.")
            if item.get("encrypted_content") is not None and not isinstance(
                item["encrypted_content"], str
            ):
                raise ValueError("Invalid encrypted Responses reasoning state.")
        else:
            raise ValueError("Unsupported item in Responses output state.")
    calls = tool_calls(message)
    canonical_calls = [
        (
            call.get("id"),
            call.get("function", {}).get("name"),
            call.get("function", {}).get("arguments"),
        )
        for call in calls
    ]
    if native_calls != canonical_calls:
        raise ValueError("Responses state does not match the stored tool calls.")


def response_content(items: list[dict[str, Any]]) -> Any:
    parts = [
        part
        for item in items
        if item.get("type") == "message"
        for part in item["content"]
    ]
    if all(part.get("type") == "text" for part in parts):
        return "".join(part["text"] for part in parts) or None
    return parts


def normalize_response_images(message: dict[str, Any], artifact_uris: set[str]) -> None:
    """Normalize visible native items once their embedded images have been staged."""
    items = response_items(message)
    if items is not None:
        for item in items:
            if item.get("type") == "message":
                _, item["content"] = normalize_assistant_content(
                    item["content"], artifact_uris=artifact_uris
                )


def restore_response_images(
    messages: list[dict[str, Any]], resolver: Callable[[str], str]
) -> list[dict[str, Any]]:
    """Resolve references inside native output metadata as well as visible text."""
    restored = copy.deepcopy(messages)
    for message in restored:
        items = response_items(message)
        if items is not None:
            for index, item in enumerate(items):
                if item.get("type") == "message":
                    items[index] = replace_image_refs([item], resolver)[0]
    return restored


def strip_response_state(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Allow an existing session to be used with the explicit legacy transport."""
    messages = copy.deepcopy(messages)
    for message in messages:
        fields = message.get("provider_specific_fields")
        if isinstance(fields, dict):
            fields.pop(STATE_KEY, None)
            if not fields:
                message.pop("provider_specific_fields")
    return messages


def _input_content(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or an array.")
    result = []
    for part in content:
        if isinstance(part, str):
            part = {"type": "text", "text": part}
        if not isinstance(part, dict):
            raise ValueError("Invalid Responses message content.")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            # Assistant history is model output, including when its optional
            # output-item id/status are omitted. Gateways validate this role
            # against output_text/refusal, not user input_text.
            result.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": part["text"],
                }
            )
        elif part.get("type") == "image_url" and isinstance(
            part.get("image_url"), dict
        ):
            image = part["image_url"]
            result.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    "detail": image.get("detail", "auto"),
                }
            )
        else:
            raise ValueError("Unsupported content for the Responses API.")
    return result


def _message_input(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay output text and attach generated images as attributed image inputs."""
    role = message["role"]
    content = _input_content(message.get("content"), role)
    if not content:
        return []
    if role != "assistant":
        return [{"type": "message", "role": role, "content": content}]
    result: list[dict[str, Any]] = []
    for part in content:
        # Assistant output messages accept output_text/refusal, not input_image.
        # Supply generated images as reference inputs, identifying their origin
        # rather than attributing the image creation to the user.
        is_image = part["type"] == "input_image"
        item_role = "user" if is_image else "assistant"
        if not result or result[-1]["role"] != item_role:
            item: dict[str, Any] = {
                "type": "message",
                "role": item_role,
                "content": [],
            }
            if is_image:
                item["content"].append(
                    {
                        "type": "input_text",
                        "text": "The following image(s) were generated by the "
                        "assistant earlier in this conversation.",
                    }
                )
            elif message.get("phase") is not None:
                item["phase"] = message["phase"]
            result.append(item)
        result[-1]["content"].append(part)
    return result


def messages_to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate legacy history, or replay all native items in their exact order."""
    result = []
    for message in messages:
        role = message.get("role")
        items = response_items(message)
        if items is not None:
            validate_response_state(message)
            for native in copy.deepcopy(items):
                if native.get("type") == "message":
                    # Keep phase, but omit output-only metadata and optional
                    # item IDs (which compatible gateways may make oversized).
                    result.extend(_message_input(native))
                    continue
                if native.get("type") == "function_call":
                    # call_id, not the optional output-item id, correlates results.
                    native.pop("id", None)
                result.append(native)
            continue
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("Unsupported message role for the Responses API.")
        result.extend(_message_input(message))
        for call in message.get("tool_calls") or []:
            result.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
    return result


def prepare_params(
    parameters: dict[str, Any], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map existing model parameters without silently changing their semantics."""
    result = copy.deepcopy(parameters)
    for name in ("input", "instructions", "previous_response_id", "conversation"):
        if name in result:
            raise ConferLLMError(
                "configuration_error", f"litellm_params.{name} is managed by ConferLLM."
            )
    chat_only = {
        "stop",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    }
    if (
        any(result.get(name) is not None for name in chat_only)
        or result.get("n", 1) != 1
    ):
        raise ConferLLMError(
            "configuration_error",
            "This model has Chat Completions-only parameters. Remove them or set "
            "the model's api_format to chat_completion.",
        )
    result.pop("n", None)
    for name in ("max_completion_tokens", "max_tokens"):
        value = result.pop(name, None)
        if value is not None:
            result.setdefault("max_output_tokens", value)
    effort = result.pop("reasoning_effort", None)
    if effort is not None:
        result.setdefault(
            "reasoning", effort if isinstance(effort, dict) else {"effort": effort}
        )
    response_format = result.pop("response_format", None)
    if isinstance(response_format, dict):
        native_format = dict(response_format)
        if native_format.get("type") == "json_schema":
            native_format = {"type": "json_schema", **native_format["json_schema"]}
        result.setdefault("text", {}).setdefault("format", native_format)
    result.setdefault("store", False)
    include = result.setdefault("include", [])
    if not isinstance(include, list):
        raise ConferLLMError(
            "configuration_error", "litellm_params.include must be an array."
        )
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    # Explicit non-strict schemas preserve the existing optional/default arguments.
    # Input validation still occurs in ToolRuntime for every provider.
    result["tools"] = [
        {"type": "function", **copy.deepcopy(tool["function"]), "strict": False}
        for tool in tools
    ]
    result["tool_choice"] = "auto"
    result["stream"] = False
    result["background"] = False
    return result


def to_model_response(raw: dict[str, Any]) -> ModelResponse:
    """Keep all calls, commentary and reasoning, not just output[0]/choices[0]."""
    from litellm.types.utils import ModelResponse

    if raw.get("status") != "completed" or raw.get("error") is not None:
        raise ConferLLMError(
            "provider_error", "The Responses API did not return a completed response."
        )
    items = copy.deepcopy(raw.get("output"))
    if not isinstance(items, list):
        raise ValueError("Responses output must be an array.")
    calls = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Responses output items must be objects.")
        kind = item.get("type")
        if kind == "function_call":
            if (
                not isinstance(item.get("call_id"), str)
                or not item["call_id"].strip()
                or not isinstance(item.get("name"), str)
                or not item["name"].strip()
                or not isinstance(item.get("arguments"), str)
                or item.get("status", "completed") != "completed"
            ):
                raise ValueError("Invalid or incomplete Responses function call.")
            calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        elif kind == "image_generation_call":
            if (
                not isinstance(item.get("result"), str)
                or item.get("status", "completed") != "completed"
            ):
                raise ValueError("Invalid Responses image generation result.")
            payload = image_payload_from_bytes(
                base64.b64decode(item["result"], validate=True)
            )
            metadata = {key: value for key, value in item.items() if key != "result"}
            item.clear()
            item.update(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_bytes_to_data_url(
                                    payload.data, payload.mime_type
                                )
                            },
                        }
                    ],
                    IMAGE_METADATA_KEY: metadata,
                }
            )
        elif kind == "message":
            if item.get("role") != "assistant" or not isinstance(
                item.get("content"), list
            ):
                raise ValueError("Invalid Responses output message.")
            normalized = []
            for part in item["content"]:
                if part.get("type") == "output_text" and isinstance(
                    part.get("text"), str
                ):
                    normalized.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "refusal" and isinstance(
                    part.get("refusal"), str
                ):
                    normalized.append({"type": "text", "text": part["refusal"]})
                else:
                    raise ValueError("Unsupported Responses output content.")
            item["content"] = normalized
        elif kind != "reasoning":
            raise ValueError("Unsupported Responses output item.")
    visible = response_content(items)
    message = {
        "role": "assistant",
        # LiteLLM's Message model exposes text in content and images separately.
        # The exact mixed ordering remains in STATE_KEY for ChatService.
        "content": (
            "".join(
                part.get("text", "") for part in visible if part.get("type") == "text"
            )
            or None
            if isinstance(visible, list)
            else visible
        ),
        "tool_calls": calls or None,
        "provider_specific_fields": {STATE_KEY: items},
    }
    if isinstance(visible, list):
        message["images"] = [
            {"index": index, **part}
            for index, part in enumerate(
                part for part in visible if part.get("type") == "image_url"
            )
        ]
    validate_response_state(message)
    usage = raw.get("usage") or {}
    mapped_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens_details": usage.get("input_tokens_details"),
        "completion_tokens_details": usage.get("output_tokens_details"),
    }
    return ModelResponse(
        id=raw.get("id"),
        model=raw.get("model"),
        created=raw.get("created_at"),
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        usage=mapped_usage,
    )
