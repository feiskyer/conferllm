"""Provider-independent normalization of every assistant output candidate."""

from __future__ import annotations

import base64
import binascii
import copy
import json
from typing import Any

from .images import (
    ImageProcessingError,
    image_bytes_to_data_url,
    image_payload_from_bytes,
)
from .responses import STATE_KEY
from .tool_protocol import tool_calls

NATIVE_MESSAGE_KEY = "conferllm_native_message"


def image_part(value: Any) -> dict[str, Any]:
    """Normalize common SDK image blocks, without reading arbitrary local paths."""
    if isinstance(value, str):
        return {"type": "image_url", "image_url": {"url": value}}
    if not isinstance(value, dict):
        raise ImageProcessingError("Provider image must be an object or URL.")
    image_url = value.get("image_url")
    if isinstance(image_url, str):
        return image_part(image_url)
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return {"type": "image_url", "image_url": copy.deepcopy(image_url)}
    if isinstance(value.get("url"), str):
        return image_part(value["url"])
    source = value.get("source")
    if isinstance(source, dict):
        if source.get("type") == "url":
            return image_part(source.get("url"))
        if source.get("type") == "base64":
            value = {
                "b64_json": source.get("data"),
                "mime_type": source.get("media_type"),
            }
    encoded = value.get("b64_json", value.get("data"))
    if isinstance(encoded, str):
        try:
            data = base64.b64decode(encoded, validate=True)
            payload = image_payload_from_bytes(
                data, declared_mime=value.get("mime_type") or value.get("media_type")
            )
        except (binascii.Error, ValueError) as error:
            raise ImageProcessingError(
                "Provider returned invalid image data."
            ) from error
        return image_part(image_bytes_to_data_url(data, payload.mime_type))
    raise ImageProcessingError("Provider image has no supported payload.")


def normalize_message(message: Any) -> dict[str, Any]:
    """Normalize typed native blocks, never parse tool-shaped ordinary text."""
    if not isinstance(message, dict):
        raise RuntimeError("Model response does not contain an assistant message.")
    result = copy.deepcopy(message)
    fields = result.get("provider_specific_fields")
    if isinstance(fields, dict) and NATIVE_MESSAGE_KEY in fields:
        native = fields[NATIVE_MESSAGE_KEY]
        if not isinstance(native, dict):
            raise ImageProcessingError("Invalid native provider message.")
        result = copy.deepcopy(native)
    result.setdefault("role", "assistant")
    if result["role"] != "assistant":
        raise ImageProcessingError("Model response has an invalid assistant role.")
    if "content" not in result:
        if result.get("tool_calls") or result.get("images"):
            result["content"] = None
        else:
            raise RuntimeError("Model response does not contain an assistant message.")
    content = result.get("content")
    block_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                raise ImageProcessingError("Provider content block must be an object.")
            kind = part.get("type")
            if kind in {"text", "output_text"}:
                parts.append({**part, "type": "text"})
            elif kind in {"image", "image_url", "output_image", "input_image"}:
                parts.append(image_part(part))
            elif kind == "refusal" and isinstance(part.get("refusal"), str):
                parts.append({"type": "text", "text": part["refusal"]})
            elif kind == "tool_use":
                block_calls.append(
                    {
                        "id": part.get("id"),
                        "type": "function",
                        "function": {
                            "name": part.get("name"),
                            "arguments": json.dumps(
                                part.get("input"), ensure_ascii=False
                            ),
                        },
                    }
                )
            elif kind == "function_call":
                block_calls.append(
                    {
                        "id": part.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": part.get("name"),
                            "arguments": part.get("arguments"),
                        },
                    }
                )
            elif kind in {"thinking", "redacted_thinking"}:
                result.setdefault("thinking_blocks", []).append(copy.deepcopy(part))
            else:
                # Unknown content must fail explicitly downstream, not disappear.
                parts.append(copy.deepcopy(part))
        result["content"] = parts or None
    if block_calls:
        existing = result.get("tool_calls") or []
        if not isinstance(existing, list):
            raise ImageProcessingError("Provider tool_calls must be an array.")
        result["tool_calls"] = [*existing, *block_calls]
    if result.get("images") is not None:
        if not isinstance(result["images"], list):
            raise ImageProcessingError("Provider images must be an array.")
        result["images"] = [image_part(image) for image in result["images"]]
    return result


def assistant_choices(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Model response does not contain an assistant message.")
    messages = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise RuntimeError("Model response choice must be an object.")
        message = normalize_message(choice.get("message"))
        try:
            calls = tool_calls(message)
        except ValueError as error:
            raise ImageProcessingError(str(error)) from error
        finish = choice.get("finish_reason")
        if finish == "tool_calls" and not calls:
            raise ImageProcessingError("Provider omitted its requested tool calls.")
        if calls and finish in {"length", "content_filter"}:
            raise ImageProcessingError(
                "Provider returned an incomplete tool call batch."
            )
        messages.append(message)
    return messages


def merge_assistant_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect visible content and all native calls, preserving response order."""
    if len(messages) == 1 and not messages[0].get("images"):
        return copy.deepcopy(messages[0])
    merged = copy.deepcopy(messages[0])
    merged.pop("images", None)
    contents: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index and contents:
            contents.append({"type": "text", "text": "\n\n"})
        content = message.get("content")
        if content is None and isinstance(message.get("refusal"), str):
            content = message["refusal"]
        if isinstance(content, str):
            contents.append({"type": "text", "text": content})
        elif isinstance(content, list):
            contents.extend(copy.deepcopy(content))
        elif content is not None:
            raise ImageProcessingError(
                "Provider content must be text or a content array."
            )
        contents.extend(copy.deepcopy(message.get("images") or []))
        calls.extend(copy.deepcopy(message.get("tool_calls") or []))
        if index:
            for key in ("annotations", "thinking_blocks", "reasoning_items"):
                if isinstance(message.get(key), list):
                    merged[key] = [
                        *(merged.get(key) or []),
                        *copy.deepcopy(message[key]),
                    ]
            if isinstance(message.get("reasoning_content"), str):
                merged["reasoning_content"] = (
                    merged.get("reasoning_content") or ""
                ) + message["reasoning_content"]
            fields = message.get("provider_specific_fields") or {}
            if isinstance(fields, dict):
                for key in (
                    STATE_KEY,
                    "thought_signatures",
                    "server_side_tool_invocations",
                ):
                    if isinstance(fields.get(key), list):
                        merged.setdefault("provider_specific_fields", {}).setdefault(
                            key, []
                        ).extend(copy.deepcopy(fields[key]))
    merged["content"] = contents or None
    if calls:
        merged["tool_calls"] = calls
    else:
        merged.pop("tool_calls", None)
    try:
        tool_calls(merged)  # Reject duplicate IDs across candidates before execution.
    except ValueError as error:
        raise ImageProcessingError(str(error)) from error
    return merged
