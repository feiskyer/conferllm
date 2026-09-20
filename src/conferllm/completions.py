"""Preserve native completion content before SDK convenience conversions lose it."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, cast

from .outputs import NATIVE_MESSAGE_KEY, image_part

if TYPE_CHECKING:
    from litellm.types.llms.vertex_ai import Candidates
    from litellm.types.utils import ModelResponse


class CompletionCapture:
    """Per-request response capture; never retain request data or credentials."""

    def __init__(self, logger_fn: Any = None) -> None:
        self.logger_fn = logger_fn
        self.payload: dict[str, Any] | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        if event.get("log_event_type") == "pre_api_call":
            self.payload = None  # A retry must not recover an older response.
        elif event.get("log_event_type") == "post_api_call":
            payload = event.get("original_response")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    payload = None
            else:
                dump = getattr(payload, "model_dump", None)
                if callable(dump):
                    payload = dump(warnings=False)
            if isinstance(payload, dict):
                self.payload = copy.deepcopy(payload)
        if callable(self.logger_fn):
            self.logger_fn(event)

    def recover_content_arrays(self) -> ModelResponse | None:
        """Recover a complete native response rejected by the SDK's str-only content."""
        from litellm.types.utils import ModelResponse

        payload = self.payload or {}
        choices = payload.get("choices")
        if (
            payload.get("object") != "chat.completion"
            or payload.get("error") is not None
            or not isinstance(choices, list)
            or not choices
            or not all(
                isinstance(choice, dict)
                and isinstance(choice.get("message"), dict)
                and choice.get("finish_reason")
                in {"stop", "tool_calls", "length", "content_filter"}
                for choice in choices
            )
            or not any(
                isinstance(choice["message"].get("content"), list) for choice in choices
            )
        ):
            return None
        # Tools remain in native metadata and pass the usual whole-batch
        # validation. No text parsing, synthetic tool calls, or network retry.
        return ModelResponse(
            id=payload.get("id"),
            model=payload.get("model"),
            created=payload.get("created"),
            usage=payload.get("usage"),
            choices=[
                {
                    "index": choice.get("index", index),
                    "finish_reason": choice["finish_reason"],
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "provider_specific_fields": {
                            NATIVE_MESSAGE_KEY: copy.deepcopy(choice["message"])
                        },
                    },
                }
                for index, choice in enumerate(choices)
            ],
        )

    def restore(
        self, response: ModelResponse, parameters: dict[str, Any]
    ) -> ModelResponse:
        """Keep native fields and isolate Gemini's per-candidate SDK conversion."""
        payload = self.payload or {}
        choices = payload.get("choices")
        if isinstance(choices, list) and len(choices) == len(response.choices):
            for choice, native in zip(response.choices, choices, strict=True):
                message = native.get("message") if isinstance(native, dict) else None
                if isinstance(message, dict):
                    fields = choice.message.provider_specific_fields or {}
                    fields[NATIVE_MESSAGE_KEY] = copy.deepcopy(message)
                    choice.message.provider_specific_fields = fields
                    # Some SDKs infer tool calls from tool-shaped ordinary text.
                    # Only the actual native tool fields authorize execution.
                    finish = native.get("finish_reason")
                    if finish in {"stop", "length", "content_filter", "tool_calls"}:
                        choice.finish_reason = finish
            return response

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return response
        if any(choice.finish_reason == "content_filter" for choice in response.choices):
            return response  # Retain the SDK's blocked-response handling.
        # LiteLLM 1.97/1.99 reuse a mutable message across candidates. Reusing the
        # existing converter once per candidate retains its signature handling,
        # without inheriting another candidate's tools, images, or reasoning.
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        response = response.model_copy(deep=True)
        response.choices = []
        call_index = 0
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or "content" not in candidate:
                continue
            candidate = {**candidate, "index": candidate.get("index", index)}
            *_, call_index = VertexGeminiConfig._process_candidates(
                [cast("Candidates", candidate)], response, parameters, call_index
            )
            message = response.choices[-1].message
            native_message = message.model_dump(exclude_none=True)
            native_message["content"] = _gemini_content(candidate["content"]["parts"])
            native_message.pop("images", None)
            fields = message.provider_specific_fields or {}
            fields[NATIVE_MESSAGE_KEY] = native_message
            message.provider_specific_fields = fields
        return response


def _gemini_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Preserve interleaved text/image order; SDK tool and signature data stay intact."""
    content: list[dict[str, Any]] = []
    for part in parts:
        handled = False
        if isinstance(part.get("text"), str):
            handled = True
            if not part.get("thought"):
                content.append({"type": "text", "text": part["text"]})
        inline = part.get("inlineData", part.get("inline_data"))
        file = part.get("fileData", part.get("file_data"))
        if isinstance(inline, dict):
            mime = inline.get("mimeType", inline.get("mime_type", ""))
            if isinstance(mime, str) and mime.startswith("image/"):
                content.append(
                    image_part({"data": inline.get("data"), "mime_type": mime})
                )
                handled = True
        if isinstance(file, dict):
            mime = file.get("mimeType", file.get("mime_type", ""))
            if isinstance(mime, str) and mime.startswith("image/"):
                content.append(image_part(file.get("fileUri", file.get("file_uri"))))
                handled = True
        if "functionCall" in part or "function_call" in part:
            handled = True  # Already converted by LiteLLM, including call signatures.
        if not handled and set(part) <= {
            "thoughtSignature",
            "thought_signature",
            "thought",
        }:
            handled = True
        if not handled:
            # Do not silently discard audio/video or future typed content.
            content.append({"type": "unsupported_gemini_part"})
    return content or None
