"""Shared stateful multimodal chat service for ConferLLM."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from .artifacts import Artifact, ArtifactError, ArtifactTransaction
from .client import LLMClient
from .errors import ConferLLMError, normalize_error
from .images import (
    ImagePayload,
    ImageProcessingError,
    image_bytes_to_data_url,
    image_payload_from_bytes,
    load_image_inputs,
    normalize_assistant_content,
    prepare_image_output_dir,
    replace_image_refs,
    replace_provider_images,
)
from .outputs import assistant_choices, merge_assistant_messages
from .responses import (
    normalize_response_images,
    response_content,
    response_items,
    restore_response_images,
)
from .session import LoadedSession, SessionStore
from .tool_protocol import tool_calls
from .tools import ToolRuntime
from .tools.common import check_cancelled

CHAT_RESPONSE_SCHEMA_VERSION = "conferllm.chat.response.v1"
MAX_TOOL_ROUNDS = 1000
DEFAULT_IMAGE_OUTPUT_DIR = Path("/tmp")


@dataclass
class _ModelTurn:
    response: dict[str, Any]
    assistant: dict[str, Any]
    text: str
    content: list[dict[str, Any]]
    tool_messages: list[dict[str, Any]]
    usage: dict[str, Any] | None
    warnings: list[str]


@dataclass(frozen=True)
class ChatResult:
    """Normalized result returned by the CLI and MCP server."""

    session_id: str
    name: str
    model: str
    answer: Any
    response: dict[str, Any] | None = None
    turn: int = 1
    content: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Return normalized assistant text."""
        if isinstance(self.answer, str):
            return self.answer
        if self.answer is None:
            return ""
        return str(self.answer)

    def to_dict(self, *, include_local_paths: bool = True) -> dict[str, Any]:
        """Return the versioned JSON result and compatibility fields."""
        content = list(self.content)
        if not content and self.text:
            content = [{"type": "text", "text": self.text}]

        artifacts: list[dict[str, Any]] = []
        for artifact in self.artifacts:
            serialized = dict(artifact)
            if artifact.get("direction") == "output":
                serialized["saved_path"] = artifact.get(
                    "exported_path"
                ) or artifact.get("local_path")
            if not include_local_paths:
                serialized.pop("local_path", None)
                serialized.pop("exported_path", None)
            artifacts.append(serialized)

        result: dict[str, Any] = {
            "schema_version": CHAT_RESPONSE_SCHEMA_VERSION,
            "ok": True,
            "session": {
                "id": self.session_id,
                "name": self.name,
                "model": self.model,
                "turn": self.turn,
            },
            "message": {
                "text": self.text,
                "content": content,
            },
            "artifacts": artifacts,
            "warnings": list(self.warnings),
            # Compatibility fields retained for one public cycle.
            "session_id": self.session_id,
            "name": self.name,
            "model": self.model,
            "answer": self.answer,
        }
        if self.response is not None:
            result["raw_response"] = self.response
            result["response"] = self.response
        return result


class ChatService:
    """Run native model/tool loops with transactional conversation storage."""

    def __init__(
        self,
        client: LLMClient,
        session_store: SessionStore | None = None,
    ) -> None:
        self.client = client
        self.session_store = session_store or SessionStore(
            client.config.get_session_root()
        )

    def chat(
        self,
        message: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        images: Sequence[str | Path] | None = None,
        image_output_dir: str | Path | None = None,
        include_raw_response: bool = False,
    ) -> ChatResult:
        """Create or continue a chat session."""
        check_cancelled()
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        if (model is None) == (session_id is None):
            raise ValueError("Exactly one of model or session_id is required.")
        if session_id is not None and name is not None:
            raise ValueError("name is only valid when creating a new session.")

        output_dir = (
            Path(image_output_dir).expanduser().resolve()
            if image_output_dir is not None
            else DEFAULT_IMAGE_OUTPUT_DIR
        )
        ordered_images = list(images or [])

        if model is not None:
            limits, warnings = self.client.validate_chat_request(
                model,
                image_count=len(ordered_images),
            )
            payloads = load_image_inputs(
                ordered_images,
                max_count=limits.max_count,
                max_bytes_per_image=limits.max_bytes_per_image,
                max_total_bytes=limits.max_total_bytes,
            )
            return self._create_session(
                model=model,
                name=name,
                message=message,
                image_payloads=payloads,
                output_dir=output_dir,
                warnings=warnings,
                include_raw_response=include_raw_response,
            )

        assert session_id is not None
        return self._continue_session(
            session_id=session_id,
            message=message,
            image_sources=ordered_images,
            output_dir=output_dir,
            include_raw_response=include_raw_response,
        )

    def _create_session(
        self,
        *,
        model: str,
        name: str | None,
        message: str,
        image_payloads: list[ImagePayload],
        output_dir: Path | None,
        warnings: list[str],
        include_raw_response: bool,
    ) -> ChatResult:
        created_at = datetime.now().astimezone()
        session_id = self.session_store.generate_session_id(created_at)
        session_name = self.session_store.derive_session_name(message, name)
        transaction = self.session_store.artifact_transaction(session_id, 1)

        with transaction:
            logical_user, provider_user = self._stage_user_message(
                transaction,
                message,
                image_payloads,
                schema_version=2,
            )
            result = self._call_model(
                model,
                [provider_user],
                transaction,
                schema_version=2,
            )
            artifacts = list(transaction.artifacts)
            active_transaction = transaction if artifacts else None
            try:
                check_cancelled()
                metadata = self.session_store.create_session(
                    session_id,
                    session_name,
                    model,
                    logical_user,
                    result.assistant,
                    response_id=self._response_id(result.response),
                    usage=result.usage,
                    created_at=created_at,
                    artifacts=artifacts,
                    artifact_transaction=active_transaction,
                    tool_messages=result.tool_messages,
                )
            except Exception as error:
                if result.tool_messages:
                    raise self._tool_failure(
                        error, transaction, result.tool_messages
                    ) from error
                raise

        public_artifacts = self._public_artifacts(
            metadata.session_id,
            artifacts,
        )
        export_warnings = self._export_output_artifacts(
            metadata.session_id,
            public_artifacts,
            output_dir,
        )
        return ChatResult(
            session_id=metadata.session_id,
            name=metadata.name,
            model=metadata.model,
            turn=1,
            answer=self._public_text(result.content, public_artifacts),
            content=self._public_content(result.content, public_artifacts),
            artifacts=public_artifacts,
            warnings=[*warnings, *result.warnings, *export_warnings],
            response=(
                self._public_response(result.response, public_artifacts)
                if include_raw_response
                else None
            ),
        )

    def _continue_session(
        self,
        *,
        session_id: str,
        message: str,
        image_sources: list[str | Path],
        output_dir: Path | None,
        include_raw_response: bool,
    ) -> ChatResult:
        with self.session_store.lock(session_id):
            loaded = self.session_store.load_session(session_id)
            history_image_count = sum(
                isinstance(item, dict)
                and item.get("type") in {"image_ref", "image_url"}
                for historical_message in loaded.messages
                if isinstance(historical_message.get("content"), list)
                for item in historical_message["content"]
            )
            limits, warnings = self.client.validate_chat_request(
                loaded.metadata.model,
                image_count=len(image_sources),
                history_image_count=history_image_count,
            )
            image_payloads = load_image_inputs(
                image_sources,
                max_count=limits.max_count,
                max_bytes_per_image=limits.max_bytes_per_image,
                max_total_bytes=limits.max_total_bytes,
            )
            self.session_store.discard_uncommitted_artifacts(loaded)
            transaction = self.session_store.artifact_transaction(
                session_id,
                loaded.next_turn,
            )

            with transaction:
                logical_user, provider_user = self._stage_user_message(
                    transaction,
                    message,
                    image_payloads,
                    schema_version=loaded.schema_version,
                )
                history = self._provider_history(loaded)
                result = self._call_model(
                    loaded.metadata.model,
                    [*history, provider_user],
                    transaction,
                    schema_version=loaded.schema_version,
                )
                artifacts = list(transaction.artifacts)
                active_transaction = transaction if artifacts else None
                try:
                    check_cancelled()
                    self.session_store.append_turn(
                        session_id,
                        loaded.next_turn,
                        logical_user,
                        result.assistant,
                        response_id=self._response_id(result.response),
                        usage=result.usage,
                        artifacts=artifacts,
                        artifact_transaction=active_transaction,
                        tool_messages=result.tool_messages,
                    )
                except Exception as error:
                    if result.tool_messages:
                        raise self._tool_failure(
                            error, transaction, result.tool_messages
                        ) from error
                    raise

        public_artifacts = self._public_artifacts(session_id, artifacts)
        export_warnings = self._export_output_artifacts(
            session_id,
            public_artifacts,
            output_dir,
        )
        return ChatResult(
            session_id=loaded.metadata.session_id,
            name=loaded.metadata.name,
            model=loaded.metadata.model,
            turn=loaded.next_turn,
            answer=self._public_text(result.content, public_artifacts),
            content=self._public_content(result.content, public_artifacts),
            artifacts=public_artifacts,
            warnings=[*warnings, *result.warnings, *export_warnings],
            response=(
                self._public_response(result.response, public_artifacts)
                if include_raw_response
                else None
            ),
        )

    def _stage_user_message(
        self,
        transaction: ArtifactTransaction,
        message: str,
        image_payloads: list[ImagePayload],
        *,
        schema_version: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not image_payloads:
            simple = {"role": "user", "content": message}
            return simple, simple

        logical_content: list[dict[str, Any]] = [{"type": "text", "text": message}]
        provider_content: list[dict[str, Any]] = [{"type": "text", "text": message}]
        for index, payload in enumerate(image_payloads):
            artifact = transaction.stage_bytes(
                payload.data,
                direction="input",
                mime_type=payload.mime_type,
                index=index,
            )
            staged_path = self._staged_path(transaction, artifact)
            provider_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(staged_path)},
                }
            )
            if schema_version == 2:
                logical_content.append(
                    {
                        "type": "image_ref",
                        "artifact_id": artifact.id,
                    }
                )
            else:
                logical_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(self._canonical_path(artifact, transaction))
                        },
                    }
                )

        return (
            {"role": "user", "content": logical_content},
            {"role": "user", "content": provider_content},
        )

    def _provider_history(self, loaded: LoadedSession) -> list[dict[str, Any]]:
        if loaded.schema_version != 2:

            def resolve_legacy(artifact_id: str) -> str:
                data = self.session_store.read_loaded_artifact(loaded, artifact_id)
                return image_bytes_to_data_url(
                    data, image_payload_from_bytes(data).mime_type
                )

            return restore_response_images(loaded.messages, resolve_legacy)

        artifacts = {artifact.id: artifact for artifact in loaded.artifacts}
        cache: dict[str, str] = {}

        def resolve(artifact_id: str) -> str:
            if artifact_id in cache:
                return cache[artifact_id]
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                raise ValueError(
                    f"Stored image reference '{artifact_id}' has no artifact record."
                )
            data = self.session_store.read_loaded_artifact(loaded, artifact_id)
            value = image_bytes_to_data_url(data, artifact.mime_type)
            cache[artifact_id] = value
            return value

        return restore_response_images(
            replace_image_refs(loaded.messages, resolve), resolve
        )

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        transaction: ArtifactTransaction,
        *,
        schema_version: int,
    ) -> _ModelTurn:
        history = list(messages)
        intermediate: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        seen: set[str] = set()
        rounds = 0
        attempted = 0
        runtime = ToolRuntime()
        try:
            with runtime:
                while True:
                    check_cancelled()
                    raw = self.client.chat(model, list(history)).model_dump()
                    check_cancelled()
                    if not isinstance(raw, dict):
                        raise RuntimeError("Model response is not a JSON object.")
                    provider_message = self._extract_assistant_message(raw)
                    try:
                        calls = tool_calls(provider_message)
                    except ValueError as error:
                        raise ImageProcessingError(str(error)) from error
                    if any(call["id"] in seen for call in calls):
                        raise ImageProcessingError("Provider repeated a tool call ID.")
                    if calls and rounds >= MAX_TOOL_ROUNDS:
                        raise ConferLLMError(
                            "tool_call_limit_exceeded",
                            f"Model exceeded {MAX_TOOL_ROUNDS} tool rounds.",
                        )
                    self._accumulate_usage(usage, self._usage(raw) or {})
                    artifact_start = len(transaction.artifacts)
                    processed = self._process_output_images(raw, transaction)
                    outputs = [
                        artifact
                        for artifact in transaction.artifacts[artifact_start:]
                        if artifact.direction == "output"
                    ]
                    assistant, text, content = self._assistant_messages(
                        processed,
                        transaction,
                        schema_version=schema_version,
                        output_artifacts=outputs,
                        allow_tool_calls=bool(calls),
                    )
                    if not calls:
                        break
                    rounds += 1
                    intermediate.append(assistant)
                    history.append(
                        self._current_provider_message(assistant, content, transaction)
                    )
                    for call in calls:
                        check_cancelled()
                        seen.add(call["id"])
                        function = call["function"]
                        attempted += 1
                        tool_result = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": function["name"],
                            "content": runtime.execute(
                                function["name"], function["arguments"]
                            ),
                        }
                        intermediate.append(tool_result)
                        history.append(tool_result)
        except Exception as error:
            if attempted:
                raise self._tool_failure(
                    error, transaction, intermediate, attempted=attempted
                ) from error
            raise
        return _ModelTurn(
            processed,
            assistant,
            text,
            content,
            intermediate,
            usage or None,
            list(runtime.warnings),
        )

    @staticmethod
    def _tool_failure(
        error: Exception,
        transaction: ArtifactTransaction,
        messages: list[dict[str, Any]],
        *,
        attempted: int | None = None,
    ) -> ConferLLMError:
        public = normalize_error(error)
        return ConferLLMError(
            public.code,
            public.message + " Tools may already have changed files or external "
            "state; those effects are not rolled back. Do not blindly retry.",
            details={
                **public.details,
                "session_id": transaction.session_id,
                "turn": transaction.turn,
                "tool_calls_attempted": (
                    attempted
                    if attempted is not None
                    else sum(message["role"] == "tool" for message in messages)
                ),
                "side_effects_may_remain": True,
            },
        )

    @staticmethod
    def _accumulate_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, dict):
                nested = total.setdefault(key, {})
                if isinstance(nested, dict):
                    ChatService._accumulate_usage(nested, value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
            elif value is not None:
                total[key] = copy.deepcopy(value)

    def _current_provider_message(
        self,
        assistant: dict[str, Any],
        content: list[dict[str, Any]],
        transaction: ArtifactTransaction,
    ) -> dict[str, Any]:
        """Replay current-turn images from staging, not uncommitted final paths."""
        message = dict(assistant)
        if any(item.get("type") == "image_ref" for item in content):
            message["content"] = content
        artifacts = {artifact.id: artifact for artifact in transaction.artifacts}

        def resolve(artifact_id: str) -> str:
            artifact = artifacts[artifact_id]
            return image_bytes_to_data_url(
                self._staged_path(transaction, artifact).read_bytes(),
                artifact.mime_type,
            )

        return restore_response_images(replace_image_refs([message], resolve), resolve)[
            0
        ]

    def _process_output_images(
        self, raw_response: dict[str, Any], transaction: ArtifactTransaction
    ) -> dict[str, Any]:
        """Stage visible images without interpreting or rewriting tool arguments."""
        processed = copy.deepcopy(raw_response)

        def save_output(payload: ImagePayload, _index: int) -> str:
            output_index = sum(
                artifact.direction == "output" for artifact in transaction.artifacts
            )
            try:
                artifact = transaction.stage_bytes(
                    payload.data,
                    direction="output",
                    mime_type=payload.mime_type,
                    index=output_index,
                )
            except ArtifactError as error:
                raise ImageProcessingError(
                    f"Failed to save generated image {output_index + 1}: {error}",
                    code=error.code,
                ) from error
            public = artifact.to_public(
                transaction.session_id,
                transaction.asset_root,
                include_local_path=False,
            )
            return str(public["uri"])

        for choice, message in zip(
            processed["choices"], assistant_choices(raw_response), strict=True
        ):
            native_items = response_items(message)
            if native_items is not None:
                native_messages = [
                    item for item in native_items if item.get("type") == "message"
                ]
                contents, _ = replace_provider_images(
                    [item["content"] for item in native_messages], save_output
                )
                for native_message, content in zip(
                    native_messages, contents, strict=True
                ):
                    native_message["content"] = content
                message["content"] = response_content(native_items)
                message.pop("images", None)
            else:
                visible = {
                    key: message[key]
                    for key in ("content", "images", "refusal")
                    if key in message
                }
                visible, _ = replace_provider_images(visible, save_output)
                message.update(visible)
            choice["message"] = message
        return processed

    def _assistant_messages(
        self,
        response: dict[str, Any],
        transaction: ArtifactTransaction,
        *,
        schema_version: int,
        output_artifacts: list[Artifact] | None = None,
        allow_tool_calls: bool = False,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        provider_message = self._extract_assistant_message(response)
        if not allow_tool_calls and provider_message.get("tool_calls"):
            raise ImageProcessingError(
                "Final assistant response contains unexecuted tool calls."
            )
        original_content = provider_message.get("content")
        if output_artifacts is None:
            output_artifacts = [
                artifact
                for artifact in transaction.artifacts
                if artifact.direction == "output"
            ]
        artifact_uris = {
            f"conferllm://sessions/{transaction.session_id}/artifacts/{artifact.id}"
            for artifact in output_artifacts
        }
        self._validate_provider_content(original_content, artifact_uris)
        separate_images = provider_message.get("images")
        if separate_images is not None:
            if not isinstance(separate_images, list):
                raise ImageProcessingError("Provider images must be an array.")
            for image in separate_images:
                if not isinstance(image, dict):
                    raise ImageProcessingError("Provider image must be an object.")
                self._validate_provider_content(
                    [{"type": "image_url", "image_url": image.get("image_url")}],
                    artifact_uris,
                )
        if original_content is None and isinstance(
            provider_message.get("refusal"), str
        ):
            original_content = provider_message["refusal"]
        text, normalized_content = normalize_assistant_content(
            original_content, artifact_uris=artifact_uris
        )
        referenced_ids = {
            item["artifact_id"]
            for item in normalized_content
            if item.get("type") == "image_ref"
        }
        # LiteLLM represents generated images in message.images for several
        # providers, including image-only replies whose content is null.
        for artifact in output_artifacts:
            if artifact.id not in referenced_ids:
                normalized_content.append(
                    {"type": "image_ref", "artifact_id": artifact.id}
                )
        if not normalized_content and not allow_tool_calls:
            raise ImageProcessingError(
                "Model response does not contain supported assistant text or embedded images."
            )

        stored_message = copy.deepcopy(provider_message)
        normalize_response_images(stored_message, artifact_uris)
        stored_message.pop("images", None)
        if allow_tool_calls and not normalized_content:
            stored_message["content"] = original_content
        elif schema_version == 2:
            if isinstance(original_content, str) and not any(
                item.get("type") == "image_ref" for item in normalized_content
            ):
                stored_message["content"] = original_content
            else:
                stored_message["content"] = normalized_content
        else:
            if not output_artifacts:
                stored_message["content"] = original_content
            else:
                paths = {
                    artifact.id: str(self._canonical_path(artifact, transaction))
                    for artifact in output_artifacts
                }
                stored_message["content"] = replace_image_refs(
                    [{"role": "assistant", "content": normalized_content}],
                    paths.__getitem__,
                )[0]["content"]
        return stored_message, text, normalized_content

    @staticmethod
    def _validate_provider_content(content: Any, artifact_uris: set[str]) -> None:
        """Fail explicitly instead of dropping unsupported output or replaying it."""
        if content is None or isinstance(content, str):
            return
        if not isinstance(content, list):
            raise ImageProcessingError(
                "Provider content must be text or a content array."
            )
        for item in content:
            if isinstance(item, str):
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if (
                        isinstance(image_url, dict)
                        and isinstance(image_url.get("url"), str)
                        and image_url["url"] in artifact_uris
                    ):
                        continue
                    raise ImageProcessingError(
                        "Provider output image could not be saved as a turn artifact."
                    )
            raise ImageProcessingError(
                "Provider returned an unsupported content block."
            )

    def _public_artifacts(
        self,
        session_id: str,
        artifacts: list[Artifact],
    ) -> list[dict[str, Any]]:
        asset_root = self.session_store.assets_path(session_id)
        return [
            artifact.to_public(
                session_id,
                asset_root,
                include_local_path=True,
            )
            for artifact in artifacts
        ]

    @staticmethod
    def _public_content(
        content: list[dict[str, Any]], artifacts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return paths for every output image, including earlier tool rounds."""
        outputs = {
            item["id"]: item for item in artifacts if item.get("direction") == "output"
        }
        seen: set[str] = set()
        result = []
        for part in content:
            if part.get("type") == "image_ref" and part.get("artifact_id") in outputs:
                artifact_id = part["artifact_id"]
                seen.add(artifact_id)
                artifact = outputs[artifact_id]
                result.append(
                    {
                        "type": "image_url",
                        "artifact_id": artifact_id,
                        "image_url": {
                            "url": artifact.get("exported_path")
                            or artifact["local_path"]
                        },
                    }
                )
            else:
                result.append(copy.deepcopy(part))
        for artifact_id, artifact in outputs.items():
            if artifact_id not in seen:
                result.append(
                    {
                        "type": "image_url",
                        "artifact_id": artifact_id,
                        "image_url": {
                            "url": artifact.get("exported_path")
                            or artifact["local_path"]
                        },
                    }
                )
        return result

    @staticmethod
    def _public_text(
        content: list[dict[str, Any]], artifacts: list[dict[str, Any]]
    ) -> str:
        pieces: list[str] = []
        previous_image = False
        for part in ChatService._public_content(content, artifacts):
            is_image = part.get("type") == "image_url"
            value = (
                str(part["image_url"]["url"])
                if is_image
                else part.get("text", "")
                if part.get("type") == "text"
                else ""
            )
            if not value:
                continue
            if (
                pieces
                and (is_image or previous_image)
                and not pieces[-1][-1].isspace()
                and not value[0].isspace()
            ):
                pieces.append("\n")
            pieces.append(value)
            previous_image = is_image
        return "".join(pieces)

    @staticmethod
    def _public_response(
        response: dict[str, Any], artifacts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        replacements = {
            artifact["uri"]: artifact.get("exported_path") or artifact["local_path"]
            for artifact in artifacts
            if artifact.get("direction") == "output"
        }

        def replace(value: Any) -> Any:
            if isinstance(value, str):
                for uri, path in replacements.items():
                    value = value.replace(uri, str(path))
                return value
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        return cast(dict[str, Any], replace(response))

    @staticmethod
    def _staged_path(
        transaction: ArtifactTransaction,
        artifact: Artifact,
    ) -> Path:
        return transaction.staging_path / PurePosixPath(artifact.relative_path).name

    @staticmethod
    def _canonical_path(
        artifact: Artifact,
        transaction: ArtifactTransaction,
    ) -> Path:
        return transaction.asset_root.joinpath(
            *PurePosixPath(artifact.relative_path).parts
        )

    @staticmethod
    def _extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
        return merge_assistant_messages(assistant_choices(response))

    @staticmethod
    def _response_id(response: dict[str, Any]) -> str | None:
        response_id = response.get("id")
        return response_id if isinstance(response_id, str) else None

    @staticmethod
    def _usage(response: dict[str, Any]) -> dict[str, Any] | None:
        usage = response.get("usage")
        return usage if isinstance(usage, dict) else None

    @staticmethod
    def _export_output_artifacts(
        session_id: str,
        artifacts: list[dict[str, Any]],
        output_dir: Path | None,
    ) -> list[str]:
        if output_dir is None:
            return []

        warnings: list[str] = []
        if not any(artifact.get("direction") == "output" for artifact in artifacts):
            return warnings
        try:
            prepare_image_output_dir(output_dir)
        except (OSError, ValueError):
            return [
                "Unable to prepare the image export directory; "
                "canonical session artifacts remain available."
            ]
        for artifact in artifacts:
            if artifact.get("direction") != "output":
                continue
            local_path = artifact.get("local_path")
            artifact_id = artifact.get("id")
            if not isinstance(local_path, str) or not isinstance(artifact_id, str):
                continue
            source = Path(local_path)
            destination = output_dir / (
                f"{session_id}-{artifact_id}{source.suffix.lower()}"
            )
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=output_dir,
                )
                os.close(descriptor)
                temporary_path = Path(temporary_name)
                shutil.copyfile(source, temporary_path)
                temporary_path.chmod(0o600)
                os.replace(temporary_path, destination)
                temporary_path = None
                artifact["exported_path"] = str(destination)
            except OSError:
                warnings.append(
                    f"Unable to export artifact '{artifact_id}'; "
                    "the canonical session artifact remains available."
                )
            finally:
                if temporary_path is not None:
                    with suppress(FileNotFoundError):
                        temporary_path.unlink()
        return warnings
