"""Shared stateful multimodal chat service for ConferLLM."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import Artifact, ArtifactError, ArtifactTransaction
from .client import LLMClient
from .images import (
    ImagePayload,
    ImageProcessingError,
    image_bytes_to_data_url,
    load_image_inputs,
    normalize_assistant_content,
    prepare_image_output_dir,
    replace_embedded_image_data,
    replace_image_refs,
)
from .session import LoadedSession, SessionStore

CHAT_RESPONSE_SCHEMA_VERSION = "conferllm.chat.response.v1"


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
    """Coordinate model calls with transactional conversation sessions."""

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
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        if (model is None) == (session_id is None):
            raise ValueError("Exactly one of model or session_id is required.")
        if session_id is not None and name is not None:
            raise ValueError("name is only valid when creating a new session.")

        output_dir = (
            Path(image_output_dir).expanduser()
            if image_output_dir is not None
            else None
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
            processed_response = self._call_model(
                model,
                [provider_user],
                transaction,
            )
            assistant_message, text, content = self._assistant_messages(
                processed_response,
                transaction,
                schema_version=2,
            )
            artifacts = list(transaction.artifacts)
            active_transaction = transaction if artifacts else None
            metadata = self.session_store.create_session(
                session_id,
                session_name,
                model,
                logical_user,
                assistant_message,
                response_id=self._response_id(processed_response),
                usage=self._usage(processed_response),
                created_at=created_at,
                artifacts=artifacts,
                artifact_transaction=active_transaction,
            )

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
            answer=text,
            content=content,
            artifacts=public_artifacts,
            warnings=[*warnings, *export_warnings],
            response=processed_response if include_raw_response else None,
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
                processed_response = self._call_model(
                    loaded.metadata.model,
                    [*history, provider_user],
                    transaction,
                )
                assistant_message, text, content = self._assistant_messages(
                    processed_response,
                    transaction,
                    schema_version=loaded.schema_version,
                )
                artifacts = list(transaction.artifacts)
                active_transaction = transaction if artifacts else None
                self.session_store.append_turn(
                    session_id,
                    loaded.next_turn,
                    logical_user,
                    assistant_message,
                    response_id=self._response_id(processed_response),
                    usage=self._usage(processed_response),
                    artifacts=artifacts,
                    artifact_transaction=active_transaction,
                )

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
            answer=text,
            content=content,
            artifacts=public_artifacts,
            warnings=[*warnings, *export_warnings],
            response=processed_response if include_raw_response else None,
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
            return loaded.messages

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

        return replace_image_refs(loaded.messages, resolve)

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        transaction: ArtifactTransaction,
    ) -> dict[str, Any]:
        raw_response = self.client.chat(model, messages).model_dump()
        if not isinstance(raw_response, dict):
            raise RuntimeError("Model response is not a JSON object.")

        def save_output(payload: ImagePayload, index: int) -> str:
            try:
                artifact = transaction.stage_bytes(
                    payload.data,
                    direction="output",
                    mime_type=payload.mime_type,
                    index=index,
                )
            except ArtifactError as error:
                raise ImageProcessingError(
                    f"Failed to save generated image {index + 1}: {error}",
                    code=error.code,
                ) from error
            public = artifact.to_public(
                transaction.session_id,
                transaction.asset_root,
                include_local_path=False,
            )
            return str(public["uri"])

        processed, _ = replace_embedded_image_data(raw_response, save_output)
        if not isinstance(processed, dict):  # pragma: no cover - type invariant
            raise RuntimeError("Model response is not a JSON object.")
        return processed

    def _assistant_messages(
        self,
        response: dict[str, Any],
        transaction: ArtifactTransaction,
        *,
        schema_version: int,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        provider_message = self._extract_assistant_message(response)
        if provider_message.get("tool_calls") or provider_message.get("function_call"):
            raise ImageProcessingError(
                "The provider requested tool execution, which ConferLLM does not perform."
            )
        original_content = provider_message.get("content")
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
        if not normalized_content:
            raise ImageProcessingError(
                "Model response does not contain supported assistant text or embedded images."
            )

        stored_message = dict(provider_message)
        stored_message.pop("images", None)
        if schema_version == 2:
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
                        "Provider output images must be embedded base64 data URLs; "
                        "remote output URLs are not fetched."
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
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Model response does not contain an assistant message.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("Model response does not contain an assistant message.")

        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise RuntimeError("Model response does not contain an assistant message.")

        assistant_message = dict(message)
        assistant_message.setdefault("role", "assistant")
        if assistant_message["role"] != "assistant":
            raise ImageProcessingError("Model response has an invalid assistant role.")
        return assistant_message

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
