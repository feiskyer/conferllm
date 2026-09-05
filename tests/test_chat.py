"""Tests for the shared stateful chat service."""

import base64
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conferllm.chat import ChatService
from conferllm.client import EffectiveImageLimits, LLMClient, ModelCapabilityError
from conferllm.config import (
    ConferLLMConfig,
    ImageLimits,
    ModelCapabilities,
    ModelConfig,
    get_default_app_dir,
)
from conferllm.images import ImageProcessingError
from conferllm.session import SessionStore

PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal-png"
JPEG_BYTES = b"\xff\xd8\xffminimal-jpeg"


def make_client() -> MagicMock:
    """Implement the real typed client contract instead of bypassing preflight."""
    client = MagicMock(spec=LLMClient)
    defaults = ImageLimits()
    client.validate_chat_request.return_value = (
        EffectiveImageLimits(
            max_count=defaults.max_count,
            max_bytes_per_image=defaults.max_bytes_per_image,
            max_total_bytes=defaults.max_total_bytes,
            capabilities_declared=False,
        ),
        [],
    )
    return client


def make_response(
    content: object = "Model answer",
    *,
    response_id: str = "response-1",
) -> MagicMock:
    """Create a LiteLLM response double."""
    response = MagicMock()
    response.model_dump.return_value = {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": 12},
    }
    return response


def test_create_session_persists_first_turn(tmp_path: Path) -> None:
    """Create a named session only after a successful provider response."""
    client = make_client()
    client.chat.return_value = make_response()
    store = SessionStore(tmp_path / "sessions")

    result = ChatService(client, store).chat(
        "Explain Raft.",
        model="reasoning",
        name="Raft notes",
    )

    assert result.name == "Raft notes"
    assert result.model == "reasoning"
    assert result.answer == "Model answer"
    loaded = store.load_session(result.session_id)
    assert loaded.metadata.name == "Raft notes"
    assert loaded.messages == [
        {"role": "user", "content": "Explain Raft."},
        {"role": "assistant", "content": "Model answer"},
    ]
    assert loaded.next_turn == 2
    assert loaded.schema_version == 2
    assert result.to_dict()["schema_version"] == "conferllm.chat.response.v1"
    assert result.response is None


def test_create_session_uses_one_timestamp_for_id_and_metadata(
    tmp_path: Path,
) -> None:
    """Keep date-based paths aligned when a provider call crosses midnight."""
    client = make_client()
    client.chat.return_value = make_response()
    store = SessionStore(tmp_path / "sessions")

    with (
        patch.object(
            store,
            "generate_session_id",
            wraps=store.generate_session_id,
        ) as generate_session_id,
        patch.object(
            store,
            "create_session",
            wraps=store.create_session,
        ) as create_session,
    ):
        result = ChatService(client, store).chat("Hello", model="reasoning")

    created_at = generate_session_id.call_args.args[0]
    assert create_session.call_args.kwargs["created_at"] == created_at
    assert result.session_id.startswith(created_at.strftime("%Y%m%d-"))


def test_continue_session_replays_history(tmp_path: Path) -> None:
    """Restore prior turns and use the session's stored model alias."""
    client = make_client()
    client.chat.side_effect = [
        make_response("First answer", response_id="response-1"),
        make_response("Second answer", response_id="response-2"),
    ]
    service = ChatService(client, SessionStore(tmp_path / "sessions"))
    first = service.chat("First question", model="reasoning")

    second = service.chat("Follow-up", session_id=first.session_id)

    assert second.session_id == first.session_id
    assert second.name == "First question"
    assert second.answer == "Second answer"
    assert client.chat.call_args_list[1].args == (
        "reasoning",
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ],
    )


def test_app_directory_move_preserves_session_and_image_replay(
    tmp_path: Path,
) -> None:
    """Relocate a v2 session without rewriting its JSONL or image metadata."""
    previous_app = tmp_path / "previous-app"
    store = SessionStore(previous_app / "sessions")
    source_image = tmp_path / "input.png"
    source_image.write_bytes(PNG_BYTES)
    input_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    output_url = "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode(
        "ascii"
    )
    client = make_client()
    client.chat.side_effect = [
        make_response(f"Generated: {output_url}"),
        make_response("Continued after the move"),
    ]
    first = ChatService(client, store).chat(
        "Describe this and generate another image.",
        model="vision",
        images=[source_image],
    )
    original_jsonl = store.session_path(first.session_id).read_bytes()
    original_artifacts = {
        artifact["id"]: store.read_artifact(first.session_id, artifact["id"])
        for artifact in first.artifacts
    }

    previous_app.rename(get_default_app_dir())
    source_image.unlink()
    relocated_store = SessionStore()
    assert relocated_store.root == get_default_app_dir() / "sessions"
    assert relocated_store.session_path(first.session_id).read_bytes() == original_jsonl
    assert {
        artifact_id: relocated_store.read_artifact(first.session_id, artifact_id)
        for artifact_id in original_artifacts
    } == original_artifacts

    continued = ChatService(client, relocated_store).chat(
        "Continue using both images.",
        session_id=first.session_id,
    )

    assert continued.session_id == first.session_id
    assert continued.answer == "Continued after the move"
    assert relocated_store.load_session(first.session_id).next_turn == 3
    replayed_urls = [
        item["image_url"]["url"]
        for message in client.chat.call_args_list[1].args[1]
        if isinstance(message["content"], list)
        for item in message["content"]
        if item["type"] == "image_url"
    ]
    assert replayed_urls == [input_url, output_url]


def test_provider_failure_does_not_create_new_session(tmp_path: Path) -> None:
    """Leave no session JSONL behind when the first model call fails."""
    client = make_client()
    client.chat.side_effect = RuntimeError("provider unavailable")
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        ChatService(client, store).chat("Hello", model="reasoning")

    assert store.list_sessions(limit=0) == []


def test_provider_failure_does_not_append_partial_turn(tmp_path: Path) -> None:
    """Keep an existing session unchanged after a failed continuation."""
    client = make_client()
    client.chat.side_effect = [
        make_response("First answer"),
        RuntimeError("provider unavailable"),
    ]
    store = SessionStore(tmp_path / "sessions")
    service = ChatService(client, store)
    first = service.chat("First question", model="reasoning")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.chat("Unstored follow-up", session_id=first.session_id)

    loaded = store.load_session(first.session_id)
    assert loaded.next_turn == 2
    assert loaded.messages == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]


@pytest.mark.parametrize(
    ("model", "session_id"),
    [
        (None, None),
        ("reasoning", "20260904-0123456789abcdef0123456789abcdef"),
    ],
)
def test_chat_requires_exactly_one_target(
    tmp_path: Path,
    model: str | None,
    session_id: str | None,
) -> None:
    """Reject ambiguous or missing model/session selection."""
    service = ChatService(MagicMock(), SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="Exactly one"):
        service.chat("Hello", model=model, session_id=session_id)


def test_name_is_rejected_for_continuation(tmp_path: Path) -> None:
    """Prevent a continuation from implicitly renaming its session."""
    service = ChatService(MagicMock(), SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="only valid"):
        service.chat(
            "Hello",
            session_id="20260904-0123456789abcdef0123456789abcdef",
            name="New name",
        )


def test_empty_message_is_rejected(tmp_path: Path) -> None:
    """Reject whitespace-only user messages before provider access."""
    service = ChatService(MagicMock(), SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="must not be empty"):
        service.chat(" \n\t ", model="reasoning")


def test_local_images_are_session_owned_and_ordered(tmp_path: Path) -> None:
    """Store ordered immutable image refs instead of original source paths."""
    first = tmp_path / "input-one.bin"
    second = tmp_path / "input-two.jpg"
    third = tmp_path / "input-three.png"
    first.write_bytes(PNG_BYTES)
    second.write_bytes(JPEG_BYTES)
    third.write_bytes(PNG_BYTES)
    client = make_client()
    client.chat.return_value = make_response()
    store = SessionStore(tmp_path / "sessions")

    result = ChatService(client, store).chat(
        "Describe this.",
        model="vision",
        images=[first, second, third],
    )

    loaded = store.load_session(result.session_id)
    user_message = loaded.messages[0]
    assert user_message["content"] == [
        {"type": "text", "text": "Describe this."},
        {"type": "image_ref", "artifact_id": "t0001-input-001"},
        {"type": "image_ref", "artifact_id": "t0001-input-002"},
        {"type": "image_ref", "artifact_id": "t0001-input-003"},
    ]
    assert [artifact.mime_type for artifact in loaded.artifacts] == [
        "image/png",
        "image/jpeg",
        "image/png",
    ]
    assert [artifact.index for artifact in loaded.artifacts] == [0, 1, 2]
    assert all(
        str(source.resolve())
        not in store.session_path(result.session_id).read_text(encoding="utf-8")
        for source in (first, second, third)
    )


def test_continuation_uses_canonical_image_after_source_changes(
    tmp_path: Path,
) -> None:
    """Replay the saved bytes after the original image is changed or deleted."""
    image = tmp_path / "input.png"
    image.write_bytes(PNG_BYTES)
    client = make_client()
    client.chat.side_effect = [
        make_response("First answer"),
        make_response("Second answer"),
    ]
    store = SessionStore(tmp_path / "sessions")
    service = ChatService(client, store)
    first = service.chat("Describe this.", model="vision", images=[image])
    image.write_bytes(JPEG_BYTES)
    image.unlink()

    service.chat("What was its format?", session_id=first.session_id)

    replayed = client.chat.call_args_list[1].args[1][0]["content"][1]["image_url"][
        "url"
    ]
    assert replayed.startswith("data:image/png;base64,")
    assert base64.b64decode(replayed.split(",", 1)[1]) == PNG_BYTES


def test_missing_image_is_rejected_before_provider_call(tmp_path: Path) -> None:
    """Return a clear error for an unreadable local image."""
    client = make_client()
    service = ChatService(client, SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="not readable"):
        service.chat(
            "Describe this.",
            model="vision",
            images=[tmp_path / "missing.png"],
        )

    client.chat.assert_not_called()


def test_generated_image_defaults_to_session_assets(tmp_path: Path) -> None:
    """Persist multiple generated images as ordered session artifacts."""
    image_bytes = PNG_BYTES
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    client = make_client()
    client.chat.return_value = make_response(f"Generated: {data_url} and {data_url}")
    store = SessionStore(tmp_path / "sessions")

    result = ChatService(client, store).chat(
        "Draw it.",
        model="image",
        include_raw_response=True,
    )

    outputs = [
        artifact for artifact in result.artifacts if artifact["direction"] == "output"
    ]
    assert [artifact["id"] for artifact in outputs] == [
        "t0001-output-001",
        "t0001-output-002",
    ]
    assert [item["type"] for item in result.content] == [
        "text",
        "image_ref",
        "text",
        "image_ref",
    ]
    assert result.response is not None
    assert data_url not in str(result.response)
    for artifact in outputs:
        image_path = Path(artifact["local_path"])
        assert image_path.parent.parent == store.assets_path(result.session_id)
        assert image_path.read_bytes() == image_bytes
    assert data_url not in store.session_path(result.session_id).read_text(
        encoding="utf-8"
    )


def test_generated_image_write_failure_does_not_create_session(
    tmp_path: Path,
) -> None:
    """Fail before persistence instead of storing an unsaved base64 image."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    client = make_client()
    client.chat.return_value = make_response(data_url)
    store = SessionStore(tmp_path / "sessions")

    with (
        patch(
            "conferllm.artifacts.os.write",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(ImageProcessingError, match="Failed to save generated image"),
    ):
        ChatService(client, store).chat("Draw it.", model="image")

    assert store.list_sessions(limit=0) == []
    assert not list((tmp_path / "sessions").rglob("*.assets"))


def test_generated_image_write_failure_does_not_append_turn(
    tmp_path: Path,
) -> None:
    """Keep an existing JSONL session unchanged when image persistence fails."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    client = make_client()
    client.chat.side_effect = [
        make_response("First answer"),
        make_response(data_url),
    ]
    store = SessionStore(tmp_path / "sessions")
    service = ChatService(client, store)
    first = service.chat("First question", model="image")
    original = store.session_path(first.session_id).read_bytes()

    with (
        patch(
            "conferllm.artifacts.os.write",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(ImageProcessingError, match="Failed to save generated image"),
    ):
        service.chat("Draw it.", session_id=first.session_id)

    assert store.session_path(first.session_id).read_bytes() == original
    assert data_url.encode() not in original


def test_second_output_write_failure_leaves_no_partial_turn(
    tmp_path: Path,
) -> None:
    """Roll back output one when output two cannot be staged."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    client = make_client()
    client.chat.return_value = make_response(data_url + data_url)
    store = SessionStore(tmp_path / "sessions")
    real_write = os.write
    calls = 0

    def fail_second_write(descriptor: int, data: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full on image two")
        return real_write(descriptor, data)

    with (
        patch("conferllm.artifacts.os.write", side_effect=fail_second_write),
        pytest.raises(ImageProcessingError, match="image 2"),
    ):
        ChatService(client, store).chat("Draw two.", model="image")

    assert store.list_sessions(limit=0) == []
    assert not list((tmp_path / "sessions").rglob("turn-0001"))


def test_text_only_capability_fails_before_reading_images_or_provider(
    tmp_path: Path,
) -> None:
    """Honor declared model capabilities before touching image sources."""
    client = LLMClient(
        ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="text-only",
                    litellm_params={"model": "openai/text"},
                    capabilities=ModelCapabilities(
                        input_modalities=["text"],
                        output_modalities=["text"],
                    ),
                )
            ]
        )
    )
    missing = tmp_path / "missing.png"

    with (
        patch.object(client, "chat") as provider,
        patch("conferllm.images.Path.read_bytes") as read_bytes,
        pytest.raises(ModelCapabilityError),
    ):
        ChatService(client, SessionStore(tmp_path / "sessions")).chat(
            "Describe it.",
            model="text-only",
            images=[missing],
        )

    read_bytes.assert_not_called()
    provider.assert_not_called()


def test_image_output_dir_exports_copies_without_replacing_canonical(
    tmp_path: Path,
) -> None:
    """Keep canonical session artifacts and return optional exported copies."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    client = make_client()
    client.chat.return_value = make_response(data_url)
    store = SessionStore(tmp_path / "sessions")

    result = ChatService(client, store).chat(
        "Draw it.",
        model="image",
        image_output_dir=tmp_path / "exports",
    )

    output = next(
        artifact for artifact in result.artifacts if artifact["direction"] == "output"
    )
    canonical = Path(output["local_path"])
    exported = Path(output["exported_path"])
    assert canonical.read_bytes() == PNG_BYTES
    assert exported.read_bytes() == PNG_BYTES
    assert canonical != exported


def test_malformed_provider_response_is_not_persisted(tmp_path: Path) -> None:
    """Require an assistant message before committing a successful turn."""
    client = make_client()
    response = MagicMock()
    response.model_dump.return_value = {"id": "empty", "choices": []}
    client.chat.return_value = response
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(RuntimeError, match="assistant message"):
        ChatService(client, store).chat("Hello", model="reasoning")

    assert store.list_sessions(limit=0) == []
