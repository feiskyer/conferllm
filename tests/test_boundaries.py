"""Persistence, concurrency, and delivery boundary checks using isolated data."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conferllm.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactTransaction,
    verify_artifact_file,
)
from conferllm.chat import ChatService
from conferllm.client import ImageLimitError, ModelCapabilityError
from conferllm.config import ConferLLMConfig, ModelCapabilities
from conferllm.errors import normalize_error
from conferllm.images import (
    ImageProcessingError,
    load_image_inputs,
    process_response_for_images,
)
from conferllm.session import SessionError, SessionStore
from conferllm.skill import install_skill
from tests.chat_fixtures import PNG, configured_client, response

DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


def test_same_session_concurrent_calls_replay_every_committed_turn(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("first", model="vision")

    entered = threading.Event()
    release = threading.Event()
    history_lengths: list[int] = []

    def provider(model: str, messages: list[dict]) -> MagicMock:
        history_lengths.append(len(messages))
        if len(history_lengths) == 1:
            entered.set()
            assert release.wait(timeout=3)
        return response()

    with (
        patch.object(client, "chat", side_effect=provider),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        try:
            second = executor.submit(
                service.chat, "second", session_id=first.session_id
            )
            assert entered.wait(timeout=3)
            third = executor.submit(service.chat, "third", session_id=first.session_id)
        finally:
            release.set()
        assert second.result(timeout=3).turn == 2
        assert third.result(timeout=3).turn == 3

    assert history_lengths == [3, 5]
    loaded = service.session_store.load_session(first.session_id)
    assert loaded.next_turn == 4


def test_model_image_limit_includes_replayed_history(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("first", model="vision", images=[DATA_URL] * 3)
    with patch.object(client, "chat") as provider, pytest.raises(ImageLimitError):
        service.chat("too many", session_id=first.session_id, images=[DATA_URL] * 2)
    provider.assert_not_called()


def test_text_prompt_requires_text_input_capability(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    client.config.model_list[0].capabilities = ModelCapabilities(
        input_modalities=["image"]
    )
    with (
        patch.object(client, "chat") as provider,
        pytest.raises(ModelCapabilityError),
    ):
        ChatService(client).chat("text", model="vision", images=[DATA_URL])
    provider.assert_not_called()
    # Model discovery must still work for an unusable-for-chat model.
    assert client.get_model_info("vision")["capabilities"]["input_modalities"] == [
        "image"
    ]


def test_tampered_history_fails_before_provider_and_preserves_jsonl(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    store = service.session_store
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("first", model="vision", images=[DATA_URL])
    original = store.session_path(first.session_id).read_bytes()
    path = store.resolve_artifact(first.session_id, "t0001-input-001")
    path.write_bytes(PNG[:-1] + b"x")
    with (
        patch.object(client, "chat") as provider,
        pytest.raises(SessionError) as failure,
    ):
        service.chat("continue", session_id=first.session_id)
    assert normalize_error(failure.value).code == "session_corrupt"
    provider.assert_not_called()
    assert store.session_path(first.session_id).read_bytes() == original


def test_export_directory_failure_is_a_warning_after_commit(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    destination = tmp_path / "not-a-directory"
    destination.write_text("user data", encoding="utf-8")
    with patch.object(client, "chat", return_value=response(DATA_URL)):
        result = service.chat("draw", model="vision", image_output_dir=destination)
    assert result.warnings
    assert service.session_store.load_session(result.session_id).next_turn == 2
    assert (
        service.session_store.read_artifact(result.session_id, "t0001-output-001")
        == PNG
    )
    assert destination.read_text(encoding="utf-8") == "user data"


def test_invalid_request_does_not_create_export_directory(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    destination = tmp_path / "exports"
    with pytest.raises(ValueError):
        ChatService(client).chat("hello", model="missing", image_output_dir=destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "content",
    [42, True, {"text": "not a content list"}, [{"type": "text", "text": 42}]],
)
def test_corrupt_history_content_is_rejected_before_provider(
    tmp_path: Path, content: object
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("first", model="vision")
    path = service.session_store.session_path(first.session_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["assistant"]["content"] = content
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with (
        patch.object(client, "chat", return_value=response()) as provider,
        pytest.raises(SessionError, match="corrupt"),
    ):
        service.chat("continue", session_id=first.session_id)
    provider.assert_not_called()


@pytest.mark.parametrize(
    "fields",
    [
        {"content": 42},
        {
            "content": "text",
            "images": [{"image_url": {"url": "https://example.invalid/x.png"}}],
        },
        {"content": None, "tool_calls": [{"id": "tool-1"}]},
    ],
)
def test_unsupported_provider_messages_do_not_become_successful_turns(
    tmp_path: Path, fields: dict
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with (
        patch.object(client, "chat", return_value=response(**fields)),
        pytest.raises(ImageProcessingError),
    ):
        service.chat("hello", model="vision")
    assert service.session_store.list_sessions() == []


def test_session_timestamp_must_match_date_in_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = datetime(2026, 9, 4, tzinfo=timezone.utc)
    session_id = store.generate_session_id(created)
    with pytest.raises(SessionError, match="date"):
        store.create_session(
            session_id,
            "name",
            "vision",
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            created_at=created + timedelta(days=1),
        )
    assert not store.session_path(session_id).exists()


def test_artifact_transaction_rejects_symlink_root_without_chmod(
    tmp_path: Path,
) -> None:
    target = tmp_path / "untouched"
    target.mkdir(mode=0o755)
    link = tmp_path / "linked-assets"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        ArtifactTransaction(
            session_id="20260904-0123456789abcdef0123456789abcdef",
            turn=1,
            asset_root=link,
            staging_parent=tmp_path,
        )
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_artifact_size_is_checked_before_reading_payload(tmp_path: Path) -> None:
    artifact = Artifact(
        id="t0001-output-001",
        direction="output",
        index=0,
        mime_type="image/png",
        size_bytes=len(PNG),
        sha256=hashlib.sha256(PNG).hexdigest(),
        relative_path="turn-0001/output-001.png",
    )
    path = tmp_path / "image.png"
    path.write_bytes(PNG * 100)
    with (
        patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")),
        pytest.raises(ArtifactError, match="size"),
    ):
        verify_artifact_file(path, artifact)


def test_aggregate_image_budget_is_checked_before_reading_next_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "next.png"
    path.write_bytes(PNG)
    with (
        patch.object(Path, "open", side_effect=AssertionError("over-budget read")),
        pytest.raises(ValueError) as failure,
    ):
        load_image_inputs(
            [DATA_URL, path],
            max_count=2,
            max_bytes_per_image=len(PNG),
            max_total_bytes=len(PNG),
        )
    assert normalize_error(failure.value).code == "image_limit_exceeded"


def test_default_app_parent_is_private_even_with_explicit_default_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = ConferLLMConfig().get_session_root()
    SessionStore(root)
    assert stat.S_IMODE(root.parent.stat().st_mode) == 0o700


def test_skill_force_refuses_home_or_workspace_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    original = tmp_path / "user-data"
    original.write_text("preserve", encoding="utf-8")
    with (
        patch(
            "conferllm.skill._write_staged_bundle",
            side_effect=AssertionError("must reject before staging or renaming"),
        ),
        pytest.raises(ValueError, match="dedicated"),
    ):
        install_skill(destination=tmp_path, force=True)
    assert original.read_text(encoding="utf-8") == "preserve"


def test_error_codes_do_not_depend_on_exception_prose() -> None:
    assert normalize_error(OSError("model not found")).code == "storage_error"
    assert normalize_error(ValueError("invalid session ID")).code == "invalid_request"


def test_base64_boundary_accepts_exact_limit() -> None:
    for data in [PNG, PNG + b"x", PNG + b"xy"]:
        encoded = "data:image/png;base64," + base64.b64encode(data).decode()
        assert (
            load_image_inputs(
                [encoded],
                max_count=1,
                max_bytes_per_image=len(data),
                max_total_bytes=len(data),
            )[0].data
            == data
        )


def test_create_permission_failure_closes_temporary_descriptor(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = datetime.now().astimezone()
    session_id = store.generate_session_id(created)
    descriptors: list[int] = []
    make_temp = tempfile.mkstemp

    def track_temp(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = make_temp(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    with (
        patch("conferllm.session.tempfile.mkstemp", side_effect=track_temp),
        patch("conferllm.session.os.fchmod", side_effect=OSError("permission failure")),
        pytest.raises(SessionError),
    ):
        store.create_session(
            session_id,
            "test",
            "vision",
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            created_at=created,
        )
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not list(store.root.rglob("*.tmp"))


def test_legacy_image_helper_cleans_up_after_chmod_failure(tmp_path: Path) -> None:
    output = tmp_path / "exports"
    original_chmod = Path.chmod

    def fail_final_chmod(path: Path, mode: int) -> None:
        if path.parent == output and path.suffix == ".png":
            raise OSError("chmod failed after rename")
        original_chmod(path, mode)

    with patch.object(Path, "chmod", fail_final_chmod), pytest.raises(OSError):
        process_response_for_images({"content": DATA_URL}, output)
    assert not list(output.iterdir())
