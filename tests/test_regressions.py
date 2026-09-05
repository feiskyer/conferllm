"""Cross-layer regressions from the architecture and reliability review."""

from __future__ import annotations

import base64
import json
import logging
import threading
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ResourceLink

from conferllm.chat import ChatResult, ChatService
from conferllm.cli import run_cli
from conferllm.client import ModelCapabilityError
from conferllm.config import ConferLLMConfig, ModelCapabilities, ModelConfig
from conferllm.errors import normalize_error
from conferllm.images import ImageProcessingError, load_image_inputs
from conferllm.server import create_mcp_server
from conferllm.session import SessionStore
from tests.chat_fixtures import PNG, configured_client, response

DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


@pytest.mark.parametrize(
    "document",
    [
        "model_list: [{litellm_params: {api_key: synthetic-secret}}]",
        "model_list: [synthetic-secret]",
        "model_list: [\napi_key: synthetic-secret: invalid",
    ],
)
def test_invalid_config_never_echoes_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    document: str,
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(document, encoding="utf-8")
    with pytest.raises(Exception) as failure:
        ConferLLMConfig.load_config(config_path)

    public = normalize_error(failure.value)
    assert public.code == "configuration_error"
    assert "synthetic-secret" not in str(public)
    assert "synthetic-secret" not in caplog.text


@pytest.mark.parametrize("document", ["false", "[]", "42", '"text"'])
def test_scalar_config_is_not_a_valid_empty_config(
    tmp_path: Path, document: str
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(document, encoding="utf-8")
    with pytest.raises(Exception) as failure:
        ConferLLMConfig.load_config(config_path)
    assert normalize_error(failure.value).code == "configuration_error"


def test_explicit_missing_config_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(Exception) as failure:
        ConferLLMConfig.load_config(tmp_path / "typo.yaml")
    assert normalize_error(failure.value).code == "configuration_error"


@pytest.mark.parametrize(
    "model_data",
    [
        {"model_name": " ", "litellm_params": {"model": "openai/example"}},
        {"model_name": "bad", "litellm_params": {}},
        {"model_name": "bad", "litellm_params": {"model": 42}},
        {"model_name": "bad", "litellm_params": {"model": " "}},
        {
            "model_name": "bad",
            "litellm_params": {"model": "openai/example", "messages": []},
        },
    ],
)
def test_invalid_model_definitions_are_rejected(model_data: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig.model_validate(model_data)


def test_duplicate_model_aliases_are_rejected() -> None:
    model = ModelConfig(model_name="same", litellm_params={"model": "openai/example"})
    with pytest.raises(ValueError, match="unique"):
        ConferLLMConfig(model_list=[model, model])


def test_provider_error_is_redacted_and_keeps_provider_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client = configured_client(tmp_path / "sessions")
    with (
        patch(
            "conferllm.client.litellm.completion",
            side_effect=RuntimeError(
                "Model not found; request api_key=synthetic-secret"
            ),
        ),
        pytest.raises(RuntimeError) as failure,
    ):
        client.chat("vision", [{"role": "user", "content": "hello"}])
    public = normalize_error(failure.value)
    assert public.code == "provider_error"
    assert "synthetic-secret" not in str(public)
    assert "synthetic-secret" not in caplog.text


def test_append_accepts_complete_json_without_final_newline(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("hello", model="vision")
        path = service.session_store.session_path(first.session_id)
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        second = service.chat("continue", session_id=first.session_id)
    loaded = service.session_store.load_session(first.session_id)
    assert second.turn == 2
    assert loaded.next_turn == 3


def test_uncommitted_artifact_directory_does_not_poison_continuation(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    store = service.session_store
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("hello", model="vision")
    original = store.session_path(first.session_id).read_bytes()
    abandoned = store.artifact_transaction(first.session_id, 2)
    abandoned.stage_bytes(PNG, direction="output", mime_type="image/png")
    abandoned.install()  # Simulate process death before JSONL commit.

    with patch.object(client, "chat", return_value=response(DATA_URL)):
        continued = service.chat("draw", session_id=first.session_id)
    assert continued.turn == 2
    assert store.session_path(first.session_id).read_bytes().startswith(original)
    assert store.read_artifact(first.session_id, "t0002-output-001") == PNG


def test_history_images_are_checked_against_current_capabilities(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("look", model="vision", images=[DATA_URL])
    client.config.model_list[0].capabilities = ModelCapabilities(
        input_modalities=["text"], output_modalities=["text"]
    )
    with (
        patch.object(client, "chat") as provider,
        pytest.raises(ModelCapabilityError),
    ):
        service.chat("what was shown?", session_id=first.session_id)
    provider.assert_not_called()


def test_history_does_not_reparse_jsonl_for_every_image(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    store = service.session_store
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("look", model="vision", images=[DATA_URL] * 3)
        with patch.object(store, "load_session", wraps=store.load_session) as load:
            service.chat("continue", session_id=first.session_id)
        assert load.call_count <= 2


def test_provider_images_field_is_normalized_and_replayed(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    generated = response(
        None,
        images=[{"type": "image_url", "image_url": {"url": DATA_URL}}],
    )
    with patch.object(client, "chat", side_effect=[generated, response()]) as provider:
        first = service.chat("draw", model="vision")
        assert first.content == [
            {"type": "image_ref", "artifact_id": "t0001-output-001"}
        ]
        service.chat("describe that image", session_id=first.session_id)
    replay = provider.call_args_list[1].args[1][1]
    assert replay["content"][0]["image_url"]["url"] == DATA_URL
    assert "images" not in replay


def test_literal_artifact_uri_is_not_interpreted_as_generated_image(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    quoted = (
        "Example URI: conferllm://sessions/"
        "20260904-0123456789abcdef0123456789abcdef/artifacts/t0001-output-001"
    )
    with patch.object(client, "chat", return_value=response(quoted)):
        result = service.chat("explain artifact URIs", model="vision")
    assert result.text == quoted
    assert result.artifacts == []


def test_invalid_generated_base64_is_not_stored(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    with (
        patch.object(
            client, "chat", return_value=response("data:image/png;base64,@@@@@")
        ),
        pytest.raises(ImageProcessingError),
    ):
        service.chat("draw", model="vision")
    assert service.session_store.list_sessions() == []


def test_data_url_limit_is_checked_before_decoding() -> None:
    with (
        patch(
            "conferllm.images.base64.b64decode",
            side_effect=AssertionError("must not decode oversized input"),
        ),
        pytest.raises(ValueError) as failure,
    ):
        load_image_inputs(
            [DATA_URL],
            max_count=1,
            max_bytes_per_image=4,
            max_total_bytes=4,
        )
    assert normalize_error(failure.value).code == "image_too_large"


async def test_mcp_factory_honors_configured_session_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "isolated-home")
    client = configured_client(tmp_path / "configured-sessions")
    service = ChatService(client)
    with patch.object(client, "chat", return_value=response()):
        first = service.chat("hello", model="vision")
    server = create_mcp_server(client)
    result = await server.call_tool("list_sessions", {})
    assert result.structured_content["sessions"][0]["session_id"] == first.session_id
    assert not (tmp_path / "isolated-home" / ".conferllm" / "sessions").exists()


async def test_mcp_text_fallback_preserves_session_contract(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    server = create_mcp_server(
        client, session_store=SessionStore(tmp_path / "sessions")
    )
    with patch.object(client, "chat", return_value=response()):
        result = await server.call_tool(
            "create_chat", {"model": "vision", "message": "hello"}
        )
    fallback = json.loads(result.content[0].text)
    assert fallback == result.structured_content
    assert fallback["session"]["id"]


async def test_mcp_compatibility_validation_keeps_stable_error_code() -> None:
    server = create_mcp_server()
    with pytest.raises(ToolError, match=r"\[invalid_request\]"):
        await server.call_tool("chat", {"message": "hello"})


async def test_mcp_inline_failure_does_not_disguise_a_committed_turn() -> None:
    result = ChatResult(
        session_id="20260904-0123456789abcdef0123456789abcdef",
        model="vision",
        name="draw",
        answer="",
        artifacts=[
            {
                "id": "t0001-output-001",
                "direction": "output",
                "mime_type": "image/png",
                "size_bytes": len(PNG),
                "uri": "conferllm://sessions/test/artifacts/t0001-output-001",
            }
        ],
    )
    service = MagicMock()
    service.chat.return_value = result
    store = MagicMock(spec=SessionStore)
    store.read_artifact.side_effect = OSError("temporary read failure")
    with patch("conferllm.server.ChatService", return_value=service):
        server = create_mcp_server(MagicMock(), session_store=store)
        returned = await server.call_tool(
            "create_chat", {"model": "vision", "message": "draw"}
        )
    assert returned.structured_content["ok"]
    assert returned.structured_content["warnings"]
    assert isinstance(returned.content[-1], ResourceLink)


async def test_mcp_resource_io_does_not_run_on_event_loop() -> None:
    event_thread = threading.get_ident()
    store = MagicMock(spec=SessionStore)
    threads: list[int] = []

    def read(session_id: str, artifact_id: str) -> bytes:
        threads.append(threading.get_ident())
        return PNG

    store.read_artifact.side_effect = read
    server = create_mcp_server(session_store=store)
    await server.read_resource(
        "conferllm://sessions/20260904-0123456789abcdef0123456789abcdef/"
        "artifacts/t0001-output-001"
    )
    assert threads and event_thread not in threads


def test_doctor_failure_returns_nonzero_exit() -> None:
    with patch("conferllm.cli._run_doctor", return_value={"ok": False}):
        code = run_cli(["doctor", "--json"], output=StringIO())
    assert code == 1


def test_human_chat_surfaces_artifacts_and_warnings(tmp_path: Path) -> None:
    result = ChatResult(
        session_id="20260904-0123456789abcdef0123456789abcdef",
        name="draw",
        model="vision",
        answer="",
        warnings=["Export failed; the canonical artifact is available."],
        artifacts=[
            {
                "id": "t0001-output-001",
                "direction": "output",
                "local_path": str(tmp_path / "canonical.png"),
            }
        ],
    )
    service = MagicMock()
    service.chat.return_value = result
    stdout, stderr = StringIO(), StringIO()
    with (
        patch("conferllm.cli._load_client"),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        code = run_cli(
            ["chat", "--model", "vision", "--prompt", "draw"],
            output=stdout,
            error_output=stderr,
        )
    assert code == 0
    assert "canonical.png" in stdout.getvalue()
    assert "Export failed" in stderr.getvalue()


def test_json_runtime_error_is_not_mixed_with_library_logs(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    output, errors = StringIO(), StringIO()
    original_disable = logging.root.manager.disable
    handler = logging.StreamHandler(errors)
    with (
        patch.object(logging.getLogger(), "handlers", [handler]),
        patch("conferllm.cli._load_client", return_value=client),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=RuntimeError("synthetic-secret"),
        ),
    ):
        code = run_cli(
            ["chat", "--model", "vision", "--prompt", "hello", "--json"],
            output=output,
            error_output=errors,
        )
    assert code == 1
    assert output.getvalue() == ""
    assert json.loads(errors.getvalue())["error"]["code"] == "provider_error"
    assert "synthetic-secret" not in errors.getvalue()
    assert logging.root.manager.disable == original_disable
