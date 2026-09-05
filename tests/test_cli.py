"""Tests for the ConferLLM command-line interface."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conferllm import __version__
from conferllm.chat import ChatResult
from conferllm.cli import run_cli
from conferllm.config import ConferLLMConfig
from conferllm.session import SessionListResult, SessionListWarning


def make_client() -> MagicMock:
    """Create a client mock with representative model data."""
    client = MagicMock()
    client.list_models.return_value = ["fast", "smart"]
    client.get_model_info.return_value = {
        "model_name": "smart",
        "provider_model": "openai/example",
        "configured_params": ["model", "api_key"],
    }
    return client


def make_chat_result() -> ChatResult:
    """Create a representative stateful chat result."""
    return ChatResult(
        session_id="20260904-0123456789abcdef0123456789abcdef",
        name="Explain this.",
        model="smart",
        answer="Model answer",
        response={
            "id": "response-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Model answer",
                    }
                }
            ],
        },
    )


def test_no_arguments_prints_help_without_starting_server() -> None:
    """Show command help instead of implicitly starting MCP."""
    output = StringIO()
    with patch("conferllm.cli.run_server") as mock_run_server:
        exit_code = run_cli([], output=output)

    assert exit_code == 0
    assert "usage: conferllm" in output.getvalue()
    assert "serve" in output.getvalue()
    mock_run_server.assert_not_called()


def test_serve_passes_options() -> None:
    """Pass explicit serve options to the server runner."""
    with patch("conferllm.cli.run_server") as mock_run_server:
        exit_code = run_cli(
            [
                "serve",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--config",
                "custom.yaml",
                "--log-level",
                "DEBUG",
            ]
        )

    assert exit_code == 0
    mock_run_server.assert_called_once_with(
        config_path=Path("custom.yaml"),
        transport="http",
        host="0.0.0.0",
        port=8080,
        log_level="DEBUG",
    )


def test_root_server_flags_are_rejected() -> None:
    """Require the explicit serve subcommand for server options."""
    with (
        patch("conferllm.cli.run_server") as mock_run_server,
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["--transport", "sse"])

    assert exc_info.value.code == 2
    mock_run_server.assert_not_called()


def test_models_text() -> None:
    """Print one configured model per line."""
    client = make_client()
    output = StringIO()
    with patch("conferllm.cli._load_client", return_value=client):
        exit_code = run_cli(["models"], output=output)

    assert exit_code == 0
    assert output.getvalue() == "fast\nsmart\n"


def test_models_json() -> None:
    """Print configured models as JSON when requested."""
    client = make_client()
    output = StringIO()
    with patch("conferllm.cli._load_client", return_value=client):
        exit_code = run_cli(["models", "--json"], output=output)

    assert exit_code == 0
    assert json.loads(output.getvalue()) == ["fast", "smart"]


def test_model_info() -> None:
    """Print non-secret model metadata as JSON."""
    client = make_client()
    output = StringIO()
    with patch("conferllm.cli._load_client", return_value=client):
        exit_code = run_cli(["model-info", "smart"], output=output)

    assert exit_code == 0
    assert json.loads(output.getvalue())["model_name"] == "smart"
    client.get_model_info.assert_called_once_with("smart")


def test_chat_creates_named_session_and_prints_identity() -> None:
    """Create a session and print its ID, name, and answer."""
    client = make_client()
    service = MagicMock()
    service.chat.return_value = make_chat_result()
    output = StringIO()

    with (
        patch("conferllm.cli._load_client", return_value=client),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            [
                "chat",
                "--model",
                "smart",
                "--name",
                "Design review",
                "--prompt",
                "Explain this.",
            ],
            output=output,
        )

    assert exit_code == 0
    assert output.getvalue() == (
        "Session: 20260904-0123456789abcdef0123456789abcdef\n"
        "Name: Explain this.\n"
        "\n"
        "Model answer\n"
    )
    service.chat.assert_called_once_with(
        "Explain this.",
        model="smart",
        session_id=None,
        name="Design review",
        images=[],
        image_output_dir=None,
        include_raw_response=False,
    )


def test_chat_continues_session_as_json() -> None:
    """Pass a session ID and emit the normalized JSON envelope."""
    client = make_client()
    service = MagicMock()
    service.chat.return_value = make_chat_result()
    output = StringIO()

    with (
        patch("conferllm.cli._load_client", return_value=client),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            [
                "chat",
                "--session",
                "20260904-0123456789abcdef0123456789abcdef",
                "--prompt",
                "Continue.",
                "--json",
            ],
            output=output,
        )

    assert exit_code == 0
    assert json.loads(output.getvalue()) == make_chat_result().to_dict()
    service.chat.assert_called_once_with(
        "Continue.",
        model=None,
        session_id="20260904-0123456789abcdef0123456789abcdef",
        name=None,
        images=[],
        image_output_dir=None,
        include_raw_response=False,
    )


def test_chat_reads_prompt_file_and_passes_images(tmp_path: Path) -> None:
    """Pass prompt-file and repeated image inputs to the shared service."""
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Compare these.", encoding="utf-8")
    image_one = tmp_path / "one.png"
    image_two = tmp_path / "two.jpg"
    image_one.write_bytes(b"one")
    image_two.write_bytes(b"two")
    output_dir = tmp_path / "generated"
    service = MagicMock()
    service.chat.return_value = make_chat_result()

    with (
        patch("conferllm.cli._load_client", return_value=make_client()),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            [
                "chat",
                "--model",
                "smart",
                "--prompt-file",
                str(prompt_path),
                "--image",
                str(image_one),
                "--image",
                str(image_two),
                "--image-output-dir",
                str(output_dir),
            ],
            output=StringIO(),
        )

    assert exit_code == 0
    service.chat.assert_called_once_with(
        "Compare these.",
        model="smart",
        session_id=None,
        name=None,
        images=[image_one, image_two],
        image_output_dir=output_dir,
        include_raw_response=False,
    )


def test_chat_requires_model_or_session() -> None:
    """Reject chat calls without a conversation target."""
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["chat", "--prompt", "Hello"])

    assert exc_info.value.code == 2


def test_chat_rejects_model_and_session_together() -> None:
    """Reject ambiguous chat calls at argument parsing time."""
    with pytest.raises(SystemExit) as exc_info:
        run_cli(
            [
                "chat",
                "--model",
                "smart",
                "--session",
                "20260904-0123456789abcdef0123456789abcdef",
                "--prompt",
                "Hello",
            ]
        )

    assert exc_info.value.code == 2


def test_sessions_list_text() -> None:
    """Print only session IDs and names in the default view."""
    first = MagicMock()
    first.session_id = "20260904-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    first.name = "Today"
    second = MagicMock()
    second.session_id = "20260903-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    second.name = "Yesterday"
    store = MagicMock()
    store.list_sessions_detailed.return_value = SessionListResult([first, second], [])
    output = StringIO()

    with patch("conferllm.cli._load_session_store", return_value=store):
        exit_code = run_cli(["sessions", "list"], output=output)

    assert exit_code == 0
    lines = output.getvalue().splitlines()
    assert lines[0].split() == ["SESSION_ID", "NAME"]
    assert lines[1].endswith("  Today")
    assert lines[2].endswith("  Yesterday")
    store.list_sessions_detailed.assert_called_once_with(
        query=None,
        model=None,
        since=None,
        until=None,
        limit=50,
    )


def test_sessions_list_filters_and_json() -> None:
    """Pass all list filters and serialize complete metadata."""
    metadata = MagicMock()
    metadata.to_dict.return_value = {
        "session_id": "20260904-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "name": "Raft",
        "model": "smart",
        "created_at": "2026-09-04T10:00:00+08:00",
    }
    store = MagicMock()
    store.list_sessions_detailed.return_value = SessionListResult([metadata], [])
    output = StringIO()

    with patch("conferllm.cli._load_session_store", return_value=store):
        exit_code = run_cli(
            [
                "sessions",
                "list",
                "--query",
                "raft",
                "--model",
                "smart",
                "--since",
                "2026-09-01",
                "--until",
                "2026-09-04",
                "--limit",
                "20",
                "--json",
            ],
            output=output,
        )

    assert exit_code == 0
    payload = json.loads(output.getvalue())
    assert payload["schema_version"] == "conferllm.sessions.response.v1"
    assert payload["sessions"][0]["name"] == "Raft"
    assert payload["warnings"] == []
    called = store.list_sessions_detailed.call_args.kwargs
    assert called["query"] == "raft"
    assert called["model"] == "smart"
    assert called["since"].isoformat() == "2026-09-01"
    assert called["until"].isoformat() == "2026-09-04"
    assert called["limit"] == 20


def test_sessions_list_uses_configured_session_root(tmp_path: Path) -> None:
    """Resolve session storage from the explicitly selected config."""
    config_path = tmp_path / "config.yaml"
    configured_root = tmp_path / "custom-sessions"
    config = ConferLLMConfig(sessions_dir=configured_root)
    store = MagicMock()
    store.list_sessions_detailed.return_value = SessionListResult([], [])

    with (
        patch(
            "conferllm.cli.ConferLLMConfig.load_config", return_value=config
        ) as load_config,
        patch("conferllm.cli.SessionStore", return_value=store) as session_store,
    ):
        exit_code = run_cli(
            ["sessions", "list", "--config", str(config_path)],
            output=StringIO(),
        )

    assert exit_code == 0
    load_config.assert_called_once_with(config_path)
    session_store.assert_called_once_with(configured_root)


def test_provider_error_returns_nonzero() -> None:
    """Report provider failures without a traceback."""
    service = MagicMock()
    service.chat.side_effect = RuntimeError("provider unavailable")
    errors = StringIO()
    with (
        patch("conferllm.cli._load_client", return_value=make_client()),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            ["chat", "--model", "smart", "--prompt", "Hello"],
            error_output=errors,
        )

    assert exit_code == 1
    assert errors.getvalue() == "conferllm: error: provider unavailable\n"


def test_chat_json_error_uses_versioned_envelope() -> None:
    """Emit a stable error object when a JSON chat call fails at runtime."""
    service = MagicMock()
    service.chat.side_effect = RuntimeError("provider unavailable")
    errors = StringIO()
    with (
        patch("conferllm.cli._load_client", return_value=make_client()),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            ["chat", "--model", "smart", "--prompt", "Hello", "--json"],
            error_output=errors,
        )

    assert exit_code == 1
    assert json.loads(errors.getvalue()) == {
        "schema_version": "conferllm.error.v1",
        "ok": False,
        "error": {
            "code": "provider_error",
            "message": "provider unavailable",
        },
    }


def test_chat_passes_include_raw_response() -> None:
    """Only include raw provider data when explicitly requested."""
    service = MagicMock()
    service.chat.return_value = make_chat_result()

    with (
        patch("conferllm.cli._load_client", return_value=make_client()),
        patch("conferllm.cli.ChatService", return_value=service),
    ):
        exit_code = run_cli(
            [
                "chat",
                "--model",
                "smart",
                "--prompt",
                "Hello",
                "--include-raw-response",
                "--json",
            ],
            output=StringIO(),
        )

    assert exit_code == 0
    assert service.chat.call_args.kwargs["include_raw_response"] is True


def test_sessions_list_reports_warnings() -> None:
    """Keep valid sessions visible while reporting corrupt headers."""
    metadata = MagicMock()
    metadata.session_id = "20260904-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    metadata.name = "Valid"
    warning = SessionListWarning(
        session_id="20260903-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        message="corrupt header",
    )
    detailed = MagicMock()
    detailed.sessions = [metadata]
    detailed.warnings = [warning]
    store = MagicMock()
    store.list_sessions_detailed.return_value = detailed
    output = StringIO()
    errors = StringIO()

    with patch("conferllm.cli._load_session_store", return_value=store):
        exit_code = run_cli(
            ["sessions", "list"],
            output=output,
            error_output=errors,
        )

    assert exit_code == 0
    assert "Valid" in output.getvalue()
    assert errors.getvalue() == "conferllm: warning: corrupt header\n"


def test_skill_install_passes_cli_options() -> None:
    """Delegate Skill installation with the selected target and destination."""
    output = StringIO()
    result = {
        "schema_version": "conferllm.skill.install.v1",
        "status": "installed",
        "message": "Installed.",
    }
    with patch("conferllm.cli._install_skill", return_value=result) as install:
        exit_code = run_cli(
            [
                "skill",
                "install",
                "--target",
                "codex",
                "--destination",
                "/tmp/conferllm-skill",
                "--force",
            ],
            output=output,
        )

    assert exit_code == 0
    install.assert_called_once_with(
        target="codex",
        destination=Path("/tmp/conferllm-skill"),
        force=True,
    )
    assert output.getvalue() == "Installed.\n"


def test_doctor_json_passes_config() -> None:
    """Delegate diagnostics and preserve its structured report."""
    output = StringIO()
    report = {"schema_version": "conferllm.doctor.v1", "ok": True}
    with patch("conferllm.cli._run_doctor", return_value=report) as doctor:
        exit_code = run_cli(
            ["doctor", "--config", "custom.yaml", "--json"],
            output=output,
        )

    assert exit_code == 0
    doctor.assert_called_once_with(Path("custom.yaml"))
    assert json.loads(output.getvalue()) == report


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Expose the package version through the executable."""
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"conferllm {__version__}\n"
