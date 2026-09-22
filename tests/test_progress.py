"""Progress logging must describe actual work without SDK noise."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

from conferllm.chat import ChatService
from conferllm.cli import run_cli
from conferllm.errors import ConferLLMError
from conferllm.progress import event, model_request
from conferllm.tools import ToolRuntime
from tests.chat_fixtures import configured_client, response
from tests.test_tool_chat import call, tool_response


def events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        record.progress
        for record in caplog.records
        if record.name == "conferllm.progress"
    ]


def test_prompt_and_tool_progress_are_logged_before_work_finishes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client = configured_client(tmp_path / "sessions")
    client.config.global_system_prompt = "Use the native tools."
    path = tmp_path / "output.txt"
    arguments = {"path": str(path), "content": "useful output"}
    replies = [
        tool_response(call("write_file", arguments), content="I will write the file."),
        response("Done."),
    ]

    def complete(**_kwargs: object) -> object:
        observed = events(caplog)
        assert observed
        assert observed[-1]["event"] == "model.request"
        assert observed[-1]["round"] == 3 - len(replies)
        return replies.pop(0)

    execute = ToolRuntime.execute

    def execute_tool(runtime: ToolRuntime, name: str, arguments: str) -> str:
        assert events(caplog)[-1]["event"] == "tool.start"
        assert not path.exists()
        return execute(runtime, name, arguments)

    with (
        patch("conferllm.client.litellm.completion", side_effect=complete),
        patch.object(ToolRuntime, "execute", execute_tool),
    ):
        result = ChatService(client).chat("Create a file.", model="vision")

    logged = events(caplog)
    assert [entry["event"] for entry in logged] == [
        "model.request",
        "model.response",
        "tool.start",
        "tool.result",
        "model.request",
        "model.response",
    ]
    assert all(entry["session"] == result.session_id for entry in logged)
    assert all(entry["turn"] == 1 and entry["model"] == "vision" for entry in logged)
    assert logged[0]["prompt"] == [
        {"role": "system", "content": "Use the native tools."},
        {"role": "user", "content": "Create a file."},
    ]
    assert logged[1]["text"] == "I will write the file."
    assert logged[1]["tool_calls"] == 1
    assert logged[2]["name"] == logged[3]["name"] == "write_file"
    assert logged[2]["call_id"] == logged[3]["call_id"] == "call-1"
    assert logged[2]["input"] == arguments
    assert logged[3]["output"]["ok"] is True
    assert logged[3]["elapsed_seconds"] >= 0
    assert logged[4]["round"] == 2
    assert "prompt" not in logged[4]
    assert logged[5]["text"] == "Done."
    assert logged[5]["tool_calls"] == 0
    assert logged[5]["elapsed_seconds"] >= 0
    assert path.read_text() == "useful output"


def test_debug_logs_each_effective_prompt_and_continuation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client = configured_client(tmp_path / "sessions")
    client.config.global_system_prompt = "Global prompt."
    client.config.model_list[0].system_prompt = "Model-specific prompt."
    service = ChatService(client)
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(call("list_directory", {"path": str(tmp_path)})),
            response("Listed."),
            response("Remembered."),
        ],
    ) as provider:
        first = service.chat("Inspect the directory.", model="vision")
        second = service.chat("Remember it.", session_id=first.session_id)
    requests = [entry for entry in events(caplog) if entry["event"] == "model.request"]
    assert [entry["round"] for entry in requests] == [1, 2, 1]
    assert [entry["turn"] for entry in requests] == [1, 1, 2]
    assert all(entry["session"] == second.session_id for entry in requests)
    for logged, actual in zip(requests, provider.call_args_list, strict=True):
        assert logged["prompt"] == actual.kwargs["messages"]
    assert "Global prompt." not in caplog.text


def test_tool_errors_are_visible_and_model_can_recover(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(
                call("read_file", "{invalid", "invalid"),
                call("missing_tool", {}, "missing"),
            ),
            response("Recovered."),
        ],
    ):
        result = ChatService(configured_client(tmp_path / "sessions")).chat(
            "Try the tools.", model="vision"
        )
    logged = events(caplog)
    assert logged[2]["input"] == "{invalid"
    errors = [
        record.progress
        for record in caplog.records
        if record.name == "conferllm.progress" and record.levelno == logging.WARNING
    ]
    assert [entry["call_id"] for entry in errors] == ["invalid", "missing"]
    assert all(entry["output"]["ok"] is False for entry in errors)
    assert errors[0]["output"]["error"] == "Invalid tool arguments."
    assert errors[1]["output"]["error"] == "Unknown tool: missing_tool"
    assert result.text == "Recovered."


@pytest.mark.parametrize(
    "arguments",
    ['{"path":' + "1" * 5000 + "}", "[" * 1200 + "]" * 1200],
    ids=["large-integer", "deeply-nested"],
)
def test_logging_cannot_preempt_tool_argument_validation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, arguments: str
) -> None:
    caplog.set_level(logging.INFO)
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            tool_response(call("read_file", arguments)),
            response("Recovered."),
        ],
    ) as provider:
        result = ChatService(configured_client(tmp_path / "sessions")).chat(
            "Try invalid arguments.", model="vision"
        )
    logged = events(caplog)
    sent = provider.call_args_list[1].kwargs["messages"]
    assert sent[-2]["tool_calls"][0]["function"]["arguments"] == arguments
    assert logged[3]["output"]["ok"] is False
    assert result.text == "Recovered."


def test_tool_logs_decode_arguments_and_display_only_useful_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    event(
        "tool.start",
        name="run_shell",
        call_id="irrelevant-provider-call-id",
        input=json.dumps({"command": "printf first\nprintf second", "cwd": "/tmp"}),
    )
    event(
        "tool.result",
        name="run_shell",
        call_id="irrelevant-provider-call-id",
        elapsed_seconds=0.125,
        output={
            "ok": True,
            "output": "first\nsecond",
            "exit_code": 0,
            "timed_out": False,
            "cwd": "/tmp",
        },
    )
    rendered = [
        record.getMessage()
        for record in caplog.records
        if record.name == "conferllm.progress"
    ]
    assert rendered == [
        "Tool run_shell\n"
        "  Input:\n"
        "    command:\n"
        "      printf first\n"
        "      printf second\n"
        "    cwd: /tmp",
        "Tool run_shell: completed (0.12s)\n  first\n  second",
    ]


def test_failed_background_and_truncated_tool_results_keep_actionable_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    event(
        "tool.result",
        name="shell_output",
        elapsed_seconds=1.25,
        output={
            "ok": False,
            "status": "failed",
            "output": "partial output",
            "error": "Command failed",
            "exit_code": -9,
            "shell_id": "background-1",
            "timed_out": True,
            "truncated": True,
            "details": [{"loc": ["command"], "msg": "Expected text", "type": "str"}],
        },
    )
    assert caplog.records[-1].getMessage() == (
        "Tool shell_output: failed (1.25s)\n"
        "  Error: Command failed\n"
        "  command: Expected text\n"
        "  partial output\n"
        "  Exit code: -9\n"
        "  Shell: background-1\n"
        "  Command timed out.\n"
        "  [output truncated by tool]"
    )


def test_argument_values_and_empty_tool_outputs_remain_readable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    event(
        "tool.start",
        name="run_shell",
        input='{"run_in_background": true, "cwd": null, "items": ["one", "two"]}',
    )
    assert caplog.records[-1].getMessage() == (
        "Tool run_shell\n"
        "  Input:\n"
        "    run in background: true\n"
        "    cwd: (none)\n"
        "    items:\n"
        "      - one\n"
        "      - two"
    )
    event(
        "tool.result",
        name="write_file",
        output={"ok": True, "path": "/tmp/empty", "output": "", "bytes_written": 0},
    )
    assert caplog.records[-1].getMessage() == (
        "Tool write_file: completed\n  Path: /tmp/empty"
    )


def test_info_only_renders_current_prompt_and_intermediate_reply(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client = configured_client(tmp_path / "sessions")
    client.config.global_system_prompt = "Use tools."
    service = ChatService(client)
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            response("Old answer."),
            tool_response(
                call("list_directory", {"path": str(tmp_path)}),
                content="I will inspect the directory.",
            ),
            response("Final answer."),
        ],
    ):
        first = service.chat("Old prompt.", model="vision")
        caplog.clear()
        service.chat("New prompt.", session_id=first.session_id)
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "System: Use tools.\n  User: New prompt." in rendered
    assert "I will inspect the directory." in rendered
    assert "Old prompt." not in rendered
    assert "Old answer." not in rendered
    assert "Final answer." not in rendered
    assert '"role"' not in rendered and '"content"' not in rendered
    assert '"ok"' not in rendered and '"event"' not in rendered
    assert "turn 2/round 2" in rendered


@pytest.mark.parametrize(
    ("control", "escaped"),
    [("\x1b", "\\x1b"), ("\r", "\\x0d")],
)
def test_terminal_control_characters_are_escaped(
    caplog: pytest.LogCaptureFixture,
    control: str,
    escaped: str,
) -> None:
    caplog.set_level(logging.INFO)
    event(
        "tool.result",
        name="read_file",
        output={"ok": True, "output": f"{control}[2Jhello\nnext"},
    )
    rendered = caplog.records[-1].getMessage()
    assert control not in rendered
    assert f"{escaped}[2Jhello\n  next" in rendered


@pytest.mark.parametrize("interrupted", [False, True])
def test_failure_logs_category_not_exception_body_and_resets_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, interrupted: bool
) -> None:
    caplog.set_level(logging.DEBUG)
    error = (
        KeyboardInterrupt("private failure body")
        if interrupted
        else RuntimeError("private failure body")
    )
    with (
        patch("conferllm.client.litellm.completion", side_effect=error),
        pytest.raises(KeyboardInterrupt if interrupted else ConferLLMError),
    ):
        ChatService(configured_client(tmp_path / "sessions")).chat(
            "Try a request.", model="vision"
        )
    failure = events(caplog)[-1]
    assert failure["event"] == "turn.error"
    assert failure["code"] == ("chat_cancelled" if interrupted else "provider_error")
    assert failure["round"] == 1
    assert "private failure body" not in caplog.text
    event("after_failure")
    assert events(caplog)[-1] == {"event": "after_failure"}


def test_progress_redacts_credentials_and_images_without_mutating_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": 'api_key="fake secret"'},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,c2VjcmV0"},
                },
            ],
            "provider_specific_fields": {"encrypted_content": "opaque state"},
        }
    ]
    original = json.dumps(messages)
    model_request("vision", messages)
    event(
        "tool.result",
        output={
            "ok": True,
            "api_key": "fake key",
            "output": '{"password": "fake password", "useful": "kept"}',
            "image": "data:image/png;base64,c2VjcmV0",
            "headers": {"Authorization": "Bearer fake-token"},
        },
    )
    for private in (
        "fake secret",
        "fake key",
        "fake password",
        "fake-token",
        "c2VjcmV0",
        "opaque state",
    ):
        assert private not in caplog.text
    assert "[image omitted]" in caplog.text
    assert "[redacted]" in caplog.text
    assert "kept" in caplog.text
    assert json.dumps(messages) == original


def test_long_text_is_not_silently_truncated(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    text = "long output\n" * 3000
    event("tool.result", output={"ok": True, "output": text})
    assert events(caplog)[0]["output"]["output"] == text


def test_json_text_keeps_its_original_formatting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    text = '{\n  "items": [1, 2]\n}'
    event("model.response", text=text)
    assert events(caplog)[0]["text"] == text


def test_concurrent_chats_keep_progress_context_separate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    service = ChatService(configured_client(tmp_path / "sessions"))
    barrier = Barrier(2)

    def complete(**_kwargs: object) -> object:
        barrier.wait(timeout=5)
        return response("Finished.")

    def chat(prompt: str) -> str:
        return service.chat(prompt, model="vision").session_id

    with (
        patch("conferllm.client.litellm.completion", side_effect=complete),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        sessions = list(executor.map(chat, ["First prompt.", "Second prompt."]))
    for prompt, session in zip(
        ["First prompt.", "Second prompt."], sessions, strict=True
    ):
        logged = [entry for entry in events(caplog) if entry["session"] == session]
        assert [entry["event"] for entry in logged] == [
            "model.request",
            "model.response",
        ]
        assert logged[0]["prompt"][-1]["content"] == prompt
        assert all(entry["round"] == 1 and entry["turn"] == 1 for entry in logged)
    event("after_threads")
    assert events(caplog)[-1] == {"event": "after_threads"}


@pytest.mark.parametrize("log_level", ["INFO", "DEBUG"])
@pytest.mark.parametrize("namespace", ["LiteLLM", "LiteLLM Router", "LiteLLM Proxy"])
def test_cli_suppresses_sdk_noise_and_emits_warnings_only_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    log_level: str,
    namespace: str,
) -> None:
    caplog.set_level(logging.INFO)
    client = configured_client(tmp_path / "sessions")
    sdk = logging.getLogger(namespace)
    caplog.set_level(logging.DEBUG, logger=sdk.name)
    stdout, stderr, sdk_stderr = StringIO(), StringIO(), StringIO()
    sdk_handler = logging.StreamHandler(sdk_stderr)

    def complete(**_kwargs: object) -> object:
        sdk.info("Wrapper: Completed Call, calling success_handler")
        sdk.debug("private SDK request dump")
        sdk.warning("useful SDK warning")
        for name in ("httpx", "httpcore", "openai"):
            logging.getLogger(name).info("unhelpful HTTP trace")
        return response("Final answer.")

    with (
        patch.object(logging.getLogger(), "handlers", []),
        patch.object(sdk, "handlers", [sdk_handler]),
        patch("conferllm.cli._load_client", return_value=client),
        patch("conferllm.client.litellm.completion", side_effect=complete),
    ):
        status = run_cli(
            [
                "chat",
                "--model",
                "vision",
                "--prompt",
                "Hello",
                "--log-level",
                log_level,
            ],
            output=stdout,
            error_output=stderr,
        )
        restored_sdk_handlers = list(sdk.handlers)
    assert status == 0
    assert "Calling model..." in stderr.getvalue()
    assert "User: Hello" in stderr.getvalue()
    assert "Model finished" in stderr.getvalue()
    assert '"event"' not in stderr.getvalue()
    if log_level == "INFO":
        assert "Final answer." not in stderr.getvalue()
    assert "Wrapper:" not in stderr.getvalue()
    assert "private SDK" not in stderr.getvalue()
    assert "unhelpful HTTP trace" not in stderr.getvalue()
    assert stderr.getvalue().count("useful SDK warning") == 1
    assert sdk_stderr.getvalue() == ""
    assert restored_sdk_handlers == [sdk_handler]
    assert '"event"' not in stdout.getvalue()
    assert "Final answer." in stdout.getvalue()


def test_client_construction_preserves_dependency_logger_state(tmp_path: Path) -> None:
    sdk = logging.getLogger("LiteLLM")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    original_handlers = list(sdk.handlers)
    original_level = sdk.level
    original_propagate = sdk.propagate
    sdk.handlers = [handler]
    sdk.setLevel(logging.DEBUG)
    sdk.propagate = False
    try:
        configured_client(tmp_path / "sessions")
        assert sdk.handlers == [handler]
        assert sdk.level == logging.DEBUG
        assert sdk.propagate is False
    finally:
        sdk.handlers = original_handlers
        sdk.setLevel(original_level)
        sdk.propagate = original_propagate


def test_repeated_cli_calls_use_their_own_stderr_and_restore_host_logging(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    first_stderr, second_stderr = StringIO(), StringIO()
    root = logging.getLogger()
    host_stream = StringIO()
    host_handler = logging.StreamHandler(host_stream)
    original_handlers = list(root.handlers)
    original_level = root.level
    root.handlers = [host_handler]
    root.setLevel(logging.ERROR)
    try:
        with (
            patch("conferllm.cli._load_client", return_value=client),
            patch(
                "conferllm.client.litellm.completion",
                side_effect=[response("First answer."), response("Second answer.")],
            ),
        ):
            first_status = run_cli(
                ["chat", "--model", "vision", "--prompt", "First prompt."],
                output=StringIO(),
                error_output=first_stderr,
            )
            second_status = run_cli(
                ["chat", "--model", "vision", "--prompt", "Second prompt."],
                output=StringIO(),
                error_output=second_stderr,
            )
        assert first_status == second_status == 0
        assert "User: First prompt." in first_stderr.getvalue()
        assert "Second prompt." not in first_stderr.getvalue()
        assert "User: Second prompt." in second_stderr.getvalue()
        assert "First prompt." not in second_stderr.getvalue()
        assert root.handlers == [host_handler]
        assert root.level == logging.ERROR
        assert host_stream.getvalue() == ""
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_default_cli_failure_emits_one_public_error(
    tmp_path: Path,
) -> None:
    client = configured_client(tmp_path / "sessions")
    stdout, stderr = StringIO(), StringIO()
    with (
        patch("conferllm.cli._load_client", return_value=client),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=RuntimeError("private body"),
        ),
    ):
        status = run_cli(
            ["chat", "--model", "vision", "--prompt", "Hello"],
            output=stdout,
            error_output=stderr,
        )
    assert status == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue().count("conferllm: error:") == 1
    assert "Error calling model" not in stderr.getvalue()
    assert "Stopped:" not in stderr.getvalue()
    assert "private body" not in stderr.getvalue()


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_quiet_modes_preserve_machine_output(
    tmp_path: Path, json_mode: bool, fail: bool
) -> None:
    client = configured_client(tmp_path / "sessions")
    stdout, stderr = StringIO(), StringIO()
    previous_disable = logging.root.manager.disable
    flags = (
        ["--json", "--log-level", "DEBUG"] if json_mode else ["--log-level", "WARNING"]
    )
    with (
        patch.object(logging.getLogger(), "handlers", []),
        patch("conferllm.cli._load_client", return_value=client),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=RuntimeError("private body") if fail else None,
            return_value=response("Final answer."),
        ),
    ):
        status = run_cli(
            ["chat", "--model", "vision", "--prompt", "Hello", *flags],
            output=stdout,
            error_output=stderr,
        )
    assert status == int(fail)
    assert logging.root.manager.disable == previous_disable
    if json_mode and fail:
        assert json.loads(stderr.getvalue())["error"]["code"] == "provider_error"
        assert stdout.getvalue() == ""
    elif not fail:
        assert stderr.getvalue() == ""
        if json_mode:
            assert json.loads(stdout.getvalue())["message"]["text"] == "Final answer."
    assert "Calling model..." not in stderr.getvalue()
    assert "private body" not in stderr.getvalue()
