"""Built-in tools exercise only temporary files and owned local subprocesses."""

from __future__ import annotations

import json
import os
import shlex
import signal
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Timer
from unittest.mock import patch

import pytest

from conferllm.errors import ConferLLMError
from conferllm.tools import TOOLS, ToolRuntime, tool_definitions
from conferllm.tools.common import MAX_OUTPUT_CHARS, cancellation_scope, truncate
from conferllm.tools.powershell import powershell_argv


def execute(runtime: ToolRuntime, name: str, **arguments: object) -> dict:
    return json.loads(runtime.execute(name, json.dumps(arguments)))


def python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "Timed out waiting for owned test process"
        time.sleep(0.01)


def test_all_native_schemas_are_independent_and_match_handlers(tmp_path: Path) -> None:
    expected = {
        "run_shell",
        "run_powershell",
        "shell_output",
        "shell_kill",
        "git_command",
        "read_file",
        "write_file",
        "append_file",
        "edit_file",
        "list_directory",
    }
    schemas = tool_definitions()
    assert {schema["function"]["name"] for schema in schemas} == expected
    with ToolRuntime(tmp_path) as runtime:
        for tool, schema in zip(TOOLS, schemas, strict=True):
            assert schema["type"] == "function"
            assert schema["function"]["parameters"]["type"] == "object"
            assert schema["function"]["parameters"]["additionalProperties"] is False
            assert callable(getattr(getattr(runtime, tool.group), tool.name))
    schemas[0]["function"]["parameters"]["properties"].clear()
    assert tool_definitions()[0]["function"]["parameters"]["properties"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("unknown", "{}"),
        ("read_file", "{bad JSON"),
        ("read_file", "[]"),
        ("read_file", "{}"),
        ("read_file", '{"path": 1}'),
        ("read_file", '{"path": "x", "offset": false}'),
        ("read_file", '{"path": "x", "offset": 0}'),
        ("read_file", '{"path": "x", "extra": "not-allowed"}'),
        ("run_shell", '{"command": "", "timeout": 1}'),
        ("run_shell", '{"command": "pwd", "timeout": "1"}'),
        ("run_shell", '{"command": "pwd", "timeout": 601}'),
        (
            "edit_file",
            '{"path": "x", "old_string": "x", "new_string": "y", "replace_all": 1}',
        ),
    ],
)
def test_invalid_arguments_are_results_without_execution(
    tmp_path: Path, name: str, arguments: str
) -> None:
    before = set(tmp_path.iterdir())
    with (
        ToolRuntime(tmp_path) as runtime,
        patch(
            "conferllm.tools.shell.subprocess.Popen",
            side_effect=AssertionError("no process"),
        ),
    ):
        result = json.loads(runtime.execute(name, arguments))
    assert result["ok"] is False
    assert set(tmp_path.iterdir()) == before


def test_file_round_trip_needs_no_prior_read_and_preserves_mode_and_crlf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "script.py"
    path.write_bytes(b"one\r\ntwo\r\n")
    path.chmod(0o755)
    with ToolRuntime(tmp_path) as runtime:
        assert execute(
            runtime, "write_file", path="script.py", content="你好\nworld\n"
        )["ok"]
        assert path.read_bytes() == "你好\r\nworld\r\n".encode()
        assert execute(runtime, "append_file", path="script.py", content="tail\n")["ok"]
        assert execute(
            runtime,
            "edit_file",
            path="script.py",
            old_string="world\ntail\n",
            new_string="changed\n",
        )["ok"]
        viewed = execute(runtime, "read_file", path="script.py", offset=2, limit=1)
        assert viewed["output"] == "     2|changed"
        assert viewed["total_lines"] == 2
        assert viewed["truncated"]
    assert path.read_bytes() == "你好\r\nchanged\r\n".encode()
    assert stat.S_IMODE(path.stat().st_mode) == 0o755


def test_file_creation_and_listing_with_real_glob_patterns(tmp_path: Path) -> None:
    with ToolRuntime(tmp_path) as runtime:
        assert execute(runtime, "write_file", path="nested/a.txt", content="a")["ok"]
        assert execute(runtime, "append_file", path="nested/cache.pyc", content="x")[
            "ok"
        ]
        assert execute(
            runtime, "edit_file", path="nested/new.txt", old_string="", new_string="new"
        )["ok"]
        listed = execute(runtime, "list_directory", path="nested", ignore=["*.pyc"])
        assert "a.txt" in listed["output"]
        assert "new.txt" in listed["output"]
        assert "cache.pyc" not in listed["output"]
        assert not listed["truncated"]
        assert execute(runtime, "list_directory", path="nested", limit=1)["truncated"]
        assert "[DIR]" in execute(runtime, "list_directory")["output"]


def test_paths_are_not_sandboxed_and_symlink_writes_preserve_the_link(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("old", encoding="utf-8")
    link = work / "link"
    link.symlink_to(target)
    with ToolRuntime(work) as runtime:
        assert execute(runtime, "write_file", path="link", content="new")["ok"]
        assert link.is_symlink()
        assert execute(runtime, "read_file", path="../outside.txt")["output"].endswith(
            "new"
        )
        assert execute(runtime, "write_file", path="~/sample.txt", content="home")["ok"]
        assert (Path.home() / "sample.txt").read_text() == "home"
        assert "[LINK]" in execute(runtime, "list_directory")["output"]
    assert target.read_text() == "new"


def test_atomic_write_failure_does_not_damage_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "original"
    path.write_text("original", encoding="utf-8")
    before = set(tmp_path.iterdir())
    with (
        ToolRuntime(tmp_path) as runtime,
        patch("conferllm.tools.file.os.replace", side_effect=OSError("disk error")),
    ):
        result = execute(runtime, "write_file", path="original", content="replacement")
    assert not result["ok"]
    assert path.read_text() == "original"
    assert set(tmp_path.iterdir()) == before


def test_exact_edit_ambiguity_and_deletion(tmp_path: Path) -> None:
    path = tmp_path / "text"
    path.write_text("one\none\nend\n", encoding="utf-8")
    with ToolRuntime(tmp_path) as runtime:
        failed = execute(
            runtime, "edit_file", path="text", old_string="one", new_string="two"
        )
        assert not failed["ok"]
        assert "2 matches" in failed["error"]
        assert execute(
            runtime,
            "edit_file",
            path="text",
            old_string="one",
            new_string="two",
            replace_all=True,
        )["ok"]
        assert execute(
            runtime, "edit_file", path="text", old_string="end", new_string=""
        )["ok"]
        assert path.read_text() == "two\ntwo\n\n"
        assert not execute(
            runtime, "edit_file", path="text", old_string="", new_string="overwrite"
        )["ok"]
        assert not execute(
            runtime, "edit_file", path="missing", old_string="missing", new_string="x"
        )["ok"]


@pytest.mark.parametrize("kind", ["missing", "directory", "binary", "too_large"])
def test_file_read_errors_are_tool_results(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "input"
    if kind == "directory":
        path.mkdir()
    elif kind == "binary":
        path.write_bytes(b"\xff\xfe")
    elif kind == "too_large":
        path.write_bytes(b"x" * 100)
    with (
        ToolRuntime(tmp_path) as runtime,
        patch("conferllm.tools.file.MAX_FILE_BYTES", 50),
    ):
        result = execute(runtime, "read_file", path="input")
    assert not result["ok"]


def test_truncation_keeps_head_tail_and_budget() -> None:
    assert truncate("short") == "short"
    result = truncate("head" + "x" * MAX_OUTPUT_CHARS + "tail")
    assert len(result) == MAX_OUTPUT_CHARS
    assert result.startswith("head") and result.endswith("tail")
    assert "truncated" in result


def test_file_output_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "long"
    path.write_text("A" * (MAX_OUTPUT_CHARS * 2), encoding="utf-8")
    with ToolRuntime(tmp_path) as runtime:
        result = execute(runtime, "read_file", path="long")
    assert result["truncated"]
    assert len(result["output"]) <= MAX_OUTPUT_CHARS


def test_foreground_shell_captures_stdout_stderr_exit_and_cwd(tmp_path: Path) -> None:
    directory = tmp_path / "nested"
    directory.mkdir()
    before = Path.cwd()
    with ToolRuntime(tmp_path) as runtime:
        result = execute(
            runtime,
            "run_shell",
            cwd="nested",
            command="pwd; printf 'stdout\\n'; printf 'stderr\\n' >&2; exit 7",
        )
        assert not result["ok"]
        assert result["exit_code"] == 7
        assert not result["timed_out"]
        assert str(directory.resolve()) in result["output"]
        assert "stdout" in result["output"] and "stderr" in result["output"]
        assert execute(runtime, "git_command", command="--version")["ok"]
        assert execute(runtime, "git_command", command="git --version")["ok"]
    assert Path.cwd() == before


def test_shell_uses_devnull_for_stdin(tmp_path: Path) -> None:
    with ToolRuntime(tmp_path) as runtime:
        result = execute(
            runtime,
            "run_shell",
            command=python_command("import sys; print(repr(sys.stdin.read()))"),
        )
    assert result["ok"]
    assert result["output"].strip() == "''"


def test_verbose_subprocess_does_not_block_or_retain_unbounded_output(
    tmp_path: Path,
) -> None:
    with ToolRuntime(tmp_path) as runtime:
        result = execute(
            runtime,
            "run_shell",
            command=python_command("print('x' * 1000000); print('THE-END')"),
        )
    assert result["ok"]
    assert "truncated" in result["output"]
    assert result["output"].endswith("THE-END\n")
    assert len(result["output"]) < MAX_OUTPUT_CHARS + 100


def test_foreground_timeout_kills_the_process_group(tmp_path: Path) -> None:
    groups: list[tuple[int, int]] = []
    killpg = os.killpg

    def record_signal(pid: int, sig: int) -> None:
        groups.append((pid, sig))
        killpg(pid, sig)

    started = time.monotonic()
    with (
        ToolRuntime(tmp_path) as runtime,
        patch("conferllm.tools.shell.os.killpg", side_effect=record_signal),
    ):
        result = execute(
            runtime,
            "run_shell",
            timeout=1,
            command=python_command(
                "import time; print('started', flush=True); time.sleep(30)"
            ),
        )
    assert time.monotonic() - started < 5
    assert result["timed_out"] and not result["ok"]
    assert "started" in result["output"]
    assert {sig for _, sig in groups} >= {signal.SIGTERM, signal.SIGKILL}
    assert all(pid != os.getpgrp() for pid, _ in groups)


def test_background_output_is_incremental_and_kill_joins_reader(tmp_path: Path) -> None:
    with ToolRuntime(tmp_path) as runtime:
        started = execute(
            runtime,
            "run_shell",
            run_in_background=True,
            command=python_command(
                "import time; print('ready', flush=True); time.sleep(30)"
            ),
        )
        shell_id = started["shell_id"]
        process = runtime.shells._background[shell_id]
        wait_until(lambda: len(process._buffer) > 0)
        first = execute(runtime, "shell_output", shell_id=shell_id, filter_str="ready")
        assert first["status"] == "running" and first["output"] == "ready"
        assert execute(runtime, "shell_output", shell_id=shell_id)["output"] == ""
        assert not execute(runtime, "shell_output", shell_id=shell_id, filter_str="[")[
            "ok"
        ]
        stopped = execute(runtime, "shell_kill", shell_id=shell_id)
        assert stopped["status"] == "terminated"
        assert process.process.poll() is not None
        assert not process._reader.is_alive()
        assert execute(runtime, "shell_kill", shell_id=shell_id)["ok"]
    assert runtime.warnings == []


def test_background_handles_do_not_cross_invocations_and_errors_clean_up(
    tmp_path: Path,
) -> None:
    runtime = ToolRuntime(tmp_path)
    with pytest.raises(RuntimeError, match="model failed"), runtime:
        started = execute(
            runtime,
            "run_shell",
            run_in_background=True,
            command=python_command("import time; time.sleep(30)"),
        )
        process = runtime.shells._background[started["shell_id"]]
        with ToolRuntime(tmp_path) as other:
            assert not execute(other, "shell_kill", shell_id=started["shell_id"])["ok"]
        raise RuntimeError("model failed")
    assert process.process.poll() is not None
    assert not process._reader.is_alive()
    assert runtime.warnings
    runtime.close()


def test_background_normal_exit_reports_status(tmp_path: Path) -> None:
    with ToolRuntime(tmp_path) as runtime:
        result = execute(runtime, "run_shell", command="exit 9", run_in_background=True)
        process = runtime.shells._background[result["shell_id"]]
        wait_until(lambda: process.process.poll() is not None)
        status = execute(runtime, "shell_output", shell_id=result["shell_id"])
        assert status["status"] == "failed" and status["exit_code"] == 9


@pytest.mark.parametrize("tool", ["run_shell", "run_powershell", "git_command"])
def test_blank_commands_are_rejected(tmp_path: Path, tool: str) -> None:
    with ToolRuntime(tmp_path) as runtime:
        assert not execute(runtime, tool, command=" \n ")["ok"]


def test_shell_missing_cwd_or_binary_is_a_result(tmp_path: Path) -> None:
    with ToolRuntime(tmp_path) as runtime:
        assert not execute(runtime, "run_shell", command="pwd", cwd="missing")["ok"]
        with patch("conferllm.tools.shell.shutil.which", return_value=None):
            assert not execute(runtime, "run_shell", command="pwd")["ok"]


def test_cancellation_stops_a_foreground_command(tmp_path: Path) -> None:
    event = Event()
    timer = Timer(0.2, event.set)
    try:
        timer.start()
        started = time.monotonic()
        with (
            cancellation_scope(event),
            ToolRuntime(tmp_path) as runtime,
            pytest.raises(ConferLLMError) as failure,
        ):
            execute(
                runtime,
                "run_shell",
                command=python_command("import time; time.sleep(30)"),
            )
        assert failure.value.code == "chat_cancelled"
        assert time.monotonic() - started < 3
    finally:
        timer.cancel()
        timer.join(timeout=1)


def test_cancellation_stops_background_even_after_stdout_closes(tmp_path: Path) -> None:
    event = Event()
    ready = tmp_path / "ready"
    with cancellation_scope(event), ToolRuntime(tmp_path) as runtime:
        started = execute(
            runtime,
            "run_shell",
            run_in_background=True,
            command=python_command(
                "import os, time; from pathlib import Path; "
                "os.close(1); os.close(2); "
                f"Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
            ),
        )
        process = runtime.shells._background[started["shell_id"]]
        wait_until(ready.exists)
        event.set()
        wait_until(lambda: process.process.poll() is not None)
    assert not process._reader.is_alive()


@pytest.mark.parametrize("executable", ["pwsh", "powershell", "powershell.exe"])
def test_powershell_discovery_and_native_arguments(executable: str) -> None:
    with patch(
        "conferllm.tools.powershell.shutil.which",
        side_effect=lambda name: f"/test/{name}" if name == executable else None,
    ):
        assert powershell_argv("Write-Output '你好'") == [
            f"/test/{executable}",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Write-Output '你好'",
        ]


def test_powershell_missing_does_not_install_anything(tmp_path: Path) -> None:
    with (
        ToolRuntime(tmp_path) as runtime,
        patch("conferllm.tools.powershell.shutil.which", return_value=None),
        patch("conferllm.tools.shell.subprocess.Popen") as spawn,
    ):
        result = execute(runtime, "run_powershell", command="Write-Output test")
    assert not result["ok"]
    assert "PowerShell executable not found" in result["error"]
    spawn.assert_not_called()


@pytest.mark.parametrize("background", [False, True])
def test_powershell_executor_uses_the_owned_process_path(
    tmp_path: Path, background: bool
) -> None:
    # The fake host checks actual process launch plumbing, not PowerShell syntax.
    host = tmp_path / "pwsh"
    host.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    host.chmod(0o755)
    with (
        ToolRuntime(tmp_path) as runtime,
        patch("conferllm.tools.powershell.shutil.which", return_value=str(host)),
    ):
        result = execute(
            runtime,
            "run_powershell",
            command="Write-Output test",
            run_in_background=background,
        )
        if background:
            process = runtime.shells._background[result["shell_id"]]
            wait_until(lambda: process.process.poll() is not None)
            result = execute(runtime, "shell_output", shell_id=result["shell_id"])
        assert result["ok"]
        assert json.loads(result["output"]) == [
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Write-Output test",
        ]
