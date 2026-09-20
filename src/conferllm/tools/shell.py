"""Noninteractive command execution with turn-owned background processes."""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import subprocess
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import Field

from .common import MAX_OUTPUT_CHARS, Arguments, resolve_path
from .powershell import powershell_argv


class ShellArguments(Arguments):
    command: str = Field(min_length=1)
    timeout: int = Field(default=120, ge=1, le=600)
    run_in_background: bool = False
    cwd: str = "."


class ShellOutputArguments(Arguments):
    shell_id: str = Field(min_length=1)
    filter_str: str = ""


class ShellKillArguments(Arguments):
    shell_id: str = Field(min_length=1)


class GitArguments(Arguments):
    command: str = Field(min_length=1)
    timeout: int = Field(default=60, ge=1, le=600)
    cwd: str = "."


class _Process:
    """Drain a process continuously while retaining only bounded unread output."""

    def __init__(
        self, argv: list[str], cwd: Path, cancel_event: threading.Event | None = None
    ) -> None:
        self._cancel_event = cancel_event
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
        )
        self._buffer = bytearray()
        self._dropped = 0
        self._lock = threading.Lock()
        self._stop_reader = threading.Event()
        self.stopped = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        try:
            self._reader.start()
        except BaseException:
            self._signal(signal.SIGKILL)
            self.process.wait()
            assert self.process.stdout is not None
            self.process.stdout.close()
            raise

    def _read(self) -> None:
        stream = self.process.stdout
        assert stream is not None
        try:
            descriptor = stream.fileno()
            eof = False
            while not self._stop_reader.is_set():
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self._signal(signal.SIGKILL)
                    break
                if eof:
                    if self.process.poll() is not None:
                        break
                    self._stop_reader.wait(0.1)
                    continue
                ready, _, _ = select.select([descriptor], [], [], 0.1)
                if not ready:
                    continue
                chunk = os.read(descriptor, 8192)
                if not chunk:
                    eof = True
                    continue
                with self._lock:
                    self._buffer.extend(chunk)
                    excess = max(0, len(self._buffer) - MAX_OUTPUT_CHARS)
                    if excess:
                        del self._buffer[:excess]
                        self._dropped += excess
        except OSError:
            pass
        finally:
            stream.close()
            if self.process.poll() is not None:
                self._signal(signal.SIGKILL)

    def _signal(self, sig: int) -> None:
        # The child is a new session leader. Signal its whole group even after
        # the shell exits, because grandchildren can keep running/hold pipes.
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, sig)

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self._signal(signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=0.5)
        finally:
            self._signal(signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=2)
            self._reader.join(timeout=0.5)
            self._stop_reader.set()
            self._reader.join(timeout=0.5)

    def output(self) -> str:
        if self.process.poll() is not None:
            self._reader.join(timeout=0.2)
        with self._lock:
            output = bytes(self._buffer).decode("utf-8", errors="replace")
            dropped = self._dropped
            self._buffer.clear()
            self._dropped = 0
        if dropped:
            output = f"[output truncated: {dropped} earlier bytes omitted]\n{output}"
        return output

    def status(self) -> str:
        code = self.process.poll()
        if self.stopped:
            return "terminated"
        if code is None:
            return "running"
        return "completed" if code == 0 else "failed"


class ShellTools:
    """Own subprocesses only for one chat invocation."""

    def __init__(
        self, cwd: Path, *, cancel_event: threading.Event | None = None
    ) -> None:
        self.cwd = cwd
        self._cancel_event = cancel_event
        self._background: dict[str, _Process] = {}

    def _run(
        self,
        argv: list[str],
        *,
        cwd: str,
        timeout: int,
        run_in_background: bool,
    ) -> dict[str, Any]:
        directory = resolve_path(cwd, self.cwd)
        process = _Process(argv, directory, self._cancel_event)
        if run_in_background:
            shell_id = uuid.uuid4().hex
            self._background[shell_id] = process
            return {
                "ok": True,
                "shell_id": shell_id,
                "status": "running",
                "cwd": str(directory),
                "output": "Use shell_output/shell_kill during this turn. "
                "The process will be stopped when this chat invocation ends.",
            }

        timed_out = False
        try:
            try:
                process.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            process.stop()
        code = process.process.returncode
        return {
            "ok": not timed_out and code == 0,
            "output": process.output(),
            "exit_code": code,
            "timed_out": timed_out,
            "cwd": str(directory),
        }

    def run_shell(
        self,
        command: str,
        timeout: int = 120,
        run_in_background: bool = False,
        cwd: str = ".",
    ) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("Command must not be empty.")
        executable = shutil.which("bash") or shutil.which("sh")
        if executable is None:
            raise FileNotFoundError("No bash or sh executable was found.")
        return self._run(
            [executable, "-c", command],
            cwd=cwd,
            timeout=timeout,
            run_in_background=run_in_background,
        )

    def run_powershell(
        self,
        command: str,
        timeout: int = 120,
        run_in_background: bool = False,
        cwd: str = ".",
    ) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("Command must not be empty.")
        return self._run(
            powershell_argv(command),
            cwd=cwd,
            timeout=timeout,
            run_in_background=run_in_background,
        )

    def git_command(
        self, command: str, timeout: int = 60, cwd: str = "."
    ) -> dict[str, Any]:
        command = command.strip()
        if not command:
            raise ValueError("Command must not be empty.")
        if re.match(r"git(?:\s|$)", command) is None:
            command = f"git {command}"
        return self.run_shell(command, timeout=timeout, cwd=cwd)

    def _get(self, shell_id: str) -> _Process:
        try:
            return self._background[shell_id]
        except KeyError:
            raise ValueError("Unknown shell_id for this chat invocation.") from None

    def shell_output(self, shell_id: str, filter_str: str = "") -> dict[str, Any]:
        pattern = re.compile(filter_str) if filter_str else None
        process = self._get(shell_id)
        output = process.output()
        if pattern is not None:
            output = "\n".join(
                line for line in output.splitlines() if pattern.search(line)
            )
        return {
            "ok": True,
            "shell_id": shell_id,
            "output": output,
            "status": process.status(),
            "exit_code": process.process.poll(),
        }

    def shell_kill(self, shell_id: str) -> dict[str, Any]:
        process = self._get(shell_id)
        process.stop()
        return self.shell_output(shell_id)

    def close(self) -> int:
        """Stop every owned process and return how many were still running."""
        running = 0
        try:
            for process in self._background.values():
                running += process.process.poll() is None
                process.stop()
        finally:
            self._background.clear()
        return running
