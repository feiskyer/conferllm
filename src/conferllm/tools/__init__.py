"""Built-in native tool schemas and their unrestricted local dispatcher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import ValidationError

from .common import Arguments, cancellation_event, check_cancelled
from .file import (
    EditFileArguments,
    FileTools,
    ListDirectoryArguments,
    ReadFileArguments,
    WriteFileArguments,
)
from .shell import (
    GitArguments,
    ShellArguments,
    ShellKillArguments,
    ShellOutputArguments,
    ShellTools,
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[Arguments]
    group: Literal["files", "shells"]

    def definition(self) -> dict[str, Any]:
        """Return a native function definition understood by LiteLLM."""
        schema = self.arguments.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


TOOLS = (
    Tool(
        "run_shell",
        "Run a noninteractive bash/sh command on the ConferLLM host. "
        "timeout is 1-600 seconds (default 120). Relative cwd uses the invocation "
        "directory. Use run_in_background for long commands, then shell_output "
        "or shell_kill; background processes stop at the end of this chat turn. "
        "Prefer dedicated file tools for reading and editing files.",
        ShellArguments,
        "shells",
    ),
    Tool(
        "run_powershell",
        "Run a noninteractive command using installed pwsh/PowerShell on the "
        "ConferLLM host. Same cwd, timeout and background lifetime as run_shell. "
        "Returns an error when PowerShell is not installed.",
        ShellArguments,
        "shells",
    ),
    Tool(
        "shell_output",
        "Read only new output and status from a background command in this "
        "chat turn. filter_str optionally selects lines using a regular expression.",
        ShellOutputArguments,
        "shells",
    ),
    Tool(
        "shell_kill",
        "Stop an owned background command and its process group; return remaining "
        "output and status. shell_id must come from this chat turn.",
        ShellKillArguments,
        "shells",
    ),
    Tool(
        "git_command",
        "Run a noninteractive Git command. The leading git is optional; timeout "
        "is 1-600 seconds (default 60). Relative cwd uses the invocation directory.",
        GitArguments,
        "shells",
    ),
    Tool(
        "read_file",
        "Read a UTF-8 file (up to 10 MiB) with display-only LINE_NUMBER| prefixes. "
        "offset is 1-indexed, limit defaults to 2000 lines. Never copy the line "
        "number prefixes into writes or edits. Relative paths use the invocation cwd.",
        ReadFileArguments,
        "files",
    ),
    Tool(
        "write_file",
        "Create or completely overwrite a UTF-8 file, creating parent directories. "
        "Prefer edit_file for targeted changes. Existing mode and line endings "
        "are preserved. Relative paths use the invocation cwd.",
        WriteFileArguments,
        "files",
    ),
    Tool(
        "append_file",
        "Append text to a UTF-8 file, creating it and its parents if missing. "
        "Preserves existing mode and line endings. Relative paths use invocation cwd.",
        WriteFileArguments,
        "files",
    ),
    Tool(
        "edit_file",
        "Replace exact old_string with new_string in a UTF-8 file. The match "
        "must be unique unless replace_all=true. Empty old_string creates only "
        "an empty/new file. No line-number prefixes. Preserves mode and line "
        "endings. Relative paths use the invocation cwd.",
        EditFileArguments,
        "files",
    ),
    Tool(
        "list_directory",
        "List directory entries, optionally ignoring filename glob patterns. "
        "limit defaults to 1000. Relative paths use the invocation cwd.",
        ListDirectoryArguments,
        "files",
    ),
)
_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def tool_definitions() -> list[dict[str, Any]]:
    """Return fresh schemas for every model without capability gating."""
    return [tool.definition() for tool in TOOLS]


class ToolRuntime:
    """One turn's tool execution context; no shared cwd or background handles."""

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = (cwd if cwd is not None else Path.cwd()).resolve()
        self.files = FileTools(self.cwd)
        self.shells = ShellTools(self.cwd, cancel_event=cancellation_event())
        self.warnings: list[str] = []

    def execute(self, name: str, arguments: str) -> str:
        """Return errors as native tool results, allowing model correction."""
        check_cancelled()
        tool = _TOOLS_BY_NAME.get(name)
        if tool is None:
            result: Any = {"ok": False, "error": f"Unknown tool: {name}"}
        else:
            try:
                parsed = tool.arguments.model_validate_json(arguments)
                target = self.files if tool.group == "files" else self.shells
                result = getattr(target, name)(**parsed.model_dump())
            except ValidationError as error:
                result = {
                    "ok": False,
                    "error": "Invalid tool arguments.",
                    "details": error.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                }
            except Exception as error:
                result = {
                    "ok": False,
                    "error": str(error),
                    "exception_type": type(error).__name__,
                }
        check_cancelled()
        return json.dumps(result, ensure_ascii=False, allow_nan=False)

    def close(self) -> None:
        stopped = self.shells.close()
        if stopped:
            self.warnings.append(
                f"Stopped {stopped} background command(s) at the end of the chat "
                "invocation; background handles do not survive across turns."
            )

    def __enter__(self) -> ToolRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
