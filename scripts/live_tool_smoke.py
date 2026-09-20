"""Opt-in, billable native-tool acceptance against one configured real model.

Run with the development interpreter, --model, and a fresh --directory.
Provider responses and tools are NOT mocked. The harness only redirects session
storage, observes real calls, and rejects actions outside its exact test scope.
It never opens the user's config or private session JSONL itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import threading
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import litellm

from conferllm import __version__, cli
from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.errors import ConferLLMError
from conferllm.session import SessionStore
from conferllm.tools import TOOLS, ToolRuntime
from conferllm.tools.common import cancellation_scope

FOREGROUND = "printf 'SHELL_SMOKE_OK\\n'"
BACKGROUND = "printf 'BACKGROUND_SMOKE_READY\\n'; sleep 300"
POWERSHELL = "Write-Output 'POWERSHELL_SMOKE_OK'"


def error_shape(response: httpx.Response) -> dict[str, Any]:
    """Expose only a finite diagnostic vocabulary, never raw provider errors."""
    try:
        payload = response.json()
        error = payload.get("error", payload)
        if not isinstance(error, dict):
            return {}
        message = str(error.get("message", "")).lower()
        vocabulary = {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "for",
            "from",
            "to",
            "on",
            "in",
            "with",
            "without",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
            "had",
            "does",
            "do",
            "not",
            "no",
            "yes",
            "true",
            "false",
            "invalid",
            "valid",
            "unsupported",
            "supported",
            "required",
            "missing",
            "unknown",
            "error",
            "input",
            "input_text",
            "input_image",
            "output",
            "output_text",
            "refusal",
            "model",
            "message",
            "messages",
            "role",
            "roles",
            "assistant",
            "user",
            "system",
            "developer",
            "tool",
            "tools",
            "function",
            "functions",
            "call",
            "calls",
            "result",
            "results",
            "response",
            "responses",
            "format",
            "schema",
            "parameter",
            "parameters",
            "argument",
            "arguments",
            "value",
            "values",
            "field",
            "fields",
            "item",
            "items",
            "array",
            "string",
            "object",
            "id",
            "call_id",
            "tool_call_id",
            "reasoning",
            "encrypted",
            "encrypted_content",
            "content",
            "status",
            "previous_response_id",
            "completed",
            "incomplete",
            "must",
            "should",
            "cannot",
            "can",
            "only",
            "at",
            "least",
            "one",
            "all",
            "each",
            "match",
            "matches",
            "matching",
            "matched",
            "mismatch",
            "found",
            "expected",
            "provided",
            "received",
            "allowed",
            "require",
            "requires",
            "contains",
            "contain",
            "after",
            "before",
            "endpoint",
            "request",
            "requests",
            "sequence",
            "conversation",
            "history",
            "validation",
            "max_output_tokens",
            "max_tokens",
            "temperature",
            "tool_choice",
            "generation",
            "count",
            "different",
            "same",
            "null",
            "empty",
            "name",
            "specified",
            "used",
            "using",
            "passed",
            "parse",
            "parsing",
            "unable",
            "failed",
            "failure",
            "check",
            "see",
            "details",
            "message_types",
            "type",
            "types",
            "out",
            "of",
            "range",
            "too",
            "many",
            "enough",
            "maximum",
            "minimum",
            "length",
            "exceed",
            "exceeds",
        }
        safe: dict[str, Any] = {
            # Only known protocol/grammar words leave this function; arbitrary
            # model IDs, URLs, quoted inputs and token-like values are discarded.
            "message_pattern": " ".join(
                token if token in vocabulary else "[filtered]"
                for token in re.findall(r"[a-z_][a-z_0-9-]*", message)
            ),
            "mentions": [
                word
                for word in (
                    "schema",
                    "required",
                    "properties",
                    "strict",
                    "tool_choice",
                    "tools",
                    "function",
                    "unsupported",
                    "not supported",
                    "model",
                    "stream",
                    "invalid",
                    "missing",
                    "additionalproperties",
                    "reasoning",
                    "encrypted",
                    "call_id",
                    "input",
                    "output",
                    "not found",
                )
                if word in message
            ],
            "tool_names": [tool.name for tool in TOOLS if tool.name in message],
            "requires_all_properties": (
                "every key in properties" in message
                or "all keys in properties" in message
            ),
            "invalid_model_message": "invalid model" in message,
        }
        known_codes = {
            "invalid_request_error",
            "invalid_function_parameters",
            "invalid_json_schema",
            "unsupported_parameter",
            "unsupported_value",
            "invalid_value",
            "model_not_found",
            "authentication_error",
            "permission_denied",
            "invalid_api_key",
            "rate_limit_exceeded",
            "invalid_argument",
            "not_found",
            "invalid_model",
            "invalid_model_name",
            "unsupported_model",
        }
        for field in ("code", "type", "status"):
            value = error.get(field)
            if isinstance(value, str) and value.lower() in known_codes:
                safe[field] = value
        parameter = error.get("param")
        if isinstance(parameter, str) and re.fullmatch(
            r"(?:tools|messages|input|output|tool_choice|parallel_tool_calls|model|stream)"
            r"(?:\[\d+\]|\.[a-z_]+)*",
            parameter,
        ):
            safe["parameter"] = parameter
        return safe
    except Exception:
        return {}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def check_scope(
    name: str, arguments: dict[str, Any], work: Path, shell_ids: set[str]
) -> None:
    """A test assertion, not a production permission mechanism."""

    def path(value: str) -> Path:
        if not isinstance(value, str) or value.startswith("~"):
            raise ValueError("Only the explicit test directory is permitted.")
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else work / candidate).resolve()

    if name in {"read_file", "write_file", "append_file", "edit_file"}:
        allowed = {work / "result.txt"}
        if name == "read_file":
            allowed.add(work / "seed.txt")
        if path(arguments.get("path", "")) not in allowed:
            raise ValueError("File action is outside the test fixture.")
    elif name == "list_directory":
        if path(arguments.get("path", ".")) != work:
            raise ValueError("Listing is outside the test fixture.")
    elif name in {"run_shell", "run_powershell", "git_command"}:
        if path(arguments.get("cwd", ".")) != work:
            raise ValueError("Command cwd is outside the test fixture.")
        command = arguments.get("command", "").strip()
        if name == "run_shell":
            expected = BACKGROUND if arguments.get("run_in_background") else FOREGROUND
            if command != expected:
                raise ValueError(
                    "Shell command differs from the prescribed smoke command."
                )
        elif name == "run_powershell":
            if command != POWERSHELL or arguments.get("run_in_background"):
                raise ValueError("PowerShell command differs from the smoke command.")
        elif command not in {"git --version", "--version"}:
            raise ValueError("Only git --version belongs to this test.")
    elif name in {"shell_output", "shell_kill"}:
        if arguments.get("shell_id") not in shell_ids:
            raise ValueError("Background handle does not belong to this test.")
    else:
        raise ValueError("Tool is outside the acceptance test.")


def prompt(work: Path) -> str:
    return f"""You are running an explicitly authorized real tool acceptance test.
Use native function calls; do not merely describe actions or invent results.
The ONLY permitted work area is {work}. Do not access anything outside it.
Do not inspect environment variables, credentials, configuration, home directories,
source repositories, or prior sessions. Do not install anything or use networking.
Use EXACTLY the commands below, with cwd="{work}". Do not add wrappers or flags.
You may batch independent calls. Calls in each batch run in the supplied order.
CRITICAL: tool results are visible only in the NEXT model response. Do not put a
dependent call in the same response as the call whose result it needs. Never guess
the nonce or a shell_id, and never use placeholders. End your response with the
read_file call before writing the nonce, and end with the background run_shell
call before using its shell_id. Wait for their actual tool results.

Complete all of these actions in this single turn:
1. read_file("{work / "seed.txt"}") and learn the nonce from the real file.
2. list_directory("{work}").
3. run_shell command={json.dumps(FOREGROUND)}, in the foreground.
4. git_command command="--version".
5. run_powershell command={json.dumps(POWERSHELL)}. Try it once even if it is
   unavailable; a missing PowerShell executable is BLOCKED, never PASS.
6. write_file("{work / "result.txt"}") containing exactly these three lines,
   ending with a newline and substituting the nonce read in step 1:
alpha=1
beta=two
nonce=<actual nonce>
7. append_file on result.txt with "gamma=three" followed by a real newline
   character, not literal backslash-n text.
8. edit_file on result.txt: old_string="beta=two", new_string="beta=TWO".
9. read_file on result.txt and verify the final four lines.
10. run_shell command={json.dumps(BACKGROUND)}, run_in_background=true.
11. shell_output using the ACTUAL returned shell_id. Observe BACKGROUND_SMOKE_READY.
12. shell_kill using that same actual shell_id while the process is running.

Do not run any command other than those specified. Do not skip a tool because
another tool can do similar work. Do not retry PowerShell or install it.
Your final response should contain the nonce, the observed markers and Git
version, and a PASS/FAIL/BLOCKED summary for all ten tool names. Be concise.
"""


def run(
    model: str,
    directory: Path,
    deadline: int,
    api_format: str | None = None,
    resume_directory: Path | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    first: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    if resume_directory is None:
        work = directory / "work"
        work.mkdir()
        sessions = directory / "sessions"
        nonce = uuid.uuid4().hex
        (work / "seed.txt").write_text(f"nonce={nonce}\n", encoding="utf-8")
    else:
        resume_directory = resume_directory.resolve()
        # Read public test receipts only. Session history is loaded exclusively
        # by the normal CLI/SessionStore continuation below.
        baseline = json.loads((resume_directory / "report.json").read_text())
        first = json.loads((resume_directory / "initial.json").read_text())
        observations = json.loads((resume_directory / "observations.json").read_text())
        if baseline["model"] != model or first["exit_code"] != 0:
            raise ValueError("Only a committed smoke turn for this model can resume.")
        work = resume_directory / "work"
        sessions = resume_directory / "sessions"
        nonce = (work / "seed.txt").read_text().strip().removeprefix("nonce=")
        events = observations["tool_events"]
    expected = f"alpha=1\nbeta=TWO\nnonce={nonce}\ngamma=three\n"
    request = prompt(work)
    (directory / "prompt.txt").write_text(request, encoding="utf-8")

    receipts: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []
    scope_failures: list[str] = []
    shell_ids: set[str] = set()
    processes: list[Any] = []
    phase = "initial" if resume_directory is None else "followup"
    cancelled = threading.Event()
    original_chat = LLMClient.chat
    original_execute = ToolRuntime.execute
    original_send = httpx.Client.send
    original_service = cli.ChatService
    original_load = cli._load_client
    started = time.monotonic()

    def checkpoint() -> None:
        write_json(
            directory / "observations.json",
            {
                "model": model,
                "phase": phase,
                "tool_events": events,
                "model_receipts": receipts,
                "http_receipts": network,
                "scope_failures": scope_failures,
            },
        )

    def observed_chat(
        client: LLMClient, alias: str, messages: list[dict[str, Any]]
    ) -> Any:
        if sum(item["phase"] == phase for item in receipts) >= 16:
            raise ConferLLMError(
                "smoke_call_budget", "Live acceptance call budget reached."
            )
        receipt: dict[str, Any] = {
            "phase": phase,
            "input_messages": len(messages),
            "history_tool_results": sum(m.get("role") == "tool" for m in messages),
            "started_at": time.time(),
        }
        receipts.append(receipt)
        checkpoint()
        print(
            json.dumps({"model": model, "phase": phase, "api_call": len(receipts)}),
            flush=True,
        )
        try:
            result = original_chat(client, alias, messages)
            data = result.model_dump()
            receipt.update(
                {
                    "response_id": data.get("id"),
                    "provider_model": data.get("model"),
                    "usage": data.get("usage"),
                    "finish_reason": (data.get("choices") or [{}])[0].get(
                        "finish_reason"
                    ),
                    "duration_s": round(time.time() - receipt["started_at"], 3),
                }
            )
            fields = (data.get("choices") or [{}])[0].get("message", {}).get(
                "provider_specific_fields"
            ) or {}
            receipt["native_output_shapes"] = [
                {"type": item.get("type"), "keys": sorted(item)}
                for item in fields.get("conferllm_responses_output", [])
            ]
            return result
        except BaseException as error:
            receipt["exception_type"] = type(error).__name__
            raise
        finally:
            checkpoint()

    def observed_send(
        client: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        before = time.monotonic()
        response = original_send(client, request, **kwargs)
        # Never record request URLs, bodies, headers, keys, or provider error text.
        receipt = {
            "phase": phase,
            "method": request.method,
            "status_code": response.status_code,
            "duration_s": round(time.monotonic() - before, 3),
            "api_format": (
                "responses"
                if request.url.path.endswith("/responses")
                else "chat_completion"
                if request.url.path.endswith("/chat/completions")
                else "provider_native"
            ),
        }
        if receipt["api_format"] in {"responses", "chat_completion"}:
            body = json.loads(request.content)
            input_items = body.get("input", [])
            if not isinstance(input_items, list):
                input_items = []
            call_ids = {
                item.get("call_id")
                for item in input_items
                if isinstance(item, dict) and item.get("type") == "function_call"
            }
            output_ids = {
                item.get("call_id")
                for item in input_items
                if isinstance(item, dict) and item.get("type") == "function_call_output"
            }
            receipt["request_shape"] = {
                "model_matches_alias": body.get("model") in {model, f"openai/{model}"},
                "input_types": [
                    item.get("type")
                    if item.get("type")
                    in {"message", "function_call", "function_call_output", "reasoning"}
                    else "other"
                    for item in input_items
                    if isinstance(item, dict)
                ],
                "input_item_keys": [
                    sorted(item) for item in input_items if isinstance(item, dict)
                ],
                "message_content_types": [
                    {
                        "role": item.get("role"),
                        "types": [
                            part.get("type")
                            for part in item.get("content", [])
                            if isinstance(part, dict)
                        ],
                    }
                    for item in input_items
                    if isinstance(item, dict) and isinstance(item.get("content"), list)
                ],
                "call_ids_match": output_ids <= call_ids,
                "tools_are_flat": all(
                    "function" not in tool for tool in body.get("tools", [])
                ),
            }
        if response.status_code >= 400:
            receipt["error_shape"] = error_shape(response)
        network.append(receipt)
        checkpoint()
        return response

    def observed_execute(runtime: ToolRuntime, name: str, arguments: str) -> str:
        try:
            parsed = json.loads(arguments)
            if phase != "initial":
                raise ValueError(
                    "Follow-up must use existing history, not new tool calls."
                )
            check_scope(name, parsed, work, shell_ids)
        except (ValueError, TypeError, AttributeError) as error:
            scope_failures.append(f"{name}: {error}")
            checkpoint()
            raise ConferLLMError("smoke_scope_violation", scope_failures[-1]) from None
        before = time.monotonic()
        raw = original_execute(runtime, name, arguments)
        result = json.loads(raw)
        events.append(
            {
                "phase": phase,
                "name": name,
                "arguments": parsed,
                "result": result,
                "duration_s": round(time.monotonic() - before, 3),
            }
        )
        if name in {"run_shell", "run_powershell"} and result.get("shell_id"):
            shell_id = result["shell_id"]
            shell_ids.add(shell_id)
            processes.append(runtime.shells._background[shell_id])
        checkpoint()
        print(
            json.dumps(
                {
                    "model": model,
                    "tool": name,
                    "ok": result.get("ok"),
                    "status": result.get("status"),
                }
            ),
            flush=True,
        )
        return raw

    class IsolatedChatService(ChatService):
        def __init__(self, client: LLMClient) -> None:
            super().__init__(client, session_store=SessionStore(sessions))

    def load_client(config_path: Path | None) -> LLMClient:
        loaded = original_load(config_path)
        if api_format is not None:
            # In-memory non-secret override only; never edit/copy the config file.
            loaded.require_model_config(model).api_format = api_format
        return loaded

    def invoke(arguments: list[str]) -> dict[str, Any]:
        output, errors = StringIO(), StringIO()
        status = cli.run_cli(arguments, output=output, error_output=errors)
        try:
            data = json.loads(output.getvalue() if status == 0 else errors.getvalue())
        except json.JSONDecodeError:
            data = {
                "ok": False,
                "error": {
                    "code": "interrupted" if status == 130 else "invalid_cli_output"
                },
            }
        return {"exit_code": status, "response": data}

    def expired(_signum: int, _frame: Any) -> None:
        cancelled.set()
        raise KeyboardInterrupt

    # This is a dedicated process, so changing its cwd cannot affect other runs.
    os.chdir(work)
    litellm.telemetry = False
    logging.disable(logging.CRITICAL)
    LLMClient.chat = observed_chat
    ToolRuntime.execute = observed_execute
    httpx.Client.send = observed_send
    cli.ChatService = IsolatedChatService
    cli._load_client = load_client
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(deadline)
    follow: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    try:
        metadata = invoke(["model-info", model])
        with cancellation_scope(cancelled):
            if resume_directory is None:
                first = invoke(
                    [
                        "chat",
                        "--model",
                        model,
                        "--name",
                        f"live-tool-smoke-{nonce[:8]}",
                        "--prompt",
                        request,
                        "--json",
                    ]
                )
            write_json(directory / "initial.json", first)
            if first["exit_code"] == 0:
                phase = "followup"
                session_id = first["response"]["session"]["id"]
                follow = invoke(
                    [
                        "chat",
                        "--session",
                        session_id,
                        "--prompt",
                        "Without calling any tools, recall from the stored conversation "
                        "the exact nonce you read and the two shell markers. Also recall "
                        "the final four-line file content and whether PowerShell actually "
                        "ran or was blocked. Do not inspect any new file. Be concise.",
                        "--json",
                    ]
                )
                write_json(directory / "followup.json", follow)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        LLMClient.chat = original_chat
        ToolRuntime.execute = original_execute
        httpx.Client.send = original_send
        cli.ChatService = original_service
        cli._load_client = original_load
        checkpoint()

    statuses: dict[str, str] = {}
    for tool in TOOLS:
        matches = [event for event in events if event["name"] == tool.name]
        if not matches:
            statuses[tool.name] = "not_called"
        elif any(event["result"].get("ok") is True for event in matches):
            statuses[tool.name] = "pass"
        elif tool.name == "run_powershell" and all(
            "PowerShell executable not found" in event["result"].get("error", "")
            for event in matches
        ):
            statuses[tool.name] = "blocked_missing_powershell"
        else:
            statuses[tool.name] = "fail"
    actual = (
        (work / "result.txt").read_text(encoding="utf-8")
        if (work / "result.txt").exists()
        else None
    )
    follow_text = (
        follow["response"].get("message", {}).get("text", "")
        if follow and follow["exit_code"] == 0
        else ""
    )
    checks = {
        "initial_completed": first.get("exit_code") == 0,
        "file_contents_exact": actual == expected,
        "foreground_marker": any(
            e["name"] == "run_shell" and e["result"].get("output") == "SHELL_SMOKE_OK\n"
            for e in events
        ),
        "background_marker_while_running": any(
            e["name"] == "shell_output"
            and e["result"].get("status") == "running"
            and "BACKGROUND_SMOKE_READY" in e["result"].get("output", "")
            for e in events
        ),
        "background_killed": any(
            e["name"] == "shell_kill" and e["result"].get("status") == "terminated"
            for e in events
        ),
        "owned_processes_cleaned": (
            baseline.get("checks", {}).get("owned_processes_cleaned", True)
            and all(
                p.process.poll() is not None and not p._reader.is_alive()
                for p in processes
            )
        ),
        "history_replayed": any(
            r["phase"] == "followup" and r["history_tool_results"] >= len(events)
            for r in receipts
        ),
        "followup_recalls_nonce": nonce in follow_text,
        "scope_respected": not scope_failures,
    }
    report = {
        "model": model,
        "version": __version__,
        "directory": str(directory),
        "resumed_initial_directory": (
            str(resume_directory) if resume_directory is not None else None
        ),
        "metadata": metadata,
        "tool_statuses": statuses,
        "checks": checks,
        "api_calls": len(receipts),
        "http_statuses": [r["status_code"] for r in network],
        "duration_s": round(time.monotonic() - started, 3),
        "initial_error": first.get("response", {}).get("error"),
        "followup_error": follow.get("response", {}).get("error") if follow else None,
        "final_file_sha256": hashlib.sha256(actual.encode()).hexdigest()
        if actual is not None
        else None,
        "powershell_available": bool(
            shutil.which("pwsh")
            or shutil.which("powershell")
            or shutil.which("powershell.exe")
        ),
        "all_checks_passed": all(checks.values()),
        "all_tools_executed_successfully": all(
            status == "pass" for status in statuses.values()
        ),
    }
    write_json(directory / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--deadline", type=int, default=600)
    parser.add_argument("--api-format", choices=["responses", "chat_completion"])
    parser.add_argument(
        "--resume-directory",
        type=Path,
        help="Continue a committed test turn without repeating any initial tools.",
    )
    arguments = parser.parse_args()
    report = run(
        arguments.model,
        arguments.directory,
        arguments.deadline,
        arguments.api_format,
        arguments.resume_directory,
    )
    if report["all_checks_passed"] and report["all_tools_executed_successfully"]:
        raise SystemExit(0)
    dependency_only = report["all_checks_passed"] and all(
        status == "pass"
        or (name == "run_powershell" and status == "blocked_missing_powershell")
        for name, status in report["tool_statuses"].items()
    )
    raise SystemExit(2 if dependency_only else 1)


if __name__ == "__main__":
    main()
