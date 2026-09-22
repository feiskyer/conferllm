"""Application progress events, separate from noisy provider diagnostics."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from textwrap import indent
from typing import Any, TextIO

from .errors import normalize_error

logger = logging.getLogger(__name__)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|(?:access|refresh)[_-]?token)",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"""((?:api[_-]?key|authorization|password|secret|(?:access|refresh)[_-]?token)"""
    r"""["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)""",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[^\s\"',;]+", re.IGNORECASE)
_DATA_URL = re.compile(r"data:[^\s;,]+;base64,[A-Za-z0-9+/=\r\n]+")
_IMAGE_KEYS = {"image_url", "b64_json", "base64", "image_base64"}
_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@contextmanager
def cli_logging(stream: TextIO, level: int) -> Iterator[None]:
    """Route one CLI invocation to its stderr and restore host logging afterward."""
    root = logging.getLogger()
    dependency_names = (
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
        "httpx",
        "httpcore",
        "openai",
    )
    loggers = [root, *(logging.getLogger(name) for name in dependency_names)]
    states = [
        (
            target,
            target.level,
            list(target.handlers),
            target.propagate,
            target.disabled,
        )
        for target in loggers
    ]
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.handlers = [handler]
    root.setLevel(level)
    for dependency in loggers[1:]:
        dependency.handlers.clear()
        dependency.setLevel(logging.WARNING)
        dependency.propagate = True
        dependency.disabled = False
    try:
        yield
    finally:
        for target, saved_level, handlers, propagate, disabled in states:
            target.handlers = handlers
            target.setLevel(saved_level)
            target.propagate = propagate
            target.disabled = disabled
        handler.close()


@dataclass
class _TurnProgress:
    session: str
    turn: int
    model: str
    round: int = 1


_TURN: ContextVar[_TurnProgress | None] = ContextVar("conferllm_progress", default=None)


@contextmanager
def turn_progress(session: str, turn: int, model: str) -> Iterator[_TurnProgress]:
    """Keep concurrent MCP requests' event identities independent."""
    progress = _TurnProgress(session, turn, model)
    token = _TURN.set(progress)
    try:
        yield progress
    except (Exception, KeyboardInterrupt) as error:
        event(
            "turn.error",
            level=logging.DEBUG,
            code=(
                normalize_error(error).code
                if isinstance(error, Exception)
                else "chat_cancelled"
            ),
            exception_type=type(error).__name__,
        )
        raise
    finally:
        _TURN.reset(token)


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth >= 32:
        return "[nested content omitted]"
    if isinstance(value, dict):
        return {
            key: (
                "[redacted]"
                if _SECRET_KEY.search(key)
                else "[image omitted]"
                if key in _IMAGE_KEYS
                else _sanitize(item, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in value]
    if isinstance(value, str):
        value = _DATA_URL.sub("[image omitted]", value)
        value = _BEARER.sub("Bearer [redacted]", value)
        value = _SECRET_TEXT.sub(r"\1[redacted]", value)
        return _CONTROL.sub(lambda match: f"\\x{ord(match[0]):02x}", value)
    return value


def _decode_arguments(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, RecursionError):
        # Malformed or oversized arguments still belong in the tool's validator.
        return value


def _value_text(value: Any) -> str:
    if isinstance(value, dict):
        return (
            "\n".join(_field(str(key), item) for key, item in value.items())
            or "(empty)"
        )
    if isinstance(value, list):
        return "\n".join("- " + _value_text(item) for item in value) or "(empty)"
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _field(label: str, value: Any) -> str:
    text = _value_text(value)
    label = label.replace("_", " ")
    if isinstance(value, (dict, list)) or "\n" in text:
        return f"{label}:\n{indent(text, '  ')}"
    return f"{label}: {text}"


def _tool_output(output: dict[str, Any]) -> str:
    lines = []
    if output.get("error"):
        lines.append(_field("Error", output["error"]))
    if output.get("details"):
        for detail in output["details"]:
            location = ".".join(str(part) for part in detail.get("loc", []))
            lines.append(_field(location or "Arguments", detail["msg"]))
    if output.get("output"):
        lines.append(str(output["output"]))
    elif output.get("path"):
        lines.append(_field("Path", output["path"]))
    if output.get("exit_code") not in (None, 0):
        lines.append(_field("Exit code", output["exit_code"]))
    if output.get("shell_id"):
        lines.append(_field("Shell", output["shell_id"]))
    if output.get("timed_out"):
        lines.append("Command timed out.")
    if output.get("truncated"):
        lines.append("[output truncated by tool]")
    return "\n".join(lines)


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "message")
        content = message.get("content")
        if role == "tool" and isinstance(content, str):
            decoded = _decode_arguments(content)
            if isinstance(decoded, dict):
                content = _tool_output(decoded)
        if isinstance(content, list):
            content = "\n".join(
                str(part["text"])
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else "[image omitted]"
                for part in content
            )
        if content:
            lines.append(_field(role.capitalize(), content))
        for call in message.get("tool_calls") or []:
            function = call["function"]
            lines.append(
                _field(
                    f"Tool {function['name']}",
                    _sanitize(_decode_arguments(function["arguments"])),
                )
            )
    return "\n".join(lines)


def _format_event(data: dict[str, Any]) -> str:
    name = data["event"]
    detail = ""
    elapsed = f" ({data['elapsed_seconds']:.2f}s)" if "elapsed_seconds" in data else ""
    if name == "model.request":
        title = "Calling model..."
        if "prompt" in data:
            detail = _prompt_text(data["prompt"])
    elif name == "model.response":
        count = data.get("tool_calls", 0)
        title = f"Model reply: {count} tool call(s)" if count else "Model finished"
        title += elapsed
        if data.get("output_images"):
            title += f"; {data['output_images']} image(s)"
        if count or logger.isEnabledFor(logging.DEBUG):
            detail = data.get("text", "")
    elif name == "tool.start":
        title = f"Tool {data.get('name', '')}"
        detail = _field("Input", data["input"])
    elif name == "tool.result":
        output = data["output"]
        status = output.get("status") or ("completed" if output["ok"] else "failed")
        title = f"Tool {data.get('name', '')}: {status}{elapsed}"
        detail = _tool_output(output)
    elif name == "turn.error":
        title = f"Stopped: {data['code']} ({data['exception_type']})"
    else:
        title = name
    if "session" in data:
        title = (
            f"[{data['model']} {data['session'][-8:]} "
            f"turn {data['turn']}/round {data['round']}] {title}"
        )
    elif "model" in data:
        title = f"[{data['model']}] {title}"
    return title + ("\n" + indent(detail, "  ") if detail else "")


def event(event_name: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Render useful content for people; retain structured fields on the record."""
    if not logger.isEnabledFor(level):
        return
    progress = _TURN.get()
    context = asdict(progress) if progress is not None else {}
    if event_name == "tool.start":
        fields["input"] = _decode_arguments(fields["input"])
    data = _sanitize({"event": event_name, **context, **fields})
    logger.log(
        level,
        "%s",
        _format_event(data),
        extra={"progress": data},
    )


def model_request(model: str, messages: list[dict[str, Any]]) -> None:
    """Log the current prompt once per turn; DEBUG also includes replayed history."""
    if not logger.isEnabledFor(logging.INFO):
        return
    progress = _TURN.get()
    fields: dict[str, Any] = {"model": model, "message_count": len(messages)}
    if progress is None or progress.round == 1 or logger.isEnabledFor(logging.DEBUG):
        if progress is not None and not logger.isEnabledFor(logging.DEBUG):
            messages = [
                message
                for message in messages[:-1]
                if message.get("role") in {"system", "developer"}
            ] + messages[-1:]
        # Omit opaque provider state, reasoning signatures, and image payloads.
        fields["prompt"] = [
            {key: value for key, value in message.items() if key in _MESSAGE_KEYS}
            for message in messages
        ]
    event("model.request", **fields)
