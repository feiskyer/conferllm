"""Shared argument validation, cancellation, and bounded tool output."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event

from pydantic import BaseModel, ConfigDict

from ..errors import ConferLLMError

MAX_OUTPUT_CHARS = 30_000
MAX_FILE_BYTES = 10 * 1024 * 1024
_CANCELLATION: ContextVar[Event | None] = ContextVar(
    "conferllm_cancellation", default=None
)


@contextmanager
def cancellation_scope(event: Event) -> Iterator[None]:
    """Carry an MCP request's cancellation into its worker thread."""
    token = _CANCELLATION.set(event)
    try:
        yield
    finally:
        _CANCELLATION.reset(token)


def cancellation_event() -> Event | None:
    return _CANCELLATION.get()


def check_cancelled() -> None:
    event = cancellation_event()
    if event is not None and event.is_set():
        raise ConferLLMError("chat_cancelled", "Chat invocation was cancelled.")


class Arguments(BaseModel):
    """Validate tool inputs without coercion or an authorization layer."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


def resolve_path(path: str, cwd: Path) -> Path:
    """Resolve user paths against a fixed invocation directory."""
    target = Path(path).expanduser()
    return (target if target.is_absolute() else cwd / target).resolve()


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and end of long tool output within a character budget."""
    if len(text) <= limit:
        return text
    marker = "\n... [output truncated] ...\n"
    remaining = limit - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head) :]
