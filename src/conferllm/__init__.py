"""ConferLLM - Unified local access to configured AI models.

Public objects are loaded on demand so session-only worker processes do not
import the provider SDK and build its model catalog before acquiring a lock.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat import ChatResult, ChatService
    from .client import LLMClient
    from .config import ConferLLMConfig, ModelConfig
    from .session import LoadedSession, SessionMetadata, SessionStore

_EXPORT_MODULES = {
    "ChatResult": ".chat",
    "ChatService": ".chat",
    "LLMClient": ".client",
    "ConferLLMConfig": ".config",
    "ModelConfig": ".config",
    "LoadedSession": ".session",
    "SessionMetadata": ".session",
    "SessionStore": ".session",
}

__all__ = [
    "ConferLLMConfig",
    "ChatResult",
    "ChatService",
    "LLMClient",
    "LoadedSession",
    "ModelConfig",
    "SessionMetadata",
    "SessionStore",
    "__version__",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))
