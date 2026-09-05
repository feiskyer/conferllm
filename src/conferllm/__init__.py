"""ConferLLM - Unified local access to configured AI models."""

from .chat import ChatResult, ChatService
from .client import LLMClient
from .config import ConferLLMConfig, ModelConfig
from .session import LoadedSession, SessionMetadata, SessionStore

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

__version__ = "0.1.0"
