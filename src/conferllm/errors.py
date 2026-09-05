"""Stable public error contract for ConferLLM interfaces."""

from __future__ import annotations

from typing import Any

ERROR_SCHEMA_VERSION = "conferllm.error.v1"


class ConferLLMError(RuntimeError):
    """An error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Include the stable code in protocol-facing error messages."""
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON error envelope."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error["details"] = self.details
        return {
            "schema_version": ERROR_SCHEMA_VERSION,
            "ok": False,
            "error": error,
        }


def normalize_error(error: Exception) -> ConferLLMError:
    """Map internal exceptions to the stable public error vocabulary."""
    if isinstance(error, ConferLLMError):
        return error

    explicit_code = getattr(error, "code", None)
    if isinstance(explicit_code, str) and explicit_code:
        details = getattr(error, "details", None)
        return ConferLLMError(
            explicit_code,
            _public_message(error),
            details=details if isinstance(details, dict) else None,
        )

    message = _public_message(error)
    # Domain errors carry explicit codes. Never infer them from prose: an
    # upstream failure mentioning "model not found" is still a provider failure,
    # and a filesystem error is not a model lookup error.
    if isinstance(error, OSError):
        code = "storage_error"
    elif isinstance(error, (TypeError, ValueError)):
        code = "invalid_request"
    else:
        code = "provider_error"

    return ConferLLMError(code, message)


def _public_message(error: Exception) -> str:
    message = str(error).strip()
    return message or type(error).__name__
