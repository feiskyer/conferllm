"""Provider-neutral token accounting, including prompt-cache reads and writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Token totals where input_tokens includes cache reads and cache writes."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def cache_hit_ratio(self) -> float | None:
        """Return the share of input tokens read from the provider cache."""
        if not self.input_tokens:
            return None
        return round(self.cached_input_tokens / self.input_tokens, 4)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible totals and the cache hit ratio."""
        return {**asdict(self), "cache_hit_ratio": self.cache_hit_ratio}


def normalize_usage(usage: Any) -> TokenUsage | None:
    """Normalize LiteLLM usage, or None when the provider reported none.

    LiteLLM already folds Anthropic and Bedrock cache reads and writes into
    ``prompt_tokens``; OpenAI, DeepSeek, and Gemini count cached tokens within
    it. ConferLLM maps Responses usage to the same names before this runs.
    """
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        details = {}
    return TokenUsage(
        input_tokens=_first_count(usage.get("prompt_tokens")),
        cached_input_tokens=_first_count(
            details.get("cached_tokens"), usage.get("cache_read_input_tokens")
        ),
        cache_write_tokens=_first_count(
            details.get("cache_write_tokens"),
            details.get("cache_creation_tokens"),
            usage.get("cache_creation_input_tokens"),
        ),
        output_tokens=_first_count(usage.get("completion_tokens")),
    )


def _first_count(*values: Any) -> int:
    return next(
        (
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        ),
        0,
    )
