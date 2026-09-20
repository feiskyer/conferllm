"""Validation of native tool-call envelopes, independent of execution."""

from typing import Any


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a whole batch before any of its calls can have side effects."""
    if message.get("function_call") is not None:
        raise ValueError(
            "The provider must use native tool_calls, not legacy function_call."
        )
    calls = message.get("tool_calls")
    if calls is None:
        return []
    if not isinstance(calls, list):
        raise ValueError("Provider tool_calls must be an array.")
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("Each tool call must be a native function call.")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in seen:
            raise ValueError("Tool call IDs must be non-empty and unique.")
        seen.add(call_id)
        function = call.get("function")
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"].strip()
            or not isinstance(function.get("arguments"), str)
        ):
            raise ValueError(
                "Tool calls require a function name and JSON argument string."
            )
    return calls
