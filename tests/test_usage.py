"""Provider-neutral token and prompt-cache accounting."""

from __future__ import annotations

from conferllm.responses import to_model_response
from conferllm.usage import TokenUsage, normalize_usage


def test_anthropic_cache_reads_and_writes_count_as_input() -> None:
    usage = {
        "prompt_tokens": 1300,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 200},
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
    }

    assert normalize_usage(usage) == TokenUsage(
        input_tokens=1300,
        cached_input_tokens=1000,
        cache_write_tokens=200,
        output_tokens=20,
    )


def test_top_level_cache_fields_are_used_without_prompt_details() -> None:
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 5,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 10,
    }

    assert normalize_usage(usage) == TokenUsage(50, 30, 10, 5)


def test_responses_usage_keeps_cache_reads_and_writes() -> None:
    raw = {
        "id": "resp_usage",
        "model": "gpt-6-astra",
        "created_at": 1,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 2000,
            "output_tokens": 10,
            "total_tokens": 2010,
            "input_tokens_details": {"cached_tokens": 1500, "cache_write_tokens": 300},
        },
    }

    usage = to_model_response(raw).model_dump()["usage"]

    assert normalize_usage(usage) == TokenUsage(2000, 1500, 300, 10)


def test_missing_usage_differs_from_reported_zero_usage() -> None:
    assert normalize_usage(None) is None
    assert normalize_usage("not usage") is None
    assert normalize_usage({}) == TokenUsage()


def test_non_integer_counts_are_ignored() -> None:
    usage = {
        "prompt_tokens": True,
        "completion_tokens": "5",
        "prompt_tokens_details": {"cached_tokens": None},
        "cache_read_input_tokens": 3,
    }

    assert normalize_usage(usage) == TokenUsage(0, 3, 0, 0)


def test_totals_add_across_calls_and_report_the_hit_ratio() -> None:
    total = TokenUsage(1000, 0, 1000, 10) + TokenUsage(1200, 1000, 200, 20)

    assert total == TokenUsage(2200, 1000, 1200, 30)
    assert total.to_dict() == {
        "input_tokens": 2200,
        "cached_input_tokens": 1000,
        "cache_write_tokens": 1200,
        "output_tokens": 30,
        "cache_hit_ratio": 0.4545,
    }
    assert TokenUsage().cache_hit_ratio is None
