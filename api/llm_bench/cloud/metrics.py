"""Shared cloud benchmark metric contract.

Schema v2 keeps the legacy `output_tokens` and `tokens_per_second` fields for
dashboard compatibility. For v2 rows they are aliases for generated output
tokens when provider usage reports them; otherwise they are a visible-text
fallback and `token_source` must make that limitation explicit.
"""

import logging
from typing import Any

from llm_bench.utils import get_current_timestamp

METRICS_SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)


def build_cloud_metrics(
    *,
    requested_tokens: int,
    generated_output_tokens: int,
    generate_time: float,
    time_to_first_token: float | None,
    times_between_tokens: list[float],
    token_source: str,
    request_mode: str,
    visible_output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    input_tokens: int | None = None,
    total_tokens: int | None = None,
    finish_reason: str | None = None,
    response_id: str | None = None,
    response_status: str | None = None,
    max_output_tokens_attempted: int | None = None,
    reasoning_effort: str | None = None,
    validation_policy: str | None = None,
    visible_text_empty: bool | None = None,
) -> dict[str, Any]:
    """Build a backwards-compatible cloud metrics document.

    `output_tokens` and `tokens_per_second` remain the legacy dashboard fields.
    For schema v2 rows they mean total generated output tokens when the provider
    reports that value, including reasoning/thinking tokens.
    """

    generated_output_tokens = int(generated_output_tokens)
    if (
        visible_output_tokens is not None
        and reasoning_tokens is not None
        and visible_output_tokens + reasoning_tokens > generated_output_tokens
    ):
        logger.warning(
            "Token split exceeds generated output: visible=%s reasoning=%s generated=%s source=%s mode=%s",
            visible_output_tokens,
            reasoning_tokens,
            generated_output_tokens,
            token_source,
            request_mode,
        )
    visible_tps = None
    if visible_output_tokens is not None:
        visible_tps = visible_output_tokens / generate_time if generate_time > 0 else 0
    generated_tps = generated_output_tokens / generate_time if generate_time > 0 else 0

    metrics: dict[str, Any] = {
        "gen_ts": get_current_timestamp(),
        "requested_tokens": requested_tokens,
        "output_tokens": generated_output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": generated_tps,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "generated_output_tokens": generated_output_tokens,
        "visible_output_tokens": visible_output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "generated_tokens_per_second": generated_tps,
        "visible_tokens_per_second": visible_tps,
        "token_source": token_source,
        "request_mode": request_mode,
        "ttft_available": time_to_first_token is not None,
    }

    def as_str(value):
        return None if value is None else str(value)

    optional = {
        "cached_input_tokens": cached_input_tokens,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "finish_reason": as_str(finish_reason),
        "response_id": as_str(response_id),
        "response_status": as_str(response_status),
        "max_output_tokens_attempted": max_output_tokens_attempted,
        "reasoning_effort": as_str(reasoning_effort),
        "validation_policy": as_str(validation_policy),
        "visible_text_empty": visible_text_empty,
    }
    metrics.update({key: value for key, value in optional.items() if value is not None})
    return metrics
