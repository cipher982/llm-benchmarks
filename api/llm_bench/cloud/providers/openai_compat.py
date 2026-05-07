import time
from collections.abc import Sequence
from typing import Optional

from llm_bench.cloud.metrics import build_cloud_metrics
from openai import OpenAI
from tiktoken import get_encoding


def _attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _budget_candidates(requested_tokens: int) -> list[int]:
    requested_tokens = int(requested_tokens)
    candidates = [requested_tokens, max(requested_tokens, 256), max(requested_tokens, 512)]
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _message_extra(message) -> dict:
    if message is None:
        return {}
    return getattr(message, "model_extra", None) or {}


def _message_field(message, name: str):
    value = _attr(message, name)
    if value:
        return value
    return _message_extra(message).get(name)


def _reasoning_text(message) -> str:
    parts = []
    for field in ("reasoning", "reasoning_content"):
        value = _message_field(message, field)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def run_chat_completion_benchmark(
    *,
    client: OpenAI,
    model: str,
    query: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    request_mode: str = "openai_compat_chat_completions",
    retry_budgets: bool = True,
    fallback_extra_bodies: Sequence[tuple[str, dict, str | None]] = (),
) -> dict:
    """Run an OpenAI-compatible chat completion and collect benchmark metrics.

    Prefer provider-reported usage tokens. Several current hosted reasoning
    models spend the first part of the token budget on hidden reasoning and may
    emit few or no visible deltas, so counting streamed text chunks understates
    generated tokens.
    """

    encoder = get_encoding("cl100k_base")
    attempts: list[tuple[str, dict | None, str | None, int]] = []
    budgets = _budget_candidates(max_tokens) if retry_budgets else [max_tokens]
    for budget in budgets:
        attempts.append((request_mode, None, None, budget))
    for mode_suffix, extra_body, reasoning_effort in fallback_extra_bodies:
        fallback_mode = f"{request_mode}_{mode_suffix}" if mode_suffix else request_mode
        for budget in budgets:
            attempts.append((fallback_mode, extra_body, reasoning_effort, budget))

    last_metrics = None
    for attempt_mode, extra_body, reasoning_effort, budget in attempts:
        time_0 = time.time()
        request_params = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": budget,
        }
        if temperature is not None:
            request_params["temperature"] = temperature
        if extra_body:
            request_params["extra_body"] = extra_body

        response = client.chat.completions.create(**request_params)

        time_1 = time.time()
        generate_time = time_1 - time_0
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        response_str = message.content or "" if message else ""
        reasoning_str = _reasoning_text(message)

        usage = getattr(response, "usage", None)
        visible_tokens = len(encoder.encode(response_str))
        if usage is not None and getattr(usage, "completion_tokens", None) is not None:
            generated_tokens = int(usage.completion_tokens)
            completion_details = _attr(usage, "completion_tokens_details")
            reasoning_tokens = _attr(completion_details, "reasoning_tokens")
            if reasoning_tokens is not None:
                reasoning_tokens = int(reasoning_tokens)
                visible_tokens = max(generated_tokens - reasoning_tokens, 0)
            elif reasoning_str:
                reasoning_tokens = max(generated_tokens - visible_tokens, 0)
            token_source = "provider_usage_completion_tokens"
        else:
            reasoning_tokens = len(encoder.encode(reasoning_str)) if reasoning_str else None
            generated_tokens = visible_tokens + (reasoning_tokens or 0)
            token_source = "tiktoken_visible_plus_reasoning_text" if reasoning_str else "tiktoken_visible_text"

        prompt_details = _attr(usage, "prompt_tokens_details")
        finish_reason = _attr(choice, "finish_reason")
        last_metrics = build_cloud_metrics(
            requested_tokens=max_tokens,
            generated_output_tokens=generated_tokens,
            visible_output_tokens=visible_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=_attr(prompt_details, "cached_tokens"),
            input_tokens=_attr(usage, "prompt_tokens"),
            total_tokens=_attr(usage, "total_tokens"),
            generate_time=generate_time,
            # Non-streaming usage is the reliable denominator for current hosted
            # reasoning models. Treat TTFT as unavailable rather than inventing a
            # streamed timing from a second request.
            time_to_first_token=None,
            times_between_tokens=[],
            token_source=token_source,
            request_mode=attempt_mode,
            finish_reason=finish_reason,
            response_id=_attr(response, "id"),
            max_output_tokens_attempted=budget,
            reasoning_effort=reasoning_effort,
            visible_text_empty=not bool(response_str.strip()),
        )
        last_metrics.update(
            {
                "output_text": response_str,
                "reasoning_output_text": reasoning_str,
            }
        )
        if response_str.strip():
            visible_target_met = visible_tokens >= max(1, int(max_tokens * 0.8))
            if finish_reason != "length" or visible_target_met:
                return last_metrics

    assert last_metrics is not None
    return last_metrics
