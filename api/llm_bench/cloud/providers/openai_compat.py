import time
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


def run_chat_completion_benchmark(
    *,
    client: OpenAI,
    model: str,
    query: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    request_mode: str = "openai_compat_chat_completions",
) -> dict:
    """Run an OpenAI-compatible chat completion and collect benchmark metrics.

    Prefer provider-reported usage tokens. Several current hosted reasoning
    models spend the first part of the token budget on hidden reasoning and may
    emit few or no visible deltas, so counting streamed text chunks understates
    generated tokens.
    """

    time_0 = time.time()

    request_params = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)

    time_1 = time.time()
    generate_time = time_1 - time_0
    response_str = ""
    if response.choices and response.choices[0].message:
        response_str = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    encoder = get_encoding("cl100k_base")
    visible_tokens = len(encoder.encode(response_str))
    if usage is not None and getattr(usage, "completion_tokens", None) is not None:
        generated_tokens = int(usage.completion_tokens)
        completion_details = _attr(usage, "completion_tokens_details")
        reasoning_tokens = _attr(completion_details, "reasoning_tokens")
        if reasoning_tokens is not None:
            reasoning_tokens = int(reasoning_tokens)
            visible_tokens = max(generated_tokens - reasoning_tokens, 0)
        token_source = "provider_usage_completion_tokens"
    else:
        generated_tokens = visible_tokens
        reasoning_tokens = None
        token_source = "tiktoken_visible_text"

    choice = response.choices[0] if response.choices else None
    prompt_details = _attr(usage, "prompt_tokens_details")
    metrics = build_cloud_metrics(
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
        request_mode=request_mode,
        finish_reason=_attr(choice, "finish_reason"),
        response_id=_attr(response, "id"),
        max_output_tokens_attempted=max_tokens,
        visible_text_empty=not bool(response_str.strip()),
    )
    metrics.update(
        {
            "output_text": response_str,
        }
    )
    return metrics
