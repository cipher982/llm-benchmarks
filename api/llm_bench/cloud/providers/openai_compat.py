import time
from typing import Optional

from llm_bench.utils import get_current_timestamp
from openai import OpenAI
from tiktoken import get_encoding


def run_chat_completion_benchmark(
    *,
    client: OpenAI,
    model: str,
    query: str,
    max_tokens: int,
    temperature: Optional[float] = None,
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
    if usage is not None and getattr(usage, "completion_tokens", None) is not None:
        completion_tokens = int(usage.completion_tokens)
    else:
        encoder = get_encoding("cl100k_base")
        completion_tokens = len(encoder.encode(response_str))

    tokens_per_second = completion_tokens / generate_time if generate_time > 0 else 0

    return {
        "gen_ts": get_current_timestamp(),
        "requested_tokens": max_tokens,
        "output_tokens": completion_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        # Non-streaming usage is the reliable denominator for current hosted
        # reasoning models. Treat TTFT as unavailable rather than inventing a
        # streamed timing from a second request.
        "time_to_first_token": None,
        "times_between_tokens": [],
        "output_text": response_str,
    }
