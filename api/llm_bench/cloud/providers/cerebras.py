import logging
import os
import time
from typing import Optional

from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig
from openai import OpenAI
from tiktoken import get_encoding

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"


def _get_client() -> OpenAI:
    base_url = os.environ.get("CEREBRAS_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set")

    return OpenAI(base_url=base_url, api_key=api_key)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Cerebras inference and return benchmark metrics."""

    assert config.provider == "cerebras", "provider must be 'cerebras'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = _get_client()

    time_0 = time.time()
    first_token_received = False
    previous_token_time: Optional[float] = None
    time_to_first_token = 0.0
    times_between_tokens = []
    response_chunks: list[str] = []
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason = None
    response_id = None

    stream = client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": run_config["query"]},
        ],
        max_completion_tokens=run_config["max_tokens"],
        temperature=config.temperature,
        stream=True,
    )

    for chunk in stream:
        # Capture usage metadata if provided on the final chunk
        usage = getattr(chunk, "usage", None)
        if usage and getattr(usage, "completion_tokens", None) is not None:
            completion_tokens = usage.completion_tokens
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        if getattr(chunk, "id", None):
            response_id = chunk.id

        if not chunk.choices:
            continue

        if getattr(chunk.choices[0], "finish_reason", None):
            finish_reason = chunk.choices[0].finish_reason

        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue

        current_time = time.time()
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        else:
            if previous_token_time is not None:
                times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time

        response_chunks.append(content)

    time_1 = time.time()
    generate_time = time_1 - time_0

    response_text = "".join(response_chunks)
    if not first_token_received:
        time_to_first_token = generate_time
    if completion_tokens is not None:
        output_tokens = int(completion_tokens)
        token_source = "provider_usage_completion_tokens"
    else:
        encoder = get_encoding("cl100k_base")
        output_tokens = len(encoder.encode(response_text))
        token_source = "tiktoken_visible_text"

    metrics = build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=output_tokens,
        visible_output_tokens=output_tokens,
        reasoning_tokens=0,
        input_tokens=prompt_tokens,
        total_tokens=total_tokens,
        generate_time=generate_time,
        time_to_first_token=time_to_first_token if first_token_received else None,
        times_between_tokens=times_between_tokens,
        token_source=token_source,
        request_mode="cerebras_chat_completions_stream",
        finish_reason=finish_reason,
        response_id=response_id,
        max_output_tokens_attempted=run_config["max_tokens"],
        visible_text_empty=not bool(response_text.strip()),
    )
    metrics.update(
        {
            "output_text": response_text,
        }
    )

    return metrics
