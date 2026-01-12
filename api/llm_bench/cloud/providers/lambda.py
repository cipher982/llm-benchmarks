"""
DEPRECATED: Lambda Labs Inference API

Lambda Labs sunset their Inference API on September 25, 2025.
The service no longer exists. See:
https://deeptalk.lambda.ai/t/sunsetting-chat-sunsetting-inference/4744

All Lambda models have been disabled in the database.
This file is kept for historical reference only.

If Lambda brings back an inference API in the future, this code
may serve as a starting point, but will likely need updates.
"""

import logging
import os
import time
from typing import Any
from typing import Dict
from typing import Tuple

from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp
from openai import OpenAI

logger = logging.getLogger(__name__)

# DEPRECATED: Lambda Inference API sunset September 25, 2025


def process_stream_response(stream, start_time: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    response_text = ""
    output_tokens = 0
    time_to_first_token = None
    times_between_tokens = []
    last_token_time = None

    for chunk in stream:
        current_time = time.time()
        output_tokens += 1

        if time_to_first_token is None:
            time_to_first_token = current_time - start_time
        elif last_token_time:
            times_between_tokens.append(current_time - last_token_time)

        last_token_time = current_time

        if chunk.choices and chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content

    # Check if tokens received is within 20% of requested
    if abs(output_tokens - max_tokens) > (max_tokens * 0.2):
        raise ValueError(f"Received {output_tokens} tokens, expected around {max_tokens}")

    metrics = {
        "gen_ts": get_current_timestamp(),
        "output_tokens": output_tokens,
        "generate_time": time.time() - start_time,
        "tokens_per_second": output_tokens / (time.time() - start_time),
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }

    return response_text, metrics


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Lambda inference and return metrics."""
    assert config.provider == "lambda", "provider must be 'lambda'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = OpenAI(
        base_url=os.environ["LAMBDA_BASE_URL"],
        api_key=os.environ["LAMBDA_API_KEY"],
    )

    start_time = time.time()
    stream = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": run_config["query"]}],
        max_tokens=run_config["max_tokens"],
        stream=True,
    )

    _, metrics = process_stream_response(stream, start_time, run_config["max_tokens"])
    metrics["requested_tokens"] = run_config["max_tokens"]

    return metrics
