import logging
import os
import time
from datetime import datetime

from llm_bench.config import CloudConfig
from openai import OpenAI
from tiktoken import get_encoding

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Fireworks inference and return metrics."""

    assert config.provider == "fireworks", "provider must be 'fireworks'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    client = OpenAI(
        base_url=os.environ["FIREWORKS_BASE_URL"],
        api_key=os.environ["FIREWORKS_API_KEY"],
    )

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_chunks = 0
    output_tokens = 0
    times_between_tokens = []
    time_to_first_token = 0
    response_str = ""

    response = client.completions.create(
        model=config.model_name,
        prompt=run_config["query"],
        max_tokens=run_config["max_tokens"],
        stream=True,
    )

    for chunk in response:
        response_content = chunk.choices[0].text
        if response_content is not None:
            current_time = time.time()
            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            else:
                assert previous_token_time is not None
                times_between_tokens.append(current_time - previous_token_time)
            previous_token_time = current_time
            response_str += response_content
            output_chunks += 1
            if len(chunk.choices) != 1:
                raise ValueError("Unexpected number of choices")
        else:
            logger.warning(f"Received empty content chunk: {chunk}")

    output_tokens = count_tokens(run_config["max_tokens"], response_str)

    time_1 = time.time()
    generate_time = time_1 - time_0
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
        "output_text": response_str,
    }

    return metrics


def count_tokens(max_tokens: int, response_str: str) -> int:
    encoder = get_encoding("cl100k_base")
    n_tokens = len(encoder.encode(response_str))
    if not 0.8 * max_tokens <= n_tokens <= 1.2 * max_tokens:
        raise ValueError(f"N Tokens {n_tokens} not within 20% of max tokens {max_tokens}")
    return max_tokens
