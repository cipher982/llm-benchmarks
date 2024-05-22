import logging
import os
import time
from datetime import datetime

from llm_bench_api.config import CloudConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Anyscale inference and return metrics."""

    assert config.provider == "anyscale", "provider must be Anyscale"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    client = OpenAI(
        base_url=os.environ["ANYSCALE_BASE_URL"],
        api_key=os.environ["ANYSCALE_API_KEY"],
    )

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_chunks = 0
    times_between_tokens = []
    time_to_first_token = 0
    response_str = ""

    stream = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": run_config["query"]}],
        max_tokens=run_config["max_tokens"],
        stream=True,
    )

    for chunk in stream:
        response = chunk.choices[0].delta  # type: ignore
        response_content = response.content if response is not None else None

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

    time_1 = time.time()
    generate_time = time_1 - time_0

    # Calculate tokens
    output_tokens = chunk.usage.completion_tokens  # type: ignore
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }

    return metrics
