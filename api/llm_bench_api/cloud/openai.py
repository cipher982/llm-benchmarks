import logging
import time
from datetime import datetime

from llm_bench_api.config import CloudConfig
from openai import OpenAI


logger = logging.getLogger(__name__)


NON_CHAT_MODELS = ["gpt-3.5-turbo-instruct"]


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenAI inference and return metrics."""

    assert config.provider == "openai", "provider must be openai"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    client = OpenAI()

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_tokens = 0
    times_between_tokens = []

    if config.model_name in NON_CHAT_MODELS:
        stream = client.completions.create(
            model=config.model_name,
            prompt=run_config["query"],
            max_tokens=run_config["max_tokens"],
            stream=True,
        )
    else:
        stream = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": run_config["query"]}],
            max_tokens=run_config["max_tokens"],
            stream=True,
        )

    time_to_first_token = None
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            current_time = time.time()
            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            else:
                assert previous_token_time is not None
                times_between_tokens.append(current_time - previous_token_time)
            previous_token_time = current_time
            output_tokens += len(chunk.choices[0].delta.content.split())

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
    }

    return metrics
