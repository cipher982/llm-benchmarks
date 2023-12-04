import logging
import time
from datetime import datetime

from llm_bench_api.config import CloudConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenAI inference and return metrics."""

    assert config.provider == "openai", "provider must be openai"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    client = OpenAI()

    # Generate
    time_0 = time.time()
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": run_config["query"]}],
        max_tokens=run_config["max_tokens"],
        stream=False,
    )
    time_1 = time.time()

    # Calculate metrics
    if response is None or response.usage is None:
        raise Exception("Response or response.usage is None")
    else:
        output_tokens = response.usage.completion_tokens
        generate_time = time_1 - time_0
        tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": [run_config["max_tokens"]],
        "output_tokens": [output_tokens],
        "generate_time": [time_1 - time_0],
        "tokens_per_second": [tokens_per_second],
    }

    return metrics
