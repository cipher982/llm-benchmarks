import logging
import time
from datetime import datetime

from anthropic import AI_PROMPT
from anthropic import Anthropic
from anthropic import HUMAN_PROMPT
from llm_bench_api.config import CloudConfig


logger = logging.getLogger(__name__)


NON_CHAT_MODELS = []


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Anthropic inference and return metrics."""

    assert config.provider == "anthropic", "provider must be openai"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    anthropic = Anthropic()

    # Generate
    time_0 = time.time()
    if config.model_name in NON_CHAT_MODELS:
        raise NotImplementedError
    else:
        response = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=run_config["max_tokens"],
            prompt=f"{HUMAN_PROMPT} {run_config['query']} {AI_PROMPT}",
        )
    time_1 = time.time()

    # Calculate metrics
    if response is None or not response.completion:
        raise Exception("No response")
    else:
        output_tokens = anthropic.count_tokens(response.completion)
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
