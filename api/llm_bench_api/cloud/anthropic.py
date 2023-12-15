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

    assert config.provider == "anthropic", "provider must be anthropic"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    anthropic = Anthropic()

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_tokens = 0
    times_between_tokens = []

    if config.model_name in NON_CHAT_MODELS:
        raise NotImplementedError
    else:
        stream = anthropic.completions.create(
            model=config.model_name,
            max_tokens_to_sample=run_config["max_tokens"],
            prompt=f"{HUMAN_PROMPT} {run_config['query']} {AI_PROMPT}",
            stream=True,
        )

    time_to_first_token = None
    for completion in stream:
        if completion.completion is not None:
            current_time = time.time()
            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            else:
                assert previous_token_time is not None
                times_between_tokens.append(current_time - previous_token_time)
            previous_token_time = current_time
            output_tokens += len(completion.completion.split())

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
