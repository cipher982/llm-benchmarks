import logging
import os
import time
from datetime import datetime

from llm_bench_api.config import CloudConfig
from openai import OpenAI

logger = logging.getLogger(__name__)

# Azure-specific model mappings to OpenAI parameters
MODEL_NAME_MAPPING = {
    "llama-2-7b-chat": {
        "api_key_env": "AZURE_L7_API_KEY",
        "post_url_env": "AZURE_L7_POST_URL",
    },
    "llama-2-13b-chat": {
        "api_key_env": "AZURE_L13_API_KEY",
        "post_url_env": "AZURE_L13_POST_URL",
    },
    "llama-2-70b-chat": {
        "api_key_env": "AZURE_L70_API_KEY",
        "post_url_env": "AZURE_L70_POST_URL",
    },
    "mistral-large": {
        "api_key_env": "AZURE_MISTRAL_L_API_KEY",
        "post_url_env": "AZURE_MISTRAL_L_POST_URL",
    },
    "cohere-cmd-r-plus": {
        "api_key_env": "AZURE_COHERE_CMD_R_PLUS_API_KEY",
        "post_url_env": "AZURE_COHERE_CMD_R_PLUS_POST_URL",
    },
}


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Azure inference using OpenAI format and return metrics."""

    assert config.provider == "azure", "provider must be 'azure'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    model_mapping = MODEL_NAME_MAPPING.get(config.model_name)
    if not model_mapping:
        raise ValueError(f"Unsupported model_name: {config.model_name}")

    client = OpenAI(
        base_url=os.environ[model_mapping["post_url_env"]],
        api_key=os.environ[model_mapping["api_key_env"]],
    )

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_tokens = 0
    times_between_tokens = []
    time_to_first_token = 0

    completion = client.chat.completions.create(
        model="azureai",
        messages=[
            {"role": "system", "content": "You are a friendly AI."},
            {"role": "user", "content": run_config["query"]},
        ],
        max_tokens=run_config["max_tokens"],
        stream=True,
    )
    logger.debug(f"Completion: {completion}")

    for chunk in completion:
        logger.debug(f"Chunk: {chunk}")
        current_time = time.time()
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        else:
            assert previous_token_time is not None
            times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time
        output_tokens += 1

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
