import logging
import os
import time
from datetime import datetime

import requests
from llm_bench_api.config import CloudConfig


logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Azure inference and return metrics."""

    assert config.provider == "azure", "provider must be azure"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Generate
    time_0 = time.time()
    # first_token_received = False
    # previous_token_time = None
    # time_to_first_token = None
    # output_tokens = 0
    # times_between_tokens = []

    api_key_mapping = {
        "llama-2-7b-chat": os.environ.get("AZURE_L7_API_KEY"),
        "llama-2-13b-chat": os.environ.get("AZURE_L13_API_KEY"),
        "llama-2-70b-chat": os.environ.get("AZURE_L70_API_KEY"),
    }

    name_url_mapping = {
        "llama-2-7b-chat": os.environ.get("AZURE_L7_POST_URL"),
        "llama-2-13b-chat": os.environ.get("AZURE_L13_POST_URL"),
        "llama-2-70b-chat": os.environ.get("AZURE_L70_POST_URL"),
    }

    api_key = api_key_mapping[config.model_name]
    model_url = name_url_mapping[config.model_name]

    assert api_key, f"API key not found for model_name: {config.model_name}"
    assert model_url, f"model_url not found for model_name: {config.model_name}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-type": "application/json",
    }

    data = {
        "prompt": run_config["query"],
        "temperature": config.temperature,
        "max_tokens": run_config["max_tokens"],
    }

    response = requests.post(
        url=model_url,
        headers=headers,
        json=data,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status code {response.status_code}: {response.text}")

    time_1 = time.time()
    generate_time = time_1 - time_0
    output_tokens = response.json()["usage"]["completion_tokens"]
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    assert output_tokens == run_config["max_tokens"], "output_tokens != max tokens"

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": -1,
        "times_between_tokens": [],
    }

    return metrics
