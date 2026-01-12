"""
DEPRECATED: RunPod Serverless Provider

RunPod is a "deploy your own inference" platform, not a standard API.
Each model requires deploying a separate serverless endpoint, which:
- Requires managing your own GPU deployments
- Has per-endpoint cold start costs
- Isn't comparable to OpenAI/Anthropic-style "call model by name" APIs

This provider has been disabled as it doesn't fit the benchmarking model
of "call API with model name, get results".

If you want to benchmark RunPod, you'd need to:
1. Deploy a serverless endpoint for each model
2. Store endpoint IDs in the database as model_id
3. Pay for each endpoint's resources
"""

import json
import logging
import os
import time

import requests
from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp

logger = logging.getLogger(__name__)

# DEPRECATED: RunPod requires per-model serverless endpoints


def process_model(config, run_config):
    url = f"https://api.runpod.ai/v2/{config.model_name}/run"
    headers = {"Authorization": os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"}
    payload = {
        "input": {
            "prompt": run_config["query"],
            "sampling_params": {
                "max_tokens": run_config["max_tokens"],
                "n": 1,
                "temperature": 0.0,
            },
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response_json = json.loads(response.text)
    status_url = f"https://api.runpod.ai/v2/{config.model_name}/stream/{response_json['id']}"
    return status_url, headers


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run RunPod inference and return metrics."""

    assert config.provider == "runpod", "provider must be 'runpod'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_tokens = 0
    times_between_tokens = []
    time_to_first_token = 0

    status_url, headers = process_model(config, run_config)

    while True:
        get_status = requests.get(status_url, headers=headers)
        status_data = get_status.json()
        if status_data["status"] == "COMPLETED":
            break
        elif get_status.status_code != 200:
            raise ValueError("An error occurred.")
        else:
            current_time = time.time()
            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            else:
                assert previous_token_time is not None
                times_between_tokens.append(current_time - previous_token_time)
            previous_token_time = current_time
            output_tokens += status_data["stream"][0]["metrics"]["output_tokens"]

    time_1 = time.time()
    generate_time = time_1 - time_0
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": get_current_timestamp(),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }

    return metrics
