import json
import logging
import time
from datetime import datetime

import boto3
from llm_bench_api.config import CloudConfig

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run BedRock inference and return metrics."""

    assert config.provider == "bedrock", "provider must be anthropic"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    # Define the request bodies for different models
    request_bodies = {
        "anthropic": {
            "max_tokens": run_config["max_tokens"],
            "messages": [{"role": "user", "content": run_config["query"]}],
            "anthropic_version": "bedrock-2023-05-31",
        },
        "amazon": {
            "inputText": f"Human: {run_config['query']}. \n\nAssistant:",
            "textGenerationConfig": {"maxTokenCount": run_config["max_tokens"]},
        },
        "meta": {
            "prompt": f"{run_config['query']}",
            "max_gen_len": run_config["max_tokens"],
        },
        "mistral": {
            "prompt": f"{run_config['query']}",
            "max_tokens": run_config["max_tokens"],
        },
    }

    # Get the request body based on the model name
    model_type = next((key for key in request_bodies if key in config.model_name), None)
    if model_type is None:
        raise ValueError(f"Model {config.model_name} not supported")
    body = request_bodies[model_type]

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    output_tokens = 0
    times_between_tokens = []

    response = bedrock.invoke_model_with_response_stream(
        body=json.dumps(body),
        modelId=config.model_name,
    )
    stream = response.get("body")
    last_chunk = None
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                last_chunk = chunk
                current_time = time.time()
                if not first_token_received:
                    time_to_first_token = current_time - time_0
                    first_token_received = True
                else:
                    assert previous_token_time is not None
                    times_between_tokens.append(current_time - previous_token_time)
                previous_token_time = current_time

    if last_chunk:
        response_metrics = json.loads(last_chunk.get("bytes").decode()).get("amazon-bedrock-invocationMetrics", {})
        output_tokens = response_metrics.get("outputTokenCount")

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
