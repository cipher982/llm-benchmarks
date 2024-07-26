import logging
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from llm_bench.config import CloudConfig

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run BedRock inference and return metrics."""

    assert config.provider == "bedrock", "provider must be bedrock"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # For some reason newer or bigger models start only in us-west-2
    REGION_MAP = {"opus": "us-west-2", "llama3-1": "us-west-2"}
    region_name = next((REGION_MAP[k] for k in REGION_MAP if k in config.model_name.lower()), "us-east-1")

    # Set up connection
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
    )

    # Prepare the messages
    messages = [{"role": "user", "content": [{"text": run_config["query"]}]}]

    # Prepare system prompts (if needed)
    system_prompts = []

    # Prepare inference config
    inference_config = {"temperature": config.temperature, "maxTokens": run_config["max_tokens"]}

    # Additional model fields
    additional_model_fields = {}

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    output_tokens = 0
    times_between_tokens = []

    try:
        response = bedrock_client.converse_stream(
            modelId=config.model_name,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields,
        )

        stream = response.get("stream")
        if stream:
            for event in stream:
                current_time = time.time()
                if "contentBlockDelta" in event:
                    if not first_token_received:
                        time_to_first_token = current_time - time_0
                        first_token_received = True
                    else:
                        assert previous_token_time is not None
                        times_between_tokens.append(current_time - previous_token_time)
                    previous_token_time = current_time
                    output_tokens += 1

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"Error: {message}")
        raise

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
