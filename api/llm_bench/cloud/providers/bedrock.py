import logging
import time

import boto3
from botocore.exceptions import ClientError
from llm_bench.cloud.metrics import build_cloud_metrics
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
    input_tokens = None
    total_tokens = None
    finish_reason = None
    token_source = "stream_chunks"
    response_id = None

    try:
        response = bedrock_client.converse_stream(
            modelId=config.model_name,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields,
        )
        response_id = response.get("ResponseMetadata", {}).get("RequestId")

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
                elif "messageStop" in event:
                    finish_reason = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    metadata = event["metadata"]
                    if "usage" in metadata and "outputTokens" in metadata["usage"]:
                        usage = metadata["usage"]
                        output_tokens = usage["outputTokens"]
                        input_tokens = usage.get("inputTokens")
                        total_tokens = usage.get("totalTokens")
                        token_source = "provider_usage_output_tokens"

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"Error: {message}")
        raise

    time_1 = time.time()
    generate_time = time_1 - time_0

    metrics = build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=output_tokens,
        visible_output_tokens=output_tokens,
        reasoning_tokens=None,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        generate_time=generate_time,
        time_to_first_token=time_to_first_token,
        times_between_tokens=times_between_tokens,
        token_source=token_source,
        request_mode="bedrock_converse_stream",
        finish_reason=finish_reason,
        response_id=response_id,
        max_output_tokens_attempted=run_config["max_tokens"],
    )

    return metrics
