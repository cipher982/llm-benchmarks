import logging
import time

import boto3
from botocore.exceptions import ClientError
from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig
from tiktoken import get_encoding

logger = logging.getLogger(__name__)


def _omit_temperature(model_name: str, reasoning_effort: str | None) -> bool:
    """Some Anthropic Bedrock modes reject sampling controls."""
    normalized = model_name.lower()
    return bool(reasoning_effort) or "claude-opus-4-7" in normalized


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

    # Additional model fields let us exercise provider-specific features such
    # as Anthropic extended thinking while keeping the normal runner generic.
    additional_model_fields = {
        **(config.misc.get("additional_model_request_fields", {}) if isinstance(config.misc, dict) else {}),
        **run_config.get("additional_model_request_fields", {}),
    }
    thinking_config = additional_model_fields.get("thinking") if isinstance(additional_model_fields, dict) else None
    reasoning_effort = None
    if isinstance(thinking_config, dict):
        reasoning_effort = thinking_config.get("type")

    # Prepare inference config. Anthropic extended thinking rejects modified
    # sampling controls, so omit temperature when thinking is enabled.
    inference_config = {"maxTokens": run_config["max_tokens"]}
    if not _omit_temperature(config.model_name, reasoning_effort):
        inference_config["temperature"] = config.temperature

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
    visible_text_parts = []
    reasoning_text_parts = []
    provider_usage_seen = False

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
                    delta = event["contentBlockDelta"].get("delta", {})
                    text_delta = delta.get("text")
                    reasoning_delta = delta.get("reasoningContent")

                    if text_delta:
                        if not first_token_received:
                            time_to_first_token = current_time - time_0
                            first_token_received = True
                        else:
                            assert previous_token_time is not None
                            times_between_tokens.append(current_time - previous_token_time)
                        previous_token_time = current_time
                        visible_text_parts.append(text_delta)
                        output_tokens += 1

                    if reasoning_delta:
                        reasoning_text = reasoning_delta.get("text")
                        if reasoning_text is None and isinstance(reasoning_delta.get("reasoningText"), dict):
                            reasoning_text = reasoning_delta["reasoningText"].get("text")
                        if reasoning_text:
                            reasoning_text_parts.append(reasoning_text)
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
                        provider_usage_seen = True

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"Error: {message}")
        raise

    time_1 = time.time()
    generate_time = time_1 - time_0
    visible_text = "".join(visible_text_parts)
    reasoning_text = "".join(reasoning_text_parts)

    visible_tokens = output_tokens
    reasoning_tokens = None
    if reasoning_text:
        encoder = get_encoding("cl100k_base")
        visible_tokens = len(encoder.encode(visible_text))
        if provider_usage_seen:
            reasoning_tokens = max(output_tokens - visible_tokens, 0)
            token_source = f"{token_source}_with_tiktoken_visible_split"
        else:
            reasoning_tokens = len(encoder.encode(reasoning_text))
            output_tokens = visible_tokens + reasoning_tokens
            token_source = "tiktoken_visible_plus_reasoning_text"

    metrics = build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=output_tokens,
        visible_output_tokens=visible_tokens,
        reasoning_tokens=reasoning_tokens,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        generate_time=generate_time,
        time_to_first_token=time_to_first_token,
        times_between_tokens=times_between_tokens,
        token_source=token_source,
        request_mode="bedrock_converse_stream",
        reasoning_effort=reasoning_effort,
        finish_reason=finish_reason,
        response_id=response_id,
        max_output_tokens_attempted=run_config["max_tokens"],
    )

    return metrics
