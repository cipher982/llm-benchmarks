import logging
import os
import time

import openai
import vertexai
from anthropic import AnthropicVertex
from google.auth import default
from google.auth import transport
from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

PROJECT_ID = "llm-bench"
REGION = "us-central1"
SECONDARY_REGION = "us-east5"
MAAS_ENDPOINT = f"{REGION}-aiplatform.googleapis.com"

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID


def _get_openai_client():
    """Get an OpenAI client configured for Vertex AI."""
    credentials, _ = default()
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)

    return openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi",
        api_key=credentials.token,
    )


def generate(config: CloudConfig, run_config: dict) -> dict:
    assert config.provider == "vertex", "provider must be Vertex"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    vertexai.init(project=PROJECT_ID)

    if "llama" in config.model_name.lower():
        logger.debug("Using Vertex/OpenAI API for Llama model")
        client = _get_openai_client()
        time_0 = time.time()
        stream = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": run_config["query"]}],
            max_tokens=run_config["max_tokens"],
            stream=True,
        )
        ttft, tbts, token_details = generate_tokens(stream, time_0, False, is_openai=True)
        generate_time = time.time() - time_0
        request_mode = "vertex_openai_chat_completions_stream"
    elif "claude" not in config.model_name.lower():
        logger.debug("Using Vertex/GenerativeModel")
        model = GenerativeModel(config.model_name)
        time_0 = time.time()
        stream = model.generate_content(
            contents=run_config["query"],
            generation_config=GenerationConfig(max_output_tokens=run_config["max_tokens"]),
            stream=True,
        )
        ttft, tbts, token_details = generate_tokens(stream, time_0, False)
        generate_time = time.time() - time_0
        request_mode = "vertex_gemini_generate_content_stream"
    else:
        logger.debug("Using Vertex/AnthropicVertex")
        # Use global endpoint for Claude 4/4.5 models (newer models only available globally)
        # Use us-east5 for opus/3-5 models, us-central1 for everything else
        model_lower = config.model_name.lower()
        if any(keyword in model_lower for keyword in ["claude-4", "sonnet-4", "opus-4"]):
            region = "global"  # Claude 4.x models use global endpoint
        elif any(keyword in model_lower for keyword in ["opus", "3-5"]):
            region = SECONDARY_REGION  # us-east5
        else:
            region = REGION  # us-central1
        client = AnthropicVertex(region=region, project_id=PROJECT_ID)
        time_0 = time.time()
        with client.messages.stream(
            max_tokens=run_config["max_tokens"],
            messages=[{"role": "user", "content": run_config["query"]}],
            model=config.model_name,
        ) as stream:
            ttft, tbts, token_details = generate_tokens(stream, time_0, True)
        generate_time = time.time() - time_0
        request_mode = "vertex_anthropic_messages_stream"

    return calculate_metrics(run_config, token_details, generate_time, ttft, tbts, request_mode)


def generate_tokens(stream, time_0, is_anthropic=False, is_openai=False):
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    times_between_tokens = []
    token_details = {
        "generated_output_tokens": 0,
        "visible_output_tokens": None,
        "reasoning_tokens": None,
        "input_tokens": None,
        "total_tokens": None,
        "token_source": "stream_chunks",
        "finish_reason": None,
        "response_id": None,
    }

    stream_iter = stream if is_anthropic or is_openai else stream

    item = None
    for item in stream_iter:
        current_time = time.time()
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        else:
            assert previous_token_time is not None
            times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time

    assert item, "No tokens received"

    if is_anthropic:
        usage = item.message.usage
        token_details.update(
            {
                "generated_output_tokens": usage.output_tokens,
                "visible_output_tokens": usage.output_tokens,
                "input_tokens": usage.input_tokens,
                "token_source": "provider_usage_output_tokens",
                "finish_reason": item.message.stop_reason,
                "response_id": item.message.id,
            }
        )
    elif is_openai:
        usage = item.usage
        token_details.update(
            {
                "generated_output_tokens": usage.completion_tokens,
                "visible_output_tokens": usage.completion_tokens,
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "token_source": "provider_usage_completion_tokens",
                "response_id": getattr(item, "id", None),
            }
        )
        if getattr(item, "choices", None):
            token_details["finish_reason"] = getattr(item.choices[0], "finish_reason", None)
    else:
        # Include both visible output tokens and thinking tokens (for Gemini 2.5+ thinking models)
        usage = item._raw_response.usage_metadata
        visible_tokens = usage.candidates_token_count
        # Add thinking tokens if present (Gemini 2.5+ models)
        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        if thoughts_tokens:
            logger.debug(f"Model used {thoughts_tokens} thinking tokens + {visible_tokens} output tokens")
        token_details.update(
            {
                "generated_output_tokens": visible_tokens + thoughts_tokens,
                "visible_output_tokens": visible_tokens,
                "reasoning_tokens": thoughts_tokens,
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
                "token_source": "provider_usage_candidates_plus_thoughts",
            }
        )
        candidates = getattr(item, "candidates", None)
        if candidates:
            token_details["finish_reason"] = getattr(candidates[0], "finish_reason", None)

    return time_to_first_token, times_between_tokens, token_details


def calculate_metrics(
    run_config,
    token_details,
    generate_time,
    time_to_first_token,
    times_between_tokens,
    request_mode,
):
    return build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=token_details["generated_output_tokens"],
        visible_output_tokens=token_details["visible_output_tokens"],
        reasoning_tokens=token_details["reasoning_tokens"],
        input_tokens=token_details["input_tokens"],
        total_tokens=token_details["total_tokens"],
        generate_time=generate_time,
        time_to_first_token=time_to_first_token,
        times_between_tokens=times_between_tokens,
        token_source=token_details["token_source"],
        request_mode=request_mode,
        finish_reason=token_details["finish_reason"],
        response_id=token_details["response_id"],
        max_output_tokens_attempted=run_config["max_tokens"],
    )
