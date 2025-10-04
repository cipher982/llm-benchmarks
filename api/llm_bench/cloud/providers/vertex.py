import logging
import os
import time
from datetime import datetime

import openai
import vertexai
from anthropic import AnthropicVertex
from google.auth import default
from google.auth import transport
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
        ttft, tbts, n_tokens = generate_tokens(stream, time_0, False, is_openai=True)
        generate_time = time.time() - time_0
    elif "claude" not in config.model_name.lower():
        logger.debug("Using Vertex/GenerativeModel")
        model = GenerativeModel(config.model_name)
        time_0 = time.time()
        stream = model.generate_content(
            contents=run_config["query"],
            generation_config=GenerationConfig(max_output_tokens=run_config["max_tokens"]),
            stream=True,
        )
        ttft, tbts, n_tokens = generate_tokens(stream, time_0, False)
        generate_time = time.time() - time_0
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
            ttft, tbts, n_tokens = generate_tokens(stream, time_0, True)
        generate_time = time.time() - time_0

    return calculate_metrics(run_config, n_tokens, generate_time, ttft, tbts)


def generate_tokens(stream, time_0, is_anthropic=False, is_openai=False):
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    times_between_tokens = []
    token_count = 0

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
        token_count = item.message.usage.output_tokens
    elif is_openai:
        token_count = item.usage.completion_tokens
    else:
        token_count = item._raw_response.usage_metadata.candidates_token_count

    return time_to_first_token, times_between_tokens, token_count


def calculate_metrics(run_config, output_tokens, generate_time, time_to_first_token, times_between_tokens):
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    return {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }
