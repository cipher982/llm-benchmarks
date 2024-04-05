import logging
import os
import time
from datetime import datetime

import tiktoken
import vertexai
from anthropic import AnthropicVertex
from llm_bench_api.config import CloudConfig
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel

logger = logging.getLogger(__name__)


PROJECT_ID = "llm-bench"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID


v1_models = ["text-bison@002", "chat-bison@002"]
anthropic_models = ["claude-3-sonnet@20240229"]


def generate(config: CloudConfig, run_config: dict) -> dict:
    assert config.provider == "vertex", "provider must be Vertex"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    if config.model_name in anthropic_models:
        return generate_anthropic(config, run_config)
    else:
        return generate_vertex(config, run_config)


def generate_vertex(config: CloudConfig, run_config: dict) -> dict:
    time_0 = time.time()
    output_tokens = 0

    vertexai.init(project=PROJECT_ID)

    if config.model_name in v1_models:
        model = TextGenerationModel.from_pretrained(config.model_name)
        stream = model.predict_streaming(
            run_config["query"],
            max_output_tokens=run_config["max_tokens"],
        )
        response_str, time_to_first_token, times_between_tokens, _ = generate_tokens(stream, time_0)
        output_tokens = count_v1_tokens(run_config["max_tokens"], response_str)
    else:
        model = GenerativeModel(config.model_name)
        stream = model.generate_content(
            run_config["query"],
            generation_config=GenerationConfig(max_output_tokens=run_config["max_tokens"]),
            stream=True,
        )
        _, time_to_first_token, times_between_tokens, chunk = generate_tokens(stream, time_0)
        output_tokens = chunk._raw_response.usage_metadata.candidates_token_count

    return calculate_metrics(run_config, output_tokens, time_0, time_to_first_token, times_between_tokens)


def generate_anthropic(config: CloudConfig, run_config: dict) -> dict:
    client = AnthropicVertex(region="us-central1", project_id=PROJECT_ID)
    time_0 = time.time()

    with client.messages.stream(
        max_tokens=run_config["max_tokens"],
        messages=[{"role": "user", "content": run_config["query"]}],
        model=config.model_name,
    ) as stream:
        response_str, time_to_first_token, times_between_tokens = generate_anthropic_tokens(stream, time_0)

    output_tokens = len(response_str.split())
    return calculate_metrics(run_config, output_tokens, time_0, time_to_first_token, times_between_tokens)


def generate_tokens(stream, time_0):
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    times_between_tokens = []
    response_str = ""

    for chunk in stream:
        current_time = time.time()
        response_str += chunk.text
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        else:
            assert previous_token_time is not None
            times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time

    return response_str, time_to_first_token, times_between_tokens, chunk


def generate_anthropic_tokens(stream, time_0):
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    times_between_tokens = []
    response_str = ""

    for text in stream.text_stream:
        current_time = time.time()
        response_str += text
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        else:
            assert previous_token_time is not None
            times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time

    return response_str, time_to_first_token, times_between_tokens


def calculate_metrics(run_config, output_tokens, time_0, time_to_first_token, times_between_tokens):
    time_1 = time.time()
    generate_time = time_1 - time_0
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


def count_v1_tokens(max_tokens: int, response_str: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    openai_tokens = len(encoder.encode(response_str))
    if not 0.9 * max_tokens <= openai_tokens <= 1.1 * max_tokens:
        raise ValueError(f"Openai tokens {openai_tokens} not within 10% of max tokens {max_tokens}")
    return max_tokens
