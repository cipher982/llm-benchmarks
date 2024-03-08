import logging
import time
from datetime import datetime

import tiktoken
import vertexai
from llm_bench_api.config import CloudConfig
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel


logger = logging.getLogger(__name__)


v1_models = ["text-bison@002", "chat-bison@002"]


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Vertex inference and return metrics."""

    assert config.provider == "vertex", "provider must be Vertex"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    vertexai.init(project="llm-bench")

    # Generate
    time_0 = time.time()
    output_tokens = 0

    if config.model_name in v1_models:
        params = {
            "max_output_tokens": run_config["max_tokens"],
        }
        model = TextGenerationModel.from_pretrained(config.model_name)
        stream = model.predict_streaming(
            run_config["query"],
            **params,
        )
        response_str, time_to_first_token, times_between_tokens, _ = generate_tokens(stream, time_0)

        # Hack as vertex token counter is broken
        output_tokens = count_v1_tokens(run_config["max_tokens"], response_str)
    else:
        params = GenerationConfig(
            max_output_tokens=run_config["max_tokens"],
        )
        model = GenerativeModel(config.model_name)
        stream = model.generate_content(
            run_config["query"],
            generation_config=params,
            stream=True,
        )

        _, time_to_first_token, times_between_tokens, chunk = generate_tokens(stream, time_0)
        output_tokens = chunk._raw_response.usage_metadata.candidates_token_count

    logger.info("Finished!!")
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


def count_v1_tokens(max_tokens: int, response_str: str) -> int:
    """
    Calculate output tokens and validate if they are within 10% of max tokens.
    A bit of a hack as vertex token counter is broken.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    openai_tokens = len(encoder.encode(response_str))
    if not 0.9 * max_tokens <= openai_tokens <= 1.1 * max_tokens:
        raise ValueError(f"Openai tokens {openai_tokens} not within 10% of max tokens {max_tokens}")
    return max_tokens
