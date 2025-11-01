import logging
import time
from datetime import datetime

import tiktoken
from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp
from openai import OpenAI

logger = logging.getLogger(__name__)


# Legacy instruct models use completions endpoint
NON_CHAT_MODELS = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"]


def _make_chat_request(client: OpenAI, config: CloudConfig, run_config: dict, use_max_tokens: bool = True):
    """Make chat completions request with appropriate token parameter."""
    request_params = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": run_config["query"]}],
        "stream": True,
    }
    
    if use_max_tokens:
        request_params["max_tokens"] = run_config["max_tokens"]
    else:
        request_params["max_completion_tokens"] = run_config["max_tokens"]
    
    return client.chat.completions.create(**request_params)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenAI inference using Chat Completions API with automatic parameter detection."""
    
    assert config.provider == "openai", "provider must be openai"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = OpenAI()
    
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    times_between_tokens = []
    time_to_first_token = 0
    response_str = ""

    # Handle legacy instruct models
    if config.model_name in NON_CHAT_MODELS:
        stream = client.completions.create(
            model=config.model_name,
            prompt=run_config["query"],
            max_tokens=run_config["max_tokens"],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                current_time = time.time()
                content = chunk.choices[0].text
                
                if not first_token_received:
                    time_to_first_token = current_time - time_0
                    first_token_received = True
                else:
                    if previous_token_time is not None:
                        times_between_tokens.append(current_time - previous_token_time)
                
                previous_token_time = current_time
                response_str += content
    else:
        # Handle chat models with automatic parameter detection
        try:
            # Try max_tokens first (works for most models)
            stream = _make_chat_request(client, config, run_config, use_max_tokens=True)
        except Exception as e:
            if "max_completion_tokens" in str(e):
                # Model requires max_completion_tokens, retry with correct parameter
                stream = _make_chat_request(client, config, run_config, use_max_tokens=False)
            else:
                # Different error, re-raise
                raise e
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                current_time = time.time()
                content = chunk.choices[0].delta.content
                
                if not first_token_received:
                    time_to_first_token = current_time - time_0
                    first_token_received = True
                else:
                    if previous_token_time is not None:
                        times_between_tokens.append(current_time - previous_token_time)
                
                previous_token_time = current_time
                response_str += content

    time_1 = time.time()
    generate_time = time_1 - time_0

    # Calculate tokens
    try:
        encoder = tiktoken.encoding_for_model(config.model_name)
        output_tokens = len(encoder.encode(response_str))
    except KeyError:
        # Fallback for models without tiktoken support
        logger.warning(f"No tiktoken encoder for {config.model_name}, using word count approximation")
        output_tokens = int(len(response_str.split()) * 1.3)
    
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