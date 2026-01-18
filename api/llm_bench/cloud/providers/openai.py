import logging
import time

import tiktoken
from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp
from openai import OpenAI

logger = logging.getLogger(__name__)


# Legacy instruct models use completions endpoint
NON_CHAT_MODELS = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"]

# Reasoning models require Responses API
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

_RESPONSES_ONLY_HINTS = (
    "only supported in v1/responses",
    "not supported in the v1/chat/completions",
    "not a chat model",
    "max_output_tokens",
    "empty chat content",
)


def _is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model that requires Responses API."""
    return any(model_name.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


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


def _make_responses_request(client: OpenAI, config: CloudConfig, run_config: dict, stream: bool = False):
    """Make Responses API request (non-streaming or streaming).

    For non-streaming, returns a Response object.
    For streaming, returns an iterator of events.
    """
    # Prefer Responses API for models that don't support chat.completions.
    # Some models (e.g. codex-max) may consume the entire max_output_tokens budget in
    # internal reasoning and emit no output text unless given a larger budget.
    max_out = int(run_config["max_tokens"])
    is_codex_max = "codex-max" in config.model_name.lower()
    if is_codex_max:
        max_out = max(max_out, 256)

    base_params = {
        "model": config.model_name,
        "input": run_config["query"],
        "max_output_tokens": max_out,
        "temperature": config.temperature,
        "stream": stream,
    }
    if is_codex_max:
        base_params["reasoning"] = {"effort": "low"}

    # Some models reject parameters like temperature/reasoning; retry by removing unsupported ones.
    attempts = [
        dict(base_params),
        {k: v for k, v in base_params.items() if k != "temperature"},
        {k: v for k, v in base_params.items() if k != "reasoning"},
        {k: v for k, v in base_params.items() if k not in ("temperature", "reasoning")},
    ]

    last_err = None
    for params in attempts:
        try:
            return client.responses.create(**params)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "unsupported parameter" in msg or "not supported" in msg:
                continue
            raise
    assert last_err is not None
    # Final fallback for codex-max: give it a much larger budget if it kept failing due to output caps.
    if is_codex_max:
        for params in attempts:
            try:
                params = dict(params)
                params["max_output_tokens"] = max(params.get("max_output_tokens", 0), 1024)
                return client.responses.create(**params)
            except Exception as e:
                last_err = e
    raise last_err


def _response_output_text(resp) -> str:
    """Extract text from a non-streaming Response object."""
    # openai>=1.x provides `output_text` convenience property for Responses.
    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out:
        return out
    try:
        dumped = resp.model_dump()
        text_parts = []
        for item in dumped.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    text_parts.append(content.get("text"))
        return "".join(text_parts)
    except Exception:
        return ""


def _process_responses_stream(stream, time_0: float) -> tuple:
    """Process streaming Responses API events.

    Returns: (response_text, time_to_first_token, times_between_tokens, response_obj)
    """
    response_text = ""
    first_token_received = False
    previous_token_time = None
    times_between_tokens = []
    time_to_first_token = 0
    response_obj = None

    for event in stream:
        event_type = getattr(event, "type", "unknown")

        # Collect text deltas
        if event_type == "response.output_text.delta":
            current_time = time.time()

            # Extract delta text
            delta = getattr(event, "delta", "")
            if delta:
                response_text += delta

                # Track timing for first token
                if not first_token_received:
                    time_to_first_token = current_time - time_0
                    first_token_received = True
                else:
                    if previous_token_time is not None:
                        times_between_tokens.append(current_time - previous_token_time)

                previous_token_time = current_time

        # Capture final response object with usage metrics
        elif event_type in ("response.completed", "response.incomplete", "response.failed"):
            response_obj = event.response
            break

    return response_text, time_to_first_token, times_between_tokens, response_obj


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenAI inference using appropriate API (Chat Completions or Responses)."""

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

    # Handle reasoning models with Responses API (streaming)
    if _is_reasoning_model(config.model_name):
        stream = _make_responses_request(client, config, run_config, stream=True)
        response_str, time_to_first_token, times_between_tokens, response_obj = _process_responses_stream(
            stream, time_0
        )

        time_1 = time.time()
        generate_time = time_1 - time_0

        # Get usage metrics from response object
        usage = getattr(response_obj, "usage", None) if response_obj else None
        if usage:
            # Use actual token counts from API
            output_tokens = usage.output_tokens
        else:
            # Fallback to estimation (should not happen)
            try:
                encoder = tiktoken.encoding_for_model(config.model_name)
                output_tokens = len(encoder.encode(response_str))
            except KeyError:
                logger.warning(f"No usage data or tiktoken encoder for {config.model_name}, using word count")
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
        # Handle chat models with automatic parameter detection and fallback to Responses API.
        try:
            stream = _make_chat_request(client, config, run_config, use_max_tokens=True)
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
            # Some models may stream non-text deltas (reasoning/tool calls) without `delta.content`.
            # If we got no visible text, retry via the Responses API to ensure we get output text.
            if not response_str.strip():
                raise RuntimeError("empty chat content")
        except Exception as e:
            msg = str(e).lower()
            if "max_completion_tokens" in msg:
                # Retry with max_completion_tokens (some models use this)
                stream = _make_chat_request(client, config, run_config, use_max_tokens=False)
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
                if not response_str.strip():
                    raise RuntimeError("empty chat content")
            elif any(h in msg for h in _RESPONSES_ONLY_HINTS):
                # Fall back to Responses API for non-chat / responses-only models (with streaming).
                stream = _make_responses_request(client, config, run_config, stream=True)
                response_str, time_to_first_token, times_between_tokens, response_obj = _process_responses_stream(
                    stream, time_0
                )
            else:
                raise e

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
