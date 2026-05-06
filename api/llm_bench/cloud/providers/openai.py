import logging
import time

import tiktoken
from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


# Legacy instruct models use completions endpoint.
NON_CHAT_MODELS = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"]

# Modern OpenAI reasoning models are best run through Responses API. Treat
# GPT-5-family aliases by prefix so new snapshots do not need code changes.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


def _reasoning_effort_candidates(model_name: str) -> list[str | None]:
    if not _is_reasoning_model(model_name):
        return [None]
    if "-pro" in model_name:
        return ["medium", "high", "xhigh", None]
    return ["low", "medium", None]


def _response_budget_candidates(model_name: str, requested_tokens: int) -> list[int]:
    requested_tokens = int(requested_tokens)
    if "-pro" in model_name or "codex-max" in model_name.lower():
        return [max(requested_tokens, 512), max(requested_tokens, 1024)]
    if _is_reasoning_model(model_name):
        return [max(requested_tokens, 128), max(requested_tokens, 512)]
    return [requested_tokens]


def _make_responses_request(
    client: OpenAI,
    config: CloudConfig,
    run_config: dict,
    *,
    max_output_tokens: int,
    reasoning_effort: str | None,
    stream: bool,
):
    base_params = {
        "model": config.model_name,
        "input": run_config["query"],
        "max_output_tokens": max_output_tokens,
        "temperature": config.temperature,
        "stream": stream,
    }
    if reasoning_effort:
        base_params["reasoning"] = {"effort": reasoning_effort}

    # Some models reject temperature or reasoning values. Retry by removing
    # optional knobs before giving up on the model.
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
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if "unsupported" in msg or "not supported" in msg or "invalid" in msg:
                continue
            raise
    raise last_err


def _process_responses_stream(stream, time_0: float) -> tuple[str, float, list[float], object | None]:
    response_text = ""
    first_token_received = False
    previous_token_time = None
    times_between_tokens = []
    time_to_first_token = 0
    response_obj = None

    for event in stream:
        event_type = getattr(event, "type", "unknown")

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if not delta:
                continue

            current_time = time.time()
            response_text += delta

            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            elif previous_token_time is not None:
                times_between_tokens.append(current_time - previous_token_time)

            previous_token_time = current_time
        elif event_type in ("response.completed", "response.incomplete", "response.failed"):
            response_obj = event.response
            break

    return response_text, time_to_first_token, times_between_tokens, response_obj


def _attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_token_count(response_text: str, model_name: str) -> int:
    try:
        encoder = tiktoken.encoding_for_model(model_name)
        return len(encoder.encode(response_text))
    except KeyError:
        logger.warning("No usage data or tiktoken encoder for %s, using word count", model_name)
        return int(len(response_text.split()) * 1.3)


def _responses_usage(response_obj, response_text: str, model_name: str) -> dict:
    usage = getattr(response_obj, "usage", None) if response_obj else None
    if usage and getattr(usage, "output_tokens", None) is not None:
        output_tokens = int(usage.output_tokens)
        output_details = _attr(usage, "output_tokens_details")
        reasoning_tokens = _attr(output_details, "reasoning_tokens")
        if reasoning_tokens is not None:
            reasoning_tokens = int(reasoning_tokens)
            visible_tokens = max(output_tokens - reasoning_tokens, 0)
        else:
            visible_tokens = _text_token_count(response_text, model_name) if response_text else 0

        input_details = _attr(usage, "input_tokens_details")
        return {
            "generated_output_tokens": output_tokens,
            "visible_output_tokens": visible_tokens,
            "reasoning_tokens": reasoning_tokens,
            "input_tokens": _attr(usage, "input_tokens"),
            "total_tokens": _attr(usage, "total_tokens"),
            "cached_input_tokens": _attr(input_details, "cached_tokens"),
            "token_source": "provider_usage_output_tokens",
        }

    visible_tokens = _text_token_count(response_text, model_name)
    return {
        "generated_output_tokens": visible_tokens,
        "visible_output_tokens": visible_tokens,
        "reasoning_tokens": None,
        "input_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "token_source": "approximate_tiktoken_or_words",
    }


def _run_responses_model(client: OpenAI, config: CloudConfig, run_config: dict) -> dict:
    last_error = None
    last_metrics = None

    for max_output_tokens in _response_budget_candidates(config.model_name, run_config["max_tokens"]):
        for reasoning_effort in _reasoning_effort_candidates(config.model_name):
            time_0 = time.time()
            try:
                stream = _make_responses_request(
                    client,
                    config,
                    run_config,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    stream=True,
                )
                response_str, time_to_first_token, times_between_tokens, response_obj = _process_responses_stream(
                    stream, time_0
                )
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "unsupported" in msg or "not supported" in msg or "invalid" in msg:
                    continue
                raise

            generate_time = time.time() - time_0
            usage_metrics = _responses_usage(response_obj, response_str, config.model_name)
            last_metrics = build_cloud_metrics(
                requested_tokens=run_config["max_tokens"],
                generated_output_tokens=usage_metrics["generated_output_tokens"],
                visible_output_tokens=usage_metrics["visible_output_tokens"],
                reasoning_tokens=usage_metrics["reasoning_tokens"],
                cached_input_tokens=usage_metrics["cached_input_tokens"],
                input_tokens=usage_metrics["input_tokens"],
                total_tokens=usage_metrics["total_tokens"],
                generate_time=generate_time,
                time_to_first_token=time_to_first_token if time_to_first_token > 0 else None,
                times_between_tokens=times_between_tokens,
                token_source=usage_metrics["token_source"],
                request_mode="openai_responses_stream",
                response_id=_attr(response_obj, "id"),
                response_status=_attr(response_obj, "status"),
                finish_reason=_attr(_attr(response_obj, "incomplete_details"), "reason"),
                max_output_tokens_attempted=max_output_tokens,
                reasoning_effort=reasoning_effort,
                visible_text_empty=not bool(response_str.strip()),
            )

            if response_str.strip():
                return last_metrics

            logger.info(
                "OpenAI Responses returned no visible text for %s with max_output_tokens=%s effort=%s",
                config.model_name,
                max_output_tokens,
                reasoning_effort,
            )

    if last_metrics and last_metrics.get("output_tokens", 0) > 0:
        return last_metrics
    if last_error:
        raise last_error
    raise RuntimeError("empty responses output text")


def _run_legacy_completion(client: OpenAI, config: CloudConfig, run_config: dict) -> dict:
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    times_between_tokens = []
    time_to_first_token = 0
    response_str = ""

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
            elif previous_token_time is not None:
                times_between_tokens.append(current_time - previous_token_time)

            previous_token_time = current_time
            response_str += content

    generate_time = time.time() - time_0
    try:
        encoder = tiktoken.encoding_for_model(config.model_name)
        output_tokens = len(encoder.encode(response_str))
    except KeyError:
        logger.warning("No tiktoken encoder for %s, using word count approximation", config.model_name)
        output_tokens = int(len(response_str.split()) * 1.3)

    return build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=output_tokens,
        visible_output_tokens=output_tokens,
        reasoning_tokens=0,
        generate_time=generate_time,
        time_to_first_token=time_to_first_token if time_to_first_token > 0 else None,
        times_between_tokens=times_between_tokens,
        token_source="tiktoken_visible_text",
        request_mode="openai_legacy_completions_stream",
        max_output_tokens_attempted=run_config["max_tokens"],
        visible_text_empty=not bool(response_str.strip()),
    )


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenAI inference using Responses API for modern text models."""

    assert config.provider == "openai", "provider must be openai"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = OpenAI()

    if config.model_name in NON_CHAT_MODELS:
        return _run_legacy_completion(client, config, run_config)
    return _run_responses_model(client, config, run_config)
