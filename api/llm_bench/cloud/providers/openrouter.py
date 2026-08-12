"""OpenRouter chat-completions adapter.

OpenRouter streams can contain many text deltas and one final usage object.
Text-delta count is therefore never a token count.  This adapter uses provider
usage when available and records a visible-text fallback explicitly.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any

from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig
from llm_bench.scheduler.routing import OR_SERVED_POLICY
from llm_bench.utils import get_current_timestamp
from openai import OpenAI
from tiktoken import get_encoding

logger = logging.getLogger(__name__)

NON_CHAT_MODELS: list[str] = []

# OpenRouter's streaming metadata has historically exposed provider display
# names rather than routing slugs. This is deliberately a small reviewed map,
# shared only by provider families that the bench can route. Unknown names stay
# unverified and therefore fail closed in the runner.
REVIEWED_PROVIDER_DISPLAY_SLUGS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "deepinfra": ("deepinfra",),
    "deep infra": ("deepinfra",),
    "fireworks": ("fireworks", "fireworks-ai"),
    "fireworks ai": ("fireworks", "fireworks-ai"),
    "together": ("together", "togetherai"),
    "together ai": ("together", "togetherai"),
    "togetherai": ("together", "togetherai"),
    "groq": ("groq",),
    "cerebras": ("cerebras",),
    "google vertex": ("google-vertex", "vertex"),
    "google-vertex": ("google-vertex", "vertex"),
    "vertex": ("google-vertex", "vertex"),
    # OpenRouter reports the Vertex lane's display name as plain "Google";
    # the ambiguity against AI Studio resolves through the expected pinned
    # slug, never by defaulting.
    "google": ("google-vertex", "google-ai-studio"),
}


def process_non_chat_model(client, config, run_config):
    raise NotImplementedError


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_extra(obj: Any) -> Mapping[str, Any]:
    extra = _attr(obj, "model_extra", {}) or {}
    return extra if isinstance(extra, Mapping) else {}


def _nested_field(obj: Any, *names: str) -> Any:
    for name in names:
        value = _attr(obj, name)
        if value is not None:
            return value
        value = _model_extra(obj).get(name)
        if value is not None:
            return value
    return None


def _route_options(config: CloudConfig) -> dict[str, Any]:
    # or-served (marketplace) lanes measure OpenRouter's default routing: the
    # user does not pin, so neither do we. The observed provider is recorded
    # from the response, not forced in the request.
    if config.misc.get("route_policy") == OR_SERVED_POLICY:
        if config.misc.get("route_reasoning_exclude"):
            return {"reasoning": {"exclude": True, "effort": "minimal"}}
        return {}
    route_slug = config.misc.get("route_provider_slug")
    if not route_slug:
        return {}
    options: dict[str, Any] = {
        "provider": {
            "only": [str(route_slug)],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }
    if config.misc.get("route_reasoning_exclude"):
        # Parity with direct lanes that measure non-reasoning visible output:
        # some OpenRouter lanes default frontier models into thinking mode,
        # which consumes the whole published-profile token budget invisibly.
        options["reasoning"] = {"exclude": True, "effort": "minimal"}
    return options


def _request_stream(client: OpenAI, config: CloudConfig, run_config: dict[str, Any]):
    model_id = config.transport_model_id or config.model_name
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": run_config["query"]}],
        "max_tokens": run_config["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_headers": {
            "HTTP-Referer": "llm-benchmarks.com",
            "X-Title": "LLM Benchmarks",
            "X-OpenRouter-Metadata": "enabled",
        },
    }
    route_options = _route_options(config)
    if route_options:
        kwargs["extra_body"] = route_options
    return client.chat.completions.create(**kwargs)


def process_chat_model(client, config, run_config):
    return _request_stream(client, config, run_config), "choices"


def _usage_metrics(usage: Any, response_text: str, reasoning_text: str, model_name: str) -> dict[str, Any]:
    encoder = get_encoding("cl100k_base")
    visible_tokens = len(encoder.encode(response_text))
    completion_tokens = _attr(usage, "completion_tokens")
    if completion_tokens is None:
        return {
            "generated_output_tokens": visible_tokens,
            "visible_output_tokens": visible_tokens,
            "reasoning_tokens": len(encoder.encode(reasoning_text)) if reasoning_text else None,
            "input_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "token_source": "tiktoken_visible_text",
        }

    generated_tokens = int(completion_tokens)
    details = _attr(usage, "completion_tokens_details")
    reasoning_tokens = _attr(details, "reasoning_tokens")
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)
        visible_tokens = max(generated_tokens - reasoning_tokens, 0)
    elif reasoning_text:
        reasoning_tokens = max(generated_tokens - visible_tokens, 0)
    if generated_tokens <= 0 and visible_tokens > 0:
        generated_tokens = visible_tokens
        reasoning_tokens = None
        token_source = "tiktoken_visible_fallback_zero_usage"
    else:
        token_source = "provider_usage_completion_tokens"

    prompt_details = _attr(usage, "prompt_tokens_details")
    return {
        "generated_output_tokens": generated_tokens,
        "visible_output_tokens": visible_tokens,
        "reasoning_tokens": reasoning_tokens,
        "input_tokens": _attr(usage, "prompt_tokens"),
        "total_tokens": _attr(usage, "total_tokens"),
        "cached_input_tokens": _attr(prompt_details, "cached_tokens"),
        "token_source": token_source,
    }


def _metadata_from_chunk(chunk: Any) -> Mapping[str, Any] | None:
    merged: dict[str, Any] = {}
    for candidate in (
        _nested_field(chunk, "openrouter_metadata"),
        _nested_field(chunk, "metadata"),
        _nested_field(chunk, "provider_metadata"),
    ):
        if isinstance(candidate, Mapping):
            merged.update(candidate)
    # The streaming API may expose the selected provider directly on the
    # chunk, while the metadata-enabled response exposes it below
    # endpoints.available[].selected. Preserve both shapes for one parser.
    direct = {
        key: value
        for key in ("provider", "provider_name", "provider_slug", "selected_provider", "selected_provider_slug")
        if (value := _nested_field(chunk, key)) is not None
    }
    for key, value in direct.items():
        merged.setdefault(key, value)
    return merged or None


def _provider_key(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def _observed_provider(
    metadata: Mapping[str, Any] | None,
    *,
    expected_slug: str | None = None,
) -> tuple[str | None, str | None]:
    if not metadata:
        return None, None
    provider = metadata.get("provider") or metadata.get("provider_name") or metadata.get("selected_provider")
    slug = metadata.get("provider_slug") or metadata.get("selected_provider_slug")
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available", []) if isinstance(endpoints, Mapping) else []
    selected = next((item for item in available if isinstance(item, Mapping) and item.get("selected")), None)
    if selected:
        provider = provider or selected.get("provider") or selected.get("provider_name") or selected.get("name")
        slug = slug or selected.get("provider_slug") or selected.get("slug")
    if provider and not slug:
        candidates = REVIEWED_PROVIDER_DISPLAY_SLUGS.get(_provider_key(provider), ())
        if expected_slug and expected_slug.casefold() in {candidate.casefold() for candidate in candidates}:
            slug = expected_slug
        elif len(candidates) == 1:
            slug = candidates[0]
    return (str(provider) if provider else None, str(slug) if slug else None)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenRouter inference and return usage-backed metrics."""

    assert (
        config.provider == "openrouter" or config.transport_provider == "openrouter"
    ), "provider or transport_provider must be 'openrouter'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    timeout_seconds = float(config.misc.get("timeout_seconds", os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")))
    client = OpenAI(
        base_url=os.environ["OPENROUTER_BASE_URL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=timeout_seconds,
        max_retries=int(config.misc.get("max_retries", 0)),
    )

    time_0 = time.time()
    response_text = ""
    reasoning_text = ""
    first_token_received = False
    previous_token_time = None
    times_between_tokens: list[float] = []
    time_to_first_token: float | None = None
    usage = None
    finish_reason = None
    response_id = None
    route_metadata = None

    stream, _ = process_chat_model(client, config, run_config)
    for chunk in stream:
        response_id = response_id or _attr(chunk, "id")
        usage = _attr(chunk, "usage") or usage
        route_metadata = _metadata_from_chunk(chunk) or route_metadata
        choices = _attr(chunk, "choices", []) or []
        if len(choices) > 1:
            raise ValueError("Unexpected number of choices")
        if not choices:
            continue
        choice = choices[0]
        finish_reason = _attr(choice, "finish_reason") or finish_reason
        delta = _attr(choice, "delta")
        content = _nested_field(delta, "content") or ""
        reasoning = _nested_field(delta, "reasoning", "reasoning_content") or ""
        if reasoning:
            reasoning_text += str(reasoning)
        if not content:
            continue
        current_time = time.time()
        if not first_token_received:
            time_to_first_token = current_time - time_0
            first_token_received = True
        elif previous_token_time is not None:
            times_between_tokens.append(current_time - previous_token_time)
        previous_token_time = current_time
        response_text += str(content)

    generate_time = time.time() - time_0
    usage_values = _usage_metrics(usage, response_text, reasoning_text, config.model_name)
    metrics = build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=usage_values["generated_output_tokens"],
        visible_output_tokens=usage_values["visible_output_tokens"],
        reasoning_tokens=usage_values["reasoning_tokens"],
        cached_input_tokens=usage_values["cached_input_tokens"],
        input_tokens=usage_values["input_tokens"],
        total_tokens=usage_values["total_tokens"],
        generate_time=generate_time,
        time_to_first_token=time_to_first_token,
        times_between_tokens=times_between_tokens,
        token_source=usage_values["token_source"],
        request_mode="openrouter_chat_completions_stream",
        finish_reason=finish_reason,
        response_id=response_id,
        response_status="incomplete" if finish_reason in {"length", "max_tokens"} else "complete",
        max_output_tokens_attempted=run_config["max_tokens"],
        visible_text_empty=not bool(response_text.strip()),
    )
    metrics.update(
        {
            "gen_ts": get_current_timestamp(),
            "output_text": response_text,
            "reasoning_output_text": reasoning_text,
        }
    )

    observed_provider, observed_provider_slug = _observed_provider(
        route_metadata,
        expected_slug=config.misc.get("route_provider_slug"),
    )
    if observed_provider is not None:
        metrics["observed_provider"] = observed_provider
    if observed_provider_slug is not None:
        metrics["observed_provider_slug"] = observed_provider_slug
    if response_id is not None:
        metrics["openrouter_response_id"] = response_id
    if config.misc.get("route_provider_slug"):
        metrics["route_provider_slug"] = config.misc["route_provider_slug"]
        metrics["route_policy"] = "pinned-provider"
        metrics["provider_metadata_verified"] = bool(observed_provider and observed_provider_slug)
    return metrics
