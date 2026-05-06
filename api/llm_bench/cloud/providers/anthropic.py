import logging
import time

from anthropic import Anthropic
from llm_bench.cloud.metrics import build_cloud_metrics
from llm_bench.config import CloudConfig

logger = logging.getLogger(__name__)


NON_CHAT_MODELS = []


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Anthropic inference using the new Messages format and return metrics, with streaming."""

    assert config.provider == "anthropic", "provider must be anthropic"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    anthropic = Anthropic()

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_tokens = 0
    times_between_tokens = []
    input_tokens = None
    cached_input_tokens = None
    finish_reason = None
    response_id = None

    if config.model_name in NON_CHAT_MODELS:
        raise NotImplementedError
    else:
        with anthropic.messages.stream(
            model=config.model_name,
            max_tokens=run_config["max_tokens"],
            messages=[{"role": "user", "content": run_config["query"]}],
        ) as stream:
            time_to_first_token = None
            for event in stream:
                current_time = time.time()
                event_type = type(event).__name__
                if event_type == "RawMessageStartEvent":
                    first_token_received = False
                elif event_type == "RawContentBlockDeltaEvent":
                    if not first_token_received:
                        time_to_first_token = current_time - time_0
                        first_token_received = True
                    else:
                        assert previous_token_time is not None
                        times_between_tokens.append(current_time - previous_token_time)
                    previous_token_time = current_time
                elif event_type == "MessageStopEvent":
                    usage = event.message.usage  # type: ignore
                    output_tokens = usage.output_tokens
                    input_tokens = usage.input_tokens
                    cached_input_tokens = getattr(usage, "cache_read_input_tokens", None)
                    finish_reason = event.message.stop_reason  # type: ignore
                    response_id = event.message.id  # type: ignore

    time_1 = time.time()
    generate_time = time_1 - time_0

    metrics = build_cloud_metrics(
        requested_tokens=run_config["max_tokens"],
        generated_output_tokens=output_tokens,
        visible_output_tokens=output_tokens,
        reasoning_tokens=None,
        cached_input_tokens=cached_input_tokens,
        input_tokens=input_tokens,
        generate_time=generate_time,
        time_to_first_token=time_to_first_token,
        times_between_tokens=times_between_tokens,
        token_source="provider_usage_output_tokens",
        request_mode="anthropic_messages_stream",
        finish_reason=finish_reason,
        response_id=response_id,
        max_output_tokens_attempted=run_config["max_tokens"],
    )

    return metrics
