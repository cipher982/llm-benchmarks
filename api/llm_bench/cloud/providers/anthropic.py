import logging
import time
from datetime import datetime

from anthropic import Anthropic
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
                    output_tokens = event.message.usage.output_tokens  # type: ignore

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
