import logging
import os

from llm_bench.cloud.providers.openai_compat import run_chat_completion_benchmark
from llm_bench.config import CloudConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run TogetherAI inference and return metrics."""
    assert config.provider == "together", "provider must be 'together'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = OpenAI(
        base_url=os.environ["TOGETHER_BASE_URL"],
        api_key=os.environ["TOGETHER_API_KEY"],
    )

    return run_chat_completion_benchmark(
        client=client,
        model=config.model_name,
        max_tokens=run_config["max_tokens"],
        query=run_config["query"],
        request_mode="together_chat_completions",
        fallback_extra_bodies=[
            ("reasoning_disabled", {"reasoning": {"enabled": False}}, "disabled"),
        ],
    )
