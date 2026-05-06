import logging
import os

from groq import Groq
from llm_bench.cloud.providers.openai_compat import run_chat_completion_benchmark
from llm_bench.config import CloudConfig

logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run Groq inference and return metrics."""

    assert config.provider == "groq", "provider must be 'groq'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return run_chat_completion_benchmark(
        client=client,
        model=config.model_name,
        max_tokens=run_config["max_tokens"],
        query=run_config["query"],
        request_mode="groq_chat_completions",
    )
