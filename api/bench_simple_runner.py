#!/usr/bin/env python3
"""
Simple benchmark runner for remote environments without MongoDB access.

Runs benchmarks for specified providers/models and posts results to HTTP ingest API.
No MongoDB dependencies, no jobs, no freshness checks.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from typing import Any
from typing import Dict
from typing import List

import dotenv
from llm_bench.config import CloudConfig
from llm_bench.http_output import log_http
from llm_bench.utils import get_current_timestamp

# Load environment
dotenv.load_dotenv(".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Benchmark defaults (match bench_headless.py)
QUERY_TEXT = "Tell a long and happy story about the history of the world."
MAX_TOKENS = 64
TEMPERATURE = 0.1

# Map provider name -> module import path
PROVIDER_MODULES: Dict[str, str] = {
    "openai": "llm_bench.cloud.providers.openai",
    "anthropic": "llm_bench.cloud.providers.anthropic",
    "bedrock": "llm_bench.cloud.providers.bedrock",
    "vertex": "llm_bench.cloud.providers.vertex",
    "anyscale": "llm_bench.cloud.providers.anyscale",
    "together": "llm_bench.cloud.providers.together",
    "openrouter": "llm_bench.cloud.providers.openrouter",
    "azure": "llm_bench.cloud.providers.azure",
    "runpod": "llm_bench.cloud.providers.runpod",
    "fireworks": "llm_bench.cloud.providers.fireworks",
    "deepinfra": "llm_bench.cloud.providers.deepinfra",
    "groq": "llm_bench.cloud.providers.groq",
    "databricks": "llm_bench.cloud.providers.databricks",
    "lambda": "llm_bench.cloud.providers.lambda",
    "cerebras": "llm_bench.cloud.providers.cerebras",
}

# Cache for imported provider modules
_PROVIDER_MODULES_CACHE: Dict[str, Any] = {}


def _load_provider_func(provider: str):
    """Load and cache provider module to avoid repeated imports."""
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unsupported provider: {provider}")

    # Return cached function if already loaded
    if provider in _PROVIDER_MODULES_CACHE:
        return _PROVIDER_MODULES_CACHE[provider]

    # Import and cache the module
    module_name = PROVIDER_MODULES[provider]
    module = __import__(module_name, fromlist=["generate"])  # type: ignore
    generate_func = module.generate

    # Cache for future use
    _PROVIDER_MODULES_CACHE[provider] = generate_func
    return generate_func


def run_single_benchmark(provider: str, model: str) -> bool:
    """
    Run a single benchmark and post result to ingest API.

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running benchmark: {provider}:{model}")

    # Build config
    run_ts = get_current_timestamp()
    model_config = CloudConfig(
        provider=provider,
        model_name=model,
        run_ts=run_ts,
        temperature=TEMPERATURE,
        misc={},
    )

    run_config = {
        "query": QUERY_TEXT,
        "max_tokens": MAX_TOKENS,
    }

    # Load provider
    try:
        generate = _load_provider_func(provider)
    except ValueError as e:
        logger.error(f"Failed to load provider: {e}")
        return False

    # Execute benchmark
    try:
        metrics = generate(model_config, run_config)
    except Exception as e:
        logger.error(f"Benchmark failed for {provider}:{model} - {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return False

    if not metrics:
        logger.error(f"Empty metrics returned for {provider}:{model}")
        return False

    # Basic validation
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for key in required:
        if key not in metrics:
            logger.error(f"Missing required metric '{key}' for {provider}:{model}")
            return False

    if metrics["tokens_per_second"] <= 0:
        logger.error(f"Invalid tokens_per_second ({metrics['tokens_per_second']}) for {provider}:{model}")
        return False

    # Post to ingest API
    success = log_http(model_config, metrics)
    if success:
        logger.info(
            f"✅ Success {provider}:{model} "
            f"(tps={metrics['tokens_per_second']:.2f}, out={metrics['output_tokens']})"
        )
    else:
        logger.error(f"Failed to post results to ingest API for {provider}:{model}")

    return success


def parse_models_arg(models_str: str) -> List[str]:
    """Parse comma-separated model list."""
    return [m.strip() for m in models_str.split(",") if m.strip()]


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark runner that posts results to HTTP ingest API")
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider name (e.g., bedrock, openai, anthropic)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model IDs (e.g., 'model1,model2')",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode, continuously looping",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Interval in minutes between runs in daemon mode (default: 30)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get models from args or env var
    models_str = args.models or os.getenv("BENCHMARK_MODELS")
    if not models_str:
        logger.error("No models specified. Use --models or set BENCHMARK_MODELS env var")
        sys.exit(1)

    models = parse_models_arg(models_str)
    if not models:
        logger.error("No valid models found after parsing")
        sys.exit(1)

    # Verify ingest API config
    if not os.getenv("INGEST_API_URL"):
        logger.error("INGEST_API_URL not set in environment")
        sys.exit(1)
    if not os.getenv("INGEST_API_KEY"):
        logger.error("INGEST_API_KEY not set in environment")
        sys.exit(1)

    logger.info(f"Provider: {args.provider}")
    logger.info(f"Models: {models}")

    if args.daemon:
        logger.info(f"Running in daemon mode (interval: {args.interval} minutes)")
        interval_seconds = args.interval * 60

        while True:
            logger.info("=== Starting benchmark cycle ===")
            success_count = 0
            fail_count = 0

            for model in models:
                if run_single_benchmark(args.provider, model):
                    success_count += 1
                else:
                    fail_count += 1

            logger.info(f"=== Cycle complete: {success_count} success, {fail_count} failed ===")
            logger.info(f"Next run in {args.interval} minutes...")
            time.sleep(interval_seconds)
    else:
        # Single run mode
        success_count = 0
        fail_count = 0

        for model in models:
            if run_single_benchmark(args.provider, model):
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"Complete: {success_count} success, {fail_count} failed")

        # Exit with error code if any failed
        if fail_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
