#!/usr/bin/env python3
"""
Simple benchmark runner for remote environments without MongoDB access.

Runs benchmarks for specified providers/models and posts results to HTTP ingest API.
No MongoDB dependencies, no jobs, no freshness checks.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import dotenv
import httpx
from llm_bench.config import CloudConfig
from llm_bench.http_output import log_http


def get_current_timestamp() -> str:
    """Return current timestamp in standard format (no heavy deps)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Load environment
dotenv.load_dotenv(".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Benchmark defaults used by the scheduler runner.
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
VARIABLE_OUTPUT_PROVIDERS = ("openai", "openrouter", "deepinfra", "fireworks", "together", "groq", "vertex")
DEFAULT_CONFIG_CACHE_PATH = "/var/lib/bedrock-bench/last_config.json"
DEFAULT_CONFIG_GRACE_SECONDS = 6 * 60 * 60
DEFAULT_CONFIG_TIMEOUT_SECONDS = 10.0

# Cache for imported provider modules
_PROVIDER_MODULES_CACHE: Dict[str, Any] = {}


class RunnerConfigAuthError(RuntimeError):
    pass


class RunnerConfigPermanentError(RuntimeError):
    pass


class RunnerConfigTransientError(RuntimeError):
    pass


@dataclass
class CycleConfig:
    models: List[str]
    interval_minutes: int | None
    source: str


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


def _validation_policy(provider: str) -> str:
    if provider in VARIABLE_OUTPUT_PROVIDERS:
        return "visible_nonzero"
    return "strict_pm10"


def _validate_metrics(provider: str, metrics: Dict[str, Any]) -> tuple[bool, str | None]:
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for key in required:
        if key not in metrics:
            return False, f"Missing required metric '{key}'"
    if metrics["tokens_per_second"] <= 0:
        return False, f"Invalid tokens_per_second ({metrics['tokens_per_second']})"
    if metrics.get("visible_text_empty") is True:
        if metrics.get("response_status") == "incomplete" or metrics.get("finish_reason") in (
            "length",
            "max_output_tokens",
        ):
            return False, "visible output empty after token budget was exhausted; retry with a larger output budget"
        return False, "visible output text is empty"
    visible_out = metrics.get("visible_output_tokens")
    if visible_out is not None and visible_out <= 0:
        return False, f"visible_output_tokens {visible_out} <= 0"
    return True, None


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
    valid, validation_error = _validate_metrics(provider, metrics)
    if not valid:
        logger.error(f"{validation_error} for {provider}:{model}")
        return False
    metrics.setdefault("validation_policy", _validation_policy(provider))

    # Post to ingest API
    success = log_http(model_config, metrics)
    if success:
        logger.info(
            f"✅ Success {provider}:{model} (tps={metrics['tokens_per_second']:.2f}, out={metrics['output_tokens']})"
        )
    else:
        logger.error(f"Failed to post results to ingest API for {provider}:{model}")

    return success


def parse_models_arg(models_str: str) -> List[str]:
    """Parse comma-separated model list."""
    return [m.strip() for m in models_str.split(",") if m.strip()]


def _config_cache_path() -> Path:
    return Path(os.getenv("RUNNER_CONFIG_CACHE_PATH", DEFAULT_CONFIG_CACHE_PATH))


def _config_grace_seconds() -> int:
    return int(os.getenv("RUNNER_CONFIG_GRACE_SECONDS", str(DEFAULT_CONFIG_GRACE_SECONDS)))


def _config_timeout_seconds() -> float:
    return float(os.getenv("RUNNER_CONFIG_TIMEOUT_SECONDS", str(DEFAULT_CONFIG_TIMEOUT_SECONDS)))


def _persist_runner_config(
    provider: str, models: List[str], interval_minutes: int | None, source_payload: dict
) -> None:
    path = _config_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": time.time(),
        "provider": provider,
        "models": models,
        "interval_minutes": interval_minutes,
        "source_payload": source_payload,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_cached_runner_config(provider: str) -> CycleConfig | None:
    path = _config_cache_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(f"Failed to read runner config cache {path}: {exc}")
        return None

    fetched_at = float(payload.get("fetched_at", 0))
    age_seconds = time.time() - fetched_at
    if age_seconds > _config_grace_seconds():
        logger.error(
            f"Runner config cache is stale ({age_seconds:.0f}s old, grace={_config_grace_seconds()}s); skipping cycle"
        )
        return None
    if payload.get("provider") != provider:
        logger.error(f"Runner config cache provider mismatch: {payload.get('provider')} != {provider}")
        return None

    models = payload.get("models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        logger.error("Runner config cache has invalid model list; skipping cycle")
        return None
    logger.warning(f"Using cached runner config ({age_seconds:.0f}s old, models={len(models)})")
    interval_minutes = payload.get("interval_minutes")
    return CycleConfig(models=models, interval_minutes=interval_minutes, source="cache")


def fetch_runner_config(provider: str) -> CycleConfig:
    config_url = os.getenv("RUNNER_CONFIG_URL")
    config_token = os.getenv("RUNNER_CONFIG_TOKEN")
    if not config_url:
        raise RunnerConfigPermanentError("RUNNER_CONFIG_URL not set")
    if not config_token:
        raise RunnerConfigAuthError("RUNNER_CONFIG_TOKEN not set")

    headers = {"Authorization": f"Bearer {config_token}"}
    try:
        response = httpx.get(config_url, headers=headers, timeout=_config_timeout_seconds())
    except httpx.RequestError as exc:
        raise RunnerConfigTransientError(f"Failed to fetch runner config: {exc}") from exc

    if response.status_code in (401, 403):
        raise RunnerConfigAuthError(f"Runner config auth failed: HTTP {response.status_code}")
    if 400 <= response.status_code < 500:
        raise RunnerConfigPermanentError(f"Runner config request failed: HTTP {response.status_code} {response.text}")
    if response.status_code >= 500:
        raise RunnerConfigTransientError(f"Runner config server error: HTTP {response.status_code} {response.text}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RunnerConfigTransientError("Runner config response was not JSON") from exc

    if payload.get("schema_version") != 1:
        raise RunnerConfigPermanentError(f"Unsupported runner config schema: {payload.get('schema_version')}")
    if payload.get("provider") != provider:
        raise RunnerConfigPermanentError(f"Runner config provider mismatch: {payload.get('provider')} != {provider}")

    models = payload.get("models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise RunnerConfigPermanentError("Runner config has invalid model list")

    interval_minutes = payload.get("interval_minutes")
    if interval_minutes is not None and not isinstance(interval_minutes, int):
        raise RunnerConfigPermanentError("Runner config interval_minutes must be an integer")

    _persist_runner_config(provider, models, interval_minutes, payload)
    logger.info(f"Fetched runner config: provider={provider}, models={len(models)}, interval={interval_minutes}")
    return CycleConfig(models=models, interval_minutes=interval_minutes, source="remote")


def resolve_cycle_config(provider: str, models_arg: str | None, default_interval_minutes: int) -> CycleConfig:
    if models_arg:
        models = parse_models_arg(models_arg)
        return CycleConfig(models=models, interval_minutes=None, source="args")

    override_enabled = os.getenv("BENCHMARK_MODELS_OVERRIDE") == "1"
    if override_enabled:
        models_str = os.getenv("BENCHMARK_MODELS")
        models = parse_models_arg(models_str or "")
        return CycleConfig(models=models, interval_minutes=None, source="env-override")

    if os.getenv("RUNNER_CONFIG_URL"):
        try:
            return fetch_runner_config(provider)
        except RunnerConfigAuthError as exc:
            logger.error(f"{exc}; refusing to use cached config")
            return CycleConfig(models=[], interval_minutes=None, source="auth-error")
        except RunnerConfigPermanentError as exc:
            logger.error(f"{exc}; refusing to use cached config")
            return CycleConfig(models=[], interval_minutes=None, source="config-error")
        except RunnerConfigTransientError as exc:
            logger.error(str(exc))
            cached = _load_cached_runner_config(provider)
            if cached:
                return cached
            return CycleConfig(models=[], interval_minutes=None, source="config-unavailable")

    models_str = os.getenv("BENCHMARK_MODELS")
    models = parse_models_arg(models_str or "")
    return CycleConfig(models=models, interval_minutes=default_interval_minutes, source="env")


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

    # Verify ingest API config
    if not os.getenv("INGEST_API_URL"):
        logger.error("INGEST_API_URL not set in environment")
        sys.exit(1)
    if not os.getenv("INGEST_API_KEY"):
        logger.error("INGEST_API_KEY not set in environment")
        sys.exit(1)

    logger.info(f"Provider: {args.provider}")

    if args.daemon:
        logger.info(f"Running in daemon mode (interval: {args.interval} minutes)")

        while True:
            cycle_config = resolve_cycle_config(args.provider, args.models, args.interval)
            models = cycle_config.models
            interval_minutes = cycle_config.interval_minutes or args.interval
            interval_seconds = interval_minutes * 60

            if not models:
                logger.warning(f"No models available from {cycle_config.source}; skipping cycle")
                logger.info(f"Next config attempt in {interval_minutes} minutes...")
                time.sleep(interval_seconds)
                continue

            logger.info(f"Models ({cycle_config.source}): {models}")
            logger.info("=== Starting benchmark cycle ===")
            success_count = 0
            fail_count = 0

            for model in models:
                if run_single_benchmark(args.provider, model):
                    success_count += 1
                else:
                    fail_count += 1

            logger.info(f"=== Cycle complete: {success_count} success, {fail_count} failed ===")
            logger.info(f"Next run in {interval_minutes} minutes...")
            time.sleep(interval_seconds)
    else:
        cycle_config = resolve_cycle_config(args.provider, args.models, args.interval)
        models = cycle_config.models
        if not models:
            logger.error("No models specified. Use --models, set BENCHMARK_MODELS, or configure RUNNER_CONFIG_URL")
            sys.exit(1)
        logger.info(f"Models ({cycle_config.source}): {models}")

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
