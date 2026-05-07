from __future__ import annotations

import os
from datetime import timedelta

SCHEDULER_TICK_SECONDS = int(os.getenv("SCHEDULER_TICK_SECONDS", "30"))
DEFAULT_DEADLINE_SECONDS = int(os.getenv("BENCHMARK_DEADLINE_SECONDS", "120"))
LEASE_GRACE_SECONDS = int(os.getenv("BENCHMARK_LEASE_GRACE_SECONDS", "60"))
DEFAULT_MAX_ATTEMPTS = int(os.getenv("BENCHMARK_MAX_ATTEMPTS", "2"))
DEFAULT_BACKOFF_SECONDS = int(os.getenv("BENCHMARK_RETRY_BACKOFF_SECONDS", str(15 * 60)))

PROVIDER_CONCURRENCY_DEFAULTS: dict[str, int] = {
    "anthropic": 2,
    "cerebras": 2,
    "deepinfra": 3,
    "fireworks": 2,
    "groq": 3,
    "openai": 2,
    "together": 2,
    "vertex": 2,
}

RETRYABLE_ERROR_KINDS = {
    "network",
    "rate_limit",
    "timeout",
    "transient_provider",
    "unknown",
}


def fresh_minutes() -> int:
    return int(os.getenv("FRESH_MINUTES", "30"))


def cadence_seconds(fresh_minutes_value: int | None = None) -> int:
    return (fresh_minutes_value if fresh_minutes_value is not None else fresh_minutes()) * 60


def provider_concurrency(provider: str) -> int:
    env_name = f"BENCHMARK_CONCURRENCY_{provider.upper()}"
    if os.getenv(env_name):
        return int(os.environ[env_name])
    return PROVIDER_CONCURRENCY_DEFAULTS.get(provider, 2)


def excluded_providers() -> set[str]:
    raw = os.getenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")
    return {provider.strip() for provider in raw.split(",") if provider.strip()}


def retry_backoff(error_kind: str | None = None) -> timedelta:
    return timedelta(seconds=DEFAULT_BACKOFF_SECONDS)


def should_retry(error_kind: str | None, attempt: int, max_attempts: int) -> bool:
    if attempt >= max_attempts:
        return False
    if not error_kind:
        return True
    return error_kind in RETRYABLE_ERROR_KINDS
