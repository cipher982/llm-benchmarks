from __future__ import annotations

import os
from datetime import timedelta

SCHEDULER_TICK_SECONDS = int(os.getenv("SCHEDULER_TICK_SECONDS", "30"))
DEFAULT_DEADLINE_SECONDS = int(os.getenv("BENCHMARK_DEADLINE_SECONDS", "120"))
LEASE_GRACE_SECONDS = int(os.getenv("BENCHMARK_LEASE_GRACE_SECONDS", "60"))
DEFAULT_MAX_ATTEMPTS = int(os.getenv("BENCHMARK_MAX_ATTEMPTS", "2"))
DEFAULT_BACKOFF_SECONDS = int(os.getenv("BENCHMARK_RETRY_BACKOFF_SECONDS", str(15 * 60)))
MAX_BACKOFF_SECONDS = int(os.getenv("BENCHMARK_RETRY_MAX_BACKOFF_SECONDS", str(6 * 60 * 60)))
DEAD_LETTER_RETRY_AFTER_SECONDS = int(os.getenv("BENCHMARK_DEAD_LETTER_RETRY_AFTER_SECONDS", str(15 * 60)))
# Retryable failures must not retry forever. Unbounded resurrection trades a
# ratchet that loses models for a queue that never lets one go.
MAX_DEAD_LETTER_REQUEUES = int(os.getenv("BENCHMARK_MAX_DEAD_LETTER_REQUEUES", "20"))
MAX_RETRY_ATTEMPTS = int(os.getenv("BENCHMARK_MAX_RETRY_ATTEMPTS", "12"))
BILLING_DEAD_LETTER_RETRY_AFTER_SECONDS = int(
    os.getenv("BENCHMARK_BILLING_DEAD_LETTER_RETRY_AFTER_SECONDS", str(6 * 60 * 60))
)

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
OPENROUTER_CONCURRENCY_DEFAULT = int(os.getenv("OPENROUTER_CONCURRENCY", "4"))

RETRYABLE_ERROR_KINDS = {
    "network",
    "rate_limit",
    "timeout",
    "transient_provider",
}

# The measurement protocol a verdict was reached under. Bump when the request
# shape changes — token budget, early stopping, retry policy, reasoning
# controls.
#
# v2 (2026-08-16): budget 64 -> 2048, the stream closes at the 64th visible
# token, and validation stopped asking whether output landed near the budget.
MEASUREMENT_PROTOCOL_VERSION = 2

# Failure kinds that describe the measurement rather than the model. A model
# that could not be measured at 64 tokens has been told nothing about itself,
# so the verdict expires when the protocol that produced it does — otherwise
# 419 models stay permanently dead-lettered for exhausting a budget that no
# longer exists, and every future protocol change needs a human with a mongosh
# session to undo it.
PROTOCOL_DEPENDENT_ERROR_KINDS = {"budget_exhausted"}

OVERLOADED_ERROR_MARKERS = (
    "overloaded",
    "model busy",
    "retry later",
    "temporarily unavailable",
)


def fresh_minutes() -> int:
    return int(os.getenv("FRESH_MINUTES", "30"))


def cadence_seconds(fresh_minutes_value: int | None = None) -> int:
    return (fresh_minutes_value if fresh_minutes_value is not None else fresh_minutes()) * 60


def provider_concurrency(provider: str) -> int:
    env_name = f"BENCHMARK_CONCURRENCY_{provider.upper()}"
    if os.getenv(env_name):
        return int(os.environ[env_name])
    return PROVIDER_CONCURRENCY_DEFAULTS.get(provider, 2)


def openrouter_concurrency() -> int:
    return max(1, int(os.getenv("OPENROUTER_CONCURRENCY", str(OPENROUTER_CONCURRENCY_DEFAULT))))


def excluded_providers() -> set[str]:
    """Providers this host runs no worker for.

    Two reasons qualify and they behave identically here: the provider runs on
    a different host (bedrock, on the EC2 runner with its IAM role), or it has
    been retired onto OpenRouter transport. Either way, admitting or scheduling
    a job for it leaves work in the queue that nothing will ever claim, which
    trips the queue invariants rather than failing visibly.

    Retirement has to be enforced here and not only by disabling rows, because
    discovery is additive: it will re-add a provider's models next time it runs,
    and without this they would quietly come back.
    """
    raw = os.getenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")
    return {provider.strip() for provider in raw.split(",") if provider.strip()}


def is_retryable_failure(error_kind: str | None, error_message: str | None = None) -> bool:
    if error_kind in RETRYABLE_ERROR_KINDS:
        return True
    if error_kind != "unknown" or not error_message:
        return False
    message = error_message.lower()
    return any(marker in message for marker in OVERLOADED_ERROR_MARKERS)


def retry_backoff(error_kind: str | None = None, attempt: int = 1) -> timedelta:
    del error_kind  # Kept in the signature for callers that classify by kind.
    multiplier = 2 ** max(0, attempt - 1)
    return timedelta(seconds=min(DEFAULT_BACKOFF_SECONDS * multiplier, MAX_BACKOFF_SECONDS))


def should_retry(
    error_kind: str | None,
    attempt: int,
    max_attempts: int,
    error_message: str | None = None,
) -> bool:
    # Provider weather is not a terminal model decision. Keep these jobs
    # recoverable so a short outage cannot permanently remove a model — but
    # bounded, so a permanently broken model cannot cycle forever.
    if is_retryable_failure(error_kind, error_message):
        return attempt < MAX_RETRY_ATTEMPTS
    # The model worked and the profile could not measure it. Retrying spends
    # money to reproduce the same result, because nothing about the next
    # attempt is different — same model, same 64-token budget.
    if error_kind == "budget_exhausted":
        return False
    if attempt >= max_attempts:
        return False
    if not error_kind:
        return True
    return error_kind == "unknown"


def endpoint_targets_enabled() -> bool:
    """Whether the scheduler measures endpoints rather than models.

    Off by default, deliberately. Endpoint targets multiply the fleet roughly
    threefold — 946 endpoints against 325 models at the time of writing — while
    OpenRouter work shares a single concurrency gate of
    ``BENCHMARK_CONCURRENCY_OPENROUTER``. Turning that on implicitly, as a side
    effect of deploying the catalogue, would burst the queue against a lane that
    cannot drain it. It is one env var so it can be turned on, watched, and
    turned off again.
    """

    return os.getenv("BENCHMARK_ENDPOINT_TARGETS", "0").strip().lower() in {"1", "true", "yes"}


def endpoint_targets_per_pass() -> int:
    """How many endpoint jobs one scheduler pass may create per provider.

    Bounds the work a pass creates, never which endpoints are eligible. The
    remainder stays eligible and sorts to the front of the next pass, because a
    bound applied to the population rather than the batch is how twelve
    DeepInfra models went unscheduled for months.
    """

    try:
        return max(1, int(os.getenv("BENCHMARK_ENDPOINT_TARGETS_PER_PASS", "25")))
    except ValueError:
        return 25


ENDPOINT_ROTATION_POLICY_VERSION = 1


def endpoint_tier_interval_seconds(provider_count: int) -> int:
    """How often one model earns an endpoint measurement opportunity.

    Provider count is a cheap popularity proxy. A tier opportunity measures one
    oldest endpoint, not every endpoint, so a popular model costs one request
    per interval instead of multiplying its faster cadence by every provider.
    """

    hot_min = int(os.getenv("BENCHMARK_ENDPOINT_HOT_PROVIDER_MIN", "8"))
    medium_min = int(os.getenv("BENCHMARK_ENDPOINT_MEDIUM_PROVIDER_MIN", "3"))
    if provider_count >= hot_min:
        hours = float(os.getenv("BENCHMARK_ENDPOINT_HOT_HOURS", "3"))
    elif provider_count >= medium_min:
        hours = float(os.getenv("BENCHMARK_ENDPOINT_MEDIUM_HOURS", "24"))
    else:
        hours = float(os.getenv("BENCHMARK_ENDPOINT_LONG_HOURS", "96"))
    return max(60, int(hours * 60 * 60))
