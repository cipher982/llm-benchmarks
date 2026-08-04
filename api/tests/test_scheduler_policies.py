from datetime import datetime
from datetime import timedelta
from datetime import timezone

from llm_bench.scheduler import policies
from llm_bench.scheduler.queue import _failure_update


def test_retryable_provider_failure_outlives_the_per_job_attempt_cap():
    # A short provider outage must not permanently remove a model, so retryable
    # kinds keep going well past max_attempts.
    assert policies.should_retry("transient_provider", attempt=5, max_attempts=2)
    assert policies.should_retry("rate_limit", attempt=2, max_attempts=2)


def test_retryable_failures_are_still_bounded():
    # But not forever. Unbounded resurrection turns a permanently broken model
    # into a job that cycles through the queue indefinitely.
    ceiling = policies.MAX_RETRY_ATTEMPTS
    assert policies.should_retry("transient_provider", attempt=ceiling - 1, max_attempts=2)
    assert not policies.should_retry("transient_provider", attempt=ceiling, max_attempts=2)


def test_unknown_overload_message_is_recoverable():
    assert policies.should_retry("unknown", attempt=3, max_attempts=2, error_message="overloaded_error")
    assert not policies.should_retry(
        "unknown", attempt=policies.MAX_RETRY_ATTEMPTS, max_attempts=2, error_message="overloaded_error"
    )


def test_retry_backoff_grows_and_is_bounded():
    assert policies.retry_backoff(attempt=2) == timedelta(seconds=policies.DEFAULT_BACKOFF_SECONDS * 2)
    assert policies.retry_backoff(attempt=100).total_seconds() == policies.MAX_BACKOFF_SECONDS


def test_failure_update_keeps_retryable_job_queued_at_attempt_cap():
    update = _failure_update(
        {
            "attempt": 2,
            "max_attempts": 2,
        },
        error_kind="timeout",
        error_message="provider timed out",
        now=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )

    assert update["$set"]["status"] == "queued"
