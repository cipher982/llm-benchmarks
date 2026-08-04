from datetime import datetime
from datetime import timedelta
from datetime import timezone

from llm_bench.scheduler import policies
from llm_bench.scheduler.queue import _failure_update


def test_retryable_provider_failure_ignores_attempt_cap():
    assert policies.should_retry("transient_provider", attempt=20, max_attempts=2)
    assert policies.should_retry("rate_limit", attempt=2, max_attempts=2)


def test_unknown_overload_message_is_recoverable():
    assert policies.should_retry("unknown", attempt=20, max_attempts=2, error_message="overloaded_error")


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
