from datetime import UTC
from datetime import datetime
from datetime import timedelta

from llm_bench.ops.catalog_quarantine import should_quarantine


def test_quarantine_requires_repeated_failures_across_time():
    now = datetime(2026, 8, 3, tzinfo=UTC)
    failures = [{"ts": now - timedelta(minutes=45)}, {"ts": now}]

    assert should_quarantine(failures, min_failures=2, min_span_minutes=30)


def test_quarantine_does_not_disable_a_single_burst():
    now = datetime(2026, 8, 3, tzinfo=UTC)
    failures = [{"ts": now - timedelta(minutes=2)}, {"ts": now}]
    assert not should_quarantine(failures, min_failures=2, min_span_minutes=30)


def test_quarantine_disables_terminal_404_hard_model():
    now = datetime(2026, 8, 16, tzinfo=UTC)
    failures = [{"ts": now, "http_status": 404, "message": "NotFoundError: No endpoints found"}]

    assert should_quarantine(
        failures,
        min_failures=3,
        min_span_minutes=30,
        health_error_kind="hard_model",
    )
    assert not should_quarantine(failures, min_failures=2, min_span_minutes=30)


def test_quarantine_disables_hard_capability_terminal_error():
    now = datetime(2026, 8, 18, tzinfo=UTC)
    failures = [
        {
            "ts": now,
            "http_status": 400,
            "message": "BadRequestError: 'claude-opus-4-7' does not support the speed parameter",
        }
    ]
    assert should_quarantine(
        failures,
        min_failures=3,
        min_span_minutes=30,
        health_error_kind="hard_capability",
    )


def test_quarantine_disables_model_with_zero_successes_and_repeated_failures():
    now = datetime(2026, 8, 18, tzinfo=UTC)
    failures = [{"ts": now - timedelta(hours=2)}, {"ts": now}]
    assert should_quarantine(
        failures,
        min_failures=2,
        min_span_minutes=0,
        last_success=None,
    )
