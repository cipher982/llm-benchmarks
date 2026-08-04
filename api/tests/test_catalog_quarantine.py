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
