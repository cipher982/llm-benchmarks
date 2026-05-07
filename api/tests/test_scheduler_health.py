from datetime import datetime
from datetime import timedelta
from datetime import timezone

from llm_bench.scheduler.health import compute_freshness_status


def test_freshness_status_thresholds():
    now = datetime(2026, 5, 7, 20, 0, tzinfo=timezone.utc)
    cadence = 60 * 60

    assert (
        compute_freshness_status(enabled=False, cadence_seconds=cadence, last_success_at=now, now=now)[0] == "disabled"
    )
    assert (
        compute_freshness_status(enabled=True, cadence_seconds=cadence, last_success_at=None, now=now)[0] == "never_run"
    )
    assert (
        compute_freshness_status(
            enabled=True,
            cadence_seconds=cadence,
            last_success_at=now - timedelta(minutes=89),
            now=now,
        )[0]
        == "fresh"
    )
    assert (
        compute_freshness_status(
            enabled=True,
            cadence_seconds=cadence,
            last_success_at=now - timedelta(minutes=91),
            now=now,
        )[0]
        == "stale"
    )
    assert (
        compute_freshness_status(
            enabled=True,
            cadence_seconds=cadence,
            last_success_at=now - timedelta(minutes=181),
            now=now,
        )[0]
        == "critical"
    )
