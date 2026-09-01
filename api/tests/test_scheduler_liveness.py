from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.scheduler import health

NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"liveness-{request.node.name}"]


def metric(db, provider, *, ago):
    db.metrics_cloud_v2.insert_one({"provider": provider, "gen_ts": NOW - ago})


def heartbeat(db, *, ago=timedelta(0)):
    db.bench_scheduler_heartbeats.insert_one({"_id": "scheduler", "updated_at": NOW - ago})


def workers(db, *providers, ago=timedelta(seconds=10)):
    """Live worker threads for these lanes.

    A provider failing and a provider's threads dying are different faults with
    different fixes, so they are separate signals: these tests are about the
    former, and a running process always has the latter.
    """
    for provider in providers:
        db.bench_scheduler_heartbeats.insert_one({"_id": f"worker:{provider}:0", "updated_at": NOW - ago})


def test_liveness_ignores_stale_completion_when_process_heartbeats_are_live(db):
    metric(db, "openai", ago=timedelta(days=30))
    heartbeat(db)
    workers(db, "openai")

    healthy, details = health.liveness_status(db, providers=["openai"], now=NOW)

    assert healthy
    assert details["provider_progress"]["openai"]["age_seconds"] == 30 * 24 * 3600


def test_liveness_stays_healthy_when_one_lane_stalls(db):
    """A stalled lane must not kill the process.

    Restarting the container does not fix Together's auth or DeepInfra's
    billing, and taking down the eight working lanes to react to the ninth
    would turn a partial failure into a total one. The stall is reported here
    and acted on by the invariant layer instead.
    """
    metric(db, "openai", ago=timedelta(minutes=2))
    metric(db, "together", ago=timedelta(hours=9))
    heartbeat(db)
    # Together's threads are alive and polling; it is Together that is broken.
    workers(db, "openai", "together")

    healthy, details = health.liveness_status(db, providers=["openai", "together"], now=NOW)

    assert healthy
    assert details["provider_progress"]["together"]["age_seconds"] == 9 * 3600
    assert details["provider_progress"]["openai"]["age_seconds"] == 120


def test_provider_progress_reports_a_lane_that_never_ran(db):
    metric(db, "openai", ago=timedelta(minutes=2))

    progress = health.provider_progress(db, providers=["openai", "vertex"], now=NOW)

    assert progress["vertex"] == {"latest_completed_at": None, "age_seconds": None}
    assert progress["openai"]["age_seconds"] == 120
