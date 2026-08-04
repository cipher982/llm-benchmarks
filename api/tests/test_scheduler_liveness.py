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


def test_liveness_rejects_stale_direct_runner_metric(db):
    metric(db, "openai", ago=timedelta(minutes=20))
    heartbeat(db)

    healthy, details = health.liveness_status(db, max_idle_seconds=900, providers=["openai"], now=NOW)

    assert not healthy
    assert details["reason"] == "benchmark completion is stale"


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

    healthy, details = health.liveness_status(db, max_idle_seconds=900, providers=["openai", "together"], now=NOW)

    assert healthy
    assert details["provider_progress"]["together"]["age_seconds"] == 9 * 3600
    assert details["provider_progress"]["openai"]["age_seconds"] == 120


def test_provider_progress_reports_a_lane_that_never_ran(db):
    metric(db, "openai", ago=timedelta(minutes=2))

    progress = health.provider_progress(db, providers=["openai", "vertex"], now=NOW)

    assert progress["vertex"] == {"latest_completed_at": None, "age_seconds": None}
    assert progress["openai"]["age_seconds"] == 120
