from datetime import datetime
from datetime import timedelta
from datetime import timezone

from llm_bench.scheduler import health
from llm_bench.scheduler.mongo import heartbeats_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name


class FakeCollection:
    def __init__(self, doc):
        self.doc = doc

    def find_one(self, query, projection=None, sort=None):
        return self.doc


def test_liveness_rejects_stale_direct_runner_metric():
    now = datetime(2026, 8, 3, tzinfo=timezone.utc)
    db = {
        metrics_collection_name(): FakeCollection({"provider": "openai", "gen_ts": now - timedelta(minutes=20)}),
        heartbeats_collection_name(): FakeCollection({"updated_at": now}),
    }

    healthy, details = health.liveness_status(
        db,
        max_idle_seconds=900,
        providers=["openai"],
        now=now,
    )

    assert not healthy
    assert details["reason"] == "benchmark completion is stale"
