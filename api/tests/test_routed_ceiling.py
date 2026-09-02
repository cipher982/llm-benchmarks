"""No setting may put more than BENCHMARK_MAX_ROUTED_JOBS_PER_DAY jobs into the routed lane.

The owner's guarantee: he must never wake up to 5x OpenRouter usage. Every
admission path — scheduled, core-set, pinned-profile, manual, probe, and
dead-letter requeue — asks the same counter, and the counter includes
requeues so a cap reset drains at the daily rate instead of bursting.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import admission
from llm_bench.ops import core_set
from llm_bench.ops import endpoint_discovery
from llm_bench.ops import invariants
from llm_bench.scheduler import cli
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import models_collection_name

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"ceiling-{request.node.name}"]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", raising=False)
    monkeypatch.delenv("BENCHMARK_CORE_SET", raising=False)
    monkeypatch.delenv("BENCHMARK_CORE_SET_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("BENCHMARK_REASONING_PUBLISH", raising=False)
    monkeypatch.delenv("BENCHMARK_EXCLUDED_PROVIDERS", raising=False)


def seed_model(db, model_id, tags, provider="openrouter"):
    db[models_collection_name()].update_one(
        {"provider": provider, "model_id": model_id},
        {"$set": {"enabled": True, "deprecated": False}},
        upsert=True,
    )
    db[endpoint_discovery.endpoints_collection_name()].insert_many(
        [
            {
                "model_id": model_id,
                "endpoint_tag": tag,
                "provider_canonical": endpoint_discovery.provider_canonical(tag),
                "enabled": True,
                "or_uptime_1d": 100,
                "or_throughput_p50": 50,
            }
            for tag in tags
        ]
    )


def finish_all(db, now):
    """Every queued job runs and succeeds instantly, as a fast fleet would."""
    for job in list(queue.jobs_collection(db).find({"status": "queued"})):
        queue.jobs_collection(db).update_one(
            {"_id": job["_id"]},
            {"$set": {"status": "success", "started_at": now, "finished_at": now, "updated_at": now}},
        )
        health.record_success(
            db,
            provider=job["provider"],
            model_id=job["model_id"],
            endpoint_tag=job.get("endpoint_tag"),
            cadence_seconds=int(job.get("cadence_seconds") or 3600),
            now=now,
        )


class TestCounter:
    def test_default_is_three_hundred(self):
        assert policies.max_routed_jobs_per_day() == 300

    def test_refuses_at_the_ceiling_and_records_the_hit(self, db, monkeypatch, capsys):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "3")
        for i in range(3):
            assert queue.enqueue_scheduled_job(db, provider="openrouter", model_id=f"m{i}", priority=1.0, now=NOW)
        assert not queue.enqueue_scheduled_job(db, provider="openrouter", model_id="m3", priority=1.0, now=NOW)
        assert queue.jobs_collection(db).count_documents({}) == 3
        hit = db.provider_state.find_one({"_id": queue.ROUTED_CEILING_STATE_ID})
        assert hit["count_24h"] == 3 and hit["ceiling"] == 3 and hit["hits"] == 1
        assert "ceiling_hit provider=openrouter jobs_24h=3 ceiling=3" in capsys.readouterr().out

    def test_the_window_reopens_after_a_day(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "1")
        assert queue.enqueue_scheduled_job(db, provider="openrouter", model_id="a", priority=1.0, now=NOW)
        assert not queue.enqueue_scheduled_job(db, provider="openrouter", model_id="b", priority=1.0, now=NOW)
        assert queue.enqueue_scheduled_job(
            db, provider="openrouter", model_id="b", priority=1.0, now=NOW + timedelta(hours=25)
        )

    def test_direct_lanes_are_not_routed(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "1")
        assert queue.enqueue_scheduled_job(db, provider="openai", model_id="a", priority=1.0, now=NOW)
        assert queue.enqueue_scheduled_job(db, provider="openai", model_id="b", priority=1.0, now=NOW)

    def test_manual_and_probe_paths_are_covered(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "1")
        queue.enqueue_manual_job(db, provider="openrouter", model_id="a", now=NOW)
        with pytest.raises(RuntimeError, match="ceiling"):
            queue.enqueue_manual_job(db, provider="openrouter", model_id="b", now=NOW)
        db[models_collection_name()].insert_one(
            {"provider": "openrouter", "model_id": "cand", "status": admission.CANDIDATE_STATUS}
        )
        assert admission.enqueue_probes(db, now=NOW, limit=10) == []


class TestMisconfiguredCoreSet:
    def test_a_core_set_ten_times_too_fast_cannot_exceed_the_ceiling(self, db, monkeypatch):
        # Thirty core endpoints every 33 minutes instead of every 5.5 hours
        # would be ~1,300 jobs/day. The lane admits 300 and no more.
        monkeypatch.setenv("BENCHMARK_CORE_SET_INTERVAL_SECONDS", str(core_set.DEFAULT_CORE_INTERVAL_SECONDS // 10))
        monkeypatch.setenv("BENCHMARK_ENDPOINT_TARGETS_PER_PASS", "100")
        for i in range(15):
            seed_model(db, f"m{i:02d}", [f"p{j}" for j in range(12)])

        created = 0
        now = NOW
        tick = timedelta(minutes=10)
        while now < NOW + timedelta(hours=24):
            for priority, model_id, tag, cadence in cli._endpoint_candidates(db, provider="openrouter", now=now):
                if queue.enqueue_scheduled_job(
                    db,
                    provider="openrouter",
                    model_id=model_id,
                    endpoint_tag=tag,
                    priority=priority,
                    now=now,
                    cadence_seconds=cadence,
                ):
                    created += 1
            finish_all(db, now)
            now += tick

        # The ledger, not the job documents: a target's job id is stable and
        # a re-enqueue replaces the document, so documents undercount.
        assert created == queue.routed_jobs_last_24h(db, provider="openrouter", now=now)
        assert created <= policies.max_routed_jobs_per_day()
        assert created >= policies.max_routed_jobs_per_day() * 0.9, "the ceiling, not the fixture, bound the count"
        assert db.provider_state.find_one({"_id": queue.ROUTED_CEILING_STATE_ID})["hits"] > 0

    def test_within_budget_the_core_set_is_untouched_by_the_ceiling(self, db):
        seed_model(db, "m", ["a", "b", "c"])
        cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        assert core_set.load(db)["projected_jobs_per_day"] < policies.max_routed_jobs_per_day()


class TestRequeueDrain:
    def test_a_cap_reset_drains_at_the_daily_rate_not_as_a_burst(self, db, monkeypatch):
        # Twelve days of billing dead letters come back eligible at once.
        old = NOW - timedelta(days=12)
        for i in range(20):
            model = f"m{i}"
            db[models_collection_name()].insert_one({"provider": "openrouter", "model_id": model, "enabled": True})
            queue.enqueue_scheduled_job(db, provider="openrouter", model_id=model, priority=1.0, now=old)
            queue.jobs_collection(db).update_one(
                {"_id": queue.scheduled_job_id("openrouter", model, None)},
                {"$set": {"status": "dead_letter", "last_attempt_error_kind": "billing", "updated_at": old}},
            )
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "5")
        first = queue.requeue_retryable_dead_letters(db, now=NOW)
        assert len(first) == 5
        assert queue.jobs_collection(db).count_documents({"status": "queued"}) == 5
        assert queue.jobs_collection(db).count_documents({"status": "dead_letter"}) == 15
        # Same tick again: nothing more. Tomorrow: the next five, oldest first.
        assert queue.requeue_retryable_dead_letters(db, now=NOW) == []
        second = queue.requeue_retryable_dead_letters(db, now=NOW + timedelta(hours=25))
        assert len(second) == 5

    def test_re_running_one_target_counts_every_admission(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "3")
        db[models_collection_name()].insert_one({"provider": "openrouter", "model_id": "m", "enabled": True})
        for i in range(3):
            at = NOW + timedelta(minutes=i)
            assert queue.enqueue_scheduled_job(db, provider="openrouter", model_id="m", priority=1.0, now=at)
            queue.jobs_collection(db).update_one(
                {"_id": queue.scheduled_job_id("openrouter", "m", None)}, {"$set": {"status": "success"}}
            )
        assert queue.jobs_collection(db).count_documents({}) == 1
        assert queue.routed_jobs_last_24h(db, provider="openrouter", now=NOW + timedelta(minutes=3)) == 3
        assert not queue.enqueue_scheduled_job(
            db, provider="openrouter", model_id="m", priority=1.0, now=NOW + timedelta(minutes=3)
        )

    def test_a_requeued_job_counts_as_an_admission(self, db):
        db[models_collection_name()].insert_one({"provider": "openrouter", "model_id": "m", "enabled": True})
        old = NOW - timedelta(days=2)
        queue.enqueue_scheduled_job(db, provider="openrouter", model_id="m", priority=1.0, now=old)
        queue.jobs_collection(db).update_one(
            {"_id": queue.scheduled_job_id("openrouter", "m", None)},
            {"$set": {"status": "dead_letter", "last_attempt_error_kind": "billing", "updated_at": old}},
        )
        assert queue.routed_jobs_last_24h(db, provider="openrouter", now=NOW) == 0
        queue.requeue_retryable_dead_letters(db, now=NOW)
        assert queue.routed_jobs_last_24h(db, provider="openrouter", now=NOW) == 1


class TestInvariant:
    def test_pages_at_the_ceiling_and_after_a_hit(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "2")
        ctx = invariants.Context(db=db, now=NOW)
        assert invariants.routed_jobs_under_daily_ceiling(ctx) == []
        queue.enqueue_scheduled_job(db, provider="openrouter", model_id="a", priority=1.0, now=NOW)
        queue.enqueue_scheduled_job(db, provider="openrouter", model_id="b", priority=1.0, now=NOW)
        [violation] = invariants.routed_jobs_under_daily_ceiling(ctx)
        assert violation.subject == "openrouter" and violation.data["jobs_24h"] == 2
        # Once the window has moved on but a refusal happened inside the day, it still pages.
        queue.enqueue_scheduled_job(db, provider="openrouter", model_id="c", priority=1.0, now=NOW)
        later = invariants.Context(db=db, now=NOW + timedelta(hours=23))
        assert len(invariants.routed_jobs_under_daily_ceiling(later)) == 1
        gone = invariants.Context(db=db, now=NOW + timedelta(hours=25))
        assert invariants.routed_jobs_under_daily_ceiling(gone) == []

    def test_is_registered_and_pages(self):
        [inv] = [i for i in invariants.INVARIANTS if i.name == "routed_jobs_under_daily_ceiling"]
        assert inv.pages is True
