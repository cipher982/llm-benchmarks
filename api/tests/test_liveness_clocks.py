"""Process liveness is independent of benchmark completion.

The watchdog exists because worker threads died on an unhandled Mongo error
while the process stayed up for eight days. Workers now heartbeat on every
poll, including an idle one, and the scheduler loop has its own heartbeat.
Those clocks detect process failure without conflating it with a long sampling
cadence or an upstream auth or billing outage.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.scheduler import health

NOW = datetime(2026, 8, 17, 14, 0, tzinfo=timezone.utc)
PROVIDERS = ["openrouter", "openai"]


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _worker(db, provider, slot, ago_seconds):
    db.bench_scheduler_heartbeats.insert_one(
        {"_id": f"worker:{provider}:{slot}", "updated_at": NOW - timedelta(seconds=ago_seconds)}
    )


def _scheduler(db, ago_seconds=5):
    db.bench_scheduler_heartbeats.insert_one({"_id": "scheduler", "updated_at": NOW - timedelta(seconds=ago_seconds)})


def _healthy_fixture(db):
    for provider in PROVIDERS:
        _worker(db, provider, 0, 10)
    _scheduler(db)


class TestIdleByDesignIsNotAFault:
    def test_no_completed_work_is_healthy_when_process_heartbeats_are_live(self, db):
        _healthy_fixture(db)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]
        assert details["provider_progress"]["openrouter"]["latest_completed_at"] is None


class TestDeadWorkersAreCaughtRegardless:
    def test_silent_workers_fail_regardless_of_recent_completions(self, db):
        """The eight-day failure: process up, threads gone."""
        for provider in PROVIDERS:
            _worker(db, provider, 0, 9999)
        _scheduler(db)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert not ok
        assert details["reason"] == "worker threads have stopped checking in"
        assert {lane["provider"] for lane in details["stale_worker_lanes"]} == set(PROVIDERS)

    def test_one_dead_lane_is_enough(self, db):
        _worker(db, "openrouter", 0, 10)
        _worker(db, "openai", 0, 9999)
        _scheduler(db)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert not ok
        assert [lane["provider"] for lane in details["stale_worker_lanes"]] == ["openai"]

    def test_a_lane_with_one_live_slot_is_alive(self, db):
        """Slots are redundant; the lane is alive if any slot is polling."""
        _worker(db, "openrouter", 0, 9999)
        _worker(db, "openrouter", 1, 10)
        _worker(db, "openai", 0, 10)
        _scheduler(db)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]

    def test_a_retired_providers_stale_heartbeat_is_ignored(self, db):
        """worker:deepinfra:* rows linger for months after retirement."""
        for provider in PROVIDERS:
            _worker(db, provider, 0, 10)
        _worker(db, "deepinfra", 0, 30 * 24 * 3600)
        _scheduler(db)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]


class TestTheControlClockIsIndependent:
    def test_a_stale_scheduler_heartbeat_fails(self, db):
        """The scheduler loop must stay observable regardless of data cadence."""
        for provider in PROVIDERS:
            _worker(db, provider, 0, 10)
        _scheduler(db, ago_seconds=3600)

        ok, details = health.liveness_status(db, providers=PROVIDERS, now=NOW)

        assert not ok
        assert details["reason"] == "scheduler heartbeat is stale"


class TestTheCheckedLaneSetTracksTheWorkerSet:
    def test_a_retired_provider_named_in_the_env_is_not_held_to_account(self, monkeypatch):
        """Otherwise the container restarts forever after a consolidation.

        A lane with no worker has no heartbeat, and liveness now reads a missing
        heartbeat as a dead thread. BENCHMARK_LIVENESS_PROVIDERS still named
        eight providers after they were retired onto OpenRouter, so the check
        would have failed on lanes that are supposed to be gone.
        """
        from llm_bench.scheduler import healthcheck

        monkeypatch.setenv("BENCHMARK_LIVENESS_PROVIDERS", "openai,vertex,openrouter,deepinfra,together")
        monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock,deepinfra,together")

        assert healthcheck._providers() == ["openai", "vertex", "openrouter"]

    def test_an_empty_result_falls_back_to_checking_every_lane(self, monkeypatch):
        from llm_bench.scheduler import healthcheck

        monkeypatch.setenv("BENCHMARK_LIVENESS_PROVIDERS", "deepinfra")
        monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "deepinfra")

        assert healthcheck._providers() is None
