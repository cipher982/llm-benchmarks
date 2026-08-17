"""Liveness asks two questions and they need two clocks.

The watchdog exists because worker threads died on an unhandled Mongo error
while the process stayed up, and the runner produced nothing for eight days with
RestartCount=0. The signal it used was completion age, which conflates "the
workers are dead" with "there is nothing due to measure" — fine at a 45-minute
scheduling period, fatal at 4.5 hours, where the runner is idle by design and a
15-minute completion limit kills it every 15 minutes forever.

Worker heartbeats answer the first question on their own clock: they are written
on every poll, including idle ones, so a dead thread goes quiet whether or not
there was work.
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


def _completion(db, ago_seconds):
    db.metrics_cloud_v2.insert_one({"provider": "openrouter", "gen_ts": NOW - timedelta(seconds=ago_seconds)})


def _healthy_fixture(db, *, completion_ago):
    for provider in PROVIDERS:
        _worker(db, provider, 0, 10)
    _scheduler(db)
    _completion(db, completion_ago)


class TestIdleByDesignIsNotAFault:
    def test_a_long_gap_between_scheduling_rounds_is_healthy(self, db):
        """4h with no completions at a 4.5h period is the configuration working."""
        _healthy_fixture(db, completion_ago=4 * 3600)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]

    def test_the_old_fixed_limit_would_have_killed_it(self, db):
        """Documents the bug: same state, 15-minute limit, dead."""
        _healthy_fixture(db, completion_ago=4 * 3600)

        ok, _ = health.liveness_status(db, max_idle_seconds=900, providers=PROVIDERS, now=NOW)

        assert not ok


class TestDeadWorkersAreCaughtRegardless:
    def test_silent_workers_fail_even_when_completions_are_recent(self, db):
        """The eight-day failure: process up, threads gone."""
        for provider in PROVIDERS:
            _worker(db, provider, 0, 9999)
        _scheduler(db)
        _completion(db, 30)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

        assert not ok
        assert details["reason"] == "worker threads have stopped checking in"
        assert {lane["provider"] for lane in details["stale_worker_lanes"]} == set(PROVIDERS)

    def test_one_dead_lane_is_enough(self, db):
        _worker(db, "openrouter", 0, 10)
        _worker(db, "openai", 0, 9999)
        _scheduler(db)
        _completion(db, 30)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

        assert not ok
        assert [lane["provider"] for lane in details["stale_worker_lanes"]] == ["openai"]

    def test_a_lane_with_one_live_slot_is_alive(self, db):
        """Slots are redundant; the lane is alive if any slot is polling."""
        _worker(db, "openrouter", 0, 9999)
        _worker(db, "openrouter", 1, 10)
        _worker(db, "openai", 0, 10)
        _scheduler(db)
        _completion(db, 30)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]

    def test_a_retired_providers_stale_heartbeat_is_ignored(self, db):
        """worker:deepinfra:* rows linger for months after retirement."""
        for provider in PROVIDERS:
            _worker(db, provider, 0, 10)
        _worker(db, "deepinfra", 0, 30 * 24 * 3600)
        _scheduler(db)
        _completion(db, 30)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

        assert ok, details["reason"]


class TestTheControlClockIsIndependent:
    def test_the_scheduler_heartbeat_limit_does_not_follow_max_idle(self, db):
        """It used to be max(max_idle_seconds, 180).

        Raising the completion budget to span a longer scheduling period would
        then have silently widened the scheduler check from 15 minutes to six
        hours — disabling, as a side effect, the check that notices the loop
        that creates the work has stopped.
        """
        for provider in PROVIDERS:
            _worker(db, provider, 0, 10)
        _scheduler(db, ago_seconds=3600)
        _completion(db, 30)

        ok, details = health.liveness_status(db, max_idle_seconds=6 * 3600, providers=PROVIDERS, now=NOW)

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
