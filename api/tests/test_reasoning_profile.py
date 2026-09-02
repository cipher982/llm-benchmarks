"""Reasoning-profile publication is a flag, off by default, and a pin with a reason."""

from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import cli
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler import runner
from llm_bench.scheduler.mongo import models_collection_name

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"reason-{request.node.name}"]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("BENCHMARK_REASONING_PUBLISH", raising=False)
    monkeypatch.setenv("BENCHMARK_CORE_SET", "0")


def seed(db, model_id="anthropic/claude-opus-5", tag="anthropic"):
    db[models_collection_name()].insert_one({"provider": "openrouter", "model_id": model_id, "enabled": True})
    db[endpoint_discovery.endpoints_collection_name()].insert_one(
        {"model_id": model_id, "endpoint_tag": tag, "provider_canonical": tag, "enabled": True}
    )
    queue.enqueue_scheduled_job(db, provider="openrouter", model_id=model_id, endpoint_tag=tag, priority=1.0, now=NOW)
    job_id = queue.scheduled_job_id("openrouter", model_id, tag)
    queue.jobs_collection(db).update_one(
        {"_id": job_id},
        {"$set": {"status": "dead_letter", "last_attempt_error_kind": "budget_exhausted", "updated_at": NOW}},
    )
    return job_id


class TestFlag:
    def test_off_by_default(self):
        assert policies.reasoning_publish_enabled() is False

    def test_sweep_does_nothing_while_off(self, db):
        seed(db)
        assert queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW) == []
        assert queue.jobs_collection(db).find_one()["status"] == "dead_letter"
        assert cli._pinned_profiles(db, provider="openrouter") == {}


class TestPin:
    def test_sweep_pins_the_target_and_frees_the_job_slot(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "1")
        job_id = seed(db)
        [pinned] = queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW)
        assert pinned == {"model_id": "anthropic/claude-opus-5", "endpoint_tag": "anthropic", "job_id": job_id}
        doc = health.health_collection(db).find_one(
            health.health_filter("openrouter", "anthropic/claude-opus-5", "anthropic")
        )
        assert doc["measurement_profile"] == policies.REASONING_PROFILE_ID
        assert "budget_exhausted" in doc["measurement_profile_reason"]
        assert doc["measurement_profile_from_job"] == job_id
        job = queue.jobs_collection(db).find_one({"_id": job_id})
        assert job["status"] == "cancelled"
        assert job["cancelled_reason"] == "pinned to reasoning profile"

    def test_sweep_skips_targets_the_catalogue_dropped(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "1")
        seed(db)
        db[models_collection_name()].update_one({"model_id": "anthropic/claude-opus-5"}, {"$set": {"enabled": False}})
        assert queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW) == []

    def test_a_reasoning_profile_failure_is_a_verdict_not_a_budget_problem(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "1")
        job_id = seed(db)
        queue.jobs_collection(db).update_one(
            {"_id": job_id}, {"$set": {"benchmark_profile_id": policies.REASONING_PROFILE_ID}}
        )
        assert queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW) == []


class TestScheduling:
    def test_pinned_target_runs_once_a_day_under_the_profile(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "1")
        seed(db)
        queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW)
        found = cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        assert [(tag, cad) for _, _, tag, cad in found] == [("anthropic", policies.REASONING_CADENCE_SECONDS)]
        assert cli._pinned_profiles(db, provider="openrouter") == {
            ("anthropic/claude-opus-5", "anthropic"): policies.REASONING_PROFILE_ID
        }
        assert queue.enqueue_scheduled_job(
            db,
            provider="openrouter",
            model_id="anthropic/claude-opus-5",
            endpoint_tag="anthropic",
            priority=1.0,
            now=NOW,
            cadence_seconds=policies.REASONING_CADENCE_SECONDS,
            benchmark_profile_id=policies.REASONING_PROFILE_ID,
        )
        job = queue.jobs_collection(db).find_one({"status": "queued"})
        assert job["benchmark_profile_id"] == policies.REASONING_PROFILE_ID
        assert job.get("sample_role", "published") == "published"

    def test_pin_is_inert_with_the_flag_off(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "1")
        seed(db)
        queue.pin_budget_exhausted_to_reasoning_profile(db, now=NOW)
        monkeypatch.setenv("BENCHMARK_REASONING_PUBLISH", "0")
        found = cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        assert [cad for _, _, _, cad in found] == [policies.endpoint_tier_interval_seconds(1)]


class TestBudget:
    def test_reasoning_profile_has_its_own_ceiling(self):
        assert runner.profile_max_tokens(policies.REASONING_PROFILE_ID) == 2048
        assert (
            runner.profile_max_cost_per_run_usd(policies.REASONING_PROFILE_ID) == runner.REASONING_MAX_COST_PER_RUN_USD
        )
        assert runner.profile_max_cost_per_run_usd(runner.DEFAULT_PROFILE_ID) == runner.MAX_COST_PER_RUN_USD

    def test_a_premium_model_gets_a_real_budget_under_the_reasoning_ceiling(self):
        price = 75e-6  # $75/M output: claude-opus-5 class
        default = runner.lane_max_tokens("openrouter", 2048, completion_price_per_token=price)
        reasoning = runner.lane_max_tokens(
            "openrouter", 2048, completion_price_per_token=price, max_cost_per_run_usd=0.05
        )
        assert default == max(runner.VISIBLE_TOKEN_MARK, int(runner.MAX_COST_PER_RUN_USD / price))
        assert reasoning == min(2048, int(0.05 / price))
        assert reasoning > default

    def test_uncapped_lanes_never_get_the_headroom(self):
        assert runner.lane_max_tokens("direct", 2048, max_cost_per_run_usd=0.05) == runner.UNCAPPED_LANE_MAX_TOKENS
