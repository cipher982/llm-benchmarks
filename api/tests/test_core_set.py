"""Core-set concentration: chosen by rule, costed to be spend-neutral, recorded daily."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import core_set
from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import cli
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import models_collection_name

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"core-{request.node.name}"]


@pytest.fixture(autouse=True)
def flat_env(monkeypatch):
    for name in (
        "BENCHMARK_CORE_SET",
        "BENCHMARK_CORE_SET_INTERVAL_SECONDS",
        "BENCHMARK_EXCLUDED_PROVIDERS",
        "BENCHMARK_ENDPOINT_BLOCK_ROTATION",
        "BENCHMARK_REASONING_PUBLISH",
    ):
        monkeypatch.delenv(name, raising=False)


def seed_model(db, model_id, endpoints):
    """endpoints: list of (tag, uptime, throughput)."""
    db[models_collection_name()].update_one(
        {"provider": "openrouter", "model_id": model_id},
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
                "or_uptime_1d": uptime,
                "or_throughput_p50": throughput,
            }
            for tag, uptime, throughput in endpoints
        ]
    )


def seed_population(db, *, models=24):
    """`models` models with decreasing provider counts, three endpoints each."""
    for i in range(models):
        count = max(1, 30 - i)
        seed_model(
            db,
            f"m{i:02d}",
            [(f"p{j}/fp8", 100 - j, 50 + j) for j in range(min(count, 3))]
            + [(f"p{j}", 99, 10) for j in range(3, count)],
        )


class TestSelection:
    def test_two_endpoints_for_the_top_ten_then_one(self, db):
        seed_population(db)
        members = core_set.select(db, now=NOW)
        by_model = {}
        for m in members:
            by_model.setdefault(m["model_id"], []).append(m["endpoint_tag"])
        assert len(by_model) == core_set.TOP_MODELS
        assert all(len(tags) == 2 for model, tags in by_model.items() if model <= "m09")
        assert all(len(tags) == 1 for model, tags in by_model.items() if model >= "m10")
        assert len(members) == 30

    def test_best_served_endpoint_wins_by_uptime_then_throughput(self, db):
        seed_model(db, "only", [("slow", 100, 10), ("fast", 100, 200), ("flaky", 90, 900)])
        [first, second] = core_set.select(db, now=NOW)
        assert (first["endpoint_tag"], second["endpoint_tag"]) == ("fast", "slow")
        assert "best-served endpoint" in first["reason"]
        assert first["provider_count"] == 3

    def test_new_models_join_the_core_with_their_reason(self, db):
        seed_population(db, models=22)
        seed_model(db, "brand/new", [("a", 100, 40)])
        db.openrouter_catalog.insert_one({"openrouter_id": "brand/new", "created": NOW - timedelta(days=3)})
        db.openrouter_catalog.insert_one({"openrouter_id": "old/model", "created": NOW - timedelta(days=300)})
        members = core_set.select(db, now=NOW)
        new = [m for m in members if m["model_id"] == "brand/new"]
        assert len(new) == 1
        assert new[0]["reason"].startswith("new on OpenRouter: created 2026-08-30")
        assert not any(m["model_id"] == "old/model" for m in members)

    def test_first_seen_counts_when_created_is_missing(self, db):
        seed_population(db, models=22)
        seed_model(db, "seen/recently", [("a", 100, 40)])
        db.openrouter_catalog.insert_one({"openrouter_id": "seen/recently", "first_seen_at": NOW - timedelta(days=2)})
        [member] = [m for m in core_set.select(db, now=NOW) if m["model_id"] == "seen/recently"]
        assert member["reason"].startswith("new to the catalogue: first seen")

    def test_excluded_lane_selects_nothing(self, db, monkeypatch):
        seed_population(db)
        monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "openrouter")
        assert core_set.select(db, now=NOW) == []

    def test_disabled_models_and_endpoints_are_never_core(self, db):
        seed_model(db, "m", [("a", 100, 40)])
        db[endpoint_discovery.endpoints_collection_name()].insert_one(
            {"model_id": "m", "endpoint_tag": "off", "enabled": False, "or_uptime_1d": 100, "or_throughput_p50": 999}
        )
        db[models_collection_name()].insert_one({"provider": "openrouter", "model_id": "gone", "enabled": False})
        db[endpoint_discovery.endpoints_collection_name()].insert_one(
            {"model_id": "gone", "endpoint_tag": "a", "enabled": True, "or_uptime_1d": 100, "or_throughput_p50": 999}
        )
        assert [(m["model_id"], m["endpoint_tag"]) for m in core_set.select(db, now=NOW)] == [("m", "a")]


class TestBudget:
    def test_concentration_never_raises_jobs_per_day(self, db):
        seed_population(db)
        rows = core_set.endpoint_rows_by_model(db, provider="openrouter")
        members = core_set.select(db, now=NOW)
        out = core_set.budget(rows, members, interval_seconds=core_set.DEFAULT_CORE_INTERVAL_SECONDS)
        assert out["core_jobs_per_day"] == pytest.approx(30 * 86400 / 19800, abs=0.01)
        assert out["tail_stretch"] >= 1.0
        assert out["projected_jobs_per_day"] <= out["baseline_jobs_per_day"] + 1e-6

    def test_todays_arithmetic(self):
        # 2026-09-02 population under the rotated tiers: 25 hot models (3h),
        # 45 medium (28h), 167 long (100h); 30 core endpoints every 5.5h.
        rows = {}
        for i in range(25):
            rows[f"hot{i}"] = [{"endpoint_tag": f"p{j}", "provider_canonical": f"p{j}"} for j in range(8)]
        for i in range(45):
            rows[f"med{i}"] = [{"endpoint_tag": f"p{j}", "provider_canonical": f"p{j}"} for j in range(3)]
        for i in range(167):
            rows[f"long{i}"] = [{"endpoint_tag": "p0", "provider_canonical": "p0"}]
        members = [{"model_id": f"hot{i // 2}", "endpoint_tag": f"p{i % 2}"} for i in range(30)]
        out = core_set.budget(rows, members, interval_seconds=19800)
        assert out["baseline_jobs_per_day"] == pytest.approx(278.65, abs=0.05)
        assert out["core_jobs_per_day"] == pytest.approx(130.91, abs=0.01)
        assert out["tail_stretch"] == pytest.approx(1.886, abs=0.001)
        assert out["projected_jobs_per_day"] == pytest.approx(out["baseline_jobs_per_day"], abs=0.05)

    def test_stretch_is_capped_when_the_core_alone_exceeds_the_baseline(self):
        rows = {
            "m": [{"endpoint_tag": "a", "provider_canonical": "a"}, {"endpoint_tag": "b", "provider_canonical": "b"}]
        }
        members = [{"model_id": "m", "endpoint_tag": "a"}]
        out = core_set.budget(rows, members, interval_seconds=60)
        assert out["tail_stretch"] == core_set.MAX_TAIL_STRETCH


class TestRecord:
    def test_computes_once_a_day_and_records_the_reasons(self, db):
        seed_population(db)
        doc = core_set.refresh_if_stale(db, now=NOW)
        assert doc["member_count"] == 30
        assert doc["computed_at"] == NOW
        assert all(m["reason"] for m in doc["members"])
        assert {"baseline_jobs_per_day", "tail_stretch", "projected_jobs_per_day"} <= set(doc)
        stored = db.provider_state.find_one({"_id": core_set.STATE_ID})
        assert stored["member_count"] == 30

        # A catalogue change inside the day does not move the set.
        seed_model(db, "late/arrival", [("a", 100, 999)])
        again = core_set.refresh_if_stale(db, now=NOW + timedelta(hours=6))
        assert again["computed_at"].replace(tzinfo=timezone.utc) == NOW

        later = core_set.refresh_if_stale(db, now=NOW + timedelta(hours=25))
        assert later["computed_at"] == NOW + timedelta(hours=25)

    def test_off_switch(self, db, monkeypatch):
        seed_population(db)
        monkeypatch.setenv("BENCHMARK_CORE_SET", "0")
        assert core_set.refresh_if_stale(db, now=NOW) is None
        assert db.provider_state.count_documents({}) == 0


class TestScheduling:
    def test_core_endpoints_get_their_own_lane_and_the_tail_is_stretched(self, db):
        seed_model(db, "m", [("a", 100, 90), ("b", 100, 80), ("c", 100, 70)])
        found = cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        cadence = {tag: cad for _, _, tag, cad in found}
        assert cadence["a"] == core_set.DEFAULT_CORE_INTERVAL_SECONDS
        assert cadence["b"] == core_set.DEFAULT_CORE_INTERVAL_SECONDS
        doc = core_set.load(db)
        tail_interval = int(policies.endpoint_tier_interval_seconds(3) * doc["tail_stretch"])
        assert cadence["c"] == tail_interval
        assert cadence["c"] > policies.endpoint_tier_interval_seconds(3)

    def test_a_core_endpoint_is_due_every_interval_not_every_rotation(self, db):
        seed_model(db, "m", [("a", 100, 90), ("b", 100, 80), ("c", 100, 70)])
        cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        health.record_success(db, provider="openrouter", model_id="m", endpoint_tag="a", cadence_seconds=19800, now=NOW)
        soon = cli._endpoint_candidates(db, provider="openrouter", now=NOW + timedelta(hours=1))
        assert "a" not in [tag for _, _, tag, _ in soon]
        later = cli._endpoint_candidates(db, provider="openrouter", now=NOW + timedelta(hours=6))
        assert "a" in [tag for _, _, tag, _ in later]

    def test_a_core_run_does_not_pace_the_tail(self, db):
        seed_model(db, "m", [("a", 100, 90), ("b", 100, 80), ("c", 100, 70)])
        cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        health.record_success(db, provider="openrouter", model_id="m", endpoint_tag="a", cadence_seconds=19800, now=NOW)
        found = cli._endpoint_candidates(db, provider="openrouter", now=NOW + timedelta(minutes=5))
        assert "c" in [tag for _, _, tag, _ in found]

    def test_off_switch_restores_the_flat_rotation(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_CORE_SET", "0")
        seed_model(db, "m", [("a", 100, 90), ("b", 100, 80), ("c", 100, 70)])
        found = cli._endpoint_candidates(db, provider="openrouter", now=NOW)
        assert {cad for _, _, _, cad in found} == {3 * policies.endpoint_tier_interval_seconds(3)}


class TestFlagships:
    def test_newest_model_per_first_party_vendor_joins_the_core(self, db):
        seed_population(db, models=22)
        seed_model(db, "anthropic/claude-opus-5", [("anthropic", 100, 40)])
        seed_model(db, "anthropic/claude-fable-5.1", [("anthropic", 100, 30), ("google-vertex", 99, 50)])
        seed_model(db, "openai/gpt-5.6-terra", [("openai", 100, 60)])
        seed_model(db, "vendor/not-first-party", [("a", 100, 999)])
        db.openrouter_catalog.insert_many(
            [
                {"openrouter_id": "anthropic/claude-opus-5", "created": NOW - timedelta(days=90)},
                {"openrouter_id": "anthropic/claude-fable-5.1", "created": NOW - timedelta(days=40)},
                {"openrouter_id": "openai/gpt-5.6-terra", "first_seen_at": NOW - timedelta(days=60)},
                # Older than the recency window, so only the vendor rule could pick it up.
                {"openrouter_id": "vendor/not-first-party", "created": NOW - timedelta(days=30)},
            ]
        )
        members = {(m["model_id"], m["endpoint_tag"]): m for m in core_set.select(db, now=NOW)}
        assert ("anthropic/claude-fable-5.1", "anthropic") in members
        assert ("anthropic/claude-opus-5", "anthropic") not in members
        assert members[("anthropic/claude-fable-5.1", "anthropic")]["reason"].startswith(
            "newest anthropic model on OpenRouter (created"
        )
        assert members[("openai/gpt-5.6-terra", "openai")]["reason"].startswith(
            "newest openai model on OpenRouter (first seen"
        )
        assert not any(m == "vendor/not-first-party" for m, _ in members)

    def test_a_flagship_without_an_enabled_endpoint_is_not_invented(self, db):
        seed_population(db, models=5)
        db.openrouter_catalog.insert_one({"openrouter_id": "google/gemini-3.8-flash", "created": NOW})
        assert not any(m["model_id"].startswith("google/") for m in core_set.select(db, now=NOW))
