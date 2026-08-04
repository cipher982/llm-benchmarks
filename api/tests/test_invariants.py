from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import invariants

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    # A database per test. mongomock shares state across clients in-process,
    # so a fixed name lets other test modules leak documents into these checks.
    return mongomock.MongoClient()[f"inv-{request.node.name}"]


def enable(db, provider, model_id, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True, **extra})


def metric(db, provider, model_name, *, ago=timedelta(minutes=5)):
    db.metrics_cloud_v2.insert_one({"provider": provider, "model_name": model_name, "run_ts": NOW - ago})


def names(violations):
    return sorted(v.subject for v in violations)


class TestNoWorkForDisabledModels:
    def test_passes_when_queue_matches_catalogue(self, db):
        enable(db, "groq", "live")
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "live", "status": "queued"})
        assert invariants.no_work_for_disabled_models(db, NOW) == []

    def test_flags_jobs_for_disabled_models(self, db):
        db.models.insert_one({"provider": "groq", "model_id": "off", "enabled": False})
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "off", "status": "queued"})
        db.bench_jobs.insert_one({"_id": "j2", "provider": "groq", "model_id": "ghost", "status": "running"})
        assert names(invariants.no_work_for_disabled_models(db, NOW)) == ["groq/ghost", "groq/off"]

    def test_ignores_terminal_jobs(self, db):
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "off", "status": "dead_letter"})
        assert invariants.no_work_for_disabled_models(db, NOW) == []


class TestEveryProviderIsProgressing:
    def test_flags_a_silent_lane_even_when_others_are_busy(self, db):
        enable(db, "openai", "gpt")
        enable(db, "together", "llama")
        for _ in range(50):
            metric(db, "openai", "gpt")
        assert names(invariants.every_provider_is_progressing(db, NOW)) == ["together"]

    def test_stale_metrics_do_not_count_as_progress(self, db):
        enable(db, "groq", "llama")
        metric(db, "groq", "llama", ago=timedelta(hours=6))
        assert names(invariants.every_provider_is_progressing(db, NOW)) == ["groq"]


class TestCatalogueIsFresh:
    def test_flags_a_catalogue_that_stopped_being_written(self, db):
        db.provider_catalog.insert_one({"provider": "groq", "last_seen_at": NOW - timedelta(days=97)})
        violations = invariants.catalogue_is_fresh(db, NOW)
        assert len(violations) == 1 and "97d old" in violations[0].detail

    def test_passes_on_a_recent_write(self, db):
        db.provider_catalog.insert_one({"provider": "groq", "last_seen_at": NOW - timedelta(hours=6)})
        assert invariants.catalogue_is_fresh(db, NOW) == []

    def test_empty_catalogue_is_a_violation_not_a_pass(self, db):
        assert len(invariants.catalogue_is_fresh(db, NOW)) == 1


class TestNoCaseDuplicateModels:
    def test_flags_the_same_model_enabled_under_two_spellings(self, db):
        enable(db, "together", "Qwen/Qwen2.5-7B-Instruct-Turbo")
        enable(db, "together", "qwen/qwen2.5-7b-instruct-turbo")
        violations = invariants.no_case_duplicate_models(db, NOW)
        assert len(violations) == 1
        assert len(violations[0].data["model_ids"]) == 2

    def test_distinct_models_are_not_duplicates(self, db):
        enable(db, "together", "Qwen/Qwen3-8B")
        enable(db, "together", "Qwen/Qwen3-32B")
        assert invariants.no_case_duplicate_models(db, NOW) == []


class TestEnabledModelsAreBeingMeasured:
    def test_flags_a_starved_model(self, db):
        enable(db, "groq", "measured")
        enable(db, "groq", "starved")
        metric(db, "groq", "measured")
        assert names(invariants.enabled_models_are_being_measured(db, NOW)) == ["groq/starved"]


class TestProviderVolumeWithinBand:
    def _baseline(self, db, provider, per_day, days=10):
        for d in range(2, 2 + days):
            for _ in range(per_day):
                metric(db, provider, "m", ago=timedelta(days=d))

    def test_flags_a_partial_collapse(self, db):
        self._baseline(db, "together", 100)
        metric(db, "together", "m", ago=timedelta(hours=3))
        assert names(invariants.provider_volume_within_band(db, NOW)) == ["together"]

    def test_normal_volume_passes(self, db):
        self._baseline(db, "together", 10)
        for _ in range(10):
            metric(db, "together", "m", ago=timedelta(hours=3))
        assert invariants.provider_volume_within_band(db, NOW) == []

    def test_insufficient_history_is_not_a_violation(self, db):
        metric(db, "new-provider", "m", ago=timedelta(days=2))
        assert invariants.provider_volume_within_band(db, NOW) == []


class TestEvaluate:
    def test_a_check_that_raises_reads_as_failed_not_green(self, db, monkeypatch):
        def boom(_db, _now):
            raise RuntimeError("mongo went away")

        monkeypatch.setattr(
            invariants,
            "INVARIANTS",
            [invariants.Invariant("explodes", "always raises", boom)],
        )
        result = invariants.evaluate(db, now=NOW)[0]
        assert result.ok is False
        assert "mongo went away" in result.error

    def test_only_filter_runs_a_subset(self, db):
        results = invariants.evaluate(db, now=NOW, only={"catalogue_is_fresh"})
        assert [r.name for r in results] == ["catalogue_is_fresh"]
