from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import desired_set
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


def snapshot(db, *, ago=timedelta(hours=6)):
    """Capture a settled desired set, i.e. one old enough to be judged against."""
    return desired_set.capture(db, now=NOW - ago)


def ctx(db, **kwargs):
    return invariants.Context(db=db, now=NOW, **kwargs)


def names(violations):
    return sorted(v.subject for v in violations)


class TestNoWorkForDisabledModels:
    def test_passes_when_queue_matches_catalogue(self, db):
        enable(db, "groq", "live")
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "live", "status": "queued"})
        assert invariants.no_work_for_disabled_models(ctx(db)) == []

    def test_flags_jobs_for_disabled_models(self, db):
        db.models.insert_one({"provider": "groq", "model_id": "off", "enabled": False})
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "off", "status": "queued"})
        db.bench_jobs.insert_one({"_id": "j2", "provider": "groq", "model_id": "ghost", "status": "running"})
        assert names(invariants.no_work_for_disabled_models(ctx(db))) == ["groq/ghost", "groq/off"]

    def test_ignores_terminal_jobs(self, db):
        db.bench_jobs.insert_one({"_id": "j1", "provider": "groq", "model_id": "off", "status": "dead_letter"})
        assert invariants.no_work_for_disabled_models(ctx(db)) == []


class TestNoCaseDuplicateModels:
    def test_flags_the_same_model_enabled_under_two_spellings(self, db):
        enable(db, "together", "Qwen/Qwen2.5-7B-Instruct-Turbo")
        enable(db, "together", "qwen/qwen2.5-7b-instruct-turbo")
        violations = invariants.no_case_duplicate_models(ctx(db))
        assert len(violations) == 1
        assert len(violations[0].data["model_ids"]) == 2

    def test_distinct_models_are_not_duplicates(self, db):
        enable(db, "together", "Qwen/Qwen3-8B")
        enable(db, "together", "Qwen/Qwen3-32B")
        assert invariants.no_case_duplicate_models(ctx(db)) == []


class TestNoJobIsStuckInQueue:
    def test_flags_work_that_never_gets_claimed(self, db):
        db.bench_jobs.insert_one(
            {"_id": "j1", "provider": "vertex", "status": "queued", "created_at": NOW - timedelta(hours=9)}
        )
        assert names(invariants.no_job_is_stuck_in_queue(ctx(db))) == ["vertex"]

    def test_recent_queued_work_is_normal(self, db):
        db.bench_jobs.insert_one(
            {"_id": "j1", "provider": "vertex", "status": "queued", "created_at": NOW - timedelta(minutes=10)}
        )
        assert invariants.no_job_is_stuck_in_queue(ctx(db)) == []


class TestEveryProviderIsProgressing:
    def test_flags_a_silent_lane_even_when_others_are_busy(self, db):
        enable(db, "openai", "gpt")
        enable(db, "together", "llama")
        snapshot(db)
        for _ in range(50):
            metric(db, "openai", "gpt")
        assert names(invariants.every_provider_is_progressing(ctx(db))) == ["together"]

    def test_stale_metrics_do_not_count_as_progress(self, db):
        enable(db, "groq", "llama")
        snapshot(db)
        metric(db, "groq", "llama", ago=timedelta(hours=6))
        assert names(invariants.every_provider_is_progressing(ctx(db))) == ["groq"]

    def test_cannot_evaluate_without_a_settled_snapshot(self, db):
        enable(db, "groq", "llama")
        with pytest.raises(invariants.CannotEvaluate):
            invariants.every_provider_is_progressing(ctx(db))


class TestDesiredModelsAreBeingMeasured:
    def test_flags_a_starved_model(self, db):
        enable(db, "groq", "measured")
        enable(db, "groq", "starved")
        snapshot(db)
        metric(db, "groq", "measured")
        assert names(invariants.desired_models_are_being_measured(ctx(db))) == ["groq/starved"]

    def test_disabling_the_starved_model_does_not_make_the_check_pass(self, db):
        """The defect both reviews found: remediation satisfying its own check.

        Disabling a model it just reported as starved is the cheapest way for a
        broken detector to go green, so the snapshot — not live catalogue state
        — has to be the denominator.
        """
        enable(db, "groq", "starved")
        snapshot(db)
        violations = invariants.desired_models_are_being_measured(ctx(db))
        assert names(violations) == ["groq/starved"]

        db.models.update_one({"model_id": "starved"}, {"$set": {"enabled": False}})

        violations = invariants.desired_models_are_being_measured(ctx(db))
        assert names(violations) == ["groq/starved"]
        assert violations[0].data["removed_since_snapshot"] is True

    def test_a_snapshot_taken_after_the_fact_is_not_used(self, db):
        """A snapshot younger than the settling window cannot license the present."""
        enable(db, "groq", "starved")
        desired_set.capture(db, now=NOW - timedelta(minutes=1))
        with pytest.raises(invariants.CannotEvaluate):
            invariants.desired_models_are_being_measured(ctx(db))


class TestDesiredSetIsNotSilentlyShrinking:
    def test_flags_mass_removal_since_the_snapshot(self, db):
        for i in range(20):
            enable(db, "together", f"m{i}")
        snapshot(db)
        db.models.update_many({"model_id": {"$in": [f"m{i}" for i in range(10)]}}, {"$set": {"enabled": False}})
        violations = invariants.desired_set_is_not_silently_shrinking(ctx(db))
        assert len(violations) == 1
        assert violations[0].data["removed_count"] == 10

    def test_a_single_demotion_is_within_tolerance(self, db):
        for i in range(20):
            enable(db, "together", f"m{i}")
        snapshot(db)
        db.models.update_one({"model_id": "m0"}, {"$set": {"enabled": False}})
        assert invariants.desired_set_is_not_silently_shrinking(ctx(db)) == []


class TestDiscoveryCompletedRecently:
    def test_no_ledger_is_unevaluable_not_a_pass(self, db):
        enable(db, "groq", "llama")
        snapshot(db)
        with pytest.raises(invariants.CannotEvaluate):
            invariants.discovery_completed_recently(ctx(db))

    def test_flags_a_provider_with_no_completed_run(self, db):
        enable(db, "groq", "llama")
        enable(db, "together", "qwen")
        snapshot(db)
        db.bench_discovery_runs.insert_one(
            {
                "provider": "groq",
                "status": "completed",
                "finished_at": NOW - timedelta(hours=3),
                "pagination_complete": True,
            }
        )
        assert names(invariants.discovery_completed_recently(ctx(db))) == ["together"]

    def test_a_partial_run_does_not_count_as_a_sync(self, db):
        """A truncated catalogue read must never become deprecation evidence."""
        enable(db, "anthropic", "claude")
        snapshot(db)
        db.bench_discovery_runs.insert_one(
            {
                "provider": "anthropic",
                "status": "completed",
                "finished_at": NOW - timedelta(hours=1),
                "pagination_complete": False,
            }
        )
        assert names(invariants.discovery_completed_recently(ctx(db))) == ["anthropic"]

    def test_a_recent_complete_run_passes(self, db):
        enable(db, "anthropic", "claude")
        snapshot(db)
        db.bench_discovery_runs.insert_one(
            {
                "provider": "anthropic",
                "status": "completed",
                "finished_at": NOW - timedelta(hours=1),
                "pagination_complete": True,
            }
        )
        assert invariants.discovery_completed_recently(ctx(db)) == []


class TestTerminalReasonsAreCurrent:
    def test_flags_an_old_reason_with_no_recheck(self, db):
        db.models.insert_one(
            {
                "provider": "deepinfra",
                "model_id": "llama",
                "enabled": False,
                "disabled_reason": "billing",
                "disabled_at": NOW - timedelta(days=40),
            }
        )
        assert names(invariants.terminal_reasons_are_current(ctx(db))) == ["deepinfra/llama"]

    def test_flags_an_overdue_recheck(self, db):
        db.models.insert_one(
            {
                "provider": "groq",
                "model_id": "llama",
                "enabled": False,
                "disabled_reason": "auth",
                "disabled_at": NOW - timedelta(days=1),
                "recheck_after": NOW - timedelta(hours=2),
            }
        )
        assert names(invariants.terminal_reasons_are_current(ctx(db))) == ["groq/llama"]

    def test_a_pending_recheck_is_not_a_violation(self, db):
        db.models.insert_one(
            {
                "provider": "groq",
                "model_id": "llama",
                "enabled": False,
                "disabled_reason": "auth",
                "disabled_at": NOW - timedelta(days=1),
                "recheck_after": NOW + timedelta(hours=2),
            }
        )
        assert invariants.terminal_reasons_are_current(ctx(db)) == []


class TestEvaluate:
    def test_a_check_that_raises_reads_as_failed_not_green(self, db, monkeypatch):
        def boom(_ctx):
            raise RuntimeError("mongo went away")

        monkeypatch.setattr(
            invariants,
            "INVARIANTS",
            [invariants.Invariant("explodes", "always raises", boom)],
        )
        result = invariants.evaluate(db, now=NOW)[0]
        assert result.ok is False
        assert "mongo went away" in result.error

    def test_missing_inputs_read_as_unevaluated_not_green(self, db):
        enable(db, "groq", "llama")
        result = next(r for r in invariants.evaluate(db, now=NOW) if r.name == "desired_models_are_being_measured")
        assert result.ok is False
        assert result.evaluated is False

    def test_every_run_is_recorded(self, db):
        invariants.evaluate(db, now=NOW, only={"no_case_duplicate_models"})
        run = db.bench_check_runs.find_one()
        assert run["threshold_version"] == invariants.THRESHOLD_VERSION
        assert [r["name"] for r in run["results"]] == ["no_case_duplicate_models"]

    def test_only_filter_runs_a_subset(self, db):
        results = invariants.evaluate(db, now=NOW, only={"no_case_duplicate_models"})
        assert [r.name for r in results] == ["no_case_duplicate_models"]
