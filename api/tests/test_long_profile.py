"""Periodic long-generation samples under `cloud-long-v1`.

Every model also gets one 512-token run per eligibility window, so a downstream
estimator can regress generate_time on generated tokens. Long rows publish to
the normal metrics collection carrying their profile id; long outcomes touch
only `long_profile_state` on health, never the model's primary health.
"""

import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import desired_set
from llm_bench.ops import invariants
from llm_bench.ops import long_profile
from llm_bench.ops.reliability_cli import _last_success_ts
from llm_bench.scheduler import health
from llm_bench.scheduler import queue
from llm_bench.scheduler import runner
from llm_bench.scheduler import worker
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.runner import RunnerResult

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

METRICS = {"output_tokens": 512, "generate_time": 8.0, "tokens_per_second": 64.0}


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _model(db, provider, model_id, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True, **extra})


def _long_row(db, provider, model_id, *, age):
    db[metrics_collection_name()].insert_one(
        {
            "provider": provider,
            "model_name": model_id,
            "benchmark_profile_id": long_profile.PROFILE_ID,
            "run_ts": NOW - age,
        }
    )


class TestSelection:
    def test_an_enabled_model_with_no_long_history_is_due(self, db):
        _model(db, "groq", "llama-fast")

        assert long_profile.enqueue_long_samples(db, now=NOW) == ["groq/llama-fast"]

        job = db.bench_jobs.find_one({"job_kind": long_profile.JOB_KIND})
        assert job["sample_role"] == long_profile.SAMPLE_ROLE
        assert job["benchmark_profile_id"] == long_profile.PROFILE_ID
        assert job["max_attempts"] == 1
        # Long rows must publish; the role is deliberately not a probe role.
        assert job["sample_role"] not in runner.NON_PUBLISHING_ROLES

    def test_a_recent_long_success_is_not_repeated(self, db):
        _model(db, "groq", "llama-fast")
        _long_row(db, "groq", "llama-fast", age=timedelta(hours=1))

        assert long_profile.enqueue_long_samples(db, now=NOW) == []

    def test_a_long_success_older_than_the_window_is_due_again(self, db):
        _model(db, "groq", "llama-fast")
        _long_row(db, "groq", "llama-fast", age=timedelta(hours=7))

        assert long_profile.enqueue_long_samples(db, now=NOW) == ["groq/llama-fast"]

    def test_a_failed_attempt_delays_one_window_but_never_blocks(self, db):
        """A model that fails only long runs retries next window, quietly."""
        _model(db, "groq", "llama-fast")
        health.record_long_profile_attempt(
            db,
            provider="groq",
            model_id="llama-fast",
            status="error",
            error_kind="timeout",
            error_message="benchmark timed out after 180s",
            now=NOW - timedelta(hours=1),
        )

        assert long_profile.enqueue_long_samples(db, now=NOW) == []
        assert long_profile.enqueue_long_samples(db, now=NOW + timedelta(hours=6)) == ["groq/llama-fast"]

    def test_zero_hours_disables_long_runs_entirely(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_LONG_PROFILE_HOURS", "0")
        _model(db, "groq", "llama-fast")

        assert not long_profile.enabled()
        assert long_profile.enqueue_long_samples(db, now=NOW) == []
        assert db.bench_jobs.count_documents({}) == 0

    def test_excluded_models_are_exempt(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_LONG_PROFILE_EXCLUDE", "openai/gpt-expensive, groq/llama-fast")
        _model(db, "groq", "llama-fast")
        _model(db, "openai", "gpt-expensive")
        _model(db, "openai", "gpt-cheap")

        assert long_profile.enqueue_long_samples(db, now=NOW) == ["openai/gpt-cheap"]

    def test_shadow_measured_models_are_exempt(self, db):
        # Already measured at 2048 tokens by the reasoning shadow pipeline.
        _model(db, "deepinfra", "MiniMaxAI/MiniMax-M2")
        db.bench_jobs.insert_one(
            {
                "_id": "deepinfra:MiniMaxAI/MiniMax-M2",
                "provider": "deepinfra",
                "model_id": "MiniMaxAI/MiniMax-M2",
                "status": "dead_letter",
                "last_attempt_error_kind": "budget_exhausted",
                "updated_at": NOW,
            }
        )
        # Covered by a recent shadow row even though the trigger has moved on.
        _model(db, "openai", "o9-reasoner")
        db["metrics_cloud_probe"].insert_one(
            {
                "provider": "openai",
                "model_name": "o9-reasoner",
                "benchmark_profile_id": "cloud-reasoning-v1",
                "run_ts": NOW - timedelta(days=2),
            }
        )
        _model(db, "groq", "llama-fast")

        assert long_profile.enqueue_long_samples(db, now=NOW) == ["groq/llama-fast"]

    def test_disabled_and_deprecated_models_are_not_selected(self, db):
        _model(db, "groq", "off-model", enabled=False)
        _model(db, "together", "old-model", deprecated=True)

        assert long_profile.enqueue_long_samples(db, now=NOW) == []

    def test_a_pending_long_job_is_not_duplicated(self, db):
        _model(db, "groq", "llama-fast")
        assert long_profile.enqueue_long_samples(db, now=NOW) == ["groq/llama-fast"]

        assert long_profile.enqueue_long_samples(db, now=NOW + timedelta(hours=8)) == []

    def test_the_cap_serves_the_stalest_model_first(self, db):
        """Bound the work per pass, never which models can be reached."""
        _model(db, "groq", "sampled-recently")
        _long_row(db, "groq", "sampled-recently", age=timedelta(hours=7))
        _model(db, "groq", "never-sampled")

        assert long_profile.enqueue_long_samples(db, now=NOW, limit=1) == ["groq/never-sampled"]
        assert long_profile.enqueue_long_samples(db, now=NOW, limit=1) == ["groq/sampled-recently"]

    def test_a_long_job_on_an_enabled_model_is_eligible(self, db):
        _model(db, "groq", "llama-fast")

        assert queue.is_model_eligible(db, provider="groq", model_id="llama-fast", sample_role="long")


class TestTimeout:
    def test_long_jobs_carry_a_scaled_deadline(self, db):
        _model(db, "groq", "llama-fast")
        long_profile.enqueue_long_samples(db, now=NOW)

        job = db.bench_jobs.find_one({"job_kind": long_profile.JOB_KIND})
        # 8x the tokens; the 120s default deadline would record timeouts that
        # say nothing about the model.
        assert job["deadline_seconds"] == 180

    def test_the_deadline_is_configurable(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_LONG_PROFILE_TIMEOUT_SECONDS", "400")
        _model(db, "groq", "llama-fast")
        long_profile.enqueue_long_samples(db, now=NOW)

        job = db.bench_jobs.find_one({"job_kind": long_profile.JOB_KIND})
        assert job["deadline_seconds"] == 400


class TestProfile:
    def test_the_long_profile_asks_for_512_tokens(self):
        assert runner.profile_max_tokens(long_profile.PROFILE_ID) == 512
        assert runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID) == 64

    def test_long_samples_publish_to_the_metrics_collection(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        captured = {}
        monkeypatch.setattr(runner, "log_mongo", lambda **kw: captured.update(kw))
        runner.log_success_mongo(object(), {}, sample_role=runner.SAMPLE_ROLE_LONG)
        assert captured["collection_name"] == metrics_collection_name()

    def test_a_long_job_writes_its_profile_and_requested_budget(self, monkeypatch):
        written = {}
        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        monkeypatch.setattr(
            runner,
            "log_success_mongo",
            lambda config, metrics, *, sample_role: written.update(metrics, _role=sample_role),
        )

        result = runner.run_benchmark_job(
            {
                "_id": "long-1",
                "provider": "groq",
                "model_id": "llama-fast",
                "sample_role": long_profile.SAMPLE_ROLE,
                "benchmark_profile_id": long_profile.PROFILE_ID,
            }
        )

        assert result.status == "success"
        assert result.sample_role == runner.SAMPLE_ROLE_LONG
        assert written["benchmark_profile_id"] == long_profile.PROFILE_ID
        assert written["requested_max_tokens"] == 512
        assert written["sample_role"] == runner.SAMPLE_ROLE_LONG

    def test_profile_and_budget_survive_the_write_layer(self, monkeypatch):
        """log_mongo copies only allowlisted keys; a mocked writer cannot see that."""
        from llm_bench.logging import _optional_metric_fields

        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        written = {}
        monkeypatch.setattr(
            runner,
            "log_success_mongo",
            lambda config, metrics, *, sample_role: written.update(metrics),
        )
        runner.run_benchmark_job(
            {
                "_id": "long-1",
                "provider": "groq",
                "model_id": "llama-fast",
                "sample_role": long_profile.SAMPLE_ROLE,
                "benchmark_profile_id": long_profile.PROFILE_ID,
            }
        )

        persisted = _optional_metric_fields(written)
        for field in ("benchmark_profile_id", "requested_max_tokens", "sample_role"):
            assert field in persisted, f"{field} is set by the runner but dropped before Mongo"
        assert persisted["benchmark_profile_id"] == long_profile.PROFILE_ID
        assert persisted["requested_max_tokens"] == 512

    def test_default_jobs_are_unchanged(self, monkeypatch):
        """A job with no profile still measures under the default rules."""
        written = {}
        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        monkeypatch.setattr(
            runner,
            "log_success_mongo",
            lambda config, metrics, *, sample_role: written.update(metrics, _role=sample_role),
        )

        result = runner.run_benchmark_job({"_id": "job-1", "provider": "groq", "model_id": "llama-fast"})

        assert result.status == "success"
        assert result.sample_role == runner.SAMPLE_ROLE_PUBLISHED
        assert written["benchmark_profile_id"] == runner.DEFAULT_PROFILE_ID
        assert written["requested_max_tokens"] == 64


class TestHealthIsolation:
    def test_long_outcomes_never_touch_primary_health(self, db):
        db.bench_model_health.insert_one(
            {
                "provider": "groq",
                "model_id": "llama-fast",
                "consecutive_failures": 0,
                "freshness_status": "fresh",
                "last_success_at": NOW - timedelta(minutes=5),
            }
        )
        health.record_long_profile_attempt(
            db,
            provider="groq",
            model_id="llama-fast",
            status="success",
            now=NOW - timedelta(hours=1),
        )
        health.record_long_profile_attempt(
            db,
            provider="groq",
            model_id="llama-fast",
            status="error",
            error_kind="timeout",
            error_message="benchmark timed out after 180s",
            now=NOW,
        )

        doc = db.bench_model_health.find_one({"provider": "groq", "model_id": "llama-fast"})
        state = doc["long_profile_state"]
        assert state["last_status"] == "error"
        assert state["last_error_kind"] == "timeout"
        assert state["last_attempt_at"] == NOW.replace(tzinfo=None) or state["last_attempt_at"] == NOW
        # A later failure must not erase the last long success.
        assert state["last_success_at"] is not None
        # Primary health is untouched by both outcomes.
        assert doc["consecutive_failures"] == 0
        assert doc["freshness_status"] == "fresh"
        assert doc["last_success_at"] == NOW.replace(tzinfo=None) - timedelta(minutes=5) or doc[
            "last_success_at"
        ] == NOW - timedelta(minutes=5)

    def test_long_only_success_does_not_satisfy_recent_counts(self, db):
        """successes_24h means the published series, not any row in the collection."""
        _long_row(db, "groq", "llama-fast", age=timedelta(hours=1))
        db["errors_cloud"].insert_one(
            {
                "provider": "groq",
                "model_name": "llama-fast",
                "ts": NOW - timedelta(hours=1),
                "error_kind": "timeout",
                "profile_id": long_profile.PROFILE_ID,
            }
        )

        successes, failures, deadline_misses = health._recent_counts(
            db, provider="groq", model_id="llama-fast", now=NOW
        )
        assert (successes, failures, deadline_misses) == (0, 0, 0)

        # Pre-profile rows carry no profile field and must still count.
        db[metrics_collection_name()].insert_one(
            {"provider": "groq", "model_name": "llama-fast", "run_ts": NOW - timedelta(minutes=30)}
        )
        successes, _, _ = health._recent_counts(db, provider="groq", model_id="llama-fast", now=NOW)
        assert successes == 1

    def test_long_only_rows_do_not_backfill_freshness(self, db):
        _model(db, "groq", "llama-fast")
        _long_row(db, "groq", "llama-fast", age=timedelta(minutes=10))

        health.backfill_from_metrics(db, cadence_seconds=1800, now=NOW)

        doc = db.bench_model_health.find_one({"provider": "groq", "model_id": "llama-fast"})
        assert doc["last_success_at"] is None
        assert doc["freshness_status"] == "never_run"

    def test_long_only_rows_are_not_lane_progress(self, db):
        _long_row(db, "groq", "llama-fast", age=timedelta(minutes=5))

        progress = health.provider_progress(db, providers=["groq"], now=NOW)
        assert progress["groq"]["latest_completed_at"] is None

        db[metrics_collection_name()].insert_one(
            {"provider": "groq", "model_name": "llama-fast", "run_ts": NOW - timedelta(minutes=5)}
        )
        progress = health.provider_progress(db, providers=["groq"], now=NOW)
        assert progress["groq"]["latest_completed_at"] is not None

    def test_long_rows_do_count_as_process_liveness(self, db):
        """Deliberate exception: a completed long run proves the process is
        alive, and restarting the container would not fix a published-series
        gap. Process liveness is not published progress."""
        _long_row(db, "groq", "llama-fast", age=timedelta(minutes=5))
        db["bench_scheduler_heartbeats"].insert_one({"_id": "scheduler", "updated_at": NOW - timedelta(minutes=1)})

        healthy, details = health.liveness_status(db, max_idle_seconds=900, now=NOW)
        assert healthy is True
        assert details["reason"] == "ok"


class TestPublishedProgressIsolation:
    """A model succeeding only at cloud-long-v1 is not being measured for the site."""

    def test_long_only_success_is_still_starved_for_the_invariants(self, db):
        db.models.insert_one({"provider": "groq", "model_id": "long-only", "enabled": True})
        desired_set.capture(db, now=NOW - timedelta(hours=6))
        db[metrics_collection_name()].insert_one(
            {
                "provider": "groq",
                "model_name": "long-only",
                "benchmark_profile_id": long_profile.PROFILE_ID,
                "run_ts": NOW - timedelta(minutes=5),
            }
        )
        ctx = invariants.Context(db=db, now=NOW)

        starved = invariants.desired_models_are_being_measured(ctx)
        assert [v.subject for v in starved] == ["groq/long-only"]

        stalled = invariants.every_provider_is_progressing(ctx)
        assert [v.subject for v in stalled] == ["groq"]

        # A published row clears both.
        db[metrics_collection_name()].insert_one(
            {"provider": "groq", "model_name": "long-only", "run_ts": NOW - timedelta(minutes=5)}
        )
        ctx = invariants.Context(db=db, now=NOW)
        assert invariants.desired_models_are_being_measured(ctx) == []
        assert invariants.every_provider_is_progressing(ctx) == []

    def test_long_success_does_not_advance_last_success_for_quarantine(self, db):
        """A long success advancing "since last success" would hide default-
        profile hard failures from catalog_quarantine's evidence window."""
        db[metrics_collection_name()].insert_one(
            {
                "provider": "groq",
                "model_name": "llama-fast",
                "benchmark_profile_id": long_profile.PROFILE_ID,
                "run_ts": NOW - timedelta(hours=1),
            }
        )
        assert _last_success_ts(db=db, provider="groq", model="llama-fast", lookback_days=30) is None

        published_at = NOW - timedelta(hours=3)
        db[metrics_collection_name()].insert_one(
            {"provider": "groq", "model_name": "llama-fast", "run_ts": published_at}
        )
        found = _last_success_ts(db=db, provider="groq", model="llama-fast", lookback_days=30)
        assert found is not None
        assert found.replace(tzinfo=timezone.utc) == published_at

    def test_lifecycle_collector_ignores_long_rows_and_long_errors(self, db):
        from llm_bench.model_lifecycle.collector import _load_error_metrics
        from llm_bench.model_lifecycle.collector import _load_success_metrics

        # Naive datetimes throughout: real Mongo returns naive UTC, and the
        # collector's pipeline comparisons cannot mix naive and aware.
        naive_now = NOW.replace(tzinfo=None)
        db[metrics_collection_name()].insert_one(
            {
                "provider": "groq",
                "model_name": "long-only",
                "benchmark_profile_id": long_profile.PROFILE_ID,
                "run_ts": naive_now - timedelta(hours=1),
            }
        )
        db[metrics_collection_name()].insert_one(
            {"provider": "groq", "model_name": "published", "run_ts": naive_now - timedelta(hours=1)}
        )
        db["errors_cloud"].insert_one(
            {
                "provider": "groq",
                "model_name": "long-only",
                "ts": naive_now - timedelta(hours=1),
                "error_kind": "timeout",
                "message": "timed out",
                "profile_id": long_profile.PROFILE_ID,
            }
        )
        db["errors_cloud"].insert_one(
            {
                "provider": "groq",
                "model_name": "published",
                "ts": naive_now - timedelta(hours=1),
                "error_kind": "timeout",
                "message": "timed out",
            }
        )
        successes = _load_success_metrics(db[metrics_collection_name()], None, naive_now)
        assert ("groq", "published", "direct") in successes
        assert not any(model == "long-only" for _, model, _ in successes)

        errors = _load_error_metrics(db["errors_cloud"], None, naive_now)
        assert ("groq", "published", "direct") in errors
        assert not any(model == "long-only" for _, model, _ in errors)

    def test_worker_routes_long_outcomes_to_long_profile_state(self, monkeypatch):
        """Success records long state, failure never reaches record_error."""
        recorded = []
        stop_event = threading.Event()
        jobs = iter(
            [
                {
                    "_id": "long:groq:m:1",
                    "provider": "groq",
                    "model_id": "m",
                    "sample_role": "long",
                    "benchmark_profile_id": long_profile.PROFILE_ID,
                    "job_kind": "long_profile",
                    "attempt": 1,
                    "deadline_seconds": 180,
                },
                {
                    "_id": "long:groq:m:2",
                    "provider": "groq",
                    "model_id": "m",
                    "sample_role": "long",
                    "benchmark_profile_id": long_profile.PROFILE_ID,
                    "job_kind": "long_profile",
                    "attempt": 1,
                    "deadline_seconds": 180,
                },
            ]
        )
        results = iter(
            [
                RunnerResult(status="success", sample_role="long"),
                RunnerResult(status="error", error_kind="timeout", error_message="timed out"),
            ]
        )

        def claim(*args, **kwargs):
            try:
                return next(jobs)
            except StopIteration:
                stop_event.set()
                return None

        class FakeClient:
            def __getitem__(self, name):
                return mongomock.MongoClient()["llm-bench"]

            def close(self):
                pass

        monkeypatch.setattr(worker, "mongo_env", lambda: ("unused", "test"))
        monkeypatch.setattr(worker, "mongo_client", lambda: FakeClient())
        monkeypatch.setattr(worker.queue, "claim_next_job", claim)
        monkeypatch.setattr(worker.queue, "mark_success", lambda *a, **k: True)
        monkeypatch.setattr(worker.queue, "mark_failure", lambda *a, **k: "dead_letter")
        monkeypatch.setattr(worker, "run_job_in_child", lambda job, *, deadline_seconds: next(results))
        monkeypatch.setattr(worker.health, "heartbeat", lambda *a, **k: None)
        monkeypatch.setattr(
            worker.health,
            "record_long_profile_attempt",
            lambda *a, **k: recorded.append(("long", k["status"])),
        )
        monkeypatch.setattr(worker.health, "record_success", lambda *a, **k: recorded.append(("primary_success",)))
        monkeypatch.setattr(worker.health, "record_error", lambda *a, **k: recorded.append(("primary_error",)))

        worker.run_worker_loop(
            provider="groq",
            slot=0,
            cadence_seconds=1800,
            stop_event=stop_event,
            idle_sleep_seconds=0,
        )

        assert recorded == [("long", "success"), ("long", "error")]


def test_workerless_provider_models_are_never_enqueued(db):
    """Bedrock has no worker on this host; a long job for it would sit queued
    forever and trip no_job_is_stuck_in_queue. Same class as the admission
    probe leak, and it appeared the moment the long sampler launched."""
    _model(db, "bedrock", "us.anthropic.claude-sonnet-4-6")
    _model(db, "groq", "qwen/qwen3.6-27b")
    enqueued = long_profile.enqueue_long_samples(db, now=NOW)
    assert enqueued == ["groq/qwen/qwen3.6-27b"]
    assert db[queue.jobs_collection_name()].count_documents({"provider": "bedrock"}) == 0
    assert db[queue.jobs_collection_name()].count_documents({"provider": "groq"}) == 1
