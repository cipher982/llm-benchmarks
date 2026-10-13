"""The catalogue is the authority on what gets benchmarked.

Jobs outlive catalogue decisions. Before this gate, 88% of queued work in
production targeted models that had been disabled hours earlier: disabling a
model stopped new scheduling but left its existing jobs cycling, and the
dead-letter sweep kept resurrecting them.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import queue


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench-test"]


def _enable(db, provider, model_id, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True, **extra})


def test_eligible_when_enabled_and_not_deprecated(db):
    _enable(db, "groq", "llama-3.1-8b")
    assert queue.is_model_eligible(db, provider="groq", model_id="llama-3.1-8b")


def test_not_eligible_when_disabled_deprecated_or_absent(db):
    db.models.insert_one({"provider": "groq", "model_id": "off", "enabled": False})
    _enable(db, "groq", "old", deprecated=True)
    assert not queue.is_model_eligible(db, provider="groq", model_id="off")
    assert not queue.is_model_eligible(db, provider="groq", model_id="old")
    assert not queue.is_model_eligible(db, provider="groq", model_id="never-heard-of-it")


# claim_next_job's eligibility gate is not covered here: it updates via an
# aggregation pipeline using $dateAdd, which mongomock cannot execute. The gate
# itself is is_model_eligible, covered above.


def test_cancel_ineligible_jobs_sweeps_the_existing_queue(db):
    _enable(db, "groq", "live")
    queue.enqueue_manual_job(db, provider="groq", model_id="live")
    queue.enqueue_manual_job(db, provider="groq", model_id="dropped")
    queue.enqueue_manual_job(db, provider="together", model_id="also-dropped")

    assert queue.cancel_ineligible_jobs(db) == 2
    assert db.bench_jobs.count_documents({"status": "cancelled"}) == 2
    assert db.bench_jobs.find_one({"model_id": "live"})["status"] == "queued"


def _endpoint(db, model_id, tag, *, enabled=True):
    db[endpoint_discovery.endpoints_collection_name()].insert_one(
        {"model_id": model_id, "endpoint_tag": tag, "enabled": enabled}
    )


def test_endpoint_catalogue_supersedes_unpinned_and_retired_scheduled_jobs(db):
    _enable(db, "openrouter", "m")
    _endpoint(db, "m", "alpha")
    _endpoint(db, "m", "retired", enabled=False)

    assert (
        queue.job_ineligibility_reason(
            db,
            {"provider": "openrouter", "model_id": "m", "job_kind": "scheduled"},
        )
        == "superseded by endpoint rotation"
    )
    assert (
        queue.job_ineligibility_reason(
            db,
            {
                "provider": "openrouter",
                "model_id": "m",
                "job_kind": "scheduled",
                "endpoint_tag": "retired",
            },
        )
        == "endpoint no longer enabled"
    )
    assert (
        queue.job_ineligibility_reason(
            db,
            {
                "provider": "openrouter",
                "model_id": "m",
                "job_kind": "scheduled",
                "endpoint_tag": "alpha",
            },
        )
        is None
    )


def test_dead_letter_endpoint_stays_parked_while_sibling_is_active(db):
    _enable(db, "openrouter", "m")
    _endpoint(db, "m", "alpha")
    _endpoint(db, "m", "beta")
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert queue.enqueue_scheduled_job(
        db, provider="openrouter", model_id="m", endpoint_tag="alpha", priority=1, now=now
    )
    db.bench_jobs.update_one(
        {"_id": "openrouter:m:alpha"},
        {
            "$set": {
                "status": "dead_letter",
                "updated_at": now - timedelta(days=8),
                "last_attempt_error_kind": "rate_limit",
            }
        },
    )
    assert queue.enqueue_scheduled_job(
        db, provider="openrouter", model_id="m", endpoint_tag="beta", priority=1, now=now
    )

    assert queue.requeue_retryable_dead_letters(db, now=now) == []
    assert db.bench_jobs.find_one({"_id": "openrouter:m:alpha"})["status"] == "dead_letter"


def test_transport_exclusions_cancel_stale_work_without_excluding_direct_lanes(db, monkeypatch):
    model_id = "anthropic/new-model"
    _enable(db, "openrouter", model_id)
    _enable(db, "bedrock", model_id)
    routed = queue.enqueue_manual_job(db, provider="openrouter", model_id=model_id)
    direct = queue.enqueue_manual_job(db, provider="bedrock", model_id=model_id)
    monkeypatch.setenv("BENCHMARK_OPENROUTER_EXCLUDED_MODEL_PREFIXES", "anthropic/")

    assert not queue.enqueue_scheduled_job(db, provider="openrouter", model_id=model_id, priority=1)
    assert queue.cancel_ineligible_jobs(db) == 1
    assert db.bench_jobs.find_one({"_id": routed})["status"] == "cancelled"
    assert db.bench_jobs.find_one({"_id": direct})["status"] == "queued"


TERMINAL_NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _terminal_job(
    db,
    *,
    provider="groq",
    model_id="m",
    age=timedelta(days=8),
    error_kind="hard_model",
    dead_letter_requeues=0,
    cadence_seconds=None,
    endpoint_tag=None,
    enabled=True,
    legacy=False,
):
    _enable(db, provider, model_id, enabled=enabled)
    if endpoint_tag:
        _endpoint(db, model_id, endpoint_tag)
    job_id = queue.scheduled_job_id(provider, model_id, endpoint_tag)
    job = {
        "_id": job_id,
        "provider": provider,
        "model_id": model_id,
        "job_kind": "scheduled",
        "status": "dead_letter",
        "created_at": TERMINAL_NOW - age - timedelta(days=1),
        "updated_at": TERMINAL_NOW - age,
        "not_before": TERMINAL_NOW - age,
        "attempt": 2,
        "max_attempts": 2,
        "dead_letter_requeues": dead_letter_requeues,
        "last_attempt_error_kind": error_kind,
        "last_attempt_error_message": "terminal failure",
    }
    if legacy:
        job.pop("job_kind")
    if cadence_seconds is not None:
        job["cadence_seconds"] = cadence_seconds
    db.bench_jobs.insert_one(job)
    return job_id


def _catalogued(db, provider, model_id, *, last_seen_at=TERMINAL_NOW - timedelta(days=1)):
    db.provider_catalog.insert_one({"provider": provider, "model_id": model_id, "last_seen_at": last_seen_at})


def _fail_recovered_job(db, job_id, *, error_kind, now):
    db.bench_jobs.update_one(
        {"_id": job_id},
        {"$set": {"status": "running", "attempt": 1}},
    )
    job = db.bench_jobs.find_one({"_id": job_id})
    return queue.mark_failure(
        db,
        job=job,
        error_kind=error_kind,
        error_message=f"{error_kind} after slow recovery",
        now=now,
    )


def test_slow_terminal_recovery_releases_a_maxed_out_dead_letter(db):
    job_id = _terminal_job(db, dead_letter_requeues=20)
    _catalogued(db, "groq", "m")

    recovered = queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)

    assert [job["_id"] for job in recovered] == [job_id]
    job = db.bench_jobs.find_one({"_id": job_id})
    assert job["status"] == "queued"
    assert job["attempt"] == 0
    assert job["max_attempts"] == 1
    assert job["dead_letter_requeues"] == 0
    assert job["terminal_recovery_active"] is True
    assert job["terminal_original_error_kind"] == "hard_model"
    assert job["last_terminal_recheck_count"] == 1


def test_legacy_scheduled_dead_letter_is_recognized_and_normalized(db):
    job_id = _terminal_job(db, legacy=True)
    _enable(db, "groq", "manual-model")
    _catalogued(db, "groq", "m")
    _catalogued(db, "groq", "manual-model")
    db.bench_jobs.insert_one(
        {
            "_id": "manual:groq:manual-model:old",
            "provider": "groq",
            "model_id": "manual-model",
            "status": "dead_letter",
            "updated_at": TERMINAL_NOW - timedelta(days=8),
            "last_attempt_error_kind": "hard_model",
        }
    )

    recovered = queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)

    assert [job["_id"] for job in recovered] == [job_id]
    assert db.bench_jobs.find_one({"_id": job_id})["job_kind"] == "scheduled"
    assert db.bench_jobs.find_one({"_id": "manual:groq:manual-model:old"})["status"] == "dead_letter"


def test_legacy_openrouter_model_job_is_superseded_by_endpoint_rotation(db):
    _enable(db, "openrouter", "m")
    _endpoint(db, "m", "alpha")
    legacy_job = {
        "_id": queue.scheduled_job_id("openrouter", "m"),
        "provider": "openrouter",
        "model_id": "m",
        "status": "dead_letter",
    }

    assert queue.job_ineligibility_reason(db, legacy_job) == "superseded by endpoint rotation"


def test_terminal_recovery_waits_for_the_job_specific_slow_interval(db):
    _terminal_job(
        db,
        age=timedelta(days=11),
        cadence_seconds=8 * 24 * 60 * 60,
    )
    _catalogued(db, "groq", "m")

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW) == []
    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW + timedelta(days=1))


def test_pending_recovery_job_is_not_enqueued_twice(db):
    _terminal_job(db)
    _catalogued(db, "groq", "m")
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW + timedelta(days=10)) == []
    assert db.bench_jobs.count_documents({"model_id": "m"}) == 1


def test_routed_ceiling_refuses_slow_recovery_without_resetting_the_job(db, monkeypatch):
    monkeypatch.setenv("BENCHMARK_MAX_ROUTED_JOBS_PER_DAY", "1")
    job_id = _terminal_job(
        db,
        provider="openrouter",
        model_id="openai/test",
        dead_letter_requeues=7,
    )
    _catalogued(db, "openrouter", "openai/test")
    queue.record_admission(db, provider="openrouter", now=TERMINAL_NOW)

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW) == []
    job = db.bench_jobs.find_one({"_id": job_id})
    assert job["status"] == "dead_letter"
    assert job["dead_letter_requeues"] == 7
    assert "last_terminal_recheck_at" not in job


def test_slow_recovery_requires_recent_source_catalogue_evidence(db):
    absent_id = _terminal_job(db, model_id="absent")
    stale_id = _terminal_job(db, model_id="stale")
    _catalogued(
        db,
        "groq",
        "stale",
        last_seen_at=TERMINAL_NOW - timedelta(days=8),
    )

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW) == []
    assert db.bench_jobs.find_one({"_id": absent_id})["status"] == "dead_letter"
    assert db.bench_jobs.find_one({"_id": stale_id})["status"] == "dead_letter"


def test_slow_recovery_does_not_resurrect_ineligible_or_budget_exhausted_jobs(db):
    disabled_id = _terminal_job(db, model_id="disabled", enabled=False)
    budget_id = _terminal_job(db, model_id="budget", error_kind="budget_exhausted")
    _catalogued(db, "groq", "disabled")
    _catalogued(db, "groq", "budget")

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW) == []
    assert db.bench_jobs.find_one({"_id": disabled_id})["status"] == "dead_letter"
    assert db.bench_jobs.find_one({"_id": budget_id})["status"] == "dead_letter"


def test_failed_recovery_is_owned_by_the_slow_sweep_until_its_next_interval(db):
    job_id = _terminal_job(db)
    _catalogued(db, "groq", "m")
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)
    failure_at = TERMINAL_NOW + timedelta(hours=1)
    db.bench_jobs.update_one(
        {"_id": job_id},
        {"$set": {"status": "running", "attempt": 1}},
    )
    job = db.bench_jobs.find_one({"_id": job_id})
    assert (
        queue.mark_failure(
            db,
            job=job,
            error_kind="rate_limit",
            error_message="provider rate limit",
            now=failure_at,
        )
        == "dead_letter"
    )

    assert (
        queue.requeue_retryable_dead_letters(
            db,
            now=TERMINAL_NOW + timedelta(hours=2),
            min_age_seconds=0,
        )
        == []
    )
    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW + timedelta(days=6)) == []
    db.provider_catalog.update_one(
        {"provider": "groq", "model_id": "m"},
        {"$set": {"last_seen_at": TERMINAL_NOW + timedelta(days=7)}},
    )

    second_cycle = queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW + timedelta(days=8))

    assert len(second_cycle) == 1
    job = db.bench_jobs.find_one({"_id": job_id})
    assert job["last_terminal_recheck_count"] == 2
    assert job["terminal_original_error_kind"] == "hard_model"


@pytest.mark.parametrize("error_kind", ["hard_model", "hard_capability"])
def test_definitive_recovery_failure_quarantines_only_its_published_provider_model(db, error_kind):
    job_id = _terminal_job(db, model_id="shared")
    _enable(db, "anthropic", "shared")
    _catalogued(db, "groq", "shared")
    _endpoint(db, "shared", "openrouter-peer")
    db.bench_model_health.insert_one(
        {
            "provider": "groq",
            "model_id": "shared",
            "endpoint_tag": None,
            "enabled": True,
            "last_success_at": None,
        }
    )
    active_id = "manual:groq:shared:queued"
    db.bench_jobs.insert_one(
        {
            "_id": active_id,
            "provider": "groq",
            "model_id": "shared",
            "job_kind": "manual",
            "status": "queued",
        }
    )
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)
    failure_at = TERMINAL_NOW + timedelta(hours=1)
    assert _fail_recovered_job(db, job_id, error_kind=error_kind, now=failure_at) == "dead_letter"

    assert queue.recheck_stale_terminal_jobs(db, now=failure_at) == []

    quarantined = db.models.find_one({"provider": "groq", "model_id": "shared"})
    assert quarantined["enabled"] is False
    assert quarantined["status"] == "rejected"
    assert quarantined["disabled_class"] == error_kind
    assert quarantined["recovery_of_published_model"] is True
    recheck_after = quarantined["recheck_after"].replace(tzinfo=timezone.utc)
    assert recheck_after == failure_at + timedelta(days=30)
    assert db.models.find_one({"provider": "anthropic", "model_id": "shared"})["enabled"] is True
    assert db.bench_endpoints.find_one({"endpoint_tag": "openrouter-peer"})["enabled"] is True
    assert db.bench_model_health.find_one({"provider": "groq"})["enabled"] is False
    assert db.bench_jobs.find_one({"_id": active_id})["status"] == "cancelled"
    assert db.bench_jobs.find_one({"_id": job_id})["status"] == "cancelled"
    batch = db.bench_mutation_batches.find_one()
    assert batch["reason"] == "terminal recovery definitive failure"


def test_auth_failure_after_recovery_stays_enabled_and_slow_paced(db):
    job_id = _terminal_job(db)
    _catalogued(db, "groq", "m")
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)
    failure_at = TERMINAL_NOW + timedelta(hours=1)
    assert _fail_recovered_job(db, job_id, error_kind="auth", now=failure_at) == "dead_letter"

    assert queue.recheck_stale_terminal_jobs(db, now=failure_at) == []
    assert queue.requeue_retryable_dead_letters(db, now=failure_at, min_age_seconds=0) == []
    assert db.models.find_one({"provider": "groq", "model_id": "m"})["enabled"] is True
    assert db.bench_jobs.find_one({"_id": job_id})["status"] == "dead_letter"


def test_endpoint_recovery_failure_does_not_quarantine_the_parent_model(db):
    job_id = _terminal_job(
        db,
        provider="openrouter",
        model_id="multi",
        endpoint_tag="broken",
    )
    _endpoint(db, "multi", "healthy")
    _catalogued(db, "openrouter", "multi")
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)
    failure_at = TERMINAL_NOW + timedelta(hours=1)
    assert _fail_recovered_job(db, job_id, error_kind="hard_model", now=failure_at) == "dead_letter"

    assert queue.recheck_stale_terminal_jobs(db, now=failure_at) == []
    assert db.models.find_one({"provider": "openrouter", "model_id": "multi"})["enabled"] is True
    assert db.bench_endpoints.find_one({"endpoint_tag": "broken"})["enabled"] is True
    assert db.bench_endpoints.find_one({"endpoint_tag": "healthy"})["enabled"] is True


def test_successful_recovery_allows_normal_scheduled_work_again(db):
    job_id = _terminal_job(db)
    _catalogued(db, "groq", "m")
    queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)
    db.bench_jobs.update_one({"_id": job_id}, {"$set": {"status": "running"}})

    assert queue.mark_success(db, job_id=job_id, now=TERMINAL_NOW + timedelta(hours=1))
    assert "terminal_recovery_active" not in db.bench_jobs.find_one({"_id": job_id})
    assert queue.enqueue_scheduled_job(
        db,
        provider="groq",
        model_id="m",
        priority=1,
        now=TERMINAL_NOW + timedelta(days=1),
    )


def test_newer_health_success_cancels_a_stale_dead_letter_for_normal_scheduling(db):
    job_id = _terminal_job(db, age=timedelta(days=2))
    db.bench_model_health.insert_one(
        {
            "provider": "groq",
            "model_id": "m",
            "endpoint_tag": None,
            "last_success_at": TERMINAL_NOW - timedelta(days=1),
        }
    )

    assert queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW) == []
    job = db.bench_jobs.find_one({"_id": job_id})
    assert job["status"] == "cancelled"
    assert job["cancelled_reason"] == "superseded by newer successful measurement"
    assert queue.enqueue_scheduled_job(
        db,
        provider="groq",
        model_id="m",
        priority=1,
        now=TERMINAL_NOW,
    )


def test_terminal_recovery_candidates_are_processed_oldest_first(db):
    oldest = _terminal_job(db, model_id="oldest", age=timedelta(days=10))
    newer = _terminal_job(db, model_id="newer", age=timedelta(days=8))
    _catalogued(db, "groq", "oldest")
    _catalogued(db, "groq", "newer")

    recovered = queue.recheck_stale_terminal_jobs(db, now=TERMINAL_NOW)

    assert [job["_id"] for job in recovered] == [oldest, newer]
