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
