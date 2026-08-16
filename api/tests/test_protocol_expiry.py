"""A verdict about the measurement expires when the measurement changes.

`budget_exhausted` says the model spent the whole token budget on hidden
reasoning and emitted nothing visible. That is a statement about the profile,
not about the model, and it is terminal on purpose — retrying it against the
same budget would fail identically forever.

The failure mode is what happens next. When the budget was raised from 64 to
2048, 419 dead letters were still holding a verdict reached against the old one,
and nothing in the sweep could tell that their reason had ceased to exist. They
would have stayed dead until someone noticed and requeued them by hand.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    database = mongomock.MongoClient()["llm-bench"]
    database.models.insert_one({"provider": "openrouter", "model_id": "deepseek/deepseek-r1", "enabled": True})
    return database


def _dead_letter(db, **extra):
    doc = {
        "_id": "job-1",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "status": "dead_letter",
        "updated_at": NOW - timedelta(minutes=1),
        "last_attempt_error_kind": "budget_exhausted",
        **extra,
    }
    db.bench_jobs.insert_one(doc)
    return doc


def test_a_verdict_from_an_older_protocol_is_requeued(db):
    _dead_letter(db, last_attempt_protocol_version=policies.MEASUREMENT_PROTOCOL_VERSION - 1)

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "queued"


def test_a_verdict_predating_the_field_is_requeued(db):
    """No recorded protocol means the protocol is unknown, not that it matches."""
    _dead_letter(db)

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "queued"


def test_a_verdict_from_the_current_protocol_stays_dead(db):
    """Otherwise this is an infinite retry loop against an unchanged budget."""
    _dead_letter(db, last_attempt_protocol_version=policies.MEASUREMENT_PROTOCOL_VERSION)

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "dead_letter"


def test_a_verdict_about_the_model_is_not_protocol_dependent(db):
    """A 404 does not become a different 404 because the budget changed."""
    _dead_letter(db, last_attempt_error_kind="hard_model")

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "dead_letter"


def test_the_protocol_clause_does_not_resurrect_a_disabled_model(db):
    db.models.update_one({"model_id": "deepseek/deepseek-r1"}, {"$set": {"enabled": False}})
    _dead_letter(db)

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "cancelled"


def test_a_failure_records_the_protocol_it_was_reached_under(db):
    job = {
        "_id": "job-2",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "status": "running",
        "attempt": 2,
        "max_attempts": 2,
    }
    db.bench_jobs.insert_one(job)

    queue.mark_failure(db, job=job, error_kind="budget_exhausted", error_message="no visible output", now=NOW)

    stored = db.bench_jobs.find_one({"_id": "job-2"})
    assert stored["last_attempt_protocol_version"] == policies.MEASUREMENT_PROTOCOL_VERSION
    assert stored["status"] == "dead_letter"


def test_requeues_are_still_bounded(db):
    """Protocol expiry must not become a way around the resurrection limit."""
    _dead_letter(db, dead_letter_requeues=policies.MAX_DEAD_LETTER_REQUEUES)

    queue.requeue_retryable_dead_letters(db, now=NOW)

    assert db.bench_jobs.find_one({"_id": "job-1"})["status"] == "dead_letter"
