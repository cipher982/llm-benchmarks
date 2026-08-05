"""A verdict nothing reads is not a verdict.

The classifier resolved error fingerprints on `error_rollups`. Retry policy and
the catalogue tools read `last_attempt_error_kind` on the job. Nothing joined
them, so a model could be correctly diagnosed as permanently gone and still be
retried forever.
"""

import mongomock
from llm_bench.ops.llm_error_classifier import propagate_classifications


def _db():
    client = mongomock.MongoClient()
    db = client["llm-bench"]
    db.error_rollups.insert_many(
        [
            {
                "fingerprint": "fp-gone",
                "provider": "openai",
                "model_name": "codex-mini-latest",
                "error_kind": "hard_model",
                "classified_by": "llm",
            },
            {
                "fingerprint": "fp-open",
                "provider": "openai",
                "model_name": "still-unknown",
                "error_kind": "unknown",
                "classified_by": "llm",
            },
        ]
    )
    return client, db


def test_a_resolved_kind_reaches_the_job_that_acts_on_it():
    client, db = _db()
    db.errors_cloud.insert_one(
        {
            "provider": "openai",
            "model_name": "codex-mini-latest",
            "fingerprint": "fp-gone",
            "error_kind": "unknown",
            "ts": 100,
        }
    )
    db.bench_jobs.insert_one(
        {
            "_id": "openai:codex-mini-latest",
            "provider": "openai",
            "model_id": "codex-mini-latest",
            "status": "dead_letter",
            "last_attempt_error_kind": "unknown",
        }
    )

    stats = propagate_classifications(client, "llm-bench")

    assert stats == {"errors_updated": 1, "jobs_updated": 1}
    assert db.bench_jobs.find_one({})["last_attempt_error_kind"] == "hard_model"
    assert db.errors_cloud.find_one({})["error_kind"] == "hard_model"


def test_a_stale_fingerprint_cannot_overwrite_a_newer_failure():
    """Only the newest error for a model is adopted.

    A model that 404'd last month and is rate-limited today must not be labelled
    hard_model — that reads as authoritative and would retire a live model.
    """
    client, db = _db()
    db.errors_cloud.insert_many(
        [
            {
                "provider": "openai",
                "model_name": "codex-mini-latest",
                "fingerprint": "fp-gone",
                "error_kind": "unknown",
                "ts": 100,
            },
            {
                "provider": "openai",
                "model_name": "codex-mini-latest",
                "fingerprint": "fp-recent-unclassified",
                "error_kind": "unknown",
                "ts": 200,
            },
        ]
    )
    db.bench_jobs.insert_one(
        {
            "_id": "openai:codex-mini-latest",
            "provider": "openai",
            "model_id": "codex-mini-latest",
            "status": "dead_letter",
            "last_attempt_error_kind": "unknown",
        }
    )

    stats = propagate_classifications(client, "llm-bench")

    assert stats["jobs_updated"] == 0
    assert db.bench_jobs.find_one({})["last_attempt_error_kind"] == "unknown"


def test_a_job_that_already_has_a_verdict_is_left_alone():
    client, db = _db()
    db.errors_cloud.insert_one(
        {
            "provider": "openai",
            "model_name": "codex-mini-latest",
            "fingerprint": "fp-gone",
            "error_kind": "unknown",
            "ts": 100,
        }
    )
    db.bench_jobs.insert_one(
        {
            "_id": "openai:codex-mini-latest",
            "provider": "openai",
            "model_id": "codex-mini-latest",
            "status": "dead_letter",
            "last_attempt_error_kind": "rate_limit",
        }
    )

    stats = propagate_classifications(client, "llm-bench")

    assert stats["jobs_updated"] == 0
    assert db.bench_jobs.find_one({})["last_attempt_error_kind"] == "rate_limit"


def test_an_unresolved_rollup_propagates_nothing():
    client, db = _db()
    db.errors_cloud.insert_one(
        {
            "provider": "openai",
            "model_name": "still-unknown",
            "fingerprint": "fp-open",
            "error_kind": "unknown",
            "ts": 100,
        }
    )

    assert propagate_classifications(client, "llm-bench") == {"errors_updated": 0, "jobs_updated": 0}
