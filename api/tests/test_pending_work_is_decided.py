"""A pool that never drains is the shape all four coverage outages shared.

Each was a guard that was correct in isolation and became a permanent stop, and
in every case the symptom was silence rather than an error. This checks the
outcome — did items leave the pool — because a saturated bound, a silent
truncation and a crash mid-pass are indistinguishable from the pool's side and
all need the same response.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import admission
from llm_bench.ops import invariants

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _ctx(db):
    return invariants.Context(db=db, now=NOW)


def candidate(db, model_id, *, age):
    db.models.insert_one(
        {
            "provider": "deepinfra",
            "model_id": model_id,
            "enabled": False,
            "status": admission.CANDIDATE_STATUS,
            "admission_started_at": NOW - age,
        }
    )


def test_a_candidate_past_its_deadline_is_a_violation(db):
    """43 models sat here for as long as admission had existed.

    The mutation cap refused an 87-change batch and applied nothing, every two
    hours. Nothing raised, nothing alerted, and each new model made it worse.
    """
    candidate(db, "stuck", age=admission.ADMISSION_DEADLINE + timedelta(days=2))

    violations = invariants.pending_work_is_being_decided(_ctx(db))

    assert [v.subject for v in violations] == ["deepinfra/stuck"]


def test_a_candidate_inside_its_deadline_is_fine(db):
    candidate(db, "fresh", age=timedelta(hours=6))

    assert invariants.pending_work_is_being_decided(_ctx(db)) == []


def test_one_missed_pass_does_not_fire(db):
    """A candidate just past the deadline is waiting, not stuck."""
    candidate(db, "borderline", age=admission.ADMISSION_DEADLINE + timedelta(hours=2))

    assert invariants.pending_work_is_being_decided(_ctx(db)) == []


def test_a_decided_candidate_leaves_the_pool(db):
    db.models.insert_one(
        {
            "provider": "deepinfra",
            "model_id": "promoted",
            "enabled": True,
            "status": admission.PROMOTED_STATUS,
            "admission_started_at": NOW - timedelta(days=30),
        }
    )

    assert invariants.pending_work_is_being_decided(_ctx(db)) == []


def test_it_is_registered_so_it_actually_runs(db):
    """An invariant nobody registered is the same as one nobody wrote."""
    assert any(i.name == "pending_work_is_being_decided" for i in invariants.INVARIANTS)


def test_the_real_deadlock_reproduces_end_to_end(db):
    """Stage the production failure and confirm the check would have caught it.

    Enough candidates that one batch cannot decide them all, all long past the
    deadline — which is exactly what the cap produced.
    """
    for i in range(60):
        candidate(db, f"c{i}", age=admission.ADMISSION_DEADLINE + timedelta(days=3))

    violations = invariants.pending_work_is_being_decided(_ctx(db))
    assert len(violations) == 60

    # Draining it clears the check, so the check tracks the fault rather than
    # the size of the pool.
    for _ in range(10):
        admission.evaluate_candidates(db, now=NOW)
        if not db.models.count_documents({"status": admission.CANDIDATE_STATUS}):
            break

    assert invariants.pending_work_is_being_decided(_ctx(db)) == []
