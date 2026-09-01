"""The checks that would have caught the three defects of 2026-08-18.

Each defect shipped while every existing invariant passed, because each one
produced counters that looked healthy: jobs enqueued, workers completed, rows
written. What they had in common was an endpoint sitting unmeasured, or a
health record disagreeing with the evidence. These tests encode all three as
regressions against the checks rather than against the code that broke.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import invariants
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
CADENCE = 10800


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def _ctx(db):
    return invariants.Context(db=db, now=NOW)


def _endpoint(db, *, model_id, tag, last_success_at=None, error_kind=None, updated_at=None):
    doc = {
        "_id": f"openrouter:{model_id}:{tag}",
        "provider": "openrouter",
        "model_id": model_id,
        "endpoint_tag": tag,
        "enabled": True,
        "cadence_seconds": CADENCE,
        "last_success_at": last_success_at,
        "last_error_kind": error_kind,
    }
    if updated_at is not None:
        doc["updated_at"] = updated_at
    db[health_collection_name()].insert_one(doc)


class TestStarvation:
    def test_the_hot_loop_is_caught(self, db):
        # The real shape: two endpoints measured constantly, the rest never.
        _endpoint(db, model_id="anthropic/claude-opus-4.8", tag="anthropic", last_success_at=NOW)
        for i in range(5):
            _endpoint(db, model_id=f"m{i}", tag="novita/fp8")
        violations = invariants.endpoint_targets_are_being_measured(_ctx(db))
        assert len(violations) == 5
        assert all("never measured" in v.detail for v in violations)

    def test_a_new_never_run_endpoint_gets_its_rotation_grace(self, db):
        _endpoint(db, model_id="new", tag="novita/fp8", updated_at=NOW)
        assert invariants.endpoint_targets_are_being_measured(_ctx(db)) == []

    def test_a_recently_measured_endpoint_is_not_a_violation(self, db):
        _endpoint(db, model_id="m", tag="groq", last_success_at=NOW - timedelta(seconds=CADENCE))
        assert invariants.endpoint_targets_are_being_measured(_ctx(db)) == []

    def test_stale_past_three_cadences_is_a_violation(self, db):
        _endpoint(db, model_id="m", tag="groq", last_success_at=NOW - timedelta(seconds=CADENCE * 4))
        violations = invariants.endpoint_targets_are_being_measured(_ctx(db))
        assert len(violations) == 1
        assert "three cadences" in violations[0].detail

    def test_budget_exhausted_is_a_measurement_question_not_starvation(self, db):
        # Reported by models_measurable_by_the_published_profile instead;
        # scheduling it harder only spends money.
        _endpoint(db, model_id="m", tag="groq", error_kind="budget_exhausted")
        assert invariants.endpoint_targets_are_being_measured(_ctx(db)) == []

    def test_disabled_parent_model_endpoints_are_not_flagged(self, db):
        _endpoint(db, model_id="disabled-model", tag="groq")
        db[invariants.models_collection_name()].insert_one(
            {"model_id": "disabled-model", "provider": "openrouter", "enabled": False}
        )
        assert invariants.endpoint_targets_are_being_measured(_ctx(db)) == []


class TestFreshnessIsBackedByEvidence:
    def test_false_fresh_from_a_fallback_is_caught(self, db):
        # 469 endpoints were in exactly this state: marked measured, no row
        # anywhere carrying the tag, because the run fell back to unpinned.
        _endpoint(db, model_id="nousresearch/hermes-4-70b", tag="nebius/fp8", last_success_at=NOW)
        violations = invariants.endpoint_freshness_is_backed_by_rows(_ctx(db))
        assert len(violations) == 1
        assert "no row carries this endpoint tag" in violations[0].detail

    def test_a_real_measurement_passes(self, db):
        _endpoint(db, model_id="m", tag="novita/fp8", last_success_at=NOW)
        db[metrics_collection_name()].insert_one({"model_name": "m", "route_endpoint_tag": "novita/fp8", "run_ts": NOW})
        assert invariants.endpoint_freshness_is_backed_by_rows(_ctx(db)) == []

    def test_a_sibling_endpoints_rows_do_not_vouch_for_this_one(self, db):
        _endpoint(db, model_id="m", tag="novita/bf16", last_success_at=NOW)
        db[metrics_collection_name()].insert_one({"model_name": "m", "route_endpoint_tag": "novita/fp8", "run_ts": NOW})
        assert len(invariants.endpoint_freshness_is_backed_by_rows(_ctx(db))) == 1

    def test_an_old_tagged_row_does_not_vouch_for_a_new_credit(self, db):
        _endpoint(db, model_id="m", tag="novita/fp8", last_success_at=NOW)
        db[metrics_collection_name()].insert_one(
            {
                "model_name": "m",
                "route_endpoint_tag": "novita/fp8",
                "run_ts": NOW - timedelta(seconds=CADENCE * 2),
            }
        )
        assert len(invariants.endpoint_freshness_is_backed_by_rows(_ctx(db))) == 1


def test_both_checks_are_registered():
    names = {inv.name for inv in invariants.INVARIANTS}
    assert "endpoint_targets_are_being_measured" in names
    assert "endpoint_freshness_is_backed_by_rows" in names


class TestVerdictIsReadFromWhicheverRecordHasOne:
    """Eight endpoints held `budget_exhausted` on their job and nothing on
    their health doc, because they failed before completions were credited per
    endpoint. Reading only the health doc reported them as starving, which
    sends someone to fix a scheduler that is working correctly.
    """

    def test_a_job_verdict_excludes_an_endpoint_the_profile_cannot_measure(self, db):
        _endpoint(db, model_id="anthropic/claude-opus-5", tag="anthropic")
        db[invariants.jobs_collection_name()].insert_one(
            {
                "_id": "openrouter:anthropic/claude-opus-5:anthropic",
                "provider": "openrouter",
                "model_id": "anthropic/claude-opus-5",
                "endpoint_tag": "anthropic",
                "status": "dead_letter",
                "last_attempt_error_kind": "budget_exhausted",
            }
        )
        assert invariants.endpoint_targets_are_being_measured(_ctx(db)) == []

    def test_an_endpoint_with_no_verdict_anywhere_is_still_starving(self, db):
        _endpoint(db, model_id="m", tag="novita/fp8")
        violations = invariants.endpoint_targets_are_being_measured(_ctx(db))
        assert len(violations) == 1
