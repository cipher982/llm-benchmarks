"""The key can say when the cap is coming; the check must listen.

Pinned against the 2026-08-20 outage: the monthly limit was reached and the
fleet went quiet for eleven days with every existing check green.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import invariants
from llm_bench.ops import openrouter_key_limit as key_limit

NOW = datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc)

# The real payload shape from /auth/key on 2026-09-02, values rounded.
STATUS = {
    "label": "sk-or-v1-675...239",
    "limit": 30,
    "limit_reset": "monthly",
    "limit_remaining": 25.48,
    "usage": 64.53,
    "usage_daily": 0.92,
    "usage_weekly": 4.52,
    "usage_monthly": 4.52,
}


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"keylimit-{request.node.name}"]


def ctx(db):
    return invariants.Context(db=db, now=NOW)


class TestHeadroom:
    def test_burn_is_the_larger_of_today_and_the_weekly_average(self):
        room = key_limit.headroom({**STATUS, "usage_daily": 0.05, "usage_weekly": 7.0}, now=NOW)
        assert room["burn_per_day"] == pytest.approx(1.0)
        assert room["days_left"] == pytest.approx(25.48)

    def test_exhausted_when_nothing_remains(self):
        room = key_limit.headroom({**STATUS, "limit_remaining": 0}, now=NOW)
        assert room["exhausted"] is True
        assert room["days_left"] == 0

    def test_unlimited_key_has_no_days_left_to_report(self):
        room = key_limit.headroom({"usage_daily": 3.0, "limit": None, "limit_remaining": None}, now=NOW)
        assert room["limited"] is False
        assert room["days_left"] is None

    def test_no_burn_yet_means_unknown_not_infinite(self):
        room = key_limit.headroom({**STATUS, "usage_daily": 0, "usage_weekly": 0}, now=NOW)
        assert room["days_left"] is None
        assert room["exhausted"] is False


class TestRecord:
    def test_stores_limit_usage_and_headroom_without_the_key(self, db):
        doc = key_limit.record_key_status(db, STATUS, now=NOW)
        stored = db.provider_state.find_one({"_id": key_limit.STATE_ID})
        assert stored["limit"] == 30
        assert stored["limit_remaining"] == pytest.approx(25.48)
        assert stored["checked_at"].replace(tzinfo=timezone.utc) == NOW
        assert stored["headroom_limited"] is True
        assert stored["headroom_days_left"] == pytest.approx(25.48 / 0.92)
        assert "api_key" not in stored and "Authorization" not in str(stored)
        assert doc["provider"] == "openrouter"

    def test_upserts_in_place(self, db):
        key_limit.record_key_status(db, STATUS, now=NOW)
        key_limit.record_key_status(db, {**STATUS, "limit_remaining": 1.0}, now=NOW + timedelta(hours=1))
        assert db.provider_state.count_documents({}) == 1
        assert db.provider_state.find_one()["limit_remaining"] == 1.0


class TestInvariant:
    def test_cannot_evaluate_without_a_reading(self, db):
        with pytest.raises(invariants.CannotEvaluate):
            invariants.openrouter_key_has_headroom(ctx(db))

    def test_cannot_evaluate_on_a_stale_reading(self, db):
        key_limit.record_key_status(db, STATUS, now=NOW - timedelta(hours=4))
        with pytest.raises(invariants.CannotEvaluate, match="stale"):
            invariants.openrouter_key_has_headroom(ctx(db))

    def test_passes_with_weeks_of_headroom(self, db):
        key_limit.record_key_status(db, STATUS, now=NOW)
        assert invariants.openrouter_key_has_headroom(ctx(db)) == []

    def test_fails_while_exhausted(self, db):
        # 2026-08-20 through 08-31: every routed call answered 403.
        key_limit.record_key_status(db, {**STATUS, "limit_remaining": 0, "usage_monthly": 30}, now=NOW)
        [violation] = invariants.openrouter_key_has_headroom(ctx(db))
        assert violation.subject == "openrouter"
        assert "exhausted" in violation.detail
        assert violation.data["remaining_usd"] == 0

    def test_fails_before_the_cap_at_the_current_burn(self, db):
        # $2 left at $1/day: two days of fleet, then eleven of silence.
        key_limit.record_key_status(
            db, {**STATUS, "limit_remaining": 2.0, "usage_daily": 1.0, "usage_weekly": 6.0}, now=NOW
        )
        [violation] = invariants.openrouter_key_has_headroom(ctx(db))
        assert "runs out in 2.0 days" in violation.detail

    def test_unlimited_key_passes(self, db):
        key_limit.record_key_status(db, {**STATUS, "limit": None, "limit_remaining": None}, now=NOW)
        assert invariants.openrouter_key_has_headroom(ctx(db)) == []

    def test_is_registered_and_pages(self):
        [inv] = [i for i in invariants.INVARIANTS if i.name == "openrouter_key_has_headroom"]
        assert inv.pages is True
        assert inv.remediable is False
