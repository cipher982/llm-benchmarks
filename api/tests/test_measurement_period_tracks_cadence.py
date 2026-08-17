"""The starvation horizon must follow the scheduling period, not restate it.

This file's own history has this failure twice: a horizon built from the wrong
number reported thirteen healthy models as starved, and the note about it warns
that a permanently red check is one people stop reading — which is how an
eight-day outage stayed invisible.

It happened a third time. MODEL_MEASUREMENT_PERIOD was a separate env var
pinned to the 45 minutes once measured on clifford. Raising FRESH_MINUTES from
30 to 180 moved the real reschedule point to 4.5h while the check kept asking
for 3h, so every model on the site would have been reported starved for doing
exactly what it was configured to do.
"""

from datetime import timedelta

from llm_bench.ops import invariants


def test_the_period_follows_fresh_minutes(monkeypatch):
    monkeypatch.setenv("FRESH_MINUTES", "180")

    # A model is rescheduled once its last success passes 1.5x the horizon.
    assert invariants.model_measurement_period() == timedelta(hours=4.5)


def test_raising_the_cadence_widens_the_starvation_horizon(monkeypatch):
    monkeypatch.setenv("FRESH_MINUTES", "30")
    tight = invariants.model_measurement_period() * invariants.MODEL_STALENESS_MULTIPLIER

    monkeypatch.setenv("FRESH_MINUTES", "180")
    wide = invariants.model_measurement_period() * invariants.MODEL_STALENESS_MULTIPLIER

    assert wide == tight * 6, "the horizon must scale with the scheduling period"


def test_the_horizon_always_exceeds_the_reschedule_point(monkeypatch):
    """Otherwise the check fires on models the scheduler has not yet re-run."""
    for fresh in ("15", "30", "60", "180", "360"):
        monkeypatch.setenv("FRESH_MINUTES", fresh)
        period = invariants.model_measurement_period()
        horizon = period * invariants.MODEL_STALENESS_MULTIPLIER
        assert horizon > period, fresh
