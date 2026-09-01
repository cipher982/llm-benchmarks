from __future__ import annotations

import pytest
from llm_bench.ops.tier_cost import Population
from llm_bench.ops.tier_cost import Tier
from llm_bench.ops.tier_cost import calculate

HISTOGRAM = {
    1: 103,
    2: 37,
    3: 39,
    4: 21,
    5: 5,
    6: 3,
    7: 4,
    8: 1,
    9: 7,
    10: 2,
    11: 1,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 1,
    20: 1,
    30: 1,
    34: 1,
}


def test_provider_rotation_exposes_full_sweep_cost_multiplier() -> None:
    population = Population(provider_histogram=HISTOGRAM, model_targets=267)
    estimates, breakdown = calculate(
        population,
        tiers=[Tier("hot", 8, None, 3), Tier("medium", 3, 7, 24), Tier("long", 1, 2, 96)],
        no_endpoint_hours=96,
        uniform_hours=[96],
        cost_per_run_usd=0.00301,
        month_days=365.25 / 12,
        monthly_budget_usd=30,
    )

    assert population.endpoint_models == 237
    assert population.endpoints == 792
    assert population.models_without_endpoints == 30

    by_policy = {row.policy: row for row in estimates}
    rotating = by_policy["provider tiers / rotate one endpoint"]
    full_sweeps = by_policy["provider tiers / full endpoint sweeps"]
    assert rotating.requests_per_day == pytest.approx(314.5)
    assert rotating.cost_per_month_usd == pytest.approx(28.8135071875)
    assert rotating.within_monthly_budget
    assert full_sweeps.requests_per_day == pytest.approx(3067.75)
    assert full_sweeps.cost_per_month_usd == pytest.approx(281.05766828125)
    assert not full_sweeps.within_monthly_budget

    by_tier = {row.tier: row for row in breakdown}
    assert (by_tier["hot"].models, by_tier["hot"].endpoints) == (25, 343)
    assert (by_tier["medium"].models, by_tier["medium"].endpoints) == (72, 272)
    assert (by_tier["long"].models, by_tier["long"].endpoints) == (140, 177)
