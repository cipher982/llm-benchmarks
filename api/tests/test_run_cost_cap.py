"""A token budget is the wrong unit to bound spend in.

2048 tokens costs a thousandth of a cent on a small open model and about 35
cents on a premium reasoning one. Three runs of openai/gpt-5.2-pro were 67% of
a day's OpenRouter spend, each burning the whole budget, while the other ~380
models together were the remaining third.

Closing the stream at the measurement mark does not rescue this on its own:
some upstreams finish the generation and bill it whatever the client does —
measured, poolside/laguna-s-2.1 billed 2,008 completion tokens against the 64
the runner actually read.
"""

from llm_bench.cloud.visible_tokens import VISIBLE_TOKEN_MARK
from llm_bench.scheduler import runner

# Roughly what the catalogue lists for a premium reasoning model: $0.347 for
# the 2,064 tokens one observed run produced.
PREMIUM = 0.347 / 2064
CHEAP = 0.02 / 1_000_000


def test_a_cheap_model_keeps_the_whole_thinking_budget():
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.lane_max_tokens("openrouter", budget, completion_price_per_token=CHEAP) == budget


def test_a_premium_model_is_cut_to_what_one_run_may_cost():
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    capped = runner.lane_max_tokens("openrouter", budget, completion_price_per_token=PREMIUM)

    assert capped < budget
    assert capped * PREMIUM <= max(runner.MAX_COST_PER_RUN_USD, VISIBLE_TOKEN_MARK * PREMIUM) + 1e-9


def test_the_floor_is_the_measurement_not_zero():
    """A model too expensive to profile is measured at the mark, not dropped."""
    absurd = 1.0  # a dollar a token
    assert runner.lane_max_tokens("openrouter", 2048, completion_price_per_token=absurd) == VISIBLE_TOKEN_MARK


def test_an_unknown_price_fails_closed():
    """A missing catalogue entry is not evidence the model is cheap.

    This asserted the opposite until two reviewers pointed at it: it returned
    the full 2048 budget and its own name claimed that was caution. That made
    the whole clamp best-effort — a model absent from the catalogue, a sentinel
    price, or one Mongo timeout would issue exactly the request the clamp exists
    to prevent.
    """
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.affordable_max_tokens(None, budget) == VISIBLE_TOKEN_MARK


def test_a_free_model_is_not_treated_as_an_unknown_one():
    """Zero is an answer from the catalogue; None is the absence of one."""
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.affordable_max_tokens(0.0, budget) == budget


def test_a_read_to_end_lane_ignores_price_and_stays_capped():
    """Those lanes are clamped for a different reason and it still applies."""
    assert runner.lane_max_tokens("bedrock", 2048, completion_price_per_token=CHEAP) == runner.UNCAPPED_LANE_MAX_TOKENS


def test_the_cap_is_bounded_by_the_profile_not_only_by_price():
    """A free model does not get an unbounded budget."""
    assert runner.affordable_max_tokens(1e-12, 2048) == 2048
