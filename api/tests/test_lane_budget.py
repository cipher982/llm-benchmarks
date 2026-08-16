"""A lane may only ask for a budget it can stop consuming.

The profile asks for 2048 tokens so a reasoning model has room to think before
emitting visible text. That is only affordable because the OpenRouter stream
loop closes at the 64th visible token — the measurement is complete there and
everything after is billed and discarded. A lane that reads to end-of-stream
would generate the full 2048 on every model, including the ones that already
answer fine at 64, on accounts the owner pays directly.
"""

from llm_bench.cloud.visible_tokens import VISIBLE_TOKEN_MARK
from llm_bench.scheduler import runner


def test_the_early_stopping_lane_gets_the_full_thinking_budget():
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.lane_max_tokens("openrouter", budget) == budget
    assert budget > VISIBLE_TOKEN_MARK, "a reasoning model needs room to think before it can be measured"


def test_a_lane_that_reads_to_end_of_stream_is_capped():
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    for lane in ("direct", "openai", "bedrock", "vertex"):
        assert runner.lane_max_tokens(lane, budget) == runner.UNCAPPED_LANE_MAX_TOKENS, lane


def test_every_early_stop_provider_actually_stops():
    """The allowlist is a claim about provider code; hold it to the claim."""
    import inspect

    from llm_bench.cloud.providers import openrouter

    for provider in runner.EARLY_STOP_PROVIDERS:
        module = {"openrouter": openrouter}[provider]
        source = inspect.getsource(module.generate)
        assert "visible_clock.crossed" in source, f"{provider} is allowed a raised budget but never stops reading"


def test_the_clamp_never_raises_a_lane_above_the_profile():
    assert runner.lane_max_tokens("openrouter", 32) == 32
    assert runner.lane_max_tokens("bedrock", 32) == 32


def test_a_native_openrouter_row_gets_the_thinking_budget():
    """It routes as "direct" — the label is about routing, not about the lane.

    Keying the clamp on decision.transport_provider sent every native OpenRouter
    row down the capped path, so the raised budget reached nothing at all: 83
    rows written after the deploy, all at 64 tokens, and reasoning models still
    exhausting it.
    """
    from llm_bench.scheduler.routing import DIRECT_TRANSPORT
    from llm_bench.scheduler.routing import RouteDecision

    decision = RouteDecision.direct("openrouter", "deepseek/deepseek-r1", reason="native")
    assert decision.transport_provider == DIRECT_TRANSPORT
    lane = runner._transport_provider(decision)
    assert lane == "openrouter"

    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.lane_max_tokens(lane, budget) == budget
