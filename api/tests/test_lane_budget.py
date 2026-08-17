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

CHEAP = 0.02 / 1_000_000


def test_the_early_stopping_lane_gets_the_full_thinking_budget():
    budget = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    assert runner.lane_max_tokens("openrouter", budget, completion_price_per_token=CHEAP) == budget
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
    assert runner.lane_max_tokens("openrouter", 32, completion_price_per_token=CHEAP) == 32
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
    assert runner.lane_max_tokens(lane, budget, completion_price_per_token=CHEAP) == budget


def test_the_stream_is_actually_closed_at_the_mark(monkeypatch):
    """The allowlist test above only greps for the break; this exercises it.

    A break without close() leaves the SSE response open and the upstream
    generating — which is the cost the early stop exists to avoid — and no test
    covered it because the fakes use a bare iterator with no close method.
    """
    from types import SimpleNamespace

    from llm_bench.cloud.providers import openrouter

    class RecordingStream:
        def __init__(self, chunks):
            self._chunks = iter(chunks)
            self.closed = False
            self.consumed = 0

        def __iter__(self):
            return self

        def __next__(self):
            chunk = next(self._chunks)
            self.consumed += 1
            return chunk

        def close(self):
            self.closed = True

    def _chunk(text):
        return SimpleNamespace(
            id="gen-1",
            usage=None,
            choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content=text, reasoning=None))],
        )

    # Far more than 64 visible tokens, delivered in many chunks.
    stream = RecordingStream([_chunk("token " * 20) for _ in range(50)])
    monkeypatch.setattr(openrouter, "process_chat_model", lambda *a, **k: (stream, None))
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: object())
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    config = SimpleNamespace(
        provider="openrouter",
        transport_provider="openrouter",
        model_name="test/model",
        misc={},
        temperature=0.1,
    )
    metrics = openrouter.generate(config, {"query": "hi", "max_tokens": 2048})

    assert stream.closed, "the stream was broken out of but never closed"
    assert stream.consumed < 50, "the whole stream was read; the early stop did nothing"
    assert metrics["time_to_64_visible_tokens_seconds"] is not None


def test_quarantine_sweeps_every_enabled_provider():
    """It swept a hardcoded ("together", "vertex").

    The catalogue became 332 OpenRouter models out of 388 and OpenRouter was
    never in that pair, so the tool reported "No quarantine candidates" while
    120 enabled models sat permanently unservable — the reassuring silence it
    exists to break.
    """
    import mongomock
    from llm_bench.ops import catalog_quarantine

    db = mongomock.MongoClient()["t"]
    db.models.insert_many(
        [
            {"provider": "openrouter", "model_id": "a/x", "enabled": True},
            {"provider": "bedrock", "model_id": "b/y", "enabled": True},
            {"provider": "together", "model_id": "c/z", "enabled": False},
            {"provider": "vertex", "model_id": "d/w", "enabled": True, "deprecated": True},
        ]
    )

    assert catalog_quarantine.default_providers(db) == ["bedrock", "openrouter"]
