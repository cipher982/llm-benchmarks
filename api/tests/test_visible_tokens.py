import tiktoken
from llm_bench.cloud.visible_tokens import VISIBLE_TOKEN_MARK
from llm_bench.cloud.visible_tokens import VisibleTokenClock

_encoder = tiktoken.get_encoding("cl100k_base")


def _text_with_at_least(n_tokens: int) -> str:
    """A visible string whose cl100k_base length is >= n_tokens."""
    text = ""
    while len(_encoder.encode(text)) < n_tokens:
        text += " benchmark"
    return text


def test_clock_is_none_before_threshold():
    clock = VisibleTokenClock(time_0=100.0)
    clock.add(_text_with_at_least(VISIBLE_TOKEN_MARK - 10), now=101.0)
    assert clock.time_to_mark is None


def test_clock_records_first_crossing_time():
    clock = VisibleTokenClock(time_0=100.0)
    clock.add(_text_with_at_least(VISIBLE_TOKEN_MARK), now=103.0)
    assert clock.time_to_mark == 3.0


def test_clock_ignores_empty_deltas_and_stops_after_crossing():
    clock = VisibleTokenClock(time_0=0.0)
    clock.add("", now=1.0)
    assert clock.time_to_mark is None
    clock.add(_text_with_at_least(VISIBLE_TOKEN_MARK), now=2.0)
    assert clock.time_to_mark == 2.0
    # The mark was crossed; later deltas must not move it.
    clock.add(_text_with_at_least(VISIBLE_TOKEN_MARK * 2), now=99.0)
    assert clock.time_to_mark == 2.0
