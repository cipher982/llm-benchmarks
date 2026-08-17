"""Time-to-64th-visible-token tracking shared by the streaming providers.

The site's headline metric is Delivered TPS
(docs/specs/delivered-tps-vision.md):

    64 visible answer tokens / time from request start to the 64th visible token

Reasoning/thinking tokens count as time, never as output, so the clock advances
only on visible content deltas. The token count is `cl100k_base` over the
accumulated visible text — the same encoder the numerator uses — so the metric
is self-consistent regardless of what the provider reports for total completion
tokens.
"""

from __future__ import annotations

import time

from tiktoken import get_encoding

# The headline denominator marker. One named constant so the runner, the
# dashboard derivation, and the spec stay in agreement.
VISIBLE_TOKEN_MARK = 64

_encoder = get_encoding("cl100k_base")


class VisibleTokenClock:
    """Accumulate visible stream text and record when it reaches the mark.

    The clock is started at the request's `time_0` and advanced with each
    visible content delta. Encoding only the pre-threshold text keeps the cost
    trivial; after the mark is crossed the clock stops encoding and keeps the
    first crossing time.

    `crossed` is the signal to stop reading the stream. Everything a provider
    generates after the 64th visible token is billed and then discarded — a
    measured DeepSeek R1 run produced 1,467 completion tokens to answer a
    question that was settled by token 64.
    """

    def __init__(self, time_0: float) -> None:
        self._time_0 = time_0
        self._text = ""
        self.crossed = False
        self.time_to_mark: float | None = None

    def add(self, delta: str, *, now: float | None = None) -> None:
        """Advance the clock with one visible content delta."""
        if self.crossed or not delta:
            return
        self._text += delta
        if len(_encoder.encode(self._text)) < VISIBLE_TOKEN_MARK:
            return
        self.crossed = True
        self.time_to_mark = (now if now is not None else time.time()) - self._time_0


def count_tokens(text: str) -> int:
    """Token count for locally accumulated stream text.

    Closing the stream early means the provider's usage chunk — which arrives
    last — never comes. Reasoning text is streamed as it is produced, so the
    thinking a model did is still fully observed; it just has to be counted
    here instead of read off a usage payload. Measured against DeepSeek R1,
    316 accumulated reasoning characters encode to 80 tokens, matching the
    provider's reported `reasoning_tokens` exactly.
    """
    return len(_encoder.encode(text)) if text else 0


# A stream can only resolve throughput as finely as its deltas. Measured
# 2026-08-17 on `openai/gpt-oss-120b`: Cerebras delivered 256 tokens in 13 SSE
# chunks and legacy `tokens_per_second` reported 3730 tok/s from a 0.069s
# post-TTFT window; Groq sent 256 chunks for identical work. Neither number is
# wrong arithmetically — the first simply has nothing to measure with.
#
# One token per chunk is full resolution. Beyond a few tokens per chunk the
# timing describes transport batching, and any rate derived from it is about
# the socket rather than the decoder.
RESOLVED_MAX_TOKENS_PER_CHUNK = 4


def stream_resolution(chunks: int, max_tokens_per_chunk: int) -> str:
    """Classify whether a stream's deltas can support a throughput number.

    `resolved`   — deltas are fine-grained enough to time generation.
    `batched`    — deltas carry many tokens; rate reflects transport batching.
    `unmeasured` — nothing visible streamed, so there is nothing to classify.
    """
    if chunks <= 0:
        return "unmeasured"
    if max_tokens_per_chunk > RESOLVED_MAX_TOKENS_PER_CHUNK:
        return "batched"
    return "resolved"
