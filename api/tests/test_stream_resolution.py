"""A throughput number is only as fine-grained as the deltas it was timed from.

Measured 2026-08-17 on `openai/gpt-oss-120b`: Cerebras returned 256 tokens in
13 SSE chunks and legacy `tokens_per_second` reported 3730 tok/s off a 0.069s
post-TTFT window. Groq sent 256 chunks for the same work. The arithmetic is
identical; only one of them had something to measure.
"""

from llm_bench.cloud.visible_tokens import stream_resolution
from llm_bench.logging import _optional_metric_fields


def test_one_token_per_chunk_is_resolved():
    assert stream_resolution(chunks=256, max_tokens_per_chunk=1) == "resolved"


def test_wide_chunks_are_batched_not_resolved():
    """The Cerebras case: ~20 tokens per delta measures the socket."""
    assert stream_resolution(chunks=13, max_tokens_per_chunk=20) == "batched"


def test_a_single_wide_chunk_taints_an_otherwise_fine_stream():
    """Resolution is bounded by the worst delta, not the average."""
    assert stream_resolution(chunks=200, max_tokens_per_chunk=64) == "batched"


def test_no_visible_content_is_unmeasured():
    assert stream_resolution(chunks=0, max_tokens_per_chunk=0) == "unmeasured"


def test_resolution_fields_survive_the_metric_writer():
    """`log_mongo` copies only registered keys; unregistered fields vanish."""
    written = _optional_metric_fields(
        {
            "visible_stream_chunks": 13,
            "max_tokens_per_chunk": 20,
            "stream_resolution": "batched",
            "not_a_registered_field": "dropped",
        }
    )
    assert written == {
        "visible_stream_chunks": 13,
        "max_tokens_per_chunk": 20,
        "stream_resolution": "batched",
    }
