from llm_bench.cloud.metrics import build_cloud_metrics


def test_build_cloud_metrics_preserves_legacy_fields_with_schema_v2_context():
    metrics = build_cloud_metrics(
        requested_tokens=64,
        generated_output_tokens=96,
        visible_output_tokens=64,
        reasoning_tokens=32,
        generate_time=2.0,
        time_to_first_token=None,
        times_between_tokens=[],
        token_source="provider_usage_output_tokens",
        request_mode="openai_responses_stream",
        finish_reason="stop",
        response_id="resp_123",
        max_output_tokens_attempted=128,
    )

    assert metrics["metrics_schema_version"] == 2
    assert metrics["output_tokens"] == 96
    assert metrics["tokens_per_second"] == 48
    assert metrics["generated_output_tokens"] == 96
    assert metrics["visible_output_tokens"] == 64
    assert metrics["reasoning_tokens"] == 32
    assert metrics["generated_tokens_per_second"] == 48
    assert metrics["visible_tokens_per_second"] == 32
    assert metrics["ttft_available"] is False
    assert metrics["token_source"] == "provider_usage_output_tokens"
    assert metrics["request_mode"] == "openai_responses_stream"


def test_time_to_64_visible_tokens_seconds_recorded_when_provided():
    metrics = build_cloud_metrics(
        requested_tokens=512,
        generated_output_tokens=64,
        generate_time=2.0,
        time_to_first_token=0.5,
        times_between_tokens=[0.01] * 63,
        time_to_64_visible_tokens_seconds=1.5,
        token_source="provider_usage_output_tokens",
        request_mode="openrouter_chat_completions_stream",
    )
    assert metrics["time_to_64_visible_tokens_seconds"] == 1.5


def test_time_to_64_visible_tokens_seconds_defaults_to_none():
    metrics = build_cloud_metrics(
        requested_tokens=64,
        generated_output_tokens=64,
        generate_time=2.0,
        time_to_first_token=None,
        times_between_tokens=[],
        token_source="provider_usage_output_tokens",
        request_mode="openai_responses_stream",
    )
    assert metrics["time_to_64_visible_tokens_seconds"] is None
