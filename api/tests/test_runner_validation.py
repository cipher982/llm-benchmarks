from llm_bench.scheduler.runner import validate_metrics


def test_validation_classifies_empty_visible_text_after_budget_exhaustion():
    ok, reason = validate_metrics(
        "openai",
        {
            "output_tokens": 128,
            "visible_output_tokens": 0,
            "visible_text_empty": True,
            "response_status": "incomplete",
            "tokens_per_second": 10,
            "generate_time": 12.8,
        },
        64,
    )

    assert ok is False
    assert "larger output budget" in reason
