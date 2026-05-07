from llm_bench.cloud.providers import bedrock
from llm_bench.config import CloudConfig


def test_bedrock_passes_additional_model_fields_and_splits_reasoning(monkeypatch):
    captured = {}

    class FakeBedrockClient:
        def converse_stream(self, **kwargs):
            captured.update(kwargs)
            return {
                "ResponseMetadata": {"RequestId": "bedrock-test"},
                "stream": [
                    {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "working through it"}}}},
                    {"contentBlockDelta": {"delta": {"text": "Final answer."}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {
                        "metadata": {
                            "usage": {
                                "inputTokens": 12,
                                "outputTokens": 80,
                                "totalTokens": 92,
                            }
                        }
                    },
                ],
            }

    monkeypatch.setattr(bedrock.boto3, "client", lambda **_: FakeBedrockClient())

    metrics = bedrock.generate(
        CloudConfig(
            provider="bedrock",
            model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            run_ts="2026-05-07T00:00:00Z",
            temperature=0.0,
        ),
        {
            "query": "Give a short answer.",
            "max_tokens": 256,
            "additional_model_request_fields": {"thinking": {"type": "enabled", "budget_tokens": 128}},
        },
    )

    assert captured["additionalModelRequestFields"] == {"thinking": {"type": "enabled", "budget_tokens": 128}}
    assert metrics["output_tokens"] == 80
    assert metrics["visible_output_tokens"] < metrics["output_tokens"]
    assert metrics["reasoning_tokens"] > 0
    assert metrics["token_source"] == "provider_usage_output_tokens_with_tiktoken_visible_split"
    assert metrics["reasoning_effort"] == "enabled"
    assert metrics["request_mode"] == "bedrock_converse_stream"
