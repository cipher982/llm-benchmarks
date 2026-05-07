from types import SimpleNamespace

from llm_bench.cloud.providers.openai_compat import run_chat_completion_benchmark


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def response(*, content, reasoning, completion_tokens, finish_reason):
    message = SimpleNamespace(content=content, model_extra={"reasoning": reasoning})
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        completion_tokens=completion_tokens,
        prompt_tokens=10,
        total_tokens=completion_tokens + 10,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage, id="resp_fake")


def test_retries_larger_budget_until_visible_output_is_complete_enough():
    client = FakeClient(
        [
            response(content="", reasoning="thinking " * 20, completion_tokens=64, finish_reason="length"),
            response(
                content="A sunrise story with enough visible output to count.",
                reasoning="thinking",
                completion_tokens=80,
                finish_reason="stop",
            ),
        ]
    )

    metrics = run_chat_completion_benchmark(
        client=client,
        model="test-model",
        query="Tell a story.",
        max_tokens=64,
        request_mode="test_provider",
    )

    assert metrics["visible_text_empty"] is False
    assert metrics["max_output_tokens_attempted"] == 256
    assert metrics["request_mode"] == "test_provider"
    assert metrics["reasoning_tokens"] > 0
    assert [call["max_tokens"] for call in client.chat.completions.calls] == [64, 256]


def test_falls_back_to_reasoning_disabled_for_hybrid_models():
    client = FakeClient(
        [
            response(content="", reasoning="thinking", completion_tokens=64, finish_reason="length"),
            response(content="", reasoning="thinking", completion_tokens=256, finish_reason="length"),
            response(content="", reasoning="thinking", completion_tokens=512, finish_reason="length"),
            response(content="Visible answer.", reasoning="", completion_tokens=20, finish_reason="stop"),
        ]
    )

    metrics = run_chat_completion_benchmark(
        client=client,
        model="test-model",
        query="Tell a story.",
        max_tokens=64,
        request_mode="together_chat_completions",
        fallback_extra_bodies=[("reasoning_disabled", {"reasoning": {"enabled": False}}, "disabled")],
    )

    assert metrics["visible_text_empty"] is False
    assert metrics["request_mode"] == "together_chat_completions_reasoning_disabled"
    assert metrics["reasoning_effort"] == "disabled"
    assert client.chat.completions.calls[-1]["extra_body"] == {"reasoning": {"enabled": False}}
