from types import SimpleNamespace

from llm_bench.cloud.providers import openrouter
from llm_bench.config import CloudConfig


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.chunks)


class FakeClient:
    def __init__(self, chunks):
        self.completions = FakeCompletions(chunks)
        self.chat = SimpleNamespace(completions=self.completions)


def _chunk(content=None, *, finish_reason=None, usage=None, metadata=None, response_id="resp-1"):
    delta = SimpleNamespace(content=content, reasoning=None, reasoning_content=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id=response_id,
        choices=[] if usage is not None and content is None and finish_reason is None else [choice],
        usage=usage,
        openrouter_metadata=metadata,
    )


def _config(**misc):
    return CloudConfig(
        provider="openrouter",
        model_name="qwen/qwen3-32b",
        run_ts="2026-08-09 00:00:00",
        temperature=0.1,
        misc=misc,
    )


def test_route_request_is_pinned_and_fallbacks_are_disabled(monkeypatch):
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient([_chunk("answer "), _chunk("text", finish_reason="stop"), _chunk(usage=usage)])
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    metrics = openrouter.generate(_config(route_provider_slug="deepinfra"), {"query": "hi", "max_tokens": 64})

    request = client.completions.calls[0]
    assert request["model"] == "qwen/qwen3-32b"
    assert request["extra_body"]["provider"] == {
        "only": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert request["stream_options"] == {"include_usage": True}
    assert request["extra_headers"]["X-OpenRouter-Metadata"] == "enabled"
    assert metrics["output_tokens"] == 4
    assert metrics["token_source"] == "provider_usage_completion_tokens"


def test_transport_model_id_is_used_without_changing_source_identity(monkeypatch):
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient([_chunk("answer", finish_reason="stop"), _chunk(usage=usage)])
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = CloudConfig(
        provider="deepinfra",
        model_name="Qwen/Qwen3-32B",
        run_ts="2026-08-09 00:00:00",
        temperature=0.1,
        transport_provider="openrouter",
        transport_model_id="qwen/qwen3-32b",
        misc={"route_provider_slug": "deepinfra"},
    )

    openrouter.generate(config, {"query": "hi", "max_tokens": 64})

    assert client.completions.calls[0]["model"] == "qwen/qwen3-32b"
    assert config.source_provider == "deepinfra"
    assert config.source_model_id == "Qwen/Qwen3-32B"


def test_stream_chunks_are_not_counted_as_tokens(monkeypatch):
    client = FakeClient([_chunk("a"), _chunk("b"), _chunk("c", finish_reason="stop")])
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    metrics = openrouter.generate(_config(), {"query": "hi", "max_tokens": 64})

    assert metrics["output_tokens"] == metrics["visible_output_tokens"]
    assert metrics["token_source"] == "tiktoken_visible_text"
    assert metrics["output_tokens"] != 3


def test_route_metadata_and_response_id_are_preserved(monkeypatch):
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient(
        [
            _chunk("answer", finish_reason="stop", metadata={"provider": "DeepInfra", "provider_slug": "deepinfra"}),
            _chunk(usage=usage),
        ]
    )
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    metrics = openrouter.generate(_config(route_provider_slug="deepinfra"), {"query": "hi", "max_tokens": 64})

    assert metrics["observed_provider"] == "DeepInfra"
    assert metrics["observed_provider_slug"] == "deepinfra"
    assert metrics["openrouter_response_id"] == "resp-1"


def test_selected_endpoint_metadata_and_timeout_are_parsed(monkeypatch):
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient(
        [
            _chunk(
                "answer",
                finish_reason="stop",
                metadata={"endpoints": {"available": [{"selected": True, "provider_name": "DeepInfra"}]}},
            ),
            _chunk(usage=usage),
        ]
    )
    observed = {}

    def make_client(**kwargs):
        observed.update(kwargs)
        return client

    monkeypatch.setattr(openrouter, "OpenAI", make_client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = _config(route_provider_slug="deepinfra")
    config.misc["timeout_seconds"] = 17

    metrics = openrouter.generate(config, {"query": "hi", "max_tokens": 64})

    assert observed["timeout"] == 17
    assert observed["max_retries"] == 0
    assert metrics["observed_provider"] == "DeepInfra"
    assert metrics["observed_provider_slug"] == "deepinfra"
    assert metrics["provider_metadata_verified"] is True


def test_google_display_name_resolves_via_expected_pinned_slug():
    from llm_bench.cloud.providers.openrouter import _observed_provider

    provider, slug = _observed_provider({"provider": "Google"}, expected_slug="google-vertex")
    assert provider == "Google"
    assert slug == "google-vertex"
    # Without a pin expectation the ambiguity must not default.
    provider, slug = _observed_provider({"provider": "Google"})
    assert slug is None


def test_or_served_metadata_verified_with_derived_slug(monkeypatch):
    """Marketplace lanes take OpenRouter's display name as authoritative and
    derive the route slug from it; the reviewed allowlist does not apply."""
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient(
        [
            _chunk("answer", finish_reason="stop", metadata={"provider": "Cloudflare"}),
            _chunk(usage=usage),
        ]
    )
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    metrics = openrouter.generate(
        _config(route_provider_slug="cloudflare", route_policy="or-served"),
        {"query": "hi", "max_tokens": 64},
    )

    assert metrics["provider_metadata_verified"] is True
    assert metrics["route_policy"] == "or-served"
    assert metrics["observed_provider"] == "Cloudflare"
    assert metrics["observed_provider_slug"] == "cloudflare"


def test_pinned_route_unknown_provider_fails_closed(monkeypatch):
    """Pinned lanes still resolve against the reviewed allowlist: an unknown
    serving provider stays unverified and fails closed."""
    usage = SimpleNamespace(completion_tokens=4, prompt_tokens=3, total_tokens=7, completion_tokens_details=None)
    client = FakeClient(
        [
            _chunk("answer", finish_reason="stop", metadata={"provider": "Cloudflare"}),
            _chunk(usage=usage),
        ]
    )
    monkeypatch.setattr(openrouter, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    metrics = openrouter.generate(
        _config(route_provider_slug="cloudflare"),
        {"query": "hi", "max_tokens": 64},
    )

    assert metrics["provider_metadata_verified"] is False
    assert metrics["route_policy"] == "pinned-provider"
