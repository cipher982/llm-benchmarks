"""Which provider lanes the daemon starts.

Lanes used to come from the adapter list, so a lane existed for every provider
the code could talk to regardless of whether anything was enabled for it. That
makes "this provider wrote no metric" ambiguous — idle by design and dead look
identical — and every per-provider health check depends on telling those apart.
"""

from llm_bench.scheduler import cli


def test_lanes_come_from_the_catalogue_not_the_adapter_list(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDER_MODULES", {"openai": object(), "groq": object(), "vertex": object()})
    monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")

    lanes = cli._worker_providers("all", {"openai": ["gpt-4o"], "groq": ["llama"]})

    assert lanes == ["groq", "openai"]


def test_excluded_providers_get_no_lane(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDER_MODULES", {"openai": object(), "bedrock": object()})
    monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")

    lanes = cli._worker_providers("all", {"openai": ["gpt-4o"], "bedrock": ["us.anthropic.claude"]})

    assert lanes == ["openai"]


def test_enabled_models_with_no_adapter_are_announced_not_dropped(monkeypatch, capsys):
    """Silently skipping them is the fallback shape this epic is removing."""
    monkeypatch.setattr(cli, "PROVIDER_MODULES", {"openai": object()})
    monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")

    lanes = cli._worker_providers("all", {"openai": ["gpt-4o"], "mystery": ["m1"]})

    assert lanes == ["openai"]
    assert "mystery" in capsys.readouterr().out


def test_probe_candidates_start_a_worker_before_first_promotion(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDER_MODULES", {"openai": object(), "openrouter": object()})
    monkeypatch.setenv("BENCHMARK_EXCLUDED_PROVIDERS", "bedrock")

    lanes = cli._worker_providers(
        "all",
        {"openai": ["gpt-4o"]},
        probe_providers={"openrouter"},
    )

    assert lanes == ["openai", "openrouter"]
