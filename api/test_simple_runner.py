#!/usr/bin/env python3
"""
Test script for bench_simple_runner.py

Tests the runner without making real API calls.
"""

import os
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

# Add api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import bench_simple_runner
from llm_bench import http_output
from llm_bench.config import CloudConfig


def test_parse_models_arg():
    """Test model parsing."""
    assert bench_simple_runner.parse_models_arg("model1,model2") == ["model1", "model2"]
    assert bench_simple_runner.parse_models_arg("model1, model2 , model3") == ["model1", "model2", "model3"]
    assert bench_simple_runner.parse_models_arg("  model1  ") == ["model1"]
    assert bench_simple_runner.parse_models_arg("") == []
    print("✅ parse_models_arg tests passed")


def test_run_single_benchmark_mock():
    """Test run_single_benchmark with mocked provider."""
    # Mock the generate function
    mock_metrics = {
        "output_tokens": 64,
        "generate_time": 1.5,
        "tokens_per_second": 42.67,
        "gen_ts": "2024-01-01 12:00:00",
    }

    mock_generate = MagicMock(return_value=mock_metrics)
    mock_log_http = MagicMock(return_value=True)

    # Mock environment
    os.environ["INGEST_API_URL"] = "http://test.local/ingest"
    os.environ["INGEST_API_KEY"] = "test-key"

    with patch.object(bench_simple_runner, "_load_provider_func", return_value=mock_generate):
        with patch.object(bench_simple_runner, "log_http", mock_log_http):
            result = bench_simple_runner.run_single_benchmark("bedrock", "test-model")

            assert result is True, "Expected successful benchmark"
            assert mock_generate.called, "Provider generate() should be called"
            assert mock_log_http.called, "log_http() should be called"

            # Verify config passed to generate
            call_args = mock_generate.call_args
            config = call_args[0][0]
            assert config.provider == "bedrock"
            assert config.model_name == "test-model"

            # Verify metrics passed to log_http
            http_call_args = mock_log_http.call_args
            assert http_call_args[0][1]["validation_policy"] == "strict_pm10"
            assert http_call_args[0][1]["output_tokens"] == mock_metrics["output_tokens"]

    print("✅ run_single_benchmark mock test passed")


def test_run_single_benchmark_validation():
    """Test that validation catches bad metrics."""
    # Mock generate function that returns invalid metrics
    mock_generate = MagicMock(return_value={"output_tokens": 64})  # Missing required fields

    os.environ["INGEST_API_URL"] = "http://test.local/ingest"
    os.environ["INGEST_API_KEY"] = "test-key"

    with patch.object(bench_simple_runner, "_load_provider_func", return_value=mock_generate):
        result = bench_simple_runner.run_single_benchmark("bedrock", "test-model")
        assert result is False, "Expected validation to fail"

    print("✅ run_single_benchmark validation test passed")


def test_load_provider_func_caching():
    """Test that provider modules are cached."""
    # Clear cache before test
    bench_simple_runner._PROVIDER_MODULES_CACHE.clear()

    # First load
    with patch("builtins.__import__") as mock_import:
        mock_module = MagicMock()
        mock_module.generate = MagicMock(name="mock_generate")
        mock_import.return_value = mock_module

        func1 = bench_simple_runner._load_provider_func("openai")
        assert mock_import.called, "First call should import"

        # Clear the mock call count
        mock_import.reset_mock()

        # Second load should use cache
        func2 = bench_simple_runner._load_provider_func("openai")
        assert not mock_import.called, "Second call should use cache"
        assert func1 is func2

    print("✅ provider caching test passed")


def test_resolve_cycle_config_fetches_remote_config(monkeypatch, tmp_path):
    """Dynamic config mode fetches and persists the remote worklist."""
    cache_path = tmp_path / "last_config.json"
    monkeypatch.setenv("RUNNER_CONFIG_URL", "https://bench-ingest.test/runner-config?provider=bedrock")
    monkeypatch.setenv("RUNNER_CONFIG_TOKEN", "config-token")
    monkeypatch.setenv("RUNNER_CONFIG_CACHE_PATH", str(cache_path))
    monkeypatch.delenv("BENCHMARK_MODELS", raising=False)
    monkeypatch.delenv("BENCHMARK_MODELS_OVERRIDE", raising=False)

    def fake_get(url, headers, timeout):
        assert url == "https://bench-ingest.test/runner-config?provider=bedrock"
        assert headers == {"Authorization": "Bearer config-token"}
        assert timeout == bench_simple_runner.DEFAULT_CONFIG_TIMEOUT_SECONDS
        return httpx.Response(
            200,
            json={
                "schema_version": 1,
                "provider": "bedrock",
                "interval_minutes": 15,
                "models": ["anthropic.claude-opus-4-7"],
                "model_metadata": {"anthropic.claude-opus-4-7": {"omit_temperature": True}},
            },
        )

    monkeypatch.setattr(bench_simple_runner.httpx, "get", fake_get)

    config = bench_simple_runner.resolve_cycle_config("bedrock", None, 30)

    assert config.source == "remote"
    assert config.models == ["anthropic.claude-opus-4-7"]
    assert config.interval_minutes == 15
    assert config.model_metadata == {"anthropic.claude-opus-4-7": {"omit_temperature": True}}
    assert cache_path.exists()


def test_resolve_cycle_config_uses_cache_on_transient_error(monkeypatch, tmp_path):
    """Network/server failures use the persisted cache inside the grace window."""
    cache_path = tmp_path / "last_config.json"
    monkeypatch.setenv("RUNNER_CONFIG_URL", "https://bench-ingest.test/runner-config?provider=bedrock")
    monkeypatch.setenv("RUNNER_CONFIG_TOKEN", "config-token")
    monkeypatch.setenv("RUNNER_CONFIG_CACHE_PATH", str(cache_path))
    bench_simple_runner._persist_runner_config(
        "bedrock",
        ["cached-model"],
        20,
        {
            "schema_version": 1,
            "provider": "bedrock",
            "models": ["cached-model"],
            "model_metadata": {"cached-model": {"omit_temperature": True}},
        },
        {"cached-model": {"omit_temperature": True}},
    )

    def fake_get(url, headers, timeout):
        return httpx.Response(503, text="temporarily unavailable")

    monkeypatch.setattr(bench_simple_runner.httpx, "get", fake_get)

    config = bench_simple_runner.resolve_cycle_config("bedrock", None, 30)

    assert config.source == "cache"
    assert config.models == ["cached-model"]
    assert config.model_metadata == {"cached-model": {"omit_temperature": True}}
    assert config.interval_minutes == 20


def test_resolve_cycle_config_does_not_use_cache_on_auth_error(monkeypatch, tmp_path):
    """Auth failures should not run stale cached work."""
    cache_path = tmp_path / "last_config.json"
    monkeypatch.setenv("RUNNER_CONFIG_URL", "https://bench-ingest.test/runner-config?provider=bedrock")
    monkeypatch.setenv("RUNNER_CONFIG_TOKEN", "config-token")
    monkeypatch.setenv("RUNNER_CONFIG_CACHE_PATH", str(cache_path))
    bench_simple_runner._persist_runner_config(
        "bedrock",
        ["cached-model"],
        20,
        {"schema_version": 1, "provider": "bedrock", "models": ["cached-model"]},
    )

    def fake_get(url, headers, timeout):
        return httpx.Response(401, text="nope")

    monkeypatch.setattr(bench_simple_runner.httpx, "get", fake_get)

    config = bench_simple_runner.resolve_cycle_config("bedrock", None, 30)

    assert config.source == "auth-error"
    assert config.models == []


def test_benchmark_models_override_wins(monkeypatch):
    """Emergency override keeps static model lists available explicitly."""
    monkeypatch.setenv("RUNNER_CONFIG_URL", "https://bench-ingest.test/runner-config?provider=bedrock")
    monkeypatch.setenv("BENCHMARK_MODELS_OVERRIDE", "1")
    monkeypatch.setenv("BENCHMARK_MODELS", "model-a,model-b")

    config = bench_simple_runner.resolve_cycle_config("bedrock", None, 30)

    assert config.source == "env-override"
    assert config.models == ["model-a", "model-b"]


def test_log_http_treats_rejected_ingest_as_failure(monkeypatch):
    """A semantic ingest rejection should not count as a successful benchmark."""
    monkeypatch.setenv("INGEST_API_URL", "https://bench-ingest.test/ingest")
    monkeypatch.setenv("INGEST_API_KEY", "ingest-token")

    def fake_post(url, json, headers, timeout):
        return httpx.Response(202, json={"status": "rejected", "reason": "model is not enabled in catalog"})

    monkeypatch.setattr(http_output.httpx, "post", fake_post)
    config = CloudConfig(provider="bedrock", model_name="disabled-model", run_ts="2026-05-11 20:00:00", temperature=0.1)

    assert http_output.log_http(config, {"tokens_per_second": 1}) is False


if __name__ == "__main__":
    print("Running bench_simple_runner tests...")
    print()

    test_parse_models_arg()
    test_run_single_benchmark_mock()
    test_run_single_benchmark_validation()
    test_load_provider_func_caching()

    print()
    print("=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
