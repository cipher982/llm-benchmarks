#!/usr/bin/env python3
"""
Test script for bench_simple_runner.py

Tests the runner without making real API calls.
"""

import os
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

# Add api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import bench_simple_runner


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
            assert http_call_args[0][1] == mock_metrics

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
