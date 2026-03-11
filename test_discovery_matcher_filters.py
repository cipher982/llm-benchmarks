#!/usr/bin/env python3
"""
Lightweight tests for discovery matcher filtering.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from llm_bench.discovery.matcher import _should_skip_openrouter_model


def test_openai_non_benchmark_filters():
    test_cases = [
        ("openai/gpt-5.4", False),
        ("openai/gpt-5.4-pro", False),
        ("openai/gpt-5.3-codex", False),
        ("openai/gpt-5.3-chat", True),
        ("openai/gpt-5.3-chat-latest", True),
        ("openai/gpt-realtime", True),
        ("openai/gpt-audio-mini", True),
        ("openai/gpt-image-1.5", True),
        ("openai/gpt-5-search-api", True),
        ("openai/o3-deep-research", True),
        ("openai/chatgpt-image-latest", True),
        ("anthropic/claude-opus-4.1", False),
    ]

    for model_id, expected in test_cases:
        result = _should_skip_openrouter_model(model_id)
        assert result == expected, f"{model_id}: expected skip={expected}, got {result}"


def main():
    test_openai_non_benchmark_filters()
    print("All discovery matcher filter tests passed")


if __name__ == "__main__":
    main()
