#!/usr/bin/env python3
"""
Lightweight routing tests for OpenAI model endpoint selection.

These checks do not make network calls. They only validate that benchmarked
OpenAI model IDs route to the expected API surface.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from llm_bench.cloud.providers.openai import _is_reasoning_model
from llm_bench.cloud.providers.openai import _is_responses_only_model


def test_reasoning_model_detection():
    """Reasoning models should go straight to the Responses API."""
    test_cases = [
        ("o1", True),
        ("o3", True),
        ("o4-mini", True),
        ("gpt-5.4", False),
        ("gpt-5.4-pro", False),
        ("gpt-5.3-codex", False),
    ]

    for model, expected in test_cases:
        result = _is_reasoning_model(model)
        assert result == expected, f"{model}: expected reasoning={expected}, got {result}"


def test_responses_only_detection():
    """Responses-only GPT-5 variants should not attempt chat.completions first."""
    test_cases = [
        ("gpt-5.4-pro", True),
        ("gpt-5.4-pro-2026-03-05", True),
        ("gpt-5.3-codex", True),
        ("gpt-5.2-codex", True),
        ("gpt-5.1-codex-mini", True),
        ("gpt-5.4", False),
        ("gpt-5.3-chat-latest", False),
        ("gpt-audio-1.5", False),
    ]

    for model, expected in test_cases:
        result = _is_responses_only_model(model)
        assert result == expected, f"{model}: expected responses_only={expected}, got {result}"


def main():
    test_reasoning_model_detection()
    test_responses_only_detection()
    print("All OpenAI routing tests passed")


if __name__ == "__main__":
    main()
