#!/usr/bin/env python3
"""
Test script for the OpenAI provider implementation with reasoning models.

This validates the actual provider code works correctly.
"""

import os
import sys

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from llm_bench.cloud.providers.openai import generate
from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp


def test_model(model_name: str, max_tokens: int = 64):
    """Test a model with the provider code."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    config = CloudConfig(
        provider="openai",
        model_name=model_name,
        run_ts=get_current_timestamp(),
        temperature=0.1,
        misc={},
    )

    run_config = {
        "query": "Tell a long and happy story about the history of the world.",
        "max_tokens": max_tokens,
    }

    try:
        metrics = generate(config, run_config)

        print(f"\n✅ SUCCESS: {model_name}")
        print("\nMetrics:")
        print(f"  - output_tokens: {metrics['output_tokens']}")
        print(f"  - generate_time: {metrics['generate_time']:.3f}s")
        print(f"  - tokens_per_second: {metrics['tokens_per_second']:.2f}")
        print(f"  - time_to_first_token: {metrics['time_to_first_token']:.3f}s")
        print(f"  - times_between_tokens count: {len(metrics['times_between_tokens'])}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {model_name}")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("OpenAI Provider Test (Reasoning Models)")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        return

    # Test models
    test_cases = [
        ("o3-mini", 256),  # Reasoning model (needs more tokens for reasoning)
        ("o4-mini", 256),  # Reasoning model
        ("gpt-4.1-mini", 64),  # Regular model
    ]

    results = {}
    for model, max_tokens in test_cases:
        success = test_model(model, max_tokens)
        results[model] = success

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model}")

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
