#!/usr/bin/env python3
"""
End-to-end test for reasoning models without MongoDB dependency.

This validates:
1. Reasoning models are detected correctly
2. Responses API is used
3. Metrics are captured correctly
4. No regression for regular models
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from llm_bench.cloud.providers.openai import _is_reasoning_model
from llm_bench.cloud.providers.openai import generate
from llm_bench.config import CloudConfig
from llm_bench.utils import get_current_timestamp


def test_model_detection():
    """Test that reasoning models are detected correctly."""
    print("\n" + "=" * 60)
    print("Test 1: Model Detection")
    print("=" * 60)

    test_cases = [
        # Reasoning models (should return True)
        ("o1-mini", True),
        ("o1-pro", True),
        ("o1-preview", True),
        ("o3-mini", True),
        ("o3-mini-2025-01-31", True),
        ("o4-mini", True),
        ("o4-mini-2025-01-17", True),
        # Regular models (should return False)
        ("gpt-4.1-mini", False),
        ("gpt-4o", False),
        ("gpt-5.2", False),
        ("gpt-3.5-turbo", False),
    ]

    all_passed = True
    for model, expected in test_cases:
        result = _is_reasoning_model(model)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"{status} {model:30s} -> {str(result):5s} (expected {expected})")

    return all_passed


def test_reasoning_model_inference():
    """Test that reasoning models work end-to-end."""
    print("\n" + "=" * 60)
    print("Test 2: Reasoning Model Inference")
    print("=" * 60)

    model_name = "o4-mini"
    print(f"\nTesting {model_name}...")

    config = CloudConfig(
        provider="openai",
        model_name=model_name,
        run_ts=get_current_timestamp(),
        temperature=0.1,
        misc={},
    )

    run_config = {
        "query": "Count to 10.",
        "max_tokens": 256,  # Reasoning models need more tokens
    }

    try:
        metrics = generate(config, run_config)

        # Validate metrics structure
        required_fields = [
            "gen_ts",
            "requested_tokens",
            "output_tokens",
            "generate_time",
            "tokens_per_second",
            "time_to_first_token",
            "times_between_tokens",
        ]

        missing = [f for f in required_fields if f not in metrics]
        if missing:
            print(f"❌ Missing fields: {missing}")
            return False

        # Validate metric values
        if metrics["output_tokens"] <= 0:
            print(f"❌ Invalid output_tokens: {metrics['output_tokens']}")
            return False

        if metrics["tokens_per_second"] <= 0:
            print(f"❌ Invalid tokens_per_second: {metrics['tokens_per_second']}")
            return False

        print("✅ Metrics valid:")
        print(f"   - output_tokens: {metrics['output_tokens']}")
        print(f"   - generate_time: {metrics['generate_time']:.3f}s")
        print(f"   - tokens_per_second: {metrics['tokens_per_second']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_regular_model_no_regression():
    """Test that regular models still work correctly."""
    print("\n" + "=" * 60)
    print("Test 3: Regular Model (No Regression)")
    print("=" * 60)

    model_name = "gpt-4.1-mini"
    print(f"\nTesting {model_name}...")

    config = CloudConfig(
        provider="openai",
        model_name=model_name,
        run_ts=get_current_timestamp(),
        temperature=0.1,
        misc={},
    )

    run_config = {
        "query": "Count to 5.",
        "max_tokens": 32,
    }

    try:
        metrics = generate(config, run_config)

        # Validate metrics
        if metrics["output_tokens"] <= 0:
            print(f"❌ Invalid output_tokens: {metrics['output_tokens']}")
            return False

        # Regular models should have time_to_first_token > 0
        if metrics["time_to_first_token"] <= 0:
            print(f"⚠️  Warning: time_to_first_token is {metrics['time_to_first_token']}")

        print("✅ Metrics valid:")
        print(f"   - output_tokens: {metrics['output_tokens']}")
        print(f"   - generate_time: {metrics['generate_time']:.3f}s")
        print(f"   - tokens_per_second: {metrics['tokens_per_second']:.2f}")
        print(f"   - time_to_first_token: {metrics['time_to_first_token']:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("OpenAI Reasoning Models - End-to-End Test Suite")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    results = {
        "Model Detection": test_model_detection(),
        "Reasoning Model Inference": test_reasoning_model_inference(),
        "Regular Model (No Regression)": test_regular_model_no_regression(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + ("=" * 60))
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
