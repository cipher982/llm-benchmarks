#!/usr/bin/env python3
"""
Test script for OpenAI Responses API with reasoning models.

This validates:
1. Responses API endpoint works with streaming
2. Token counting and timing metrics are captured
3. Response format is correctly parsed
4. Comparison with chat completions (if applicable)
"""

import os
import time

from openai import OpenAI

# Test configuration
TEST_MODELS = [
    "o3-mini",  # Reasoning model
    "o4-mini",  # Reasoning model
    "gpt-4.1-mini",  # Regular model for comparison
]

QUERY = "Tell a long and happy story about the history of the world."
MAX_OUTPUT_TOKENS = 256  # Reasoning models need more tokens for internal reasoning


def test_responses_api(model: str):
    """Test the /v1/responses endpoint with streaming."""
    print(f"\n{'='*60}")
    print(f"Testing Responses API: {model}")
    print(f"{'='*60}")

    client = OpenAI()

    try:
        time_start = time.time()
        first_token_time = None
        response_text = ""
        event_count = 0

        print("Request params:")
        print(f"  - input: {QUERY[:50]}...")
        print(f"  - max_output_tokens: {MAX_OUTPUT_TOKENS}")
        print("\nStreaming events:")

        # Create streaming response
        stream = client.responses.create(
            model=model,
            input=QUERY,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        # Process stream events
        response_obj = None
        for event in stream:
            event_count += 1
            event_type = getattr(event, "type", "unknown")

            # Track first token timing
            if first_token_time is None and event_type == "response.output_text.delta":
                first_token_time = time.time() - time_start
                print(f"  [{event_count}] {event_type} (FIRST TOKEN at {first_token_time:.3f}s)")

            # Collect text deltas
            if event_type == "response.output_text.delta":
                # Event has delta attribute with text
                if hasattr(event, "delta"):
                    response_text += event.delta
            elif event_type in ("response.completed", "response.incomplete", "response.failed"):
                print(f"  [{event_count}] {event_type}")
                # Final response has usage info
                response_obj = event.response
                break
            else:
                print(f"  [{event_count}] {event_type}")

        time_end = time.time()
        total_time = time_end - time_start

        # Get usage from final response
        usage = getattr(response_obj, "usage", None) if response_obj else None

        print("\nResults:")
        print(f"  - Total events: {event_count}")
        print(f"  - Response text length: {len(response_text)} chars")
        print(f"  - Response text: {response_text[:100]}...")
        print(f"  - Total time: {total_time:.3f}s")
        print(
            f"  - Time to first token: {first_token_time:.3f}s" if first_token_time else "  - Time to first token: N/A"
        )

        if usage:
            print("\nUsage metrics:")
            print(f"  - input_tokens: {usage.input_tokens}")
            print(f"  - output_tokens: {usage.output_tokens}")
            print(f"  - total_tokens: {usage.total_tokens}")

            if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
                reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)
                print(f"  - reasoning_tokens: {reasoning_tokens}")

            tokens_per_second = usage.output_tokens / total_time if total_time > 0 else 0
            print(f"  - tokens_per_second: {tokens_per_second:.2f}")
        else:
            print("\n⚠️  No usage metrics available")

        # Check response status
        if response_obj:
            status = getattr(response_obj, "status", "unknown")
            print(f"\nResponse status: {status}")
            if status == "incomplete":
                incomplete_details = getattr(response_obj, "incomplete_details", None)
                if incomplete_details:
                    print(f"  Incomplete reason: {incomplete_details}")

        print(f"\n✅ SUCCESS: {model}")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {model}")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_chat_completions(model: str):
    """Test the /v1/chat/completions endpoint for comparison."""
    print(f"\n{'='*60}")
    print(f"Testing Chat Completions API: {model}")
    print(f"{'='*60}")

    client = OpenAI()

    try:
        time_start = time.time()
        first_token_time = None
        response_text = ""
        chunk_count = 0

        print("Request params:")
        print(f"  - messages: [{{'role': 'user', 'content': '{QUERY[:50]}...'}}]")
        print(f"  - max_completion_tokens: {MAX_OUTPUT_TOKENS}")
        print("\nStreaming chunks:")

        # Create streaming response
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": QUERY}],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        # Process stream chunks
        for chunk in stream:
            chunk_count += 1

            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content

                if content:
                    # Track first token timing
                    if first_token_time is None:
                        first_token_time = time.time() - time_start
                        print(f"  [Chunk {chunk_count}] First content at {first_token_time:.3f}s")

                    response_text += content

        time_end = time.time()
        total_time = time_end - time_start

        print("\nResults:")
        print(f"  - Total chunks: {chunk_count}")
        print(f"  - Response text length: {len(response_text)} chars")
        print(f"  - Response text: {response_text[:100]}...")
        print(f"  - Total time: {total_time:.3f}s")
        print(
            f"  - Time to first token: {first_token_time:.3f}s" if first_token_time else "  - Time to first token: N/A"
        )

        print(f"\n✅ SUCCESS: {model}")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {model}")
        print(f"  {type(e).__name__}: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("OpenAI Responses API Test Script")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        return

    results = {}

    # Test each model with Responses API
    for model in TEST_MODELS:
        success = test_responses_api(model)
        results[f"{model} (responses)"] = success
        time.sleep(1)  # Brief delay between tests

    # Try chat completions for comparison (will fail for o-series)
    print(f"\n{'='*60}")
    print("Comparison: Chat Completions API")
    print(f"{'='*60}")

    for model in ["gpt-4.1-mini"]:  # Only test regular models
        success = test_chat_completions(model)
        results[f"{model} (chat)"] = success
        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")


if __name__ == "__main__":
    main()
