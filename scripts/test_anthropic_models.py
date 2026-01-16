#!/usr/bin/env python3
"""
Test script to verify Anthropic API access and list available models.

This helps you:
1. Verify your ANTHROPIC_API_KEY is set correctly
2. Test connectivity to Anthropic's API
3. Discover the exact model IDs available to you

Usage:
    python scripts/test_anthropic_models.py
    
    # Test a specific model
    python scripts/test_anthropic_models.py --test-model claude-4-sonnet-20250929

Requirements:
    pip install anthropic
"""

import argparse
import os
import sys
from datetime import datetime

try:
    from anthropic import Anthropic
except ImportError:
    print("❌ Error: anthropic package not installed")
    print("   Install it with: pip install anthropic")
    sys.exit(1)


def test_api_connection():
    """Test basic connection to Anthropic API."""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Get your API key from: https://console.anthropic.com/")
        return False
    
    print(f"API Key: {api_key[:10]}...{api_key[-4:]} (length: {len(api_key)})")
    print()
    
    try:
        client = Anthropic(api_key=api_key)
        print("✅ Successfully created Anthropic client")
        return client
    except Exception as e:
        print(f"❌ Failed to create Anthropic client: {e}")
        return None


def test_model(client, model_id: str):
    """Test a specific model with a simple query."""
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_id}")
    print('='*80)
    
    try:
        print("Sending test message...")
        start_time = datetime.now()
        
        # Use non-streaming for simplicity
        message = client.messages.create(
            model=model_id,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
            ]
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Model {model_id} is working!")
        print(f"\nResponse:")
        print(f"  Content: {message.content[0].text if message.content else 'N/A'}")
        print(f"  Input tokens: {message.usage.input_tokens}")
        print(f"  Output tokens: {message.usage.output_tokens}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Stop reason: {message.stop_reason}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Model test failed: {error_msg}")
        
        if "model:" in error_msg.lower() or "not found" in error_msg.lower():
            print("\n💡 This might mean:")
            print("   - The model ID is incorrect")
            print("   - The model is not available to your API key")
            print("   - The model name has changed")
        elif "authentication" in error_msg.lower():
            print("\n💡 Check your ANTHROPIC_API_KEY")
        elif "rate" in error_msg.lower():
            print("\n💡 You may have hit rate limits")
        
        return False


def list_known_models():
    """List models we know about from the codebase."""
    
    known_models = [
        ("claude-2.1", "Claude 2.1", "Legacy"),
        ("claude-3-haiku-20240307", "Claude 3 Haiku", "Fast & affordable"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "Latest Haiku"),
        ("claude-3-sonnet-20240229", "Claude 3 Sonnet", "Balanced"),
        ("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet", "Previous flagship"),
        ("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet", "Latest in your DB"),
        ("claude-3-opus-20240229", "Claude 3 Opus", "Most capable (expensive)"),
    ]
    
    print("\n" + "="*80)
    print("Known Anthropic Models")
    print("="*80)
    print(f"{'Model ID':<35} {'Name':<25} {'Notes':<20}")
    print("-"*80)
    
    for model_id, name, notes in known_models:
        print(f"{model_id:<35} {name:<25} {notes:<20}")
    
    print("\nFor Claude Sonnet 4.5 (released Sep 29, 2025), likely IDs:")
    print("  - claude-4-sonnet-20250929")
    print("  - claude-4-5-sonnet-20250929")
    print("\nVerify at: https://docs.anthropic.com/en/docs/models-overview")


def main():
    parser = argparse.ArgumentParser(
        description="Test Anthropic API access and available models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--test-model",
        help="Test a specific model ID (e.g., claude-4-sonnet-20250929)"
    )
    
    parser.add_argument(
        "--test-all-known",
        action="store_true",
        help="Test all known models from the codebase"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Anthropic API Model Test")
    print("="*80)
    print()
    
    # Test API connection
    client = test_api_connection()
    if not client:
        sys.exit(1)
    
    # List known models
    list_known_models()
    
    # Test specific model
    if args.test_model:
        success = test_model(client, args.test_model)
        sys.exit(0 if success else 1)
    
    # Test all known models
    if args.test_all_known:
        print("\n" + "="*80)
        print("Testing All Known Models")
        print("="*80)
        
        models_to_test = [
            "claude-3-5-sonnet-20240620",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ]
        
        results = {}
        for model_id in models_to_test:
            print()
            success = test_model(client, model_id)
            results[model_id] = success
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        for model_id, success in results.items():
            status = "✅ Working" if success else "❌ Failed"
            print(f"{status:<15} {model_id}")
        
        sys.exit(0)
    
    # Default: instructions
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("\n1. Test a specific model:")
    print("   python scripts/test_anthropic_models.py --test-model claude-4-sonnet-20250929")
    print("\n2. Test all known models:")
    print("   python scripts/test_anthropic_models.py --test-all-known")
    print("\n3. Once you find the correct model ID, add it to the database:")
    print("   python scripts/add_model_to_db.py --provider anthropic --model-id <MODEL_ID>")
    print()


if __name__ == "__main__":
    main()