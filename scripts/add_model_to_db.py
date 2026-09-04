#!/usr/bin/env python3
"""
Script to add a model to the LLM Benchmarks MongoDB database.

This provides a Python alternative to using the seed_model.js script.

Usage:
    python scripts/add_model_to_db.py --provider anthropic --model-id claude-4-sonnet-20250929

Requirements:
    - pymongo installed (pip install pymongo)
    - MONGODB_URI environment variable set
"""

import argparse
import os
import sys
from datetime import datetime

try:
    from pymongo import MongoClient
except ImportError:
    print("❌ Error: pymongo not installed")
    print("   Install it with: pip install pymongo")
    sys.exit(1)


def add_model_to_db(provider: str, model_id: str, enabled: bool = True) -> bool:
    """Add a model to the MongoDB models collection."""
    
    # Get MongoDB configuration from environment
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("❌ Error: MONGODB_URI environment variable not set")
        print("   Example: export MONGODB_URI='mongodb+srv://user:pass@cluster.mongodb.net'")
        return False
    
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    coll_name = os.getenv("MONGODB_COLLECTION_MODELS", "models")
    
    print(f"Configuration:")
    print(f"  Provider:   {provider}")
    print(f"  Model ID:   {model_id}")
    print(f"  Enabled:    {enabled}")
    print(f"  Database:   {db_name}")
    print(f"  Collection: {coll_name}")
    print()
    
    # Create the document
    doc = {
        "provider": provider,
        "model_id": model_id,
        "enabled": enabled,
        "added_at": datetime.now(),
    }
    
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(uri)
        
        # Get collection
        coll = client[db_name][coll_name]
        
        # Check if model already exists
        existing = coll.find_one({"provider": provider, "model_id": model_id})
        if existing:
            print(f"⚠️  Warning: Model already exists in database")
            print(f"   Existing document: {existing}")
            
            response = input("\nUpdate it? (y/N): ").strip().lower()
            if response == 'y':
                result = coll.update_one(
                    {"provider": provider, "model_id": model_id},
                    {"$set": {"enabled": enabled, "updated_at": datetime.now()}}
                )
                print(f"✅ Model updated successfully! (matched: {result.matched_count}, modified: {result.modified_count})")
            else:
                print("Skipped update.")
                return True
        else:
            # Insert the document
            print("Inserting model into database...")
            result = coll.insert_one(doc)
            print(f"✅ Model added successfully! ID: {result.inserted_id}")
        
        # Verify
        print("\nVerifying...")
        found = coll.find_one({"provider": provider, "model_id": model_id})
        if found:
            print("Document in database:")
            for key, value in found.items():
                if key != "_id":
                    print(f"  {key}: {value}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def list_models(provider: str = None):
    """List all models in the database."""
    
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("❌ Error: MONGODB_URI environment variable not set")
        return False
    
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    coll_name = os.getenv("MONGODB_COLLECTION_MODELS", "models")
    
    try:
        client = MongoClient(uri)
        coll = client[db_name][coll_name]
        
        query = {"enabled": True}
        if provider:
            query["provider"] = provider
        
        models = list(coll.find(query, {"provider": 1, "model_id": 1, "enabled": 1, "_id": 0}))
        
        if not models:
            print(f"No enabled models found" + (f" for provider: {provider}" if provider else ""))
            return True
        
        print(f"\nEnabled models" + (f" for provider: {provider}" if provider else "") + ":")
        print(f"{'Provider':<20} {'Model ID':<50} {'Enabled':<10}")
        print("-" * 80)
        
        for model in sorted(models, key=lambda x: (x.get('provider', ''), x.get('model_id', ''))):
            provider_name = model.get('provider', 'N/A')
            model_id = model.get('model_id', 'N/A')
            enabled = model.get('enabled', False)
            print(f"{provider_name:<20} {model_id:<50} {enabled!s:<10}")
        
        print(f"\nTotal: {len(models)} models")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add or manage models in the LLM Benchmarks database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add Claude Sonnet 4.5
  python scripts/add_model_to_db.py --provider anthropic --model-id claude-4-sonnet-20250929
  
  # Add a disabled model (won't be benchmarked)
  python scripts/add_model_to_db.py --provider openai --model-id gpt-5 --disabled
  
  # List all enabled Anthropic models
  python scripts/add_model_to_db.py --list --provider anthropic
  
  # List all enabled models
  python scripts/add_model_to_db.py --list

Environment Variables:
  MONGODB_URI                  - MongoDB connection string (required)
  MONGODB_DB                   - Database name (default: llm-bench)
  MONGODB_COLLECTION_MODELS    - Collection name (default: models)
        """
    )
    
    parser.add_argument("--provider", help="Provider name (e.g., anthropic, openai)")
    parser.add_argument("--model-id", help="Model identifier (e.g., claude-4-sonnet-20250929)")
    parser.add_argument("--disabled", action="store_true", help="Add model as disabled")
    parser.add_argument("--list", action="store_true", help="List models instead of adding")
    
    args = parser.parse_args()
    
    if args.list:
        success = list_models(provider=args.provider)
        sys.exit(0 if success else 1)
    
    if not args.provider or not args.model_id:
        parser.print_help()
        print("\n❌ Error: --provider and --model-id are required (unless using --list)")
        sys.exit(1)
    
    # Special check for Claude Sonnet 4.5
    if args.provider == "anthropic" and "claude-4" in args.model_id:
        print("⚠️  Note: Please verify this is the correct model ID from Anthropic's documentation:")
        print("   https://docs.anthropic.com/en/docs/models-overview")
        print()
    
    enabled = not args.disabled
    success = add_model_to_db(args.provider, args.model_id, enabled)
    
    if success:
        print("\n" + "="*80)
        print("Next Steps:")
        print("="*80)
        if args.provider == "anthropic":
            print("1. Ensure ANTHROPIC_API_KEY is set in your environment")
        print("2. Test the model with a benchmark run:")
        print(f"   python api/bench_headless.py --providers {args.provider} --limit 1")
        print("3. Check for any errors in the errors_cloud collection")
        print()
        print("To list all enabled models for this provider:")
        print(f"   python scripts/add_model_to_db.py --list --provider {args.provider}")
        print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()