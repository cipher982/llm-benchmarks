"""
Simple database model reader for benchmarking service.
Replaces models.json with a single function call.
"""
import os
from typing import Dict, List
from pymongo import MongoClient


def get_models_from_database() -> Dict[str, List[str]]:
    """
    Get enabled models from database.
    Returns: {provider: [model_ids]} - same format as models.json
    """
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable not set")
    
    client = MongoClient(mongodb_uri)
    db = client['llm-bench'] 
    collection = db['models']
    
    # Get all enabled models
    models = collection.find({"enabled": True}, {"provider": 1, "model_id": 1, "_id": 0})
    
    # Group by provider
    result = {}
    for model in models:
        provider = model["provider"]
        if provider not in result:
            result[provider] = []
        result[provider].append(model["model_id"])
    
    client.close()
    return result


def load_provider_models() -> Dict[str, List[str]]:
    """
    Load models from database.
    This is the only function the benchmarking service needs to call.
    """
    return get_models_from_database()