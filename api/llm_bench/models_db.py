"""
Model catalog loader from MongoDB only.
Returns a dict: {provider: [model_ids]}.
"""
import os
from typing import Dict, List

from pymongo import MongoClient


def load_provider_models() -> Dict[str, List[str]]:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")
    coll_name = os.getenv("MONGODB_COLLECTION_MODELS", "models")

    client = MongoClient(uri)
    try:
        coll = client[db_name][coll_name]
        cursor = coll.find({"enabled": True}, {"provider": 1, "model_id": 1, "_id": 0})
        result: Dict[str, List[str]] = {}
        for doc in cursor:
            provider = doc.get("provider")
            model_id = doc.get("model_id")
            if not provider or not model_id:
                continue
            result.setdefault(provider, []).append(model_id)
        return result
    finally:
        client.close()
