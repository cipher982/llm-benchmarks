"""
Model catalog loader from MongoDB only.
Returns a dict: {provider: [model_ids]}.
"""
import os
from typing import Dict, List, Tuple

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
        cursor = coll.find(
            {"enabled": True, "deprecated": {"$ne": True}},
            {"provider": 1, "model_id": 1, "created_at": 1, "_id": 0},
        )
        grouped: Dict[str, List[Tuple[str, int, str]]] = {}
        for doc in cursor:
            provider = doc.get("provider")
            model_id = doc.get("model_id")
            if not provider or not model_id:
                continue
            created_at = doc.get("created_at")
            created_ms = int(created_at.timestamp() * 1000) if hasattr(created_at, "timestamp") else 0
            grouped.setdefault(provider, []).append((model_id, created_ms, str(model_id)))

        # Deterministic ordering:
        # - newest created_at first (prioritize newly-added models)
        # - then model_id lexicographically (stable)
        result: Dict[str, List[str]] = {}
        for provider, rows in grouped.items():
            rows.sort(key=lambda r: (-r[1], r[2]))
            result[provider] = [r[0] for r in rows]
        return result
    finally:
        client.close()
