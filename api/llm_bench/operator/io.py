"""
I/O operations for AI Operator - loading signals and storing decisions.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

from pymongo import MongoClient

from ..model_lifecycle.collector import LifecycleSnapshot, collect_lifecycle_snapshots
from .engine import OperatorDecision


def load_snapshots(
    *,
    provider_filter: Optional[List[str]] = None,
    now: Optional[datetime] = None,
    client: Optional[MongoClient] = None
) -> List[LifecycleSnapshot]:
    """
    Load lifecycle snapshots from MongoDB.

    This is a thin wrapper around collector.collect_lifecycle_snapshots()
    for consistency with the operator module pattern.

    Args:
        provider_filter: Optional list of providers to filter
        now: Current time (defaults to utcnow)
        client: Optional MongoDB client (creates one if not provided)

    Returns:
        List of lifecycle snapshots
    """
    return collect_lifecycle_snapshots(
        provider_filter=provider_filter,
        now=now,
        client=client
    )


def store_decisions(
    decisions: List[OperatorDecision],
    *,
    collection_name: str = "model_status",
    client: Optional[MongoClient] = None
) -> int:
    """
    Store operator decisions in MongoDB model_status collection.

    Updates existing documents or inserts new ones with operator_decision field.

    Args:
        decisions: List of operator decisions to store
        collection_name: Name of collection (default: model_status)
        client: Optional MongoDB client (creates one if not provided)

    Returns:
        Number of documents updated
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")

    close_client = False
    if client is None:
        client = MongoClient(uri)
        close_client = True

    try:
        collection = client[db_name][collection_name]
        updated_count = 0

        for decision in decisions:
            # Build operator_decision subdocument
            operator_decision_doc = {
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "suggested_at": decision.suggested_at,
                "suggested_by": decision.suggested_by,
                "status": decision.status,
            }

            if decision.executed_at:
                operator_decision_doc["executed_at"] = decision.executed_at

            # Upsert into model_status
            result = collection.update_one(
                {
                    "provider": decision.provider,
                    "model_id": decision.model_id
                },
                {
                    "$set": {
                        "operator_decision": operator_decision_doc,
                        "last_updated": datetime.now(timezone.utc)
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )

            if result.modified_count > 0 or result.upserted_id is not None:
                updated_count += 1

        return updated_count

    finally:
        if close_client:
            client.close()


def load_pending_decisions(
    *,
    collection_name: str = "model_status",
    client: Optional[MongoClient] = None
) -> List[dict]:
    """
    Load pending operator decisions from MongoDB.

    Args:
        collection_name: Name of collection (default: model_status)
        client: Optional MongoDB client (creates one if not provided)

    Returns:
        List of documents with pending operator decisions
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")

    close_client = False
    if client is None:
        client = MongoClient(uri)
        close_client = True

    try:
        collection = client[db_name][collection_name]

        # Find documents with pending decisions
        query = {
            "operator_decision.status": "pending",
            "operator_decision.action": {"$exists": True}
        }

        projection = {
            "provider": 1,
            "model_id": 1,
            "operator_decision": 1,
            "_id": 0
        }

        return list(collection.find(query, projection))

    finally:
        if close_client:
            client.close()
