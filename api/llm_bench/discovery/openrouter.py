"""
OpenRouter catalog fetcher.

Fetches the full model catalog from OpenRouter's free public API and stores it in MongoDB.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List

import httpx
from pymongo import MongoClient

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_PROVIDERS_URL = "https://openrouter.ai/api/v1/providers"


async def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """
    Fetch the full model catalog from OpenRouter API.

    Returns a list of model dictionaries with structure:
    {
        "id": "anthropic/claude-opus-4",
        "name": "Anthropic: Claude Opus 4",
        "created": 1234567890,
        "context_length": 200000,
        "pricing": {
            "prompt": "0.000015",
            "completion": "0.000075"
        },
        ...
    }

    Note: This API is free and requires no authentication.
    """
    logger.info(f"Fetching OpenRouter catalog from {OPENROUTER_API_URL}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(OPENROUTER_API_URL, params={"output_modalities": "text"})
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        logger.info(f"Fetched {len(models)} models from OpenRouter")
        return models


async def fetch_openrouter_providers() -> List[Dict[str, Any]]:
    """Fetch OpenRouter's current hosting-provider registry."""

    logger.info(f"Fetching OpenRouter provider catalog from {OPENROUTER_PROVIDERS_URL}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(OPENROUTER_PROVIDERS_URL)
        response.raise_for_status()
        data = response.json()
        providers = data.get("data", [])
        if not isinstance(providers, list):
            raise ValueError("OpenRouter providers response has no data list")
        logger.info(f"Fetched {len(providers)} providers from OpenRouter")
        return [provider for provider in providers if isinstance(provider, dict) and provider.get("slug")]


def store_catalog_in_db(
    models: List[Dict[str, Any]],
    uri: str | None = None,
    db_name: str | None = None,
) -> int:
    """
    Store OpenRouter catalog in MongoDB.

    Updates existing models or inserts new ones. Tracks first_seen_at and last_seen_at.

    Returns the number of models stored.
    """
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    db_name = db_name or os.getenv("MONGODB_DB", "llm-bench")
    collection_name = "openrouter_catalog"

    client = MongoClient(uri)
    try:
        db = client[db_name]
        collection = db[collection_name]

        now = datetime.now(timezone.utc)
        stored_count = 0

        for model in models:
            openrouter_id = model.get("id")
            if not openrouter_id:
                logger.warning(f"Skipping model with no ID: {model}")
                continue

            # Extract key fields
            name = model.get("name", "")
            org = openrouter_id.split("/")[0] if "/" in openrouter_id else ""

            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0))
            completion_price = float(pricing.get("completion", 0))

            context_length = model.get("context_length", 0)
            created = model.get("created")
            created_date = datetime.fromtimestamp(created, tz=timezone.utc) if created else None
            architecture = model.get("architecture") or {}

            # Check if model already exists
            existing = collection.find_one({"openrouter_id": openrouter_id})

            if existing:
                # Update last_seen_at
                collection.update_one(
                    {"openrouter_id": openrouter_id},
                    {
                        "$set": {
                            "name": name,
                            "org": org,
                            "pricing": {
                                "prompt": prompt_price,
                                "completion": completion_price,
                            },
                            "context_length": context_length,
                            "created": created_date,
                            "canonical_slug": model.get("canonical_slug"),
                            "architecture": architecture,
                            "supported_parameters": model.get("supported_parameters") or [],
                            "input_modalities": architecture.get("input_modalities") or [],
                            "output_modalities": architecture.get("output_modalities") or ["text"],
                            "last_seen_at": now,
                        }
                    },
                )
            else:
                # Insert new model
                collection.insert_one(
                    {
                        "openrouter_id": openrouter_id,
                        "name": name,
                        "org": org,
                        "pricing": {
                            "prompt": prompt_price,
                            "completion": completion_price,
                        },
                        "context_length": context_length,
                        "created": created_date,
                        "canonical_slug": model.get("canonical_slug"),
                        "architecture": architecture,
                        "supported_parameters": model.get("supported_parameters") or [],
                        "input_modalities": architecture.get("input_modalities") or [],
                        "output_modalities": architecture.get("output_modalities") or ["text"],
                        "first_seen_at": now,
                        "last_seen_at": now,
                        # Matching fields will be added by matcher.py
                        "matched_provider": None,
                        "matched_model_id": None,
                        "match_confidence": None,
                        "match_reasoning": None,
                    }
                )

            stored_count += 1

        logger.info(f"Stored {stored_count} models in {collection_name}")
        return stored_count

    finally:
        client.close()


def store_providers_in_db(
    providers: List[Dict[str, Any]],
    uri: str | None = None,
    db_name: str | None = None,
) -> int:
    """Upsert OpenRouter providers for route and provenance discovery."""

    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    db_name = db_name or os.getenv("MONGODB_DB", "llm-bench")
    client = MongoClient(uri)
    try:
        collection = client[db_name]["openrouter_providers"]
        now = datetime.now(timezone.utc)
        stored_count = 0
        for provider in providers:
            slug = provider.get("slug")
            if not slug:
                continue
            collection.update_one(
                {"slug": slug},
                {
                    "$set": {
                        "slug": slug,
                        "name": provider.get("name", slug),
                        "privacy_policy_url": provider.get("privacy_policy_url"),
                        "terms_of_service_url": provider.get("terms_of_service_url"),
                        "status_page_url": provider.get("status_page_url"),
                        "headquarters": provider.get("headquarters"),
                        "datacenters": provider.get("datacenters") or [],
                        "last_seen_at": now,
                    },
                    "$setOnInsert": {"first_seen_at": now},
                },
                upsert=True,
            )
            stored_count += 1
        collection.create_index("slug", unique=True)
        logger.info(f"Stored {stored_count} providers in openrouter_providers")
        return stored_count
    finally:
        client.close()


def get_catalog_from_db(
    uri: str | None = None,
    db_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the OpenRouter catalog from MongoDB.

    Returns a list of model dictionaries.
    """
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    db_name = db_name or os.getenv("MONGODB_DB", "llm-bench")
    collection_name = "openrouter_catalog"

    client = MongoClient(uri)
    try:
        db = client[db_name]
        collection = db[collection_name]

        # Return all models, sorted by last_seen_at (most recent first)
        cursor = collection.find({}).sort("last_seen_at", -1)
        models = list(cursor)

        logger.info(f"Retrieved {len(models)} models from {collection_name}")
        return models

    finally:
        client.close()


def get_our_models_from_db(
    uri: str | None = None,
    db_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve our current models from the models collection.

    Returns a list of {provider, model_id, enabled, deprecated} dicts.
    """
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    db_name = db_name or os.getenv("MONGODB_DB", "llm-bench")
    collection_name = os.getenv("MONGODB_COLLECTION_MODELS", "models")

    client = MongoClient(uri)
    try:
        db = client[db_name]
        collection = db[collection_name]

        cursor = collection.find({}, {"provider": 1, "model_id": 1, "enabled": 1, "deprecated": 1, "_id": 0})
        models = list(cursor)

        logger.info(f"Retrieved {len(models)} models from our models collection")
        return models

    finally:
        client.close()
