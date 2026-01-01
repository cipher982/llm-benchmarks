"""
LLM-based matcher to map OpenRouter models to direct provider APIs.

Uses LLM reasoning to guess the correct provider and model ID, then relies on
API feedback loop to correct mistakes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# Configuration
OPENAI_MODEL = os.getenv("DISCOVERY_LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Supported direct providers (from spec)
SUPPORTED_PROVIDERS = [
    "anthropic",
    "openai",
    "bedrock",
    "vertex",
    "together",
    "groq",
    "fireworks",
    "deepinfra",
    "cerebras",
]


@dataclass
class ModelMatch:
    """Result of matching an OpenRouter model to a direct provider."""
    openrouter_id: str
    openrouter_name: str
    provider: str
    model_id: str
    confidence: float
    reasoning: str


MATCHING_SYSTEM_PROMPT = f"""You are a model discovery assistant for an LLM benchmarking service.

Your job is to match OpenRouter model IDs to their direct provider APIs.

SUPPORTED PROVIDERS: {', '.join(SUPPORTED_PROVIDERS)}

IMPORTANT RULES:
1. Only suggest providers from the supported list above
2. If you're unsure, set confidence < 0.7
3. Your guess doesn't need to be perfect - the API will correct it with clear error messages
4. For Bedrock (AWS):
   - Anthropic Claude models: Use "us.anthropic.*" prefix (e.g., "us.anthropic.claude-opus-4-5-20251101-v1:0")
   - Meta Llama 3.2+: Use "us.meta.*" prefix (e.g., "us.meta.llama4-maverick-17b-instruct-v1:0")
   - Meta Llama 3.1 and older: Use "meta.*" prefix (e.g., "meta.llama3-1-70b-instruct-v1:0")
   - Always add "-v1:0" suffix
5. For OpenAI: Use standard model names (e.g., "gpt-4", "gpt-3.5-turbo")
6. For Anthropic: Use format like "claude-opus-4-20250514"
7. For Vertex (Google): Models are "gemini-*" or "claude-*" (if via Anthropic partnership)

Common patterns:
- "anthropic/claude-opus-4" → provider: "anthropic", model_id: "claude-opus-4-20250514"
- "anthropic/claude-opus-4" → provider: "bedrock", model_id: "us.anthropic.claude-opus-4-20250514-v1:0"
- "openai/gpt-4" → provider: "openai", model_id: "gpt-4"
- "meta-llama/llama-3.3-70b-instruct" → provider: "bedrock", model_id: "us.meta.llama3-3-70b-instruct-v1:0"
- "google/gemini-2.0-flash" → provider: "vertex", model_id: "gemini-2.0-flash"

If model is NOT available via any direct provider, return confidence = 0.0 and explain why.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "provider": "anthropic|openai|bedrock|...",
  "model_id": "...",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of match"
}}
"""


async def call_llm_for_match(openrouter_model: Dict[str, Any]) -> dict:
    """
    Call OpenAI LLM to match OpenRouter model to direct provider.

    Returns parsed JSON dict with provider, model_id, confidence, reasoning.
    Raises exception on API failure.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    # Handle both DB format (openrouter_id) and API format (id)
    openrouter_id = openrouter_model.get("openrouter_id") or openrouter_model.get("id", "")
    openrouter_name = openrouter_model.get("name", "")
    context_length = openrouter_model.get("context_length", 0)

    prompt = f"""Match this OpenRouter model to a direct provider API:

OpenRouter ID: {openrouter_id}
OpenRouter Name: {openrouter_name}
Context Length: {context_length}

Which direct provider offers this model? Guess the model ID they use.

Respond with JSON only (no markdown):
{{"provider": "...", "model_id": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    request_body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": MATCHING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"]["content"]

        # Parse JSON response (handle markdown code fences)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)


async def match_single_model(
    openrouter_model: Dict[str, Any],
    our_models: List[Dict[str, Any]],
) -> Optional[ModelMatch]:
    """
    Match a single OpenRouter model to a direct provider.

    Args:
        openrouter_model: Model dict from OpenRouter catalog (from DB, has openrouter_id field)
        our_models: List of models we already have in our DB

    Returns:
        ModelMatch if this is a new model (not in our DB), None if we already have it
    """
    # Handle both DB format (openrouter_id) and API format (id)
    openrouter_id = openrouter_model.get("openrouter_id") or openrouter_model.get("id", "")
    openrouter_name = openrouter_model.get("name", "")

    # Check if we already benchmark this model
    # Simple heuristic: Check if the model_id appears in our DB
    # (More sophisticated matching could be added later)
    for our_model in our_models:
        if our_model.get("enabled") and not our_model.get("deprecated"):
            # Check for obvious matches in model_id
            our_model_id = our_model.get("model_id", "")
            if openrouter_id.endswith(our_model_id) or our_model_id in openrouter_id:
                logger.debug(f"Skipping {openrouter_id} - already have {our_model_id}")
                return None

    # New model - ask LLM for match
    try:
        match_result = await call_llm_for_match(openrouter_model)

        provider = match_result.get("provider", "")
        model_id = match_result.get("model_id", "")
        confidence = match_result.get("confidence", 0.0)
        reasoning = match_result.get("reasoning", "")

        # Validate provider is supported
        if provider not in SUPPORTED_PROVIDERS:
            logger.warning(f"LLM suggested unsupported provider '{provider}' for {openrouter_id}")
            return None

        # Low confidence matches should be flagged
        if confidence < 0.5:
            logger.info(f"Low confidence match for {openrouter_id}: {provider}/{model_id} (confidence: {confidence})")
            return None

        return ModelMatch(
            openrouter_id=openrouter_id,
            openrouter_name=openrouter_name,
            provider=provider,
            model_id=model_id,
            confidence=confidence,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"Failed to match {openrouter_id}: {e}")
        return None


async def match_to_direct_providers(
    openrouter_models: List[Dict[str, Any]],
    our_models: List[Dict[str, Any]],
    batch_size: int = 10,
    max_matches: Optional[int] = None,
) -> List[ModelMatch]:
    """
    Match OpenRouter models to direct providers using LLM reasoning.

    Args:
        openrouter_models: Models from OpenRouter catalog
        our_models: Models we already have in our DB
        batch_size: Number of concurrent LLM calls
        max_matches: Optional limit on number of matches to return

    Returns:
        List of ModelMatch objects for new models worth adding
    """
    logger.info(f"Matching {len(openrouter_models)} OpenRouter models to direct providers")

    matches = []
    total_processed = 0

    # Process in batches to avoid overwhelming the API
    for i in range(0, len(openrouter_models), batch_size):
        batch = openrouter_models[i:i + batch_size]

        tasks = [match_single_model(model, our_models) for model in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            if result is not None:
                matches.append(result)

        total_processed += len(batch)
        logger.info(f"Processed {total_processed}/{len(openrouter_models)} models, found {len(matches)} new matches")

        # Stop if we've hit the max
        if max_matches and len(matches) >= max_matches:
            logger.info(f"Reached max_matches limit of {max_matches}, stopping")
            break

        # Small delay between batches to be nice to the API
        if i + batch_size < len(openrouter_models):
            await asyncio.sleep(0.5)

    logger.info(f"Matching complete: {len(matches)} new models discovered")
    return matches


def store_matches_in_db(
    matches: List[ModelMatch],
    uri: str | None = None,
    db_name: str | None = None,
) -> int:
    """
    Store match results in the openrouter_catalog collection.

    Updates the matched_provider, matched_model_id, match_confidence, and match_reasoning fields.

    Returns the number of matches stored.
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

        stored_count = 0

        for match in matches:
            result = collection.update_one(
                {"openrouter_id": match.openrouter_id},
                {
                    "$set": {
                        "matched_provider": match.provider,
                        "matched_model_id": match.model_id,
                        "match_confidence": match.confidence,
                        "match_reasoning": match.reasoning,
                    }
                }
            )

            if result.modified_count > 0:
                stored_count += 1

        logger.info(f"Stored {stored_count} matches in {collection_name}")
        return stored_count

    finally:
        client.close()
