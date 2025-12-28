#!/usr/bin/env python3
"""
LLM-based error classification for llm-benchmarks.

Classifies unique error fingerprints (not individual errors) using Claude Haiku or GPT-4o-mini.
This reduces LLM API calls by 100-1000x compared to per-error classification.

Usage:
    from llm_bench.ops.llm_error_classifier import classify_unclassified_rollups

    # Classify all unclassified rollups
    results = await classify_unclassified_rollups()

    # Or run as standalone script
    python -m api.llm_bench.ops.llm_error_classifier
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pymongo import MongoClient

from llm_bench.ops.error_taxonomy import ErrorKind


@dataclass
class ClassificationResult:
    """Result of LLM classification."""
    fingerprint: str
    error_kind: ErrorKind
    confidence: float
    reasoning: str
    classified_at: datetime


# Classification prompt template
CLASSIFICATION_SYSTEM_PROMPT = """You are classifying LLM API errors. For each error, output a JSON line with:
- kind: auth|billing|rate_limit|hard_model|hard_capability|transient_provider|network|unknown
- confidence: 0.0-1.0
- reasoning: Brief explanation (one sentence)

Categories:
- auth: Authentication/authorization (401, 403, API keys, credentials, AWS profiles, security tokens)
- billing: Payment issues (402, credits, invoices, payment required, inference prohibited)
- rate_limit: Throttling (429, quota, too many requests, rate limit)
- hard_model: Model doesn't exist (404 model not found, deprecated, removed, no endpoints)
- hard_capability: Wrong API/parameters (unsupported features, wrong endpoint, parameter mismatch, "use responses API", "not a chat model", max_output_tokens vs max_completion_tokens)
- transient_provider: Server errors (5xx, internal server error, service unavailable)
- network: Connection issues (timeout, DNS, connection reset, temporarily unavailable, connection error)
- unknown: Cannot determine from the error message

Important:
- hard_capability means OUR code needs updating (API version change, endpoint migration required)
- hard_model means the provider removed/deprecated the model (model ID doesn't exist anymore)
- Distinguish between temporary server issues (transient_provider) and connection problems (network)

Respond with ONLY valid JSON lines, one per error. No markdown, no extra text."""


def build_classification_prompt(errors: list[dict]) -> str:
    """Build the user prompt for batch error classification."""
    lines = ["Classify these errors:\n"]

    for i, err in enumerate(errors, 1):
        fp = err.get("fingerprint", "unknown")[:16]
        provider = err.get("provider", "?")
        model = err.get("model_name", "?")
        stage = err.get("stage", "?")
        samples = err.get("sample_messages", [])
        sample = samples[0] if samples else "no message"

        lines.append(f"{i}. [{fp}] {provider}:{model} ({stage})")
        lines.append(f"   Message: {sample[:500]}")
        lines.append("")

    return "\n".join(lines)


async def call_anthropic_classifier(prompt: str, system_prompt: str) -> list[dict]:
    """Call Claude Haiku for classification."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    request_body = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 4096,
        "temperature": 0,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        # Extract text from content blocks
        text = ""
        for content in result.get("content", []):
            if content.get("type") == "text":
                text += content.get("text", "")

        return parse_classification_response(text)


async def call_openai_classifier(prompt: str, system_prompt: str) -> list[dict]:
    """Call GPT-4o-mini for classification."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    request_body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"]["content"]
        return parse_classification_response(text)


def parse_classification_response(text: str) -> list[dict]:
    """Parse LLM response into classification results."""
    results = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Remove markdown code fences if present
        if line.startswith("```"):
            continue

        try:
            obj = json.loads(line)
            if "kind" in obj and "confidence" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Skip malformed lines
            continue

    return results


async def classify_batch(
    rollups: list[dict],
    prefer_anthropic: bool = True
) -> list[dict]:
    """Classify a batch of error rollups using LLM."""
    if not rollups:
        return []

    prompt = build_classification_prompt(rollups)

    # Try Anthropic first if preferred and key is available
    if prefer_anthropic and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return await call_anthropic_classifier(prompt, CLASSIFICATION_SYSTEM_PROMPT)
        except Exception as e:
            print(f"Anthropic classification failed: {e}")
            # Fall through to OpenAI if available

    # Try OpenAI as fallback
    if os.getenv("OPENAI_API_KEY"):
        return await call_openai_classifier(prompt, CLASSIFICATION_SYSTEM_PROMPT)

    raise ValueError("No API key available (need ANTHROPIC_API_KEY or OPENAI_API_KEY)")


def update_rollups_with_classifications(
    client: MongoClient,
    db_name: str,
    rollups_collection: str,
    rollups: list[dict],
    classifications: list[dict]
) -> dict[str, int]:
    """Update rollups collection with LLM classifications."""
    stats = {"updated": 0, "skipped": 0, "errors": 0}

    if len(rollups) != len(classifications):
        print(f"Warning: {len(rollups)} rollups but {len(classifications)} classifications")

    collection = client[db_name][rollups_collection]
    now = datetime.now(timezone.utc)

    for rollup, classification in zip(rollups, classifications):
        fingerprint = rollup.get("fingerprint")
        if not fingerprint:
            stats["skipped"] += 1
            continue

        try:
            kind = classification.get("kind", "unknown")
            confidence = float(classification.get("confidence", 0.0))
            reasoning = classification.get("reasoning", "")

            # Validate kind
            if kind not in {k.value for k in ErrorKind}:
                print(f"Invalid kind '{kind}' for {fingerprint[:16]}, using unknown")
                kind = ErrorKind.UNKNOWN.value

            # Update the rollup
            collection.update_one(
                {"fingerprint": fingerprint},
                {
                    "$set": {
                        "error_kind": kind,
                        "classification_confidence": confidence,
                        "classification_reasoning": reasoning,
                        "classified_at": now,
                        "classified_by": "llm"
                    }
                }
            )
            stats["updated"] += 1
        except Exception as e:
            print(f"Error updating {fingerprint[:16]}: {e}")
            stats["errors"] += 1

    return stats


async def classify_unclassified_rollups(
    batch_size: int = 50,
    max_rollups: Optional[int] = None,
    prefer_anthropic: bool = True
) -> dict[str, Any]:
    """
    Find and classify unclassified error rollups via LLM.

    Args:
        batch_size: Number of rollups to classify per LLM call
        max_rollups: Maximum total rollups to process (None = unlimited)
        prefer_anthropic: Try Anthropic first, fall back to OpenAI

    Returns:
        Statistics about the classification run
    """
    # Get MongoDB connection
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    rollups_collection = os.getenv("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")

    if not uri:
        raise ValueError("MONGODB_URI not set")

    client = MongoClient(uri)

    try:
        collection = client[db_name][rollups_collection]

        # Find unclassified rollups (error_kind is null or "unknown")
        query = {
            "$or": [
                {"error_kind": {"$exists": False}},
                {"error_kind": None},
                {"error_kind": "unknown"}
            ]
        }

        # Limit if requested
        cursor = collection.find(query).sort("count", -1)  # Process most frequent first
        if max_rollups:
            cursor = cursor.limit(max_rollups)

        unclassified = list(cursor)

        if not unclassified:
            return {
                "status": "success",
                "total_unclassified": 0,
                "processed": 0,
                "updated": 0,
                "skipped": 0,
                "errors": 0
            }

        print(f"Found {len(unclassified)} unclassified rollups")

        # Process in batches
        total_stats = {"updated": 0, "skipped": 0, "errors": 0}

        for i in range(0, len(unclassified), batch_size):
            batch = unclassified[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} rollups)...")

            try:
                classifications = await classify_batch(batch, prefer_anthropic)
                stats = update_rollups_with_classifications(
                    client, db_name, rollups_collection, batch, classifications
                )

                total_stats["updated"] += stats["updated"]
                total_stats["skipped"] += stats["skipped"]
                total_stats["errors"] += stats["errors"]

                print(f"  Updated: {stats['updated']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
            except Exception as e:
                print(f"Batch classification failed: {e}")
                total_stats["errors"] += len(batch)

        return {
            "status": "success",
            "total_unclassified": len(unclassified),
            "processed": len(unclassified),
            **total_stats
        }

    finally:
        client.close()


# CLI interface
async def main_async():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Classify unclassified error rollups using LLM")
    parser.add_argument("--batch-size", type=int, default=50, help="Rollups per LLM call")
    parser.add_argument("--max-rollups", type=int, help="Maximum rollups to process")
    parser.add_argument("--use-openai", action="store_true", help="Prefer OpenAI over Anthropic")
    args = parser.parse_args()

    print("Starting LLM error classification...")
    results = await classify_unclassified_rollups(
        batch_size=args.batch_size,
        max_rollups=args.max_rollups,
        prefer_anthropic=not args.use_openai
    )

    print("\nClassification complete:")
    print(f"  Total unclassified: {results['total_unclassified']}")
    print(f"  Processed: {results['processed']}")
    print(f"  Updated: {results['updated']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors: {results['errors']}")


def main():
    """Synchronous wrapper for async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
