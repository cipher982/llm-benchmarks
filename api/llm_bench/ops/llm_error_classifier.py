#!/usr/bin/env python3
"""
LLM-based error classification for llm-benchmarks.

`error_taxonomy.classify_error` keys on HTTP status alone and returns UNKNOWN
for anything without one, on the understanding that this module resolves the
rest. It had no caller anywhere — only the example in this docstring — so 946
rollups sat unclassified, and `unknown` was a permanent state rather than a
queue. It now runs on a loop in the daemon.

Classifies unique error fingerprints rather than individual errors, which is
what makes it affordable: a few hundred fingerprints stand in for hundreds of
thousands of rows.

Configuration:
    LLM_CLASSIFIER_MODEL - OpenRouter model (default: openai/gpt-5.6-luna)
    OPENROUTER_API_KEY - Required for classification

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
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional

import httpx
from pymongo import MongoClient

from llm_bench.ops.error_taxonomy import ErrorKind

# Personal-funded OpenRouter, matching the identity normalizer and the standing
# provider routing. This used to call api.openai.com with a work key.
#
# A non-reasoning model on purpose. This answers with one JSON line per error,
# and a reasoning model handed a list-shaped answer spends its budget thinking
# and returns empty content — the same failure that cost three attempts when the
# identity consolidator was first written.
CLASSIFIER_MODEL = os.getenv("LLM_CLASSIFIER_MODEL", "openai/gpt-5.6-luna")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
- kind: auth|billing|rate_limit|hard_model|hard_capability|transient_provider|network|budget_exhausted|unknown
- confidence: 0.0-1.0
- reasoning: Brief explanation (one sentence)

Categories:
- auth: Authentication/authorization (401, 403, API keys, credentials, AWS profiles, security tokens)
- billing: Payment issues (402, credits, invoices, payment required, inference prohibited)
- rate_limit: Throttling (429, quota, too many requests, rate limit)
- hard_model: Model doesn't exist (404 model not found, deprecated, removed, no endpoints)
- hard_capability: Wrong API/parameters (unsupported features, wrong endpoint, parameter mismatch,
  "use responses API", "not a chat model", max_output_tokens vs max_completion_tokens)
- transient_provider: Server errors (5xx, internal server error, service unavailable)
- network: Connection issues (timeout, DNS, connection reset, temporarily unavailable, connection error)
- budget_exhausted: The model answered but spent the whole output budget on reasoning before
  emitting any visible text. Nothing is broken; the benchmark profile cannot measure this model.
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


@dataclass
class LLMUsage:
    """Track LLM API usage for cost monitoring."""

    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 1

    def cost_estimate_usd(self) -> float:
        """Estimate cost based on model pricing."""
        # Pricing as of Dec 2024 (per 1M tokens)
        pricing = {
            "gpt-5-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "o1-mini": {"input": 3.0, "output": 12.0, "reasoning": 3.0},
            "o1": {"input": 15.0, "output": 60.0, "reasoning": 15.0},
        }
        rates = pricing.get(self.model, {"input": 0.15, "output": 0.60, "reasoning": 0.0})
        cost = (self.input_tokens * rates["input"] + self.output_tokens * rates["output"]) / 1_000_000
        if self.reasoning_tokens > 0:
            cost += (self.reasoning_tokens * rates.get("reasoning", 0.0)) / 1_000_000
        return cost


async def call_openai_classifier(prompt: str, system_prompt: str) -> tuple[list[dict], LLMUsage]:
    """Ask the classifier model. Returns (classifications, usage)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    request_body = {
        "model": CLASSIFIER_MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_body,
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"].get("content") or ""

        usage_data = result.get("usage", {})
        reasoning_tokens = usage_data.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        usage = LLMUsage(
            model=result.get("model", CLASSIFIER_MODEL),
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return parse_classification_response(text), usage


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


async def classify_batch(rollups: list[dict]) -> tuple[list[dict], Optional[LLMUsage]]:
    """Classify a batch of error rollups. Returns (classifications, usage)."""
    if not rollups:
        return [], None
    return await call_openai_classifier(build_classification_prompt(rollups), CLASSIFICATION_SYSTEM_PROMPT)


def update_rollups_with_classifications(
    client: MongoClient, db_name: str, rollups_collection: str, rollups: list[dict], classifications: list[dict]
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
                        "classified_by": "llm",
                    }
                },
            )
            stats["updated"] += 1
        except Exception as e:
            print(f"Error updating {fingerprint[:16]}: {e}")
            stats["errors"] += 1

    return stats


async def classify_unclassified_rollups(
    batch_size: int = 50,
    max_rollups: Optional[int] = None,
) -> dict[str, Any]:
    """
    Find and classify unclassified error rollups via LLM.

    Args:
        batch_size: Number of rollups to classify per LLM call
        max_rollups: Maximum total rollups to process (None = unlimited)

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
        query = {"$or": [{"error_kind": {"$exists": False}}, {"error_kind": None}, {"error_kind": "unknown"}]}

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
                "errors": 0,
            }

        print(f"Found {len(unclassified)} unclassified rollups")

        # Process in batches
        total_stats = {"updated": 0, "skipped": 0, "errors": 0}
        total_usage = {
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "cost_estimate_usd": 0.0,
        }

        for i in range(0, len(unclassified), batch_size):
            batch = unclassified[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} rollups)...")

            try:
                classifications, usage = await classify_batch(batch)
                stats = update_rollups_with_classifications(client, db_name, rollups_collection, batch, classifications)

                total_stats["updated"] += stats["updated"]
                total_stats["skipped"] += stats["skipped"]
                total_stats["errors"] += stats["errors"]

                # Aggregate usage
                if usage:
                    total_usage["model"] = usage.model
                    total_usage["input_tokens"] += usage.input_tokens
                    total_usage["output_tokens"] += usage.output_tokens
                    total_usage["total_tokens"] += usage.total_tokens
                    total_usage["api_calls"] += 1
                    total_usage["cost_estimate_usd"] += usage.cost_estimate_usd()

                print(f"  Updated: {stats['updated']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
                if usage:
                    print(f"  Tokens: {usage.input_tokens} in + {usage.output_tokens} out = {usage.total_tokens}")
            except Exception as e:
                print(f"Batch classification failed: {e}")
                total_stats["errors"] += len(batch)

        return {
            "status": "success",
            "total_unclassified": len(unclassified),
            "processed": len(unclassified),
            **total_stats,
            "llm_usage": total_usage,
        }

    finally:
        client.close()


def propagate_classifications(client: MongoClient, db_name: str) -> dict[str, int]:
    """Copy resolved kinds onto the error rows and jobs that act on them.

    The classifier writes its verdict on a rollup, keyed by fingerprint. Nothing
    reads rollups. Retry policy reads `last_attempt_error_kind` on the job, the
    catalogue tools read the same field, and the error history is read per row.
    So a model could be correctly diagnosed as permanently gone and still be
    retried forever, because the diagnosis and the decision were looking at
    different documents.

    A job is only adopted from the newest error recorded for that model, and
    only when the job still says `unknown`. Matching on anything older would let
    a stale fingerprint overwrite a newer, different failure — which is worse
    than leaving it unknown, because it would look authoritative.
    """
    # Canonical helpers, not hand-written env names. Inventing
    # `MONGODB_COLLECTION_JOBS` here read a different variable from the one the
    # scheduler uses, so this silently updated nothing while reporting success.
    from llm_bench.scheduler.mongo import error_rollups_collection_name
    from llm_bench.scheduler.mongo import errors_collection_name
    from llm_bench.scheduler.mongo import jobs_collection_name

    db = client[db_name]
    rollups = db[error_rollups_collection_name()]
    errors = db[errors_collection_name()]
    jobs = db[jobs_collection_name()]

    resolved = {
        row["fingerprint"]: row["error_kind"]
        for row in rollups.find(
            {"classified_by": "llm", "error_kind": {"$nin": [None, "unknown"]}},
            {"fingerprint": 1, "error_kind": 1},
        )
        if row.get("fingerprint")
    }
    if not resolved:
        return {"errors_updated": 0, "jobs_updated": 0}

    stats = {"errors_updated": 0, "jobs_updated": 0}
    for fingerprint, kind in resolved.items():
        result = errors.update_many(
            {"fingerprint": fingerprint, "error_kind": {"$in": [None, "unknown"]}},
            {"$set": {"error_kind": kind, "reclassified_by": "llm"}},
        )
        stats["errors_updated"] += result.modified_count

    for job in jobs.find(
        {"status": "dead_letter", "last_attempt_error_kind": "unknown"},
        {"provider": 1, "model_id": 1},
    ):
        newest = errors.find_one(
            {"provider": job["provider"], "model_name": job["model_id"]},
            {"fingerprint": 1},
            sort=[("ts", -1)],
        )
        kind = resolved.get((newest or {}).get("fingerprint"))
        if not kind:
            continue
        jobs.update_one(
            {"_id": job["_id"]},
            {"$set": {"last_attempt_error_kind": kind, "reclassified_by": "llm"}},
        )
        stats["jobs_updated"] += 1

    return stats


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
