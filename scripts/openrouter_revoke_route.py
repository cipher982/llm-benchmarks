#!/usr/bin/env python3
"""Record a monotonic OpenRouter route revocation generation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from datetime import timezone

from llm_bench.scheduler.mongo import route_revocations_collection_name
from pymongo import MongoClient


def revoke_route(
    db,
    *,
    provider: str,
    model_id: str,
    reason: str,
    evidence_sha256: str | None = None,
    operator: str = "operator",
) -> dict:
    collection = db[route_revocations_collection_name()]
    latest = collection.find_one(
        {"source_provider": provider, "source_model_id": model_id},
        sort=[("generation", -1)],
    )
    generation = int(latest.get("generation", 0)) + 1 if latest else 1
    record = {
        "source_provider": provider,
        "source_model_id": model_id,
        "generation": generation,
        "reason": reason,
        "operator": operator,
        "evidence_sha256": evidence_sha256,
        "revoked_at": datetime.now(timezone.utc),
    }
    collection.insert_one(record)
    record.pop("_id", None)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-sha256")
    parser.add_argument("--operator", default=os.environ.get("USER", "operator"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")
    if not args.apply:
        print({"preview": True, "provider": args.provider, "model_id": args.model_id, "reason": args.reason})
        return 0
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is required with --apply")
    client = MongoClient(uri)
    try:
        result = revoke_route(
            client[os.environ.get("MONGODB_DB", "llm-bench")],
            provider=args.provider,
            model_id=args.model_id,
            reason=args.reason,
            evidence_sha256=args.evidence_sha256,
            operator=args.operator,
        )
    finally:
        client.close()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
