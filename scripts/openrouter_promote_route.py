#!/usr/bin/env python3
"""Promote one OpenRouter candidate only from immutable passing evidence.

The paired canary is report-only. This command is the only supported bridge
from a candidate to an active route. It hashes the canary artifact, copies the
statistical gates into the route record, and gives the evidence a finite
expiry. MongoDB writes still require both ``--apply`` and ``--yes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any

from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.routing import RouteDecision
from pymongo import MongoClient

from scripts.openrouter_paired_canary import evaluate


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_candidate(report: dict[str, Any], provider: str, model_id: str) -> dict[str, Any]:
    for decision in report.get("decisions", []):
        if (
            isinstance(decision, dict)
            and decision.get("source_provider") == provider
            and decision.get("source_model_id") == model_id
        ):
            if decision.get("state") != "candidate" or decision.get("transport_provider") != "openrouter":
                raise ValueError(f"{provider}/{model_id} is not an OpenRouter candidate")
            return dict(decision)
    raise ValueError(f"no route decision for {provider}/{model_id}")


def promote_decision(
    candidate: dict[str, Any],
    canary: dict[str, Any],
    *,
    evidence_path: Path,
    evidence_uri: str | None = None,
    now: datetime | None = None,
    expires_hours: float = 24.0,
) -> dict[str, Any]:
    """Build and validate one active route from a passing canary artifact."""

    evaluation = canary.get("evaluation")
    pairs = canary.get("pairs")
    if not isinstance(evaluation, dict) or not isinstance(pairs, list):
        raise ValueError("canary is missing raw pairs or evaluation")
    required_pairs = int(evaluation.get("required_pairs", 0))
    if required_pairs != 30 or len(pairs) != 30:
        raise ValueError("promotion requires exactly 30 raw canary pairs")
    direct_first = sum(1 for pair in pairs if pair.get("order") == ["direct", "openrouter"])
    route_first = sum(1 for pair in pairs if pair.get("order") == ["openrouter", "direct"])
    if direct_first != 15 or route_first != 15:
        raise ValueError("canary order is not balanced")
    if candidate.get("source_provider") != canary.get("source_provider") or candidate.get(
        "source_model_id"
    ) != canary.get("source_model_id"):
        raise ValueError("canary source identity does not match candidate")
    if candidate.get("route_model_id") != canary.get("route_model_id"):
        raise ValueError("canary route model does not match candidate")
    pricing = canary.get("pricing")
    if not isinstance(pricing, dict):
        raise ValueError("canary pricing evidence is missing")
    recomputed = evaluate(
        pairs,
        min_route_tps_ratio=0.8,
        max_route_ttft_ratio=1.5,
        required_pairs=required_pairs,
        min_success_rate=0.95,
        max_route_error_delta=0.05,
        max_route_cost_ratio=1.10,
        pricing=pricing,
        bootstrap_seed=int(canary.get("seed", 0)),
        expected_route_provider_slug=str(candidate["route_provider_slug"]),
    )
    if recomputed.get("canary_state") != "passed" or recomputed.get("promotion_valid") is not True:
        raise ValueError("raw canary pairs do not pass all promotion gates")
    evaluation = recomputed
    if expires_hours <= 0:
        raise ValueError("expires_hours must be positive")
    now = now or datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=expires_hours)
    if not evidence_uri or not evidence_uri.startswith("s3://"):
        raise ValueError("promotion requires a durable s3:// evidence URI")
    digest = sha256_file(evidence_path)

    route = dict(candidate)
    route.update(
        {
            "state": "active",
            "transport_provider": "openrouter",
            "route_policy": "pinned-provider",
            "canary_id": canary.get("canary_id"),
            "canary_state": "passed",
            "canary_successes": int(evaluation["successful_pairs"]),
            "canary_required_successes": int(evaluation["required_successful_pairs"]),
            "canary_cost_status": "verified",
            "canary_promotion_gate": "passed",
            "canary_evidence_uri": evidence_uri,
            "canary_evidence_sha256": digest,
            "canary_tps_ci95_lower": float(evaluation["route_tps_ratio_ci95"][0]),
            "canary_ttft_ci95_upper": float(evaluation["route_ttft_ratio_ci95"][1]),
            "canary_cost_ci95_upper": float(evaluation["route_cost_ratio_ci95"][1]),
            "canary_route_error_delta": float(evaluation["route_error_delta"]),
            "canary_route_metadata_verified": int(evaluation["route_metadata_verified"]),
            "promoted_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "recheck_at": expires_at.isoformat(),
        }
    )
    decision = RouteDecision.from_snapshot(
        str(route["source_provider"]),
        str(route["source_model_id"]),
        route,
        now=now,
    )
    if decision.transport_provider != "openrouter":
        raise ValueError(f"promoted route failed fail-closed validation: {decision.reason}")
    return route


def apply_promotion(route: dict[str, Any], *, client: MongoClient, db_name: str) -> None:
    collection = client[db_name][route_decisions_collection_name()]
    key = {
        "source_provider": route["source_provider"],
        "source_model_id": route["source_model_id"],
        "route_snapshot_at": route["route_snapshot_at"],
    }
    collection.update_one(key, {"$set": route, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}}, upsert=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--canary-json", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-uri")
    parser.add_argument("--expires-hours", type=float, default=24.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")
    route = promote_decision(
        _find_candidate(load_json(args.decisions_json), args.provider, args.model_id),
        load_json(args.canary_json),
        evidence_path=args.canary_json,
        evidence_uri=args.evidence_uri,
        expires_hours=args.expires_hours,
    )
    write_json(args.output, route)
    print(json.dumps({"state": route["state"], "expires_at": route["expires_at"]}, sort_keys=True))
    print(f"wrote {args.output}")
    if args.apply:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is required with --apply")
        client = MongoClient(uri)
        try:
            apply_promotion(route, client=client, db_name=os.environ.get("MONGODB_DB", "llm-bench"))
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
