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
from llm_bench.scheduler.routing import OR_SERVED_POLICY
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
    revocation_generation: int | None = None,
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
    direct_hashes = {
        str(pair.get("attempts", {}).get("direct", {}).get("effective_request_hash"))
        for pair in pairs
        if pair.get("attempts", {}).get("direct", {}).get("effective_request_hash")
    }
    routed_hashes = {
        str(pair.get("attempts", {}).get("openrouter", {}).get("effective_request_hash"))
        for pair in pairs
        if pair.get("attempts", {}).get("openrouter", {}).get("effective_request_hash")
    }
    if len(direct_hashes) != 1 or len(routed_hashes) != 1:
        raise ValueError("promotion requires stable direct and routed effective request hashes")
    if not canary.get("profile_hash"):
        raise ValueError("promotion requires effective request profile evidence")
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
        # Site policy 2026-08-12: the published number for non-direct-provider
        # lanes is "what OpenRouter serves", so parity is informational, not a
        # promotion gate. Relax via env for the consolidation wave; strict
        # defaults remain for the historical same-provider parity standard.
        min_route_tps_ratio=float(os.environ.get("BENCHMARK_ROUTE_MIN_TPS_RATIO", "0.8")),
        max_route_ttft_ratio=float(os.environ.get("BENCHMARK_ROUTE_MAX_TTFT_RATIO", "1.5")),
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
    if revocation_generation is None:
        raise ValueError("promotion requires the current route revocation generation")
    candidate_generation = int(candidate.get("route_revocation_generation", 0) or 0)
    generation = int(revocation_generation)
    if generation != candidate_generation or generation < 0:
        raise ValueError("promotion generation must match the candidate's current generation")

    route = dict(candidate)
    route.update(
        {
            "state": "active",
            "terminal_state": "route-approved",
            "audit_reason": "paired-canary-promoted",
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
            "canary_ttft_ci95_upper": (
                float(evaluation["route_ttft_ratio_ci95"][1])
                if evaluation.get("route_ttft_ratio_ci95") is not None
                else None
            ),
            "canary_ttft_waived_direct_unmeasured": bool(evaluation.get("ttft_waived_direct_unmeasured")),
            "canary_cost_ci95_upper": float(evaluation["route_cost_ratio_ci95"][1]),
            "canary_route_error_delta": float(evaluation["route_error_delta"]),
            "canary_route_metadata_verified": int(evaluation["route_metadata_verified"]),
            "profile_hash": canary.get("profile_hash"),
            "direct_effective_request_hash": next(iter(direct_hashes)),
            "routed_effective_request_hash": next(iter(routed_hashes)),
            "promoted_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "recheck_at": expires_at.isoformat(),
            "route_revocation_generation": generation,
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


def promote_or_served_route(
    candidate: dict[str, Any],
    canary: dict[str, Any],
    *,
    evidence_path: Path,
    evidence_uri: str | None = None,
    now: datetime | None = None,
    expires_hours: float = 72.0,
    revocation_generation: int | None = None,
) -> dict[str, Any]:
    """Build and validate one or-served route from routed-only canary evidence.

    Marketplace lanes (site policy 2026-08-12) measure what OpenRouter serves
    without pinning to the source provider. The evidence standard is serving
    reliability + verified, stable provider metadata; no direct lane and no
    parity statistics are involved, so the fail-closed re-validation skips the
    paired-canary statistical gates.
    """
    evaluation = canary.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("mode") != "report-only-routed-canary":
        raise ValueError("or-served promotion requires routed-only canary evidence")
    if evaluation.get("promotion_valid") is not True or evaluation.get("canary_state") != "passed":
        raise ValueError("routed-only canary did not pass")
    observed_slug = evaluation.get("observed_provider_slug")
    if not observed_slug:
        raise ValueError("or-served canary did not record a stable observed provider")
    if candidate.get("source_provider") != canary.get("source_provider") or candidate.get(
        "source_model_id"
    ) != canary.get("source_model_id"):
        raise ValueError("canary source identity does not match candidate")
    if candidate.get("route_model_id") != canary.get("route_model_id"):
        raise ValueError("canary route model does not match candidate")
    if expires_hours <= 0:
        raise ValueError("expires_hours must be positive")
    now = now or datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=expires_hours)
    if not evidence_uri or not evidence_uri.startswith("s3://"):
        raise ValueError("promotion requires a durable s3:// evidence URI")
    digest = sha256_file(evidence_path)
    if revocation_generation is None:
        raise ValueError("promotion requires the current route revocation generation")
    candidate_generation = int(candidate.get("route_revocation_generation", 0) or 0)
    generation = int(revocation_generation)
    if generation != candidate_generation or generation < 0:
        raise ValueError("promotion generation must match the candidate's current generation")

    route = dict(candidate)
    route.update(
        {
            "state": "active",
            "terminal_state": "route-approved",
            "audit_decision": "route-or",
            "audit_reason": "or-served-canary-promoted",
            "transport_provider": "openrouter",
            "route_policy": OR_SERVED_POLICY,
            "route_provider_slug": observed_slug,
            "observed_provider_slug": observed_slug,
            "observed_provider": canary.get("observed_provider") or "",
            "provider_metadata_verified": True,
            "canary_id": canary.get("canary_id"),
            "canary_state": "passed",
            "canary_successes": int(evaluation["successful"]),
            "canary_required_successes": int(evaluation["required"]),
            "canary_promotion_gate": "passed",
            "canary_evidence_uri": evidence_uri,
            "canary_evidence_sha256": digest,
            "promoted_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "recheck_at": expires_at.isoformat(),
            "route_revocation_generation": generation,
        }
    )
    decision = RouteDecision.from_snapshot(
        str(route["source_provider"]),
        str(route["source_model_id"]),
        route,
        now=now,
        require_promotion_evidence=False,
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
    parser.add_argument("--revocation-generation", type=int, required=True)
    parser.add_argument("--or-served", action="store_true", help="promote from routed-only canary evidence")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")
    candidate = _find_candidate(load_json(args.decisions_json), args.provider, args.model_id)
    canary = load_json(args.canary_json)
    if args.or_served:
        route = promote_or_served_route(
            candidate,
            canary,
            evidence_path=args.canary_json,
            evidence_uri=args.evidence_uri,
            expires_hours=args.expires_hours,
            revocation_generation=args.revocation_generation,
        )
    else:
        route = promote_decision(
            candidate,
            canary,
            evidence_path=args.canary_json,
            evidence_uri=args.evidence_uri,
            expires_hours=args.expires_hours,
            revocation_generation=args.revocation_generation,
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
