#!/usr/bin/env python3
"""Materialize an OpenRouter audit into guarded route-decision records.

The audit report is evidence, not activation. Every source row gets one
decision record. Verified availability rows become ``candidate`` records with
an ``availability_passed`` state; all other rows remain direct. The scheduler
requires an active record and a passed measurement canary before it can route
anything through OpenRouter.

By default this command writes JSON only. MongoDB writes require both
``--apply`` and ``--yes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.routing import ROUTE_DECISION_VERSION
from pymongo import MongoClient

DEFAULT_CANARY_REQUIRED = 2

TERMINAL_REASON_MAP = {
    "bedrock-out-of-scope": "direct-policy-excluded",
    "no-exact-or-ambiguous-model-id": "direct-no-match",
    "catalog-evidence-incomplete": "direct-unknown",
    "endpoint-evidence-missing": "direct-unknown",
    "identity-evidence-missing": "direct-unknown",
    "source-provider-not-listed": "direct-incompatible",
    "protocol-incompatible": "direct-incompatible",
    "observed-provider-mismatch": "direct-probe-failed",
    "probe-failed-or-incomplete": "direct-probe-failed",
    "probe-evidence-incomplete": "direct-probe-failed",
    "visible-output-empty": "direct-probe-failed",
    "budget-exhausted": "direct-unknown",
}


def terminal_state(row: dict[str, Any], *, catalog_scope: dict[str, Any] | None) -> str:
    """Map audit evidence to the stable terminal state used by runtime/reporting."""

    reason = str(row.get("reason_class") or "unknown")
    if reason == "no-exact-or-ambiguous-model-id":
        if not isinstance(catalog_scope, dict) or catalog_scope.get("scope") != "global":
            return "direct-unknown"
    if reason == "ambiguous-model-id":
        return "direct-ambiguous"
    return TERMINAL_REASON_MAP.get(reason, "direct-unknown")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _timestamp(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _probe(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    probe = evidence.get("probe")
    return probe if isinstance(probe, dict) else {}


def _route_probe_id(source_key: str, snapshot_at: str) -> str:
    digest = hashlib.sha256(f"{snapshot_at}:{source_key}".encode("utf-8")).hexdigest()[:20]
    return f"coverage:{digest}"


def materialize_row(
    row: dict[str, Any],
    *,
    snapshot_at: str,
    audit_path: str,
    canary_required: int = DEFAULT_CANARY_REQUIRED,
    catalog_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one audit row into a scheduler route decision."""

    source_provider = str(row.get("provider") or "")
    source_model_id = str(row.get("model_id") or "")
    source_key = str(row.get("source_key") or f"{source_provider}/{source_model_id}")
    base: dict[str, Any] = {
        "source_provider": source_provider,
        "source_model_id": source_model_id,
        "route_decision_version": ROUTE_DECISION_VERSION,
        "route_snapshot_at": snapshot_at,
        "source_audit": audit_path,
        "audit_decision": row.get("decision") or "keep-direct",
        "audit_reason": row.get("reason_class") or "unknown",
        "state": "direct",
        "transport_provider": "direct",
        "route_policy": "direct",
        "terminal_state": terminal_state(row, catalog_scope=catalog_scope),
        "route_revocation_generation": 0,
    }

    if row.get("decision") != "route-or":
        return base

    probe = _probe(row)
    route_model_id = row.get("or_model_id")
    route_provider_slug = row.get("route_provider_slug") or probe.get("route_provider_slug")
    observed_provider = row.get("observed_provider") or probe.get("observed_provider")
    observed_provider_slug = row.get("observed_provider_slug") or probe.get("observed_provider_slug")
    metadata_verified = probe.get("provider_metadata_verified") is True
    if not all((route_model_id, route_provider_slug, observed_provider_slug, metadata_verified)):
        base["audit_reason"] = "incomplete-audit-evidence"
        base["terminal_state"] = "direct-unknown"
        return base

    base.update(
        {
            "state": "candidate",
            "transport_provider": "openrouter",
            "route_model_id": str(route_model_id),
            "route_provider_slug": str(route_provider_slug),
            "observed_provider": str(observed_provider or ""),
            "observed_provider_slug": str(observed_provider_slug),
            "provider_metadata_verified": True,
            "route_policy": "pinned-provider",
            "route_probe_id": _route_probe_id(source_key, snapshot_at),
            "canary_state": "availability_passed",
            "canary_successes": 0,
            "canary_required_successes": max(1, int(canary_required)),
            "route_revocation_generation": 0,
        }
    )
    return base


def materialize_report(
    report: dict[str, Any],
    *,
    audit_path: str,
    canary_required: int = DEFAULT_CANARY_REQUIRED,
    expected_source_count: int | None = None,
) -> dict[str, Any]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("audit report must contain a rows list")
    snapshot_at = _timestamp(report.get("generated_at"))
    decisions = []
    catalog_scope = report.get("catalog_scope")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("audit rows must be objects")
        key = (str(row.get("provider") or ""), str(row.get("model_id") or ""))
        if not all(key):
            raise ValueError("audit rows require provider and model_id")
        if key in seen:
            raise ValueError(f"duplicate audit row for {key[0]}/{key[1]}")
        seen.add(key)
        decisions.append(
            materialize_row(
                row,
                snapshot_at=snapshot_at,
                audit_path=audit_path,
                canary_required=canary_required,
                catalog_scope=catalog_scope if isinstance(catalog_scope, dict) else None,
            )
        )
    if expected_source_count is not None and len(decisions) != expected_source_count:
        raise ValueError(f"expected {expected_source_count} source rows, got {len(decisions)}")
    decisions.sort(key=lambda item: (item["source_provider"], item["source_model_id"]))
    counts: dict[str, int] = {}
    terminal_counts: dict[str, int] = {}
    for decision in decisions:
        key = "candidate" if decision["state"] == "candidate" else "direct"
        counts[key] = counts.get(key, 0) + 1
        terminal = "route-candidate" if key == "candidate" else decision.get("terminal_state", "direct-unknown")
        terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_snapshot": audit_path,
        "audit_generated_at": snapshot_at,
        "route_decision_version": ROUTE_DECISION_VERSION,
        "canary_required_successes": max(1, int(canary_required)),
        "counts": counts,
        "terminal_counts": terminal_counts,
        "source_count": len(decisions),
        "decisions": decisions,
    }


def apply_decisions(
    report: dict[str, Any],
    *,
    client: MongoClient,
    db_name: str,
) -> dict[str, int]:
    """Upsert decisions without overwriting a passed canary."""

    collection = client[db_name][route_decisions_collection_name()]
    inserted_or_updated = 0
    preserved_passed = 0
    for decision in report["decisions"]:
        key = {
            "source_provider": decision["source_provider"],
            "source_model_id": decision["source_model_id"],
            "route_snapshot_at": decision["route_snapshot_at"],
        }
        existing = collection.find_one(key, {"canary_state": 1})
        if existing and existing.get("canary_state") == "passed":
            preserved_passed += 1
            continue
        collection.update_one(
            key,
            {
                "$set": decision,
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
        inserted_or_updated += 1
    return {"inserted_or_updated": inserted_or_updated, "preserved_passed": preserved_passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary-required", type=int, default=DEFAULT_CANARY_REQUIRED)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--apply", action="store_true", help="Upsert decisions into MongoDB")
    parser.add_argument("--yes", action="store_true", help="Confirm the MongoDB write")
    args = parser.parse_args()

    if args.canary_required < 1:
        raise ValueError("--canary-required must be at least 1")
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")

    report = materialize_report(
        load_json(args.audit_json),
        audit_path=str(args.audit_json),
        canary_required=args.canary_required,
        expected_source_count=args.expected_source_count,
    )
    write_json(args.output, report)
    print(json.dumps(report["counts"], sort_keys=True))
    print(f"wrote {args.output}")

    if args.apply:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is required with --apply")
        db_name = os.environ.get("MONGODB_DB", "llm-bench")
        client = MongoClient(uri)
        try:
            result = apply_decisions(report, client=client, db_name=db_name)
        finally:
            client.close()
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
