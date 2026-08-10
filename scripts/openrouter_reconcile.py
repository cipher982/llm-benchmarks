#!/usr/bin/env python3
"""Reconcile one frozen source/catalog audit into an immutable run artifact.

The command is deliberately separate from activation. It verifies that the
source set is complete and that no availability candidate is being presented as
a terminal 241-row result. MongoDB application is optional and idempotent by
the run id derived from the evidence hashes.
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

from llm_bench.scheduler.mongo import route_audit_collection_name
from llm_bench.scheduler.mongo import route_reconciliation_collection_name
from pymongo import MongoClient

TERMINAL_STATES = {
    "route-approved",
    "direct-no-match",
    "direct-ambiguous",
    "direct-incompatible",
    "direct-probe-failed",
    "direct-canary-failed",
    "direct-policy-excluded",
    "direct-unknown",
}


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


def stable_id(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconcile(
    report: dict[str, Any],
    decisions: dict[str, Any],
    *,
    source_snapshot_hash: str,
    catalog_snapshot_hash: str,
    alias_rule_version: str,
    profile_hash: str,
    expected_source_count: int,
    previous: dict[str, Any] | None = None,
    previous_snapshot_hash: str | None = None,
) -> dict[str, Any]:
    rows = decisions.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("decisions artifact must contain a decisions list")
    if decisions.get("finalized") is not True:
        raise ValueError("reconciliation requires a finalized decisions artifact")
    if len(rows) != expected_source_count:
        raise ValueError(f"expected {expected_source_count} decisions, got {len(rows)}")
    keys = [(str(row.get("source_provider")), str(row.get("source_model_id"))) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("decisions contain duplicate source rows")
    candidates = [row for row in rows if row.get("state") == "candidate"]
    if candidates:
        raise ValueError(f"final reconciliation cannot contain {len(candidates)} route candidates; run canaries first")
    for row in rows:
        terminal = row.get("terminal_state")
        if terminal not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {terminal}")
        if terminal == "route-approved":
            if row.get("state") != "active" or row.get("transport_provider") != "openrouter":
                raise ValueError("route-approved row must be an active OpenRouter route")
        elif row.get("state") != "direct" or row.get("transport_provider") != "direct":
            raise ValueError(f"{terminal} row must remain on direct transport")
        if terminal == "route-approved" and row.get("profile_hash") != profile_hash:
            raise ValueError("reconciliation profile hash does not match an approved route")
    current_by_key = {(str(row.get("source_provider")), str(row.get("source_model_id"))): row for row in rows}
    previous_rows = previous.get("decisions", []) if isinstance(previous, dict) else []
    previous_by_key = {
        (str(row.get("source_provider")), str(row.get("source_model_id"))): row
        for row in previous_rows
        if isinstance(row, dict)
    }
    new_keys = sorted(current_by_key.keys() - previous_by_key.keys())
    removed_keys = sorted(previous_by_key.keys() - current_by_key.keys())
    changed_keys = sorted(
        key
        for key in current_by_key.keys() & previous_by_key.keys()
        if stable_id(current_by_key[key]) != stable_id(previous_by_key[key])
    )
    stale_keys = sorted(
        key
        for key in current_by_key.keys() & previous_by_key.keys()
        if previous_by_key[key].get("terminal_state") == "route-approved"
        and current_by_key[key].get("terminal_state") != "route-approved"
    )
    basis = {
        "source_snapshot_hash": source_snapshot_hash,
        "catalog_snapshot_hash": catalog_snapshot_hash,
        "alias_rule_version": alias_rule_version,
        "profile_hash": profile_hash,
        "source_count": expected_source_count,
        "decision_snapshot_hash": stable_id(rows),
    }
    if previous_snapshot_hash:
        basis["previous_snapshot_hash"] = previous_snapshot_hash
    run_id = f"reconcile:{stable_id(basis)[:24]}"
    terminal_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("terminal_state") or "direct-unknown")
        terminal_counts[state] = terminal_counts.get(state, 0) + 1
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "basis": basis,
        "source_count": expected_source_count,
        "terminal_counts": terminal_counts,
        "route_count": terminal_counts.get("route-approved", 0) + terminal_counts.get("active", 0),
        "direct_count": sum(value for key, value in terminal_counts.items() if key.startswith("direct-")),
        "report_snapshot": report.get("generated_at"),
        "delta": {
            "baseline": previous is None,
            "new": [f"{provider}/{model_id}" for provider, model_id in new_keys],
            "changed": [f"{provider}/{model_id}" for provider, model_id in changed_keys],
            "stale": [f"{provider}/{model_id}" for provider, model_id in stale_keys],
            "removed": [f"{provider}/{model_id}" for provider, model_id in removed_keys],
        },
        "decisions": rows,
    }


def apply_reconciliation(artifact: dict[str, Any], *, client: MongoClient, db_name: str) -> None:
    db = client[db_name]
    run_id = artifact["run_id"]
    db[route_reconciliation_collection_name()].replace_one({"run_id": run_id}, artifact, upsert=True)
    for row in artifact["decisions"]:
        key = {
            "run_id": run_id,
            "source_provider": row["source_provider"],
            "source_model_id": row["source_model_id"],
        }
        db[route_audit_collection_name()].replace_one(key, {**key, "decision": row}, upsert=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--catalog-snapshot", type=Path, required=True)
    parser.add_argument("--alias-rule-version", default="or-alias-v1")
    parser.add_argument("--profile-hash", required=True)
    parser.add_argument("--previous-reconciliation", type=Path)
    parser.add_argument("--expected-source-count", type=int, default=241)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")
    audit = load_json(args.audit_json)
    decisions = load_json(args.decisions_json)
    previous = load_json(args.previous_reconciliation) if args.previous_reconciliation else None
    artifact = reconcile(
        audit,
        decisions,
        source_snapshot_hash=sha256_file(args.source_snapshot),
        catalog_snapshot_hash=sha256_file(args.catalog_snapshot),
        alias_rule_version=args.alias_rule_version,
        profile_hash=args.profile_hash,
        expected_source_count=args.expected_source_count,
        previous=previous,
        previous_snapshot_hash=sha256_file(args.previous_reconciliation) if args.previous_reconciliation else None,
    )
    write_json(args.output, artifact)
    print(json.dumps({"run_id": artifact["run_id"], "terminal_counts": artifact["terminal_counts"]}, sort_keys=True))
    print(f"wrote {args.output}")
    if args.apply:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is required with --apply")
        client = MongoClient(uri)
        try:
            apply_reconciliation(artifact, client=client, db_name=os.environ.get("MONGODB_DB", "llm-bench"))
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
