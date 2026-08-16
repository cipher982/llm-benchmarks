#!/usr/bin/env python3
"""Disable non-core source rows that did not earn an OpenRouter route.

This is the final, reversible half of the marketplace migration. Direct core
providers stay enabled; source rows with a passing routed canary stay enabled
under their historical provider identity; every other enabled non-core row is
disabled with a mutation-batch before-image. Existing active route snapshots
for those rows are superseded so they cannot send traffic back to a dead direct
provider.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any

from llm_bench.ops.mutations import MutationBatch
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.queue import cancel_ineligible_jobs
from llm_bench.scheduler.routing import DIRECT_PROVIDERS
from pymongo import MongoClient


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def passing_keys(canary_dir: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in sorted(canary_dir.glob("*__canary.json")):
        payload = load_json(path)
        evaluation = payload.get("evaluation") or {}
        if evaluation.get("promotion_valid") is True:
            provider = str(payload.get("source_provider") or "")
            model_id = str(payload.get("source_model_id") or "")
            if provider and model_id:
                keys.add((provider, model_id))
    return keys


def canary_keys(canary_dir: Path) -> set[tuple[str, str]]:
    """Return every candidate with a canary result, pass or fail."""
    keys: set[tuple[str, str]] = set()
    for path in sorted(canary_dir.glob("*__canary.json")):
        payload = load_json(path)
        provider = str(payload.get("source_provider") or "")
        model_id = str(payload.get("source_model_id") or "")
        if not provider or not model_id:
            raise ValueError(f"canary artifact {path} has no source identity")
        keys.add((provider, model_id))
    return keys


def wave_keys(decisions_path: Path) -> set[tuple[str, str]]:
    """Return every source row covered by a complete build report.

    A retirement pass must be tied to the report that produced the canaries.
    The canary directory alone cannot distinguish a complete wave from a
    partial download or a provider-scoped rerun.
    """
    report = load_json(decisions_path)
    if not isinstance(report, dict):
        raise ValueError("wave decisions must be a JSON object")
    keys: set[tuple[str, str]] = set()
    for row in report.get("decisions", []):
        if not isinstance(row, dict):
            continue
        provider = str(row.get("source_provider") or "")
        model_id = str(row.get("source_model_id") or "")
        if provider and model_id:
            keys.add((provider, model_id))
    for row in report.get("skipped", []):
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "")
        provider, separator, model_id = source.partition("/")
        if separator and provider and model_id:
            keys.add((provider, model_id))
    if not keys:
        raise ValueError("wave decisions contain no source rows")
    return keys


def candidate_keys(decisions_path: Path) -> set[tuple[str, str]]:
    """Return the candidate rows that require a canary artifact."""
    report = load_json(decisions_path)
    if not isinstance(report, dict):
        raise ValueError("wave decisions must be a JSON object")
    keys = {
        (str(row.get("source_provider") or ""), str(row.get("source_model_id") or ""))
        for row in report.get("decisions", [])
        if isinstance(row, dict) and row.get("source_provider") and row.get("source_model_id")
    }
    if not keys:
        raise ValueError("wave decisions contain no candidate rows")
    return keys


def active_route_keys(db) -> set[tuple[str, str]]:
    """Return non-core rows already protected by an active OpenRouter route."""
    return {
        (str(row.get("source_provider") or ""), str(row.get("source_model_id") or ""))
        for row in db[route_decisions_collection_name()].find(
            {"state": "active", "transport_provider": "openrouter"},
            {"source_provider": 1, "source_model_id": 1, "_id": 0},
        )
        if row.get("source_provider") and row.get("source_model_id")
    }


def _open_db() -> tuple[MongoClient, Any]:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is required")
    client = MongoClient(uri)
    return client, client[os.environ.get("MONGODB_DB", "llm-bench")]


def retire_unrouted(
    *,
    db,
    wave_id: str,
    passing: set[tuple[str, str]],
    covered: set[tuple[str, str]],
    apply: bool,
) -> dict[str, Any]:
    rows = list(
        db[models_collection_name()].find(
            {"enabled": True, "deprecated": {"$ne": True}, "provider": {"$nin": sorted(DIRECT_PROVIDERS)}},
            {"provider": 1, "model_id": 1, "_id": 0},
        )
    )
    active_routes = active_route_keys(db)
    current_keys = {
        (str(row.get("provider") or ""), str(row.get("model_id") or ""))
        for row in rows
        if row.get("provider") and row.get("model_id")
    }
    protected = passing | active_routes
    uncovered = sorted(current_keys - (covered | active_routes))
    if uncovered:
        raise ValueError(
            "refusing incomplete marketplace wave; enabled rows are outside the build report: "
            + ", ".join(f"{provider}/{model_id}" for provider, model_id in uncovered[:10])
        )
    retirements = sorted(
        [
            {"provider": str(row.get("provider") or ""), "model_id": str(row.get("model_id") or "")}
            for row in rows
            if (str(row.get("provider") or ""), str(row.get("model_id") or "")) in covered
            and (str(row.get("provider") or ""), str(row.get("model_id") or "")) not in protected
        ],
        key=lambda row: (row["provider"], row["model_id"]),
    )
    result: dict[str, Any] = {
        "wave_id": wave_id,
        "passing_enabled": len(passing),
        "enabled_non_core": len(rows),
        "covered_by_wave": len(covered),
        "active_routes_protected": len(active_routes & current_keys),
        "retirements": len(retirements),
        "by_provider": {},
        "applied_batches": [],
        "superseded_routes": 0,
        "cancelled_jobs": 0,
    }
    for row in retirements:
        result["by_provider"][row["provider"]] = result["by_provider"].get(row["provider"], 0) + 1
    if not apply:
        return result

    now = utc_now()
    remaining = list(retirements)
    reason = f"OpenRouter marketplace migration {wave_id}: no passing routed canary"
    while remaining:
        batch = MutationBatch(db=db, reason=reason, actor="openrouter_marketplace_wave")
        selected: list[dict[str, str]] = []
        for row in remaining:
            if not batch.has_room_for(row["provider"]):
                continue
            batch.set_model_fields(
                provider=row["provider"],
                model_id=row["model_id"],
                enabled=False,
                disabled_class="openrouter_unrouted",
                disabled_reason=reason,
                disabled_at=now,
                recheck_after=now + timedelta(days=30),
                disabled_by="openrouter_marketplace_wave",
                migration_wave_id=wave_id,
            )
            selected.append(row)
        if not selected:
            raise RuntimeError("mutation caps made no progress")
        applied = batch.apply(now=now)
        result["applied_batches"].append(applied)
        selected_keys = {(row["provider"], row["model_id"]) for row in selected}
        remaining = [row for row in remaining if (row["provider"], row["model_id"]) not in selected_keys]

    route_collection = db[route_decisions_collection_name()]
    for row in retirements:
        update = route_collection.update_many(
            {"source_provider": row["provider"], "source_model_id": row["model_id"], "state": "active"},
            {
                "$set": {
                    "state": "superseded",
                    "superseded_at": now,
                    "superseded_reason": "source-row-disabled-without-openrouter-route",
                    "superseded_by": wave_id,
                }
            },
        )
        result["superseded_routes"] += update.modified_count

    result["cancelled_jobs"] = cancel_ineligible_jobs(db, now=now)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True, help="complete build report for this wave")
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")

    passing = passing_keys(args.canary_dir)
    covered = wave_keys(args.decisions)
    expected_canaries = candidate_keys(args.decisions)
    actual_canaries = canary_keys(args.canary_dir)
    if actual_canaries != expected_canaries:
        missing = sorted(expected_canaries - actual_canaries)
        extra = sorted(actual_canaries - expected_canaries)
        raise ValueError("refusing incomplete marketplace canary wave; " f"missing={missing[:10]} extra={extra[:10]}")
    client, db = _open_db()
    try:
        result = retire_unrouted(db=db, wave_id=args.wave_id, passing=passing, covered=covered, apply=args.apply)
    finally:
        client.close()
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
