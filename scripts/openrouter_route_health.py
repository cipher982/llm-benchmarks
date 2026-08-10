#!/usr/bin/env python3
"""Apply an explicit cooldown or recovery transition to one route record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.routing import cooldown_route_snapshot
from llm_bench.scheduler.routing import recover_route_snapshot
from pymongo import MongoClient


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def apply_route_health(route: dict[str, Any], *, client: MongoClient, db_name: str) -> None:
    collection = client[db_name][route_decisions_collection_name()]
    key = {
        "source_provider": route["source_provider"],
        "source_model_id": route["source_model_id"],
        "route_snapshot_at": route["route_snapshot_at"],
    }
    collection.update_one(key, {"$set": route}, upsert=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cooldown-reason")
    mode.add_argument("--recover-probe-id")
    parser.add_argument("--recovery-probe-passed", action="store_true")
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.yes:
        raise ValueError("--apply requires --yes")
    route = load_json(args.route_json)
    if args.cooldown_reason:
        route = cooldown_route_snapshot(
            route,
            failure_reason=args.cooldown_reason,
            cooldown_seconds=args.cooldown_seconds,
        )
    else:
        route = recover_route_snapshot(
            route,
            recovery_probe_id=args.recover_probe_id,
            recovery_probe_passed=args.recovery_probe_passed,
        )
    write_json(args.output, route)
    print(json.dumps({"state": route["state"], "health": route.get("route_health_state")}, sort_keys=True))
    print(f"wrote {args.output}")
    if args.apply:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is required with --apply")
        client = MongoClient(uri)
        try:
            apply_route_health(route, client=client, db_name=os.environ.get("MONGODB_DB", "llm-bench"))
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
