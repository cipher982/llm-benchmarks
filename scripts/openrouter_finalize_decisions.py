#!/usr/bin/env python3
"""Convert candidate decisions plus promoted route snapshots into terminal rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_bench.scheduler.routing import RouteDecision


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def finalize(decisions: dict[str, Any], active_routes: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = decisions.get("decisions")
    if not isinstance(source_rows, list):
        raise ValueError("decisions artifact must contain a decisions list")
    source_by_key = {
        (str(row.get("source_provider")), str(row.get("source_model_id"))): row
        for row in source_rows
        if isinstance(row, dict)
    }
    active_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for route in active_routes:
        if route.get("state") != "active":
            continue
        key = (str(route.get("source_provider")), str(route.get("source_model_id")))
        if key in active_by_key:
            raise ValueError(f"duplicate active route for {key[0]}/{key[1]}")
        decision = RouteDecision.from_snapshot(key[0], key[1], route)
        if decision.transport_provider != "openrouter":
            raise ValueError(f"active route failed validation for {key[0]}/{key[1]}: {decision.reason}")
        source = source_by_key.get(key)
        if not source or source.get("state") != "candidate" or source.get("transport_provider") != "openrouter":
            raise ValueError(f"active route has no matching OpenRouter candidate for {key[0]}/{key[1]}")
        for field in ("route_model_id", "route_provider_slug", "route_probe_id", "observed_provider_slug"):
            if route.get(field) != source.get(field):
                raise ValueError(f"active route evidence does not match candidate {key[0]}/{key[1]}: {field}")
        active_by_key[key] = route
    output: list[dict[str, Any]] = []
    for original in source_rows:
        row = dict(original)
        key = (str(row.get("source_provider")), str(row.get("source_model_id")))
        active = active_by_key.get(key)
        if active:
            row.update(active)
            row["terminal_state"] = "route-approved"
            row["audit_reason"] = "paired-canary-promoted"
        elif row.get("state") == "candidate":
            row.update(
                {
                    "state": "direct",
                    "transport_provider": "direct",
                    "route_policy": "direct",
                    "terminal_state": "direct-unknown",
                    "audit_reason": "canary-not-run-or-not-promoted",
                    "route_model_id": None,
                    "route_provider_slug": None,
                }
            )
        else:
            row.setdefault("terminal_state", "direct-unknown")
        output.append(row)
    output.sort(key=lambda item: (str(item.get("source_provider")), str(item.get("source_model_id"))))
    counts: dict[str, int] = {}
    for row in output:
        state = str(row.get("terminal_state") or "direct-unknown")
        counts[state] = counts.get(state, 0) + 1
    return {
        "schema_version": 1,
        "finalized": True,
        "source_count": len(output),
        "terminal_counts": counts,
        "decisions": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--active-route", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(load_json(args.decisions_json), [load_json(path) for path in args.active_route])
    write_json(args.output, result)
    print(json.dumps(result["terminal_counts"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
