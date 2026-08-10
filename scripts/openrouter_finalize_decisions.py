#!/usr/bin/env python3
"""Convert candidate decisions plus promoted route snapshots into terminal rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
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


def _failure_reason(evaluation: dict[str, Any]) -> str:
    explicit = evaluation.get("failure_reason")
    if explicit:
        return str(explicit)
    if int(evaluation.get("successful_pairs", 0) or 0) < int(evaluation.get("required_successful_pairs", 0) or 0):
        return "insufficient-successful-pairs"
    for key, reason in (
        ("error_valid", "error-gate-failed"),
        ("output_valid", "output-gate-failed"),
        ("performance_valid", "performance-gate-failed"),
        ("metadata_valid", "metadata-gate-failed"),
        ("cost_valid", "cost-gate-failed"),
    ):
        if evaluation.get(key) is False:
            return reason
    return "promotion-gate-failed"


def _recheck_at(canary: dict[str, Any]) -> str | None:
    value = canary.get("recheck_at")
    if value:
        return str(value)
    generated = canary.get("generated_at")
    if not generated:
        return None
    try:
        timestamp = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (timestamp + timedelta(hours=24)).isoformat()


def finalize(
    decisions: dict[str, Any],
    active_routes: list[dict[str, Any]],
    canaries: list[dict[str, Any]] | None = None,
    *,
    canary_paths: list[str] | None = None,
) -> dict[str, Any]:
    source_rows = decisions.get("decisions")
    if not isinstance(source_rows, list):
        raise ValueError("decisions artifact must contain a decisions list")
    source_by_key = {
        (str(row.get("source_provider")), str(row.get("source_model_id"))): row
        for row in source_rows
        if isinstance(row, dict)
    }
    active_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    canary_by_key: dict[tuple[str, str], tuple[dict[str, Any], str | None]] = {}
    for index, canary in enumerate(canaries or []):
        key = (str(canary.get("source_provider")), str(canary.get("source_model_id")))
        if key in canary_by_key:
            raise ValueError(f"duplicate canary for {key[0]}/{key[1]}")
        path = canary_paths[index] if canary_paths and index < len(canary_paths) else None
        canary_by_key[key] = (canary, path)
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
        canary_entry = canary_by_key.get(key)
        if active:
            row.update(active)
            row["terminal_state"] = "route-approved"
            row["audit_reason"] = "paired-canary-promoted"
        elif row.get("state") == "candidate":
            if canary_entry:
                canary, canary_path = canary_entry
                evaluation = canary.get("evaluation", {})
                if evaluation.get("promotion_valid") is True:
                    raise ValueError(f"passing canary has no active route for {key[0]}/{key[1]}")
                row["canary_artifact_path"] = canary_path
                row["canary_state"] = evaluation.get("canary_state", "failed")
                row["canary_cost_status"] = evaluation.get("cost_status", "unverified")
                row["canary_failure_reason"] = _failure_reason(evaluation)
                row["canary_generated_at"] = canary.get("generated_at")
                row["canary_recheck_at"] = _recheck_at(canary)
                row["canary_budget"] = canary.get("canary_budget") or canary.get("budget")
                row["canary_evaluation"] = evaluation
                terminal_state = "direct-canary-failed"
                audit_reason = "paired-canary-failed"
            else:
                terminal_state = "direct-unknown"
                audit_reason = "canary-not-run"
            row.update(
                {
                    "state": "direct",
                    "transport_provider": "direct",
                    "route_policy": "direct",
                    "terminal_state": terminal_state,
                    "audit_reason": audit_reason,
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
    parser.add_argument("--canary", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        load_json(args.decisions_json),
        [load_json(path) for path in args.active_route],
        [load_json(path) for path in args.canary],
        canary_paths=[str(path) for path in args.canary],
    )
    write_json(args.output, result)
    print(json.dumps(result["terminal_counts"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
