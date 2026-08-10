#!/usr/bin/env python3
"""Run every reviewed candidate through the bounded paired canary.

Candidates whose original direct provider is not configured locally receive a
structured preflight failure artifact. This closes the evidence loop without
pretending that an OpenRouter-only call proves replacement equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from math import ceil
from pathlib import Path
from typing import Any

try:
    from scripts.openrouter_budget import reserve_daily_budget
    from scripts.openrouter_paired_canary import QUERY_TEXT
    from scripts.openrouter_paired_canary import run_canary
except ModuleNotFoundError:  # direct ``uv run python scripts/...``
    from openrouter_budget import reserve_daily_budget
    from openrouter_paired_canary import QUERY_TEXT
    from openrouter_paired_canary import run_canary


DIRECT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _filename(provider: str, model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{provider}__{model_id}") + "__canary.json"


def _pricing_filename(provider: str, model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{provider}__{model_id}") + ".json"


def _failure(candidate: dict[str, Any], pricing: dict[str, Any], reason: str) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc)
    timed_out = reason.startswith("local-canary-timeout")
    return {
        "schema_version": 1,
        "mode": "preflight-failed-paired-canary",
        "generated_at": generated_at.isoformat(),
        "recheck_at": (generated_at + timedelta(hours=24)).isoformat(),
        "canary_budget": {
            "deadline_seconds": 120,
            "attempts_started": 1 if timed_out else 0,
            "retries_allowed": 1,
            "retries_observed": 1 if timed_out else 0,
            "closure": "bounded-window" if timed_out else "preflight",
        },
        "source_provider": candidate["source_provider"],
        "source_model_id": candidate["source_model_id"],
        "route_model_id": candidate.get("route_model_id"),
        "route_provider_slug": candidate.get("route_provider_slug"),
        "pairs_requested": 30,
        "pairs": [],
        "pricing": pricing,
        "evaluation": {
            "canary_state": "failed",
            "promotion_valid": False,
            "output_valid": False,
            "performance_valid": False,
            "metadata_valid": False,
            "error_valid": False,
            "cost_status": "unverified",
            "cost_valid": False,
            "successful_pairs": 0,
            "required_pairs": 30,
            "required_successful_pairs": 29,
            "failure_reason": reason,
        },
    }


def _measured_budget(result: dict[str, Any], *, deadline_seconds: int) -> dict[str, Any]:
    attempts = [
        attempt
        for pair in result.get("pairs", [])
        for attempt in (pair.get("attempts", {}) or {}).values()
        if isinstance(attempt, dict)
    ]
    return {
        "deadline_seconds": deadline_seconds,
        "attempts_started": len(attempts),
        "retries_allowed": 1,
        "retries_observed": sum(int(attempt.get("retry_count", 0) or 0) for attempt in attempts),
        "closure": "measured",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--pricing-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--deadline-seconds", type=int, default=120)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--batch-max-cost-usd", type=float, default=50.0)
    parser.add_argument("--daily-max-cost-usd", type=float, default=50.0)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--only", action="append", dest="only_sources", help="source provider/model key to run")
    parser.add_argument("--mark-failed", help="write terminal failure artifacts without sending requests")
    args = parser.parse_args()
    if args.pairs != 30:
        raise ValueError("promotion canaries must use exactly 30 paired requests")

    report = load_json(args.decisions)
    candidates = [row for row in report.get("decisions", []) if row.get("state") == "candidate"]
    if args.only_sources:
        selected = set(args.only_sources)
        candidates = [
            row for row in candidates if f"{row.get('source_provider')}/{row.get('source_model_id')}" in selected
        ]
    allowed = set(args.providers or DIRECT_KEY_ENV)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        provider = str(candidate["source_provider"])
        model_id = str(candidate["source_model_id"])
        output = args.output_dir / _filename(provider, model_id)
        pricing_path = args.pricing_dir / _pricing_filename(provider, model_id)
        pricing = load_json(pricing_path)
        key_env = DIRECT_KEY_ENV.get(provider)
        if provider not in allowed:
            result = _failure(candidate, pricing, "provider-not-enabled-for-local-canary")
        elif args.mark_failed:
            result = _failure(candidate, pricing, args.mark_failed)
        elif not key_env or not os.environ.get(key_env):
            result = _failure(candidate, pricing, f"missing-local-credential:{key_env or 'unknown'}")
        else:
            max_rate = max(
                float(pricing.get("direct", {}).get("input_per_token", 0) or 0),
                float(pricing.get("direct", {}).get("output_per_token", 0) or 0),
                float(pricing.get("openrouter", {}).get("input_per_token", 0) or 0),
                float(pricing.get("openrouter", {}).get("output_per_token", 0) or 0),
            )
            estimated_input_tokens = max(1, ceil(len(QUERY_TEXT) / 4))
            estimated_cost = args.pairs * 2 * (estimated_input_tokens + args.max_tokens) * max_rate
            if estimated_cost > args.max_cost_usd:
                result = _failure(candidate, pricing, f"estimated-cost-exceeds-cap:{estimated_cost:.6f}")
            else:
                ledger = reserve_daily_budget(
                    args.daily_ledger,
                    amount_usd=min(args.max_cost_usd, estimated_cost),
                    batch_max_usd=args.batch_max_cost_usd,
                    daily_max_usd=args.daily_max_cost_usd,
                    operation=f"paired-canary:{provider}/{model_id}",
                )
                try:
                    result = run_canary(
                        report,
                        provider=provider,
                        model_id=model_id,
                        pairs_count=args.pairs,
                        max_tokens=args.max_tokens,
                        deadline_seconds=args.deadline_seconds,
                        seed=int.from_bytes(
                            hashlib.sha256(f"{provider}/{model_id}".encode("utf-8")).digest()[:4],
                            "big",
                        ),
                        min_route_tps_ratio=0.8,
                        max_route_ttft_ratio=1.5,
                        required_pairs=30,
                        min_success_rate=0.95,
                        max_route_error_delta=0.05,
                        max_route_cost_ratio=1.10,
                        pricing=pricing,
                    )
                    result["budget"] = {
                        "estimated_input_tokens": estimated_input_tokens,
                        "estimated_cost_usd": estimated_cost,
                        "daily_reserved_usd": ledger["reserved_usd"],
                        "daily_ledger": str(args.daily_ledger),
                    }
                    result["canary_budget"] = _measured_budget(result, deadline_seconds=args.deadline_seconds)
                except Exception as exc:  # preserve a terminal artifact for this row
                    result = _failure(candidate, pricing, f"canary-exception:{type(exc).__name__}:{exc}")
        write_json(output, result)
        evaluation = result.get("evaluation", {})
        summary = {
            "source": f"{provider}/{model_id}",
            "route": candidate.get("route_model_id"),
            "state": evaluation.get("canary_state"),
            "promotion_valid": evaluation.get("promotion_valid"),
            "reason": evaluation.get("failure_reason"),
            "output": str(output),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        results.append(summary)
    write_json(args.output_dir / "_summary.json", {"candidates": len(candidates), "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
