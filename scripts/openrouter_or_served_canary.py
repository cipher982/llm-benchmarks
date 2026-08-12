#!/usr/bin/env python3
"""Routed-only canary for or-served (marketplace) lanes.

Site policy 2026-08-12: for providers that are not consumed directly
(openai/vertex/bedrock stay direct), the site measures what OpenRouter
actually serves, without pinning to the source provider. Evidence for one
route is therefore: the routed lane serves reliably, the provider metadata
verifies, and the observed serving provider is stable across attempts.

No direct lane is measured, so no direct-provider API key is needed, and
no parity ratio is computed. This is deliberately lighter than the paired
canary: parity was the old invariant, and it no longer applies to
marketplace lanes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from llm_bench.scheduler.routing import OR_SERVED_POLICY

from scripts.openrouter_budget import DEFAULT_BATCH_MAX_USD
from scripts.openrouter_budget import DEFAULT_DAILY_MAX_USD
from scripts.openrouter_budget import reserve_daily_budget
from scripts.openrouter_paired_canary import QUERY_TEXT
from scripts.openrouter_paired_canary import _active_canary_decision
from scripts.openrouter_paired_canary import _attempt
from scripts.openrouter_paired_canary import _candidate

DEFAULT_ATTEMPTS = 12
DEFAULT_MIN_SUCCESS_RATE = 0.95
DEFAULT_MAX_TOKENS = 64


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_routed_canary(
    report: dict[str, Any],
    *,
    provider: str,
    model_id: str,
    attempts_count: int,
    max_tokens: int,
    deadline_seconds: int,
    min_success_rate: float,
    expected_observed_provider_slug: str | None = None,
) -> dict[str, Any]:
    """Run N routed attempts and evaluate serving reliability + metadata."""
    if attempts_count < 3:
        raise ValueError("attempts_count must be at least 3")
    candidate = _candidate(report, provider, model_id)
    canary_id = f"or-served:{provider}:{model_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    route_decision = _active_canary_decision(candidate, canary_id)

    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + deadline_seconds
    for index in range(1, attempts_count + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            attempts.append(
                {
                    "attempt_index": index,
                    "status": "error",
                    "stage": "deadline",
                    "error": "canary deadline exceeded",
                }
            )
            continue
        attempt = _attempt(
            route_decision,
            max_tokens=max_tokens,
            deadline_seconds=max(1.0, remaining),
        )
        attempt["attempt_index"] = index
        attempts.append(attempt)

    successes = [a for a in attempts if a.get("status") == "success"]
    success_rate = len(successes) / len(attempts)
    observed = {
        str(a.get("metrics", {}).get("observed_provider_slug"))
        for a in successes
        if a.get("metrics", {}).get("observed_provider_slug")
    }
    metadata_verified = (
        all(a.get("metrics", {}).get("provider_metadata_verified") is True for a in successes) if successes else False
    )
    stable_observed = len(observed) == 1 and bool(observed)
    observed_slug = next(iter(observed)) if stable_observed else None

    reasons: list[str] = []
    if success_rate < min_success_rate:
        reasons.append(f"success rate {success_rate:.2f} below {min_success_rate}")
    if not metadata_verified:
        reasons.append("provider metadata was not verified on every success")
    if not stable_observed:
        reasons.append(f"observed serving provider was not stable: {sorted(observed)}")
    if expected_observed_provider_slug and observed_slug != expected_observed_provider_slug:
        reasons.append(f"observed {observed_slug!r} != expected {expected_observed_provider_slug!r}")

    passed = not reasons
    evaluation = {
        "mode": "report-only-routed-canary",
        "canary_state": "passed" if passed else "failed",
        "promotion_valid": passed,
        "successful": len(successes),
        "required": attempts_count,
        "success_rate": round(success_rate, 4),
        "observed_provider_slug": observed_slug,
        "provider_metadata_verified": bool(metadata_verified),
        "route_policy": OR_SERVED_POLICY,
        "reasons": reasons,
    }

    return {
        "schema_version": 1,
        "mode": "report-only-routed-canary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canary_id": canary_id,
        "source_provider": provider,
        "source_model_id": model_id,
        "route_model_id": route_decision.route_model_id,
        "route_provider_slug": observed_slug or candidate.get("route_provider_slug"),
        "observed_provider_slug": observed_slug,
        "observed_provider": (str(successes[0].get("metrics", {}).get("observed_provider") or "") if successes else ""),
        "route_policy": OR_SERVED_POLICY,
        "benchmark_profile_id": "cloud-default-v1",
        "max_tokens": max_tokens,
        "attempts_requested": attempts_count,
        "deadline_seconds": deadline_seconds,
        "pairs": [],
        "attempts": attempts,
        "evaluation": evaluation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--deadline-seconds", type=int, default=180)
    parser.add_argument("--expected-observed-provider-slug", type=str, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--batch-max-cost-usd", type=float, default=DEFAULT_BATCH_MAX_USD)
    parser.add_argument("--daily-max-cost-usd", type=float, default=DEFAULT_DAILY_MAX_USD)
    parser.add_argument("--daily-ledger-json", type=Path, required=True)
    parser.add_argument("--pricing-json", type=Path)
    args = parser.parse_args()

    pricing = load_json(args.pricing_json) if args.pricing_json else {}
    or_pricing = pricing.get("openrouter") or {}
    max_rate = max(
        float(or_pricing.get("input_per_token", 0) or 0),
        float(or_pricing.get("output_per_token", 0) or 0),
    )
    estimated_input_tokens = max(1, len(QUERY_TEXT) // 4)
    estimated_cost = args.attempts * (estimated_input_tokens + args.max_tokens) * max_rate
    ledger = reserve_daily_budget(
        args.daily_ledger_json,
        amount_usd=min(args.max_cost_usd, max(0.001, estimated_cost)),
        batch_max_usd=args.batch_max_cost_usd,
        daily_max_usd=args.daily_max_cost_usd,
        operation=f"or-served-canary:{args.provider}/{args.model_id}",
    )

    result = run_routed_canary(
        load_json(args.decisions_json),
        provider=args.provider,
        model_id=args.model_id,
        attempts_count=args.attempts,
        max_tokens=args.max_tokens,
        deadline_seconds=args.deadline_seconds,
        min_success_rate=args.min_success_rate,
        expected_observed_provider_slug=args.expected_observed_provider_slug,
    )
    result["budget"] = {
        "estimated_cost_usd": estimated_cost,
        "max_cost_usd": args.max_cost_usd,
        "daily_ledger": str(args.daily_ledger_json),
        "daily_reserved_usd": ledger["reserved_usd"],
    }
    write_json(args.output, result)
    print(json.dumps(result["evaluation"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
