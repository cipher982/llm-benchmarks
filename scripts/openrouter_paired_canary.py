#!/usr/bin/env python3
"""Run a bounded paired direct-versus-OpenRouter measurement canary.

This command uses the benchmark runner's production adapters and validation
policy, but never writes published metrics or route decisions. It emits a JSON
evidence artifact. Cost remains an explicit gate because the current direct
provider metric contract does not expose a comparable cost field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any

from llm_bench.scheduler import runner
from llm_bench.scheduler.routing import RouteDecision

try:
    from scripts.openrouter_budget import DEFAULT_BATCH_MAX_USD
    from scripts.openrouter_budget import DEFAULT_DAILY_MAX_USD
    from scripts.openrouter_budget import reserve_daily_budget
except ModuleNotFoundError:  # direct ``python scripts/openrouter_paired_canary.py``
    from openrouter_budget import DEFAULT_BATCH_MAX_USD
    from openrouter_budget import DEFAULT_DAILY_MAX_USD
    from openrouter_budget import reserve_daily_budget

QUERY_TEXT = "Tell a long and happy story about the history of the world."
DEFAULT_MAX_TOKENS = 64
DEFAULT_REQUIRED_PAIRS = 30
DEFAULT_PAIRS = DEFAULT_REQUIRED_PAIRS
DEFAULT_MIN_SUCCESS_RATE = 0.95
DEFAULT_MIN_ROUTE_TPS_RATIO = 0.8
DEFAULT_MAX_ROUTE_TTFT_RATIO = 1.5
DEFAULT_MAX_ROUTE_ERROR_DELTA = 0.05
DEFAULT_MAX_ROUTE_COST_RATIO = 1.10
DEFAULT_BOOTSTRAP_SAMPLES = 10_000

METRIC_FIELDS = (
    "output_tokens",
    "generated_output_tokens",
    "visible_output_tokens",
    "reasoning_tokens",
    "input_tokens",
    "cached_input_tokens",
    "total_tokens",
    "tokens_per_second",
    "visible_tokens_per_second",
    "generate_time",
    "time_to_first_token",
    "finish_reason",
    "response_status",
    "response_id",
    "openrouter_response_id",
    "observed_provider",
    "observed_provider_slug",
    "provider_metadata_verified",
    "token_source",
    "validation_policy",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in METRIC_FIELDS if key in metrics}


def _candidate(report: dict[str, Any], provider: str, model_id: str) -> dict[str, Any]:
    for decision in report.get("decisions", []):
        if (
            isinstance(decision, dict)
            and decision.get("source_provider") == provider
            and decision.get("source_model_id") == model_id
        ):
            if decision.get("state") != "candidate" or decision.get("transport_provider") != "openrouter":
                raise ValueError(f"{provider}/{model_id} is not an OpenRouter candidate")
            return decision
    raise ValueError(f"no route decision for {provider}/{model_id}")


def _active_canary_decision(candidate: dict[str, Any], canary_id: str) -> RouteDecision:
    snapshot = dict(candidate)
    required = max(1, int(snapshot.get("canary_required_successes", DEFAULT_PAIRS)))
    snapshot.update(
        {
            "state": "active",
            "canary_id": canary_id,
            "canary_state": "passed",
            "canary_successes": required,
            "canary_required_successes": required,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
    )
    decision = RouteDecision.from_snapshot(
        str(candidate["source_provider"]),
        str(candidate["source_model_id"]),
        snapshot,
        require_promotion_evidence=False,
    )
    if decision.transport_provider != "openrouter":
        raise ValueError(f"candidate did not resolve to OpenRouter: {decision.reason}")
    return decision


def _attempt(
    decision: RouteDecision,
    *,
    max_tokens: int,
    deadline_seconds: int,
) -> dict[str, Any]:
    effective_request = {
        "transport_provider": decision.transport_provider,
        "transport_model_id": decision.transport_model_id,
        "route_provider_slug": decision.route_provider_slug,
        "source_provider": decision.source_provider,
        "source_model_id": decision.source_model_id,
        "query": QUERY_TEXT,
        "max_tokens": max_tokens,
        "temperature": runner.TEMPERATURE,
        "protocol_version": runner.PROTOCOL_VERSION,
    }
    global_deadline = time.monotonic() + max(0.1, float(deadline_seconds))
    errors: list[str] = []
    stage = "generate"
    last_retry_count = 0
    for retry_count in range(2):
        remaining_seconds = global_deadline - time.monotonic()
        if remaining_seconds <= 0:
            errors.append("attempt deadline exhausted before retry")
            break
        timeout_seconds = min(45.0, max(0.1, remaining_seconds))
        last_retry_count = retry_count
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _timeout(_signum, _frame):
            raise TimeoutError(f"attempt exceeded {timeout_seconds:.1f}s")

        try:
            signal.signal(signal.SIGALRM, _timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            _, metrics = runner._generate_and_validate(
                decision,
                run_ts=datetime.now(timezone.utc).isoformat(),
                run_config={"query": QUERY_TEXT, "max_tokens": max_tokens},
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            return {
                "status": "success",
                "metrics": _safe_metrics(metrics),
                "effective_request": effective_request,
                "effective_request_hash": stable_hash(effective_request),
                "retry_count": retry_count,
                "retry_errors": errors,
            }
        except runner.AttemptFailure as exc:
            stage = exc.stage
            errors.append(exc.message)
        except Exception as exc:  # noqa: BLE001 - preserve per-attempt evidence
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    return {
        "status": "error",
        "stage": stage,
        "error": errors[-1] if errors else "attempt failed",
        "retry_count": last_retry_count,
        "retry_errors": errors,
        "effective_request": effective_request,
        "effective_request_hash": stable_hash(effective_request),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_ci(values: list[float], *, seed: int, samples: int = DEFAULT_BOOTSTRAP_SAMPLES) -> list[float] | None:
    """Return a deterministic percentile bootstrap CI for paired effects."""

    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = [median(rng.choice(values) for _ in values) for _ in range(samples)]
    lower = _percentile(estimates, 0.025)
    upper = _percentile(estimates, 0.975)
    assert lower is not None and upper is not None
    return [lower, upper]


def _price_attempt(attempt: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    metrics = attempt.get("metrics", {})
    input_tokens = metrics.get("input_tokens")
    output_tokens = metrics.get("generated_output_tokens", metrics.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    try:
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cached = min(input_tokens, max(0, int(metrics.get("cached_input_tokens") or 0)))
        input_rate = float(pricing["input_per_token"])
        output_rate = float(pricing["output_per_token"])
        cached_rate = float(pricing.get("cached_input_per_token", input_rate))
    except (KeyError, TypeError, ValueError):
        return None
    return (input_tokens - cached) * input_rate + cached * cached_rate + output_tokens * output_rate


def _planned_orders(pairs_count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    direct_first = [["direct", "openrouter"] for _ in range(pairs_count // 2)]
    route_first = [["openrouter", "direct"] for _ in range(pairs_count // 2)]
    orders = direct_first + route_first
    if pairs_count % 2:
        extra = ["direct", "openrouter"]
        rng.shuffle(extra)
        orders.append(extra)
    rng.shuffle(orders)
    return orders


def evaluate(
    pairs: list[dict[str, Any]],
    *,
    min_route_tps_ratio: float,
    max_route_ttft_ratio: float,
    required_pairs: int = DEFAULT_REQUIRED_PAIRS,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    max_route_error_delta: float = DEFAULT_MAX_ROUTE_ERROR_DELTA,
    max_route_cost_ratio: float = DEFAULT_MAX_ROUTE_COST_RATIO,
    pricing: dict[str, dict[str, Any]] | None = None,
    bootstrap_seed: int = 0,
    expected_route_provider_slug: str | None = None,
) -> dict[str, Any]:
    paired_tps: list[float] = []
    paired_ttft: list[float] = []
    paired_cost: list[float] = []
    direct_ttft_measured = 0
    direct_failures = 0
    route_failures = 0
    route_metadata_verified = 0
    successful_pairs = sum(
        1
        for pair in pairs
        if pair["attempts"].get("direct", {}).get("status") == "success"
        and pair["attempts"].get("openrouter", {}).get("status") == "success"
    )
    for pair in pairs:
        direct_attempt = pair["attempts"].get("direct", {})
        route_attempt = pair["attempts"].get("openrouter", {})
        if direct_attempt.get("status") != "success":
            direct_failures += 1
        if route_attempt.get("status") != "success":
            route_failures += 1
        if route_attempt.get("status") != "success" or direct_attempt.get("status") != "success":
            continue
        direct_metrics = direct_attempt.get("metrics", {})
        route_metrics = route_attempt.get("metrics", {})
        # Each evidence stream accumulates independently: a lane that does not
        # measure one metric must not erase the pair's other evidence.
        try:
            direct_tps = float(direct_metrics["tokens_per_second"])
            route_tps = float(route_metrics["tokens_per_second"])
            if direct_tps > 0 and route_tps > 0:
                paired_tps.append(route_tps / direct_tps)
        except (KeyError, TypeError, ValueError):
            pass
        direct_ttft_raw = direct_metrics.get("time_to_first_token")
        if direct_ttft_raw is not None:
            direct_ttft_measured += 1
        try:
            direct_ttft = float(direct_ttft_raw)
            route_ttft = float(route_metrics["time_to_first_token"])
            if direct_ttft > 0 and route_ttft > 0:
                paired_ttft.append(route_ttft / direct_ttft)
        except (KeyError, TypeError, ValueError):
            pass
        if (
            route_metrics.get("provider_metadata_verified") is True
            and route_metrics.get("observed_provider_slug")
            and (
                expected_route_provider_slug is None
                or route_metrics.get("observed_provider_slug") == expected_route_provider_slug
            )
        ):
            route_metadata_verified += 1
        if pricing:
            direct_cost = _price_attempt(direct_attempt, pricing.get("direct", {}))
            route_cost = _price_attempt(route_attempt, pricing.get("openrouter", {}))
            if direct_cost is not None and route_cost is not None and direct_cost > 0:
                paired_cost.append(route_cost / direct_cost)

    route_tps_ratio = median(paired_tps) if paired_tps else None
    route_ttft_ratio = median(paired_ttft) if paired_ttft else None
    route_cost_ratio = median(paired_cost) if paired_cost else None
    required_successful_pairs = max(1, ceil(required_pairs * min_success_rate))
    output_valid = successful_pairs >= required_successful_pairs
    tps_ci95 = _bootstrap_ci(paired_tps, seed=bootstrap_seed + 1)
    ttft_ci95 = _bootstrap_ci(paired_ttft, seed=bootstrap_seed + 2)
    cost_ci95 = _bootstrap_ci(paired_cost, seed=bootstrap_seed + 3)
    route_error_rate = route_failures / len(pairs) if pairs else 1.0
    direct_error_rate = direct_failures / len(pairs) if pairs else 1.0
    route_error_delta = route_error_rate - direct_error_rate
    metadata_valid = route_metadata_verified == successful_pairs and successful_pairs > 0
    # The TTFT bound is enforceable only when the direct lane measures TTFT.
    # A provider client with no TTFT capture (non-streaming direct call) makes
    # the bound structurally unmeasurable, and the published direct data for
    # that provider carries no TTFT either, so routing cannot corrupt it.
    ttft_waived_direct_unmeasured = successful_pairs > 0 and direct_ttft_measured == 0
    ttft_gate_valid = ttft_waived_direct_unmeasured or (ttft_ci95 is not None and ttft_ci95[1] <= max_route_ttft_ratio)
    performance_valid = tps_ci95 is not None and tps_ci95[0] >= min_route_tps_ratio and ttft_gate_valid
    error_valid = route_error_delta <= max_route_error_delta
    cost_status = "verified" if pricing and len(paired_cost) == successful_pairs else "unverified"
    cost_valid = cost_status == "verified" and cost_ci95 is not None and cost_ci95[1] <= max_route_cost_ratio
    promotion_valid = output_valid and performance_valid and metadata_valid and error_valid and cost_valid
    return {
        "successful_pairs": successful_pairs,
        "required_pairs": len(pairs),
        "required_successful_pairs": required_successful_pairs,
        "output_valid": output_valid,
        "route_tps_ratio": route_tps_ratio,
        "route_ttft_ratio": route_ttft_ratio,
        "route_cost_ratio": route_cost_ratio,
        "route_tps_ratio_ci95": tps_ci95,
        "route_ttft_ratio_ci95": ttft_ci95,
        "route_cost_ratio_ci95": cost_ci95,
        "min_route_tps_ratio": min_route_tps_ratio,
        "max_route_ttft_ratio": max_route_ttft_ratio,
        "max_route_cost_ratio": max_route_cost_ratio,
        "direct_error_rate": direct_error_rate,
        "route_error_rate": route_error_rate,
        "route_error_delta": route_error_delta,
        "max_route_error_delta": max_route_error_delta,
        "route_metadata_verified": route_metadata_verified,
        "direct_ttft_measured_pairs": direct_ttft_measured,
        "ttft_waived_direct_unmeasured": ttft_waived_direct_unmeasured,
        "metadata_valid": metadata_valid,
        "performance_valid": performance_valid,
        "error_valid": error_valid,
        "cost_status": cost_status,
        "cost_valid": cost_valid,
        "promotion_valid": promotion_valid,
        "canary_state": (
            "passed"
            if promotion_valid
            else (
                "measurement_passed_cost_unverified"
                if output_valid and performance_valid and cost_status == "unverified"
                else "failed"
            )
        ),
    }


def run_canary(
    report: dict[str, Any],
    *,
    provider: str,
    model_id: str,
    pairs_count: int,
    max_tokens: int,
    deadline_seconds: int,
    seed: int,
    min_route_tps_ratio: float,
    max_route_ttft_ratio: float,
    required_pairs: int = DEFAULT_REQUIRED_PAIRS,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    max_route_error_delta: float = DEFAULT_MAX_ROUTE_ERROR_DELTA,
    max_route_cost_ratio: float = DEFAULT_MAX_ROUTE_COST_RATIO,
    pricing: dict[str, dict[str, Any]] | None = None,
    expected_route_provider_slug: str | None = None,
) -> dict[str, Any]:
    if pairs_count < 1:
        raise ValueError("pairs_count must be at least 1")
    if required_pairs < 1 or pairs_count < required_pairs:
        raise ValueError("pairs_count must be at least required_pairs")
    if pairs_count > 30 or required_pairs > 30:
        raise ValueError("promotion canaries are capped at 30 paired requests")
    if pricing is None:
        raise ValueError("paired canary requires direct and OpenRouter pricing evidence")
    candidate = _candidate(report, provider, model_id)
    canary_id = f"canary:{provider}:{model_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    route_decision = _active_canary_decision(candidate, canary_id)
    direct_decision = RouteDecision.direct(provider, model_id, reason="paired-canary-direct")
    pairs: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + deadline_seconds
    for pair_index, order in enumerate(_planned_orders(pairs_count, seed), start=1):
        attempts: dict[str, Any] = {}
        for transport in order:
            if time.monotonic() >= deadline:
                attempts[transport] = {"status": "error", "stage": "deadline", "error": "canary deadline exceeded"}
                continue
            decision = direct_decision if transport == "direct" else route_decision
            attempts[transport] = _attempt(
                decision,
                max_tokens=max_tokens,
                deadline_seconds=deadline_seconds,
            )
        pairs.append({"pair_index": pair_index, "order": order, "attempts": attempts})

    return {
        "schema_version": 1,
        "mode": "report-only-paired-canary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canary_id": canary_id,
        "source_provider": provider,
        "source_model_id": model_id,
        "route_model_id": candidate["route_model_id"],
        "route_provider_slug": candidate["route_provider_slug"],
        "benchmark_profile_id": "cloud-default-v1",
        "profile_hash": stable_hash(
            {
                "profile_id": "cloud-default-v1",
                "query": QUERY_TEXT,
                "max_tokens": max_tokens,
                "temperature": runner.TEMPERATURE,
                "protocol_version": runner.PROTOCOL_VERSION,
            }
        ),
        "max_tokens": max_tokens,
        "pairs_requested": pairs_count,
        "deadline_seconds": deadline_seconds,
        "seed": seed,
        "pricing": pricing,
        "pairs": pairs,
        "evaluation": evaluate(
            pairs,
            min_route_tps_ratio=min_route_tps_ratio,
            max_route_ttft_ratio=max_route_ttft_ratio,
            required_pairs=required_pairs,
            min_success_rate=min_success_rate,
            max_route_error_delta=max_route_error_delta,
            max_route_cost_ratio=max_route_cost_ratio,
            pricing=pricing,
            bootstrap_seed=seed,
            expected_route_provider_slug=expected_route_provider_slug or str(candidate["route_provider_slug"]),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--required-pairs", type=int, default=DEFAULT_REQUIRED_PAIRS)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--pricing-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--deadline-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-route-tps-ratio", type=float, default=DEFAULT_MIN_ROUTE_TPS_RATIO)
    parser.add_argument("--max-route-ttft-ratio", type=float, default=DEFAULT_MAX_ROUTE_TTFT_RATIO)
    parser.add_argument("--max-route-error-delta", type=float, default=DEFAULT_MAX_ROUTE_ERROR_DELTA)
    parser.add_argument("--max-route-cost-ratio", type=float, default=DEFAULT_MAX_ROUTE_COST_RATIO)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--batch-max-cost-usd", type=float, default=DEFAULT_BATCH_MAX_USD)
    parser.add_argument("--daily-max-cost-usd", type=float, default=DEFAULT_DAILY_MAX_USD)
    parser.add_argument("--daily-ledger-json", type=Path)
    args = parser.parse_args()
    pricing = load_json(args.pricing_json) if args.pricing_json else None
    if not isinstance(pricing, dict):
        raise ValueError("--pricing-json is required for bounded promotion canaries")
    if args.max_cost_usd <= 0:
        raise ValueError("--max-cost-usd must be positive")
    if not args.daily_ledger_json:
        raise ValueError("live canaries require --daily-ledger-json shared budget ledger")
    if args.max_cost_usd > args.batch_max_cost_usd:
        raise ValueError("--max-cost-usd cannot exceed --batch-max-cost-usd")
    # A deliberately conservative upper bound. The actual usage cost is
    # recorded per attempt and the evaluation still requires complete pricing.
    max_rate = max(
        float(pricing.get("direct", {}).get("input_per_token", 0) or 0),
        float(pricing.get("direct", {}).get("output_per_token", 0) or 0),
        float(pricing.get("openrouter", {}).get("input_per_token", 0) or 0),
        float(pricing.get("openrouter", {}).get("output_per_token", 0) or 0),
    )
    estimated_input_tokens = max(1, ceil(len(QUERY_TEXT) / 4))
    estimated_cost = args.pairs * 2 * (estimated_input_tokens + args.max_tokens) * max_rate
    if estimated_cost > args.max_cost_usd:
        raise ValueError(f"estimated canary cost ${estimated_cost:.4f} exceeds cap ${args.max_cost_usd:.4f}")
    ledger = reserve_daily_budget(
        args.daily_ledger_json,
        amount_usd=min(args.max_cost_usd, estimated_cost),
        batch_max_usd=args.batch_max_cost_usd,
        daily_max_usd=args.daily_max_cost_usd,
        operation=f"paired-canary:{args.provider}/{args.model_id}",
    )
    result = run_canary(
        load_json(args.decisions_json),
        provider=args.provider,
        model_id=args.model_id,
        pairs_count=args.pairs,
        max_tokens=args.max_tokens,
        deadline_seconds=args.deadline_seconds,
        seed=args.seed,
        min_route_tps_ratio=args.min_route_tps_ratio,
        max_route_ttft_ratio=args.max_route_ttft_ratio,
        required_pairs=args.required_pairs,
        min_success_rate=args.min_success_rate,
        max_route_error_delta=args.max_route_error_delta,
        max_route_cost_ratio=args.max_route_cost_ratio,
        pricing=pricing,
    )
    result["budget"] = {
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_cost_usd": estimated_cost,
        "max_cost_usd": args.max_cost_usd,
        "batch_max_cost_usd": args.batch_max_cost_usd,
        "daily_max_cost_usd": args.daily_max_cost_usd,
        "daily_ledger": str(args.daily_ledger_json),
        "daily_reserved_usd": ledger["reserved_usd"],
    }
    write_json(args.output, result)
    print(json.dumps(result["evaluation"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
