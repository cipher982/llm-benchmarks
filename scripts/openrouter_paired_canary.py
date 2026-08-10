#!/usr/bin/env python3
"""Run a bounded paired direct-versus-OpenRouter measurement canary.

This command uses the benchmark runner's production adapters and validation
policy, but never writes published metrics or route decisions. It emits a JSON
evidence artifact. Cost remains an explicit gate because the current direct
provider metric contract does not expose a comparable cost field.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from datetime import timezone
from pathlib import Path
from statistics import median
from typing import Any

from llm_bench.scheduler import runner
from llm_bench.scheduler.routing import RouteDecision

QUERY_TEXT = "Tell a long and happy story about the history of the world."
DEFAULT_MAX_TOKENS = 64
DEFAULT_PAIRS = 2
DEFAULT_MIN_ROUTE_TPS_RATIO = 0.5
DEFAULT_MAX_ROUTE_TTFT_RATIO = 3.0

METRIC_FIELDS = (
    "output_tokens",
    "visible_output_tokens",
    "reasoning_tokens",
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
        }
    )
    decision = RouteDecision.from_snapshot(
        str(candidate["source_provider"]),
        str(candidate["source_model_id"]),
        snapshot,
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
    run_ts = datetime.now(timezone.utc).isoformat()
    try:
        _, metrics = runner._generate_and_validate(
            decision,
            run_ts=run_ts,
            run_config={"query": QUERY_TEXT, "max_tokens": max_tokens},
            max_tokens=max_tokens,
            timeout_seconds=min(45.0, max(5.0, deadline_seconds / 3.0))
            if decision.transport_provider == "openrouter"
            else None,
        )
        return {"status": "success", "metrics": _safe_metrics(metrics)}
    except runner.AttemptFailure as exc:
        return {"status": "error", "stage": exc.stage, "error": exc.message}
    except Exception as exc:  # noqa: BLE001 - preserve per-attempt evidence
        return {"status": "error", "stage": "generate", "error": f"{type(exc).__name__}: {exc}"}


def _ratio(numerator: list[float], denominator: list[float]) -> float | None:
    if not numerator or not denominator:
        return None
    base = median(denominator)
    return median(numerator) / base if base > 0 else None


def _planned_orders(pairs_count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for _ in range(pairs_count):
        order = ["direct", "openrouter"]
        rng.shuffle(order)
        orders.append(order)
    if pairs_count > 1 and len({tuple(order) for order in orders}) == 1:
        orders[-1] = list(reversed(orders[-1]))
    return orders


def evaluate(
    pairs: list[dict[str, Any]],
    *,
    min_route_tps_ratio: float,
    max_route_ttft_ratio: float,
) -> dict[str, Any]:
    direct = [
        pair["attempts"]["direct"] for pair in pairs if pair["attempts"].get("direct", {}).get("status") == "success"
    ]
    routed = [
        pair["attempts"]["openrouter"]
        for pair in pairs
        if pair["attempts"].get("openrouter", {}).get("status") == "success"
    ]
    direct_tps = [float(item["metrics"]["tokens_per_second"]) for item in direct]
    route_tps = [float(item["metrics"]["tokens_per_second"]) for item in routed]
    direct_ttft = [
        float(item["metrics"]["time_to_first_token"])
        for item in direct
        if item["metrics"].get("time_to_first_token") is not None
    ]
    route_ttft = [
        float(item["metrics"]["time_to_first_token"])
        for item in routed
        if item["metrics"].get("time_to_first_token") is not None
    ]
    successful_pairs = sum(
        1
        for pair in pairs
        if pair["attempts"].get("direct", {}).get("status") == "success"
        and pair["attempts"].get("openrouter", {}).get("status") == "success"
    )
    route_tps_ratio = _ratio(route_tps, direct_tps)
    route_ttft_ratio = _ratio(route_ttft, direct_ttft)
    output_valid = successful_pairs == len(pairs)
    performance_valid = (
        route_tps_ratio is not None
        and route_tps_ratio >= min_route_tps_ratio
        and (route_ttft_ratio is None or route_ttft_ratio <= max_route_ttft_ratio)
    )
    return {
        "successful_pairs": successful_pairs,
        "required_pairs": len(pairs),
        "output_valid": output_valid,
        "route_tps_ratio": route_tps_ratio,
        "route_ttft_ratio": route_ttft_ratio,
        "min_route_tps_ratio": min_route_tps_ratio,
        "max_route_ttft_ratio": max_route_ttft_ratio,
        "performance_valid": performance_valid,
        "cost_status": "unverified",
        "canary_state": "measurement_passed_cost_unverified" if output_valid and performance_valid else "failed",
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
) -> dict[str, Any]:
    if pairs_count < 1:
        raise ValueError("pairs_count must be at least 1")
    candidate = _candidate(report, provider, model_id)
    canary_id = f"canary:{provider}:{model_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    route_decision = _active_canary_decision(candidate, canary_id)
    direct_decision = RouteDecision.direct(provider, model_id, reason="paired-canary-direct")
    pairs: list[dict[str, Any]] = []
    for pair_index, order in enumerate(_planned_orders(pairs_count, seed), start=1):
        attempts: dict[str, Any] = {}
        for transport in order:
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
        "max_tokens": max_tokens,
        "pairs": pairs,
        "evaluation": evaluate(
            pairs,
            min_route_tps_ratio=min_route_tps_ratio,
            max_route_ttft_ratio=max_route_ttft_ratio,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--deadline-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-route-tps-ratio", type=float, default=DEFAULT_MIN_ROUTE_TPS_RATIO)
    parser.add_argument("--max-route-ttft-ratio", type=float, default=DEFAULT_MAX_ROUTE_TTFT_RATIO)
    args = parser.parse_args()
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
    )
    write_json(args.output, result)
    print(json.dumps(result["evaluation"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
