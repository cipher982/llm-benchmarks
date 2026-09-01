#!/usr/bin/env python3
"""Ballpark OpenRouter benchmark costs for uniform and provider-count tier policies.

With ``MONGODB_URI`` set, the command reads the enabled OpenRouter model and
endpoint population directly. For offline comparisons, pass the provider-count
histogram and model target count explicitly::

    uv run python api/llm_bench/ops/tier_cost.py --histogram '1:103,2:37,8:1' --model-targets 141

The rotating tier policy schedules one oldest provider endpoint per model at
each interval. The full-sweep comparison schedules every provider endpoint at
each interval and exposes the linear provider-count cost multiplier.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable

from pymongo import MongoClient

DEFAULT_COST_PER_RUN_USD = 0.00301
DEFAULT_MONTH_DAYS = 365.25 / 12


@dataclass(frozen=True, slots=True)
class Population:
    provider_histogram: dict[int, int]
    model_targets: int

    @property
    def endpoint_models(self) -> int:
        return sum(self.provider_histogram.values())

    @property
    def endpoints(self) -> int:
        return sum(providers * models for providers, models in self.provider_histogram.items())

    @property
    def models_without_endpoints(self) -> int:
        return max(0, self.model_targets - self.endpoint_models)

    @property
    def current_targets(self) -> int:
        """Current scheduler shape: every model target plus every endpoint target."""
        return self.model_targets + self.endpoints

    @property
    def deduplicated_targets(self) -> int:
        """Endpoint targets where available, otherwise one unpinned model target."""
        return self.endpoints + self.models_without_endpoints


@dataclass(frozen=True, slots=True)
class Tier:
    name: str
    minimum_providers: int
    maximum_providers: int | None
    interval_hours: float

    def includes(self, provider_count: int) -> bool:
        return provider_count >= self.minimum_providers and (
            self.maximum_providers is None or provider_count <= self.maximum_providers
        )


@dataclass(frozen=True, slots=True)
class Estimate:
    policy: str
    requests_per_day: float
    cost_per_day_usd: float
    cost_per_month_usd: float
    within_monthly_budget: bool


@dataclass(frozen=True, slots=True)
class TierEstimate:
    tier: str
    provider_range: str
    interval_hours: float
    models: int
    endpoints: int
    rotating_requests_per_day: float
    rotating_endpoint_revisit_hours_min: float
    rotating_endpoint_revisit_hours_max: float


def parse_histogram(value: str) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        providers_text, separator, models_text = item.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(f"histogram item {item!r} must be PROVIDERS:MODELS")
        try:
            providers = int(providers_text)
            models = int(models_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"histogram item {item!r} must contain integers") from exc
        if providers < 1 or models < 0:
            raise argparse.ArgumentTypeError("provider counts must be positive and model counts non-negative")
        histogram[providers] = histogram.get(providers, 0) + models
    if not histogram:
        raise argparse.ArgumentTypeError("histogram must contain at least one provider-count bucket")
    return histogram


def parse_hours(value: str) -> list[float]:
    try:
        hours = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("uniform hours must be comma-separated numbers") from exc
    if not hours or any(hour <= 0 for hour in hours):
        raise argparse.ArgumentTypeError("uniform hours must all be positive")
    return hours


def load_population_from_mongo() -> Population:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set unless --histogram and --model-targets are supplied")
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    models_collection = os.getenv("MONGODB_COLLECTION_MODELS", "models")
    endpoints_collection = os.getenv("MONGODB_COLLECTION_ENDPOINTS", "bench_endpoints")
    client = MongoClient(uri)
    try:
        db = client[db_name]
        model_targets = db[models_collection].count_documents({"provider": "openrouter", "enabled": True})
        rows = db[endpoints_collection].aggregate(
            [
                {"$match": {"enabled": True}},
                {"$group": {"_id": "$model_id", "providers": {"$sum": 1}}},
                {"$group": {"_id": "$providers", "models": {"$sum": 1}}},
            ]
        )
        histogram = {int(row["_id"]): int(row["models"]) for row in rows}
    finally:
        client.close()
    if not histogram:
        raise RuntimeError("no enabled endpoint targets found")
    return Population(provider_histogram=histogram, model_targets=model_targets)


def estimate(
    policy: str,
    requests_per_day: float,
    *,
    cost_per_run_usd: float,
    month_days: float,
    monthly_budget_usd: float,
) -> Estimate:
    daily = requests_per_day * cost_per_run_usd
    monthly = daily * month_days
    return Estimate(
        policy=policy,
        requests_per_day=requests_per_day,
        cost_per_day_usd=daily,
        cost_per_month_usd=monthly,
        within_monthly_budget=monthly <= monthly_budget_usd,
    )


def tier_breakdown(population: Population, tiers: Iterable[Tier]) -> list[TierEstimate]:
    results: list[TierEstimate] = []
    for tier in tiers:
        buckets = [
            (providers, models)
            for providers, models in population.provider_histogram.items()
            if tier.includes(providers)
        ]
        if not buckets:
            continue
        models = sum(models for _, models in buckets)
        endpoints = sum(providers * models for providers, models in buckets)
        provider_counts = [providers for providers, _ in buckets]
        results.append(
            TierEstimate(
                tier=tier.name,
                provider_range=(
                    f"{tier.minimum_providers}+"
                    if tier.maximum_providers is None
                    else f"{tier.minimum_providers}-{tier.maximum_providers}"
                ),
                interval_hours=tier.interval_hours,
                models=models,
                endpoints=endpoints,
                rotating_requests_per_day=models * 24 / tier.interval_hours,
                rotating_endpoint_revisit_hours_min=min(provider_counts) * tier.interval_hours,
                rotating_endpoint_revisit_hours_max=max(provider_counts) * tier.interval_hours,
            )
        )
    return results


def calculate(
    population: Population,
    *,
    tiers: list[Tier],
    no_endpoint_hours: float,
    uniform_hours: list[float],
    cost_per_run_usd: float,
    month_days: float,
    monthly_budget_usd: float,
) -> tuple[list[Estimate], list[TierEstimate]]:
    estimates: list[Estimate] = []
    for hours in uniform_hours:
        label = f"uniform current targets / {format_hours(hours)}"
        estimates.append(
            estimate(
                label,
                population.current_targets * 24 / hours,
                cost_per_run_usd=cost_per_run_usd,
                month_days=month_days,
                monthly_budget_usd=monthly_budget_usd,
            )
        )
        estimates.append(
            estimate(
                f"uniform deduplicated / {format_hours(hours)}",
                population.deduplicated_targets * 24 / hours,
                cost_per_run_usd=cost_per_run_usd,
                month_days=month_days,
                monthly_budget_usd=monthly_budget_usd,
            )
        )

    breakdown = tier_breakdown(population, tiers)
    no_endpoint_requests = population.models_without_endpoints * 24 / no_endpoint_hours
    rotating_requests = sum(row.rotating_requests_per_day for row in breakdown) + no_endpoint_requests
    full_sweep_requests = no_endpoint_requests
    for tier in tiers:
        full_sweep_requests += sum(
            providers * models * 24 / tier.interval_hours
            for providers, models in population.provider_histogram.items()
            if tier.includes(providers)
        )
    estimates.extend(
        [
            estimate(
                "provider tiers / rotate one endpoint",
                rotating_requests,
                cost_per_run_usd=cost_per_run_usd,
                month_days=month_days,
                monthly_budget_usd=monthly_budget_usd,
            ),
            estimate(
                "provider tiers / full endpoint sweeps",
                full_sweep_requests,
                cost_per_run_usd=cost_per_run_usd,
                month_days=month_days,
                monthly_budget_usd=monthly_budget_usd,
            ),
        ]
    )
    return estimates, breakdown


def format_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:g}m"
    if hours >= 24:
        return f"{hours / 24:g}d"
    return f"{hours:g}h"


def print_report(
    population: Population,
    estimates: list[Estimate],
    breakdown: list[TierEstimate],
    *,
    cost_per_run_usd: float,
    monthly_budget_usd: float,
) -> None:
    print(
        f"Population: {population.model_targets} model targets, {population.endpoint_models} endpoint models, "
        f"{population.endpoints} endpoints, {population.models_without_endpoints} models without endpoints"
    )
    print(f"Assumptions: ${cost_per_run_usd:.5f}/request, ${monthly_budget_usd:.2f}/month budget")
    print()
    print(f"{'Policy':48} {'req/day':>10} {'$/day':>9} {'$/month':>10} {'budget':>8}")
    print("-" * 91)
    for row in estimates:
        budget = "yes" if row.within_monthly_budget else "NO"
        print(
            f"{row.policy:48} {row.requests_per_day:10.1f} "
            f"${row.cost_per_day_usd:8.2f} ${row.cost_per_month_usd:9.2f} {budget:>8}"
        )
    print()
    print("Rotating tier detail (one oldest endpoint per model opportunity):")
    print(
        f"{'Tier':10} {'providers':>9} {'models':>7} {'endpoints':>10} "
        f"{'interval':>9} {'req/day':>9} {'endpoint revisit':>18}"
    )
    print("-" * 87)
    for row in breakdown:
        revisit = (
            f"{format_hours(row.rotating_endpoint_revisit_hours_min)}-"
            f"{format_hours(row.rotating_endpoint_revisit_hours_max)}"
        )
        print(
            f"{row.tier:10} {row.provider_range:>9} {row.models:7d} {row.endpoints:10d} "
            f"{format_hours(row.interval_hours):>9} {row.rotating_requests_per_day:9.1f} {revisit:>18}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histogram", type=parse_histogram, help="Offline provider histogram, e.g. '1:103,2:37,8:1'")
    parser.add_argument("--model-targets", type=int, help="Enabled OpenRouter model-level target count")
    parser.add_argument("--cost-per-run", type=float, default=DEFAULT_COST_PER_RUN_USD)
    parser.add_argument("--monthly-budget", type=float, default=30.0)
    parser.add_argument("--month-days", type=float, default=DEFAULT_MONTH_DAYS)
    parser.add_argument("--uniform-hours", type=parse_hours, default=parse_hours("0.5,3,24,48,96"))
    parser.add_argument("--hot-min-providers", type=int, default=8)
    parser.add_argument("--medium-min-providers", type=int, default=3)
    parser.add_argument("--hot-hours", type=float, default=3.0)
    parser.add_argument("--medium-hours", type=float, default=24.0)
    parser.add_argument("--long-hours", type=float, default=96.0)
    parser.add_argument("--no-endpoint-hours", type=float, default=96.0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.histogram is None) != (args.model_targets is None):
        raise SystemExit("--histogram and --model-targets must be supplied together")
    population = (
        Population(provider_histogram=args.histogram, model_targets=args.model_targets)
        if args.histogram is not None
        else load_population_from_mongo()
    )
    if args.medium_min_providers < 2 or args.hot_min_providers <= args.medium_min_providers:
        raise SystemExit("tier boundaries must satisfy 2 <= medium-min-providers < hot-min-providers")
    positive_values = (
        args.cost_per_run,
        args.monthly_budget,
        args.month_days,
        args.hot_hours,
        args.medium_hours,
        args.long_hours,
        args.no_endpoint_hours,
    )
    if any(value <= 0 for value in positive_values):
        raise SystemExit("cost, budget, month length, and intervals must be positive")
    tiers = [
        Tier("hot", args.hot_min_providers, None, args.hot_hours),
        Tier("medium", args.medium_min_providers, args.hot_min_providers - 1, args.medium_hours),
        Tier("long", 1, args.medium_min_providers - 1, args.long_hours),
    ]
    estimates, breakdown = calculate(
        population,
        tiers=tiers,
        no_endpoint_hours=args.no_endpoint_hours,
        uniform_hours=args.uniform_hours,
        cost_per_run_usd=args.cost_per_run,
        month_days=args.month_days,
        monthly_budget_usd=args.monthly_budget,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "population": {
                        "provider_histogram": population.provider_histogram,
                        "model_targets": population.model_targets,
                        "endpoint_models": population.endpoint_models,
                        "endpoints": population.endpoints,
                        "models_without_endpoints": population.models_without_endpoints,
                    },
                    "assumptions": {
                        "cost_per_run_usd": args.cost_per_run,
                        "monthly_budget_usd": args.monthly_budget,
                        "month_days": args.month_days,
                    },
                    "estimates": [asdict(row) for row in estimates],
                    "tier_breakdown": [asdict(row) for row in breakdown],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print_report(
        population,
        estimates,
        breakdown,
        cost_per_run_usd=args.cost_per_run,
        monthly_budget_usd=args.monthly_budget,
    )


if __name__ == "__main__":
    main()
