"""Core-set concentration: enough samples on the endpoints that matter to publish.

The rotation tiers spread ~280 jobs a day across ~800 endpoints, which gives a
hot model's endpoint about four samples a week and a long-tail endpoint one.
Publication (dashboard `endpointPublication.ts`) wants eight deduplicated
samples across four UTC blocks for a preliminary figure and thirty across all
six for an official one, so under the flat rotation no endpoint could ever
publish — measured 2026-09-02: 762 endpoints, 0 preliminary, 0 official.

This module picks a small core — the best-served endpoints of the most widely
served models, plus anything new on OpenRouter — and samples each of them
every 5.5 hours (4.36/day: the official gate inside a week, and a 5.5h step
walks all six 4h blocks). The rest of the catalogue is slowed by exactly the
factor that keeps total jobs per day at or below what the rotation alone
would create, so concentration is a redistribution, never a spend increase.

The selection is recomputed once a day from `bench_endpoints` and the
OpenRouter catalogue and written to `provider_state` with a reason per
member, the arithmetic, and a timestamp. Nothing here is a hand-written list:
a later agent can read why each endpoint is in, and `BENCHMARK_CORE_SET=0`
restores the flat rotation.
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import models_collection_name

STATE_ID = "openrouter:core_set"
PROVIDER_STATE_COLLECTION = "provider_state"
CATALOG_COLLECTION = "openrouter_catalog"
CORE_SET_POLICY_VERSION = 1

# 5.5h: 4.36 samples/day per core endpoint, so thirty deduplicated samples
# arrive inside the seven-day publication window, and successive samples land
# 1.5 blocks apart so six cycles visit every 4h UTC block.
DEFAULT_CORE_INTERVAL_SECONDS = int(5.5 * 60 * 60)
RECOMPUTE_AFTER = timedelta(hours=24)

# The most widely served models, by provider count (the same popularity proxy
# the tiers use). The first ten contribute two endpoints, the next ten one.
TOP_MODELS = 20
DEEP_MODELS = 10
RECENT_DAYS = 14

# The stretch cannot be less than one (that would speed the tail up) and is
# capped so a pathological population cannot park the tail for months.
MAX_TAIL_STRETCH = 50.0

DAY_SECONDS = 86400.0


def enabled() -> bool:
    return os.getenv("BENCHMARK_CORE_SET", "1").strip().lower() not in {"0", "false", "no"}


def core_interval_seconds() -> int:
    try:
        return max(60, int(os.getenv("BENCHMARK_CORE_SET_INTERVAL_SECONDS", str(DEFAULT_CORE_INTERVAL_SECONDS))))
    except ValueError:
        return DEFAULT_CORE_INTERVAL_SECONDS


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _served_rank(row: dict[str, Any]) -> tuple[float, float, str]:
    """Best-served first: OpenRouter's own uptime, then its throughput, then a stable name."""
    uptime = row.get("or_uptime_1d")
    throughput = row.get("or_throughput_p50")
    return (
        -(float(uptime) if isinstance(uptime, (int, float)) else -1.0),
        -(float(throughput) if isinstance(throughput, (int, float)) else -1.0),
        str(row.get("endpoint_tag") or ""),
    )


def endpoint_rows_by_model(db: Database, *, provider: str) -> dict[str, list[dict[str, Any]]]:
    """Enabled endpoints of enabled, non-deprecated models, grouped by model."""
    enabled_models = set(
        db[models_collection_name()].distinct(
            "model_id",
            {"provider": provider, "enabled": True, "deprecated": {"$ne": True}},
        )
    )
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in db[endpoint_discovery.endpoints_collection_name()].find(
        {"enabled": True, "model_id": {"$in": sorted(enabled_models)}},
        {
            "model_id": 1,
            "endpoint_tag": 1,
            "provider_canonical": 1,
            "or_uptime_1d": 1,
            "or_throughput_p50": 1,
        },
    ):
        if row.get("model_id") and row.get("endpoint_tag"):
            rows_by_model.setdefault(row["model_id"], []).append(row)
    return rows_by_model


def provider_count(rows: list[dict[str, Any]]) -> int:
    return len(
        {row.get("provider_canonical") or endpoint_discovery.provider_canonical(row["endpoint_tag"]) for row in rows}
    )


def _recent_models(db: Database, *, now: datetime) -> dict[str, str]:
    """Model ids new on OpenRouter inside the recency window, with why."""
    cutoff = now - timedelta(days=RECENT_DAYS)
    out: dict[str, str] = {}
    for doc in db[CATALOG_COLLECTION].find(
        {"$or": [{"created": {"$gte": cutoff}}, {"first_seen_at": {"$gte": cutoff}}]},
        {"openrouter_id": 1, "matched_model_id": 1, "created": 1, "first_seen_at": 1},
    ):
        created = _as_utc(doc.get("created"))
        first_seen = _as_utc(doc.get("first_seen_at"))
        if created is not None and created >= cutoff:
            why = f"new on OpenRouter: created {created.date().isoformat()}"
        elif first_seen is not None:
            why = f"new to the catalogue: first seen {first_seen.date().isoformat()}"
        else:
            continue
        for key in ("openrouter_id", "matched_model_id"):
            model_id = doc.get(key)
            if model_id and model_id not in out:
                out[model_id] = why
    return out


def select(db: Database, *, provider: str = "openrouter", now: datetime | None = None) -> list[dict[str, Any]]:
    """The core members, each carrying the reason it was chosen.

    Every other enqueuer skips `policies.excluded_providers()`; so does this.
    The check is on the lane, not on an endpoint's upstream: `groq` as an
    OpenRouter endpoint tag is routed work this host runs a worker for, and it
    is exactly the kind of endpoint the site wants sampled well.
    """
    now = now or datetime.now(timezone.utc)
    if provider in policies.excluded_providers():
        return []
    rows_by_model = endpoint_rows_by_model(db, provider=provider)
    if not rows_by_model:
        return []

    ranked = sorted(
        ((model_id, rows, provider_count(rows)) for model_id, rows in rows_by_model.items()),
        key=lambda item: (-item[2], item[0]),
    )

    members: list[dict[str, Any]] = []
    chosen: set[tuple[str, str]] = set()

    def add(model_id: str, row: dict[str, Any], reason: str, count: int) -> None:
        key = (model_id, row["endpoint_tag"])
        if key in chosen:
            return
        chosen.add(key)
        members.append(
            {
                "model_id": model_id,
                "endpoint_tag": row["endpoint_tag"],
                "provider_canonical": row.get("provider_canonical")
                or endpoint_discovery.provider_canonical(row["endpoint_tag"]),
                "reason": reason,
                "provider_count": count,
                "or_uptime_1d": row.get("or_uptime_1d"),
                "or_throughput_p50": row.get("or_throughput_p50"),
            }
        )

    for rank, (model_id, rows, count) in enumerate(ranked[:TOP_MODELS], start=1):
        take = 2 if rank <= DEEP_MODELS else 1
        for row in sorted(rows, key=_served_rank)[:take]:
            add(
                model_id,
                row,
                f"rank {rank} model by provider count ({count} providers); "
                "best-served endpoint by OpenRouter uptime then throughput",
                count,
            )

    for model_id, why in sorted(_recent_models(db, now=now).items()):
        rows = rows_by_model.get(model_id)
        if not rows:
            continue
        best = sorted(rows, key=_served_rank)[0]
        add(model_id, best, why + "; best-served endpoint", provider_count(rows))

    return members


def budget(
    rows_by_model: dict[str, list[dict[str, Any]]],
    members: list[dict[str, Any]],
    *,
    interval_seconds: int,
) -> dict[str, Any]:
    """Jobs-per-day arithmetic that makes concentration spend-neutral.

    baseline   what the flat rotation would create: one opportunity per model
               per tier interval, every model.
    core       len(members) * 86400 / interval.
    tail       the rotation over models that still have a non-core endpoint.
    stretch    tail / (baseline - core), floored at 1, so
               core + tail / stretch <= baseline.
    """
    core_keys = {(m["model_id"], m["endpoint_tag"]) for m in members}
    baseline = 0.0
    tail = 0.0
    for model_id, rows in rows_by_model.items():
        per_day = DAY_SECONDS / policies.endpoint_tier_interval_seconds(provider_count(rows))
        baseline += per_day
        if any((model_id, row["endpoint_tag"]) not in core_keys for row in rows):
            tail += per_day
    core = len(members) * DAY_SECONDS / interval_seconds
    headroom = baseline - core
    if tail <= 0:
        stretch = 1.0
    elif headroom <= 0:
        stretch = MAX_TAIL_STRETCH
    else:
        stretch = min(MAX_TAIL_STRETCH, max(1.0, tail / headroom))
    return {
        "baseline_jobs_per_day": round(baseline, 2),
        "core_jobs_per_day": round(core, 2),
        "tail_jobs_per_day_unstretched": round(tail, 2),
        "tail_stretch": round(stretch, 4),
        "projected_jobs_per_day": round(core + tail / stretch, 2),
    }


def compute(db: Database, *, provider: str = "openrouter", now: datetime | None = None) -> dict[str, Any]:
    """Select, cost, and record the core set."""
    now = now or datetime.now(timezone.utc)
    interval = core_interval_seconds()
    members = select(db, provider=provider, now=now)
    arithmetic = budget(endpoint_rows_by_model(db, provider=provider), members, interval_seconds=interval)
    doc: dict[str, Any] = {
        "provider": provider,
        "kind": "core_set",
        "policy_version": CORE_SET_POLICY_VERSION,
        "computed_at": now,
        "core_interval_seconds": interval,
        "member_count": len(members),
        "members": members,
        **arithmetic,
    }
    db[PROVIDER_STATE_COLLECTION].update_one({"_id": STATE_ID}, {"$set": doc}, upsert=True)
    doc["_id"] = STATE_ID
    return doc


def load(db: Database) -> dict[str, Any] | None:
    return db[PROVIDER_STATE_COLLECTION].find_one({"_id": STATE_ID})


def refresh_if_stale(
    db: Database, *, provider: str = "openrouter", now: datetime | None = None
) -> dict[str, Any] | None:
    """The current core set, recomputed once a day. None when the feature is off."""
    if not enabled():
        return None
    now = now or datetime.now(timezone.utc)
    doc = load(db)
    if doc is not None:
        computed_at = _as_utc(doc.get("computed_at"))
        fresh = computed_at is not None and now - computed_at < RECOMPUTE_AFTER
        same_policy = doc.get("policy_version") == CORE_SET_POLICY_VERSION
        same_interval = doc.get("core_interval_seconds") == core_interval_seconds()
        if fresh and same_policy and same_interval:
            return doc
    return compute(db, provider=provider, now=now)


def members_by_target(doc: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not doc:
        return {}
    return {(m["model_id"], m["endpoint_tag"]): m for m in doc.get("members", [])}
