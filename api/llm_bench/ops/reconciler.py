"""Retire models the providers have actually stopped serving.

The dangerous half of keeping a catalogue in sync. Adding a model that turns out
to be wrong costs a chart line; removing one that is still live loses a series
the site has been building for months, and the loss is quiet.

So absence is only counted across *complete* discovery runs. A run that errored,
stopped short of pagination, or never happened proves nothing about what the
provider offers, and the ledger exists precisely so those can be told apart —
before it, a filter change and a provider deletion looked identical.

Three consecutive complete runs, not three calendar days. Polling daily with
jitter means "three days" can be two observations, and a deprecation resting on
two observations is a coin flip on one bad night.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.ops import invariants
from llm_bench.ops import mutations
from llm_bench.scheduler.mongo import models_collection_name

# Consecutive complete runs a model must be missing from before it is retired.
REQUIRED_ABSENT_RUNS = int(os.getenv("BENCHMARK_ABSENT_RUNS_TO_DEPRECATE", "3"))

# Providers with no discovery authority. Absence from a catalogue that is never
# read is not evidence, so their models are never retired this way.
UNCOVERED_PROVIDERS = frozenset({"bedrock", "vertex"})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass
class Retirement:
    provider: str
    model_id: str
    reason: str

    @property
    def subject(self) -> str:
        return f"{self.provider}/{self.model_id}"


def complete_run_times(db: Database, *, provider: str, limit: int = REQUIRED_ABSENT_RUNS) -> list[datetime]:
    """When this provider's last N fully-successful catalogue reads finished."""
    runs = db[invariants.discovery_runs_collection_name()].find(
        {"provider": provider, "status": "completed", "pagination_complete": True},
        {"finished_at": 1},
        sort=[("finished_at", -1)],
        limit=limit,
    )
    return [stamp for stamp in (_as_utc(r.get("finished_at")) for r in runs) if stamp is not None]


def find_retirements(db: Database, *, now: datetime | None = None) -> list[Retirement]:
    """Enabled models absent from every one of the last N complete runs."""
    now = now or utcnow()
    seen: dict[tuple[str, str], datetime] = {}
    for row in db.provider_catalog.find({}, {"provider": 1, "model_id": 1, "last_seen_at": 1}):
        stamp = _as_utc(row.get("last_seen_at"))
        if stamp is not None:
            seen[(row["provider"], row["model_id"])] = stamp

    cutoffs: dict[str, datetime | None] = {}
    retirements: list[Retirement] = []

    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1},
    ):
        provider, model_id = doc["provider"], doc["model_id"]
        if provider in UNCOVERED_PROVIDERS:
            continue

        if provider not in cutoffs:
            runs = complete_run_times(db, provider=provider)
            # Not enough complete history to judge absence yet.
            cutoffs[provider] = runs[-1] if len(runs) >= REQUIRED_ABSENT_RUNS else None
        cutoff = cutoffs[provider]
        if cutoff is None:
            continue

        last_seen = seen.get((provider, model_id))
        if last_seen is None or last_seen < cutoff:
            retirements.append(
                Retirement(
                    provider=provider,
                    model_id=model_id,
                    reason=(
                        f"absent from the last {REQUIRED_ABSENT_RUNS} complete discovery runs "
                        f"(last seen {last_seen.isoformat() if last_seen else 'never'})"
                    ),
                )
            )
    return retirements


def retire(db: Database, *, now: datetime | None = None, dry_run: bool = True) -> list[Retirement]:
    """Deprecate models the provider has stopped listing.

    Applies as one bounded batch, so a discovery regression that empties a
    provider's catalogue hits the cap and retires nothing. That is the failure
    this guards against: the run looks successful, and every model vanishes.
    """
    now = now or utcnow()
    retirements = find_retirements(db, now=now)
    if dry_run or not retirements:
        return retirements

    batch = mutations.MutationBatch(db=db, reason="absent from provider catalogue", actor="reconciler")
    for item in retirements:
        batch.set_model_fields(
            provider=item.provider,
            model_id=item.model_id,
            enabled=False,
            deprecated=True,
            disabled_class="provider_retired",
            disabled_reason=item.reason,
            disabled_at=now,
        )
    batch.apply(now=now)
    return retirements


def summarize(db: Database, *, now: datetime | None = None) -> dict[str, Any]:
    """What the reconciler would do, without doing any of it."""
    now = now or utcnow()
    retirements = find_retirements(db, now=now)
    by_provider: dict[str, int] = {}
    for item in retirements:
        by_provider[item.provider] = by_provider.get(item.provider, 0) + 1
    return {
        "retirement_count": len(retirements),
        "by_provider": by_provider,
        "subjects": [item.subject for item in retirements[:50]],
    }
