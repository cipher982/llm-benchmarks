"""Immutable snapshots of what the system intended to measure.

Phase 0's first design evaluated every invariant against live catalogue state,
which let remediation satisfy the check it was remediating: disable a stale
model and "every enabled model is fresh" passes, disable a provider and coverage
ratios improve. The state is internally consistent and says nothing about
whether the site is correct.

Fixing that needs the denominator to be a record the checks cannot edit. A
snapshot is captured on a schedule, never modified, and evaluation reads the
newest snapshot that is already older than the window being judged. An action
taken now therefore cannot change the verdict on the window it was taken in —
it can only change a later one, where the difference between two snapshots is
itself the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler.mongo import collection_name
from llm_bench.scheduler.mongo import models_collection_name

# A snapshot must be settled before it can be judged against, otherwise an
# action and the evidence for it can be written in the same window.
DEFAULT_MIN_SNAPSHOT_AGE = timedelta(hours=1)


def desired_set_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_DESIRED_SET", "bench_desired_set")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DesiredSet:
    """What the catalogue said should be measured, at one instant."""

    captured_at: datetime
    models: tuple[tuple[str, str], ...]
    providers: tuple[str, ...]

    @property
    def model_count(self) -> int:
        return len(self.models)

    def models_for(self, provider: str) -> tuple[str, ...]:
        return tuple(model_id for p, model_id in self.models if p == provider)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def ensure_indexes(db: Database) -> None:
    """Indexes the desired set and the catalogue it snapshots both depend on."""
    db[desired_set_collection_name()].create_index([("captured_at", -1)])

    # The plain (provider, model_id) unique index is case-sensitive, so
    # Qwen/Qwen2.5-7B-Instruct-Turbo and qwen/qwen2.5-7b-instruct-turbo are
    # separate documents that can both be enabled — which benchmarks the model
    # twice and draws it twice on the site. Five pairs were live on 2026-08-04.
    #
    # Scoped to enabled rows rather than the whole collection: 28 duplicate
    # groups exist among disabled documents, and those are worth keeping. They
    # are small, they carry display names for historical metric rows, and old
    # URLs still resolve through them. The rule that matters is that no model
    # is *enabled* under two spellings.
    db[models_collection_name()].create_index(
        [("provider", 1), ("model_id", 1)],
        name="uniq_enabled_provider_model_ci",
        unique=True,
        collation={"locale": "en", "strength": 2},
        partialFilterExpression={"enabled": True},
    )


def capture(db: Database, *, now: datetime | None = None) -> DesiredSet:
    """Write one immutable snapshot of the current intent. Never updates."""
    now = now or utcnow()
    models = sorted(
        (doc["provider"], doc["model_id"])
        for doc in db[models_collection_name()].find(
            {"enabled": True, "deprecated": {"$ne": True}},
            {"provider": 1, "model_id": 1},
        )
        if doc.get("provider") and doc.get("model_id")
    )
    providers = sorted({provider for provider, _ in models})
    db[desired_set_collection_name()].insert_one(
        {
            "captured_at": now,
            "models": [{"provider": p, "model_id": m} for p, m in models],
            "providers": providers,
            "model_count": len(models),
        }
    )
    return DesiredSet(captured_at=now, models=tuple(models), providers=tuple(providers))


def _hydrate(doc: dict[str, Any]) -> DesiredSet:
    models = tuple(
        (entry["provider"], entry["model_id"])
        for entry in doc.get("models", [])
        if entry.get("provider") and entry.get("model_id")
    )
    return DesiredSet(
        captured_at=_as_utc(doc.get("captured_at")) or utcnow(),
        models=models,
        providers=tuple(doc.get("providers") or sorted({p for p, _ in models})),
    )


def for_evaluation(
    db: Database,
    *,
    now: datetime | None = None,
    min_age: timedelta = DEFAULT_MIN_SNAPSHOT_AGE,
) -> DesiredSet | None:
    """The newest snapshot old enough to judge the present against.

    Returns None when no settled snapshot exists. Callers must treat that as an
    inability to evaluate, not as a pass — an unknown denominator is exactly the
    condition this module exists to make visible.
    """
    now = now or utcnow()
    doc = db[desired_set_collection_name()].find_one(
        {"captured_at": {"$lte": now - min_age}},
        sort=[("captured_at", -1)],
    )
    return _hydrate(doc) if doc else None


def drift(db: Database, *, since: DesiredSet, now: datetime | None = None) -> dict[str, list[str]]:
    """What changed between a snapshot and live intent.

    A large removal between two snapshots is the signature of remediation
    shrinking its own denominator, so it is reported rather than inferred.
    """
    now = now or utcnow()
    current = capture_view(db)
    was, is_now = set(since.models), set(current)
    return {
        "removed": sorted(f"{p}/{m}" for p, m in was - is_now),
        "added": sorted(f"{p}/{m}" for p, m in is_now - was),
    }


def capture_view(db: Database) -> tuple[tuple[str, str], ...]:
    """Live intent without writing a snapshot."""
    return tuple(
        sorted(
            (doc["provider"], doc["model_id"])
            for doc in db[models_collection_name()].find(
                {"enabled": True, "deprecated": {"$ne": True}},
                {"provider": 1, "model_id": 1},
            )
            if doc.get("provider") and doc.get("model_id")
        )
    )
