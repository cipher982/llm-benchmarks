"""Continuous assertions over production state.

Every failure found during the 2026-08-04 recovery was two components that
individually worked, disagreeing — or a component reporting success while doing
nothing. The runner had 29 passing unit tests throughout an eight-day outage.
Isolated tests do not see that class of failure; assertions over live state do.

Each invariant here is derived from a specific failure that actually happened,
and returns the offending records rather than a boolean, so a caller can either
remediate or report with evidence.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name

# Catalogue refresh runs daily; two missed runs is a real signal, one is a blip.
CATALOGUE_MAX_AGE = timedelta(hours=48)
# How far past its cadence a model may fall before it counts as starved.
MODEL_STALENESS_MULTIPLIER = 4
# Trailing window used to establish what "normal" volume looks like.
VOLUME_BASELINE_DAYS = 14
# A provider may drop to this fraction of its trailing median before it is a fault.
VOLUME_FLOOR_RATIO = 0.4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Violation:
    """One offending record, with enough context to act on it."""

    subject: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    name: str
    description: str
    ok: bool
    violations: list[Violation]
    remediable: bool
    checked_at: datetime
    error: str | None = None

    @property
    def summary(self) -> str:
        if self.error:
            return f"{self.name}: ERRORED — {self.error}"
        if self.ok:
            return f"{self.name}: ok"
        return f"{self.name}: {len(self.violations)} violation(s)"


@dataclass
class Invariant:
    name: str
    description: str
    check: Callable[[Database, datetime], list[Violation]]
    # True when a violation can be corrected by a reversible, logged action.
    remediable: bool = False


def _enabled_models(db: Database) -> list[dict[str, Any]]:
    return list(
        db[models_collection_name()].find(
            {"enabled": True, "deprecated": {"$ne": True}},
            {"provider": 1, "model_id": 1},
        )
    )


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def no_work_for_disabled_models(db: Database, now: datetime) -> list[Violation]:
    """Queue and catalogue must agree about what should be benchmarked.

    On 2026-08-04, 223 of 252 queued or running jobs (88%) targeted models that
    had been disabled hours earlier. Demotion stopped new scheduling but never
    reached work already queued, and the dead-letter sweep kept resurrecting it.
    """
    eligible = {(m["provider"], m["model_id"]) for m in _enabled_models(db)}
    violations = []
    for job in db[jobs_collection_name()].find(
        {"status": {"$in": ["queued", "running"]}},
        {"provider": 1, "model_id": 1, "status": 1},
    ):
        key = (job.get("provider"), job.get("model_id"))
        if key not in eligible:
            violations.append(
                Violation(
                    subject=f"{job.get('provider')}/{job.get('model_id')}",
                    detail=f"job {job['_id']} is {job.get('status')} but the model is not enabled",
                    data={"job_id": job["_id"], "provider": job.get("provider")},
                )
            )
    return violations


def every_provider_is_progressing(db: Database, now: datetime) -> list[Violation]:
    """Each provider lane must be producing, not just the fleet as a whole.

    liveness_status reports healthy when any one provider has recent data, so a
    busy OpenAI lane can hide dead Together, DeepInfra and Vertex workers.
    """
    providers = {m["provider"] for m in _enabled_models(db)}
    cutoff = now - timedelta(hours=2)
    recent = set(db[metrics_collection_name()].distinct("provider", {"run_ts": {"$gte": cutoff}}))
    return [
        Violation(
            subject=provider,
            detail="has enabled models but wrote no metric in the last 2h",
            data={"provider": provider},
        )
        for provider in sorted(providers - recent)
    ]


def enabled_models_are_being_measured(db: Database, now: datetime, *, cadence_seconds: int = 1800) -> list[Violation]:
    """An enabled model with no recent data is either starved or silently broken."""
    horizon = now - timedelta(seconds=cadence_seconds * MODEL_STALENESS_MULTIPLIER)
    fresh = set(db[metrics_collection_name()].distinct("model_name", {"run_ts": {"$gte": horizon}}))
    return [
        Violation(
            subject=f"{m['provider']}/{m['model_id']}",
            detail="enabled but no successful measurement within the staleness horizon",
            data={"provider": m["provider"], "model_id": m["model_id"]},
        )
        for m in _enabled_models(db)
        if m["model_id"] not in fresh
    ]


def catalogue_is_fresh(db: Database, now: datetime) -> list[Violation]:
    """provider_catalog must be actively written.

    It silently stopped being updated on 2026-04-29 when the Sauron discovery
    job was registered enabled=False, and stayed stale for three months while
    still looking like a live data source.
    """
    newest = db.provider_catalog.find_one(sort=[("last_seen_at", -1)])
    stamp = (newest or {}).get("last_seen_at")
    if stamp is None:
        return [Violation(subject="provider_catalog", detail="no last_seen_at on any document")]
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age = now - stamp
    if age > CATALOGUE_MAX_AGE:
        return [
            Violation(
                subject="provider_catalog",
                detail=f"newest last_seen_at is {age.days}d old; discovery is not writing",
                data={"last_seen_at": stamp.isoformat()},
            )
        ]
    return []


def dead_letters_are_not_accumulating(db: Database, now: datetime) -> list[Violation]:
    """A growing dead-letter pile means terminal failures have no path back.

    This is the ratchet that decayed coverage to 11.7% with no single incident.
    """
    jobs = db[jobs_collection_name()]
    total = jobs.count_documents({"status": "dead_letter"})
    week_ago = now - timedelta(days=7)
    recent = jobs.count_documents({"status": "dead_letter", "updated_at": {"$gte": week_ago}})
    if total and recent > total * 0.5:
        return [
            Violation(
                subject="bench_jobs",
                detail=f"{recent} of {total} dead letters were created in the last 7d",
                data={"total": total, "recent": recent},
            )
        ]
    return []


def no_case_duplicate_models(db: Database, now: datetime) -> list[Violation]:
    """The unique index on (provider, model_id) is case-sensitive.

    Qwen/Qwen2.5-7B-Instruct-Turbo and qwen/qwen2.5-7b-instruct-turbo are
    separate documents and can both be enabled, which benchmarks the model twice
    and shows it twice on the site.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for m in _enabled_models(db):
        seen.setdefault((m["provider"], m["model_id"].lower()), []).append(m["model_id"])
    return [
        Violation(
            subject=f"{provider}/{lowered}",
            detail=f"enabled under {len(ids)} spellings: {', '.join(sorted(ids))}",
            data={"provider": provider, "model_ids": sorted(ids)},
        )
        for (provider, lowered), ids in sorted(seen.items())
        if len(ids) > 1
    ]


def provider_volume_within_band(db: Database, now: datetime) -> list[Violation]:
    """Catch partial collapse: a provider still writing, but far less than usual.

    Total row count stayed plausible during the degradation because Bedrock kept
    running on a different host while every clifford lane thinned out.
    """
    metrics = db[metrics_collection_name()]
    baseline_start = now - timedelta(days=VOLUME_BASELINE_DAYS)
    day_ago = now - timedelta(days=1)

    daily: dict[str, list[int]] = {}
    for row in metrics.aggregate(
        [
            {"$match": {"run_ts": {"$gte": baseline_start, "$lt": day_ago}}},
            {
                "$group": {
                    "_id": {
                        "p": "$provider",
                        "d": {"$dateToString": {"format": "%Y-%m-%d", "date": "$run_ts"}},
                    },
                    "n": {"$sum": 1},
                }
            },
        ]
    ):
        daily.setdefault(row["_id"]["p"], []).append(row["n"])

    today = {
        row["_id"]: row["n"]
        for row in metrics.aggregate(
            [
                {"$match": {"run_ts": {"$gte": day_ago}}},
                {"$group": {"_id": "$provider", "n": {"$sum": 1}}},
            ]
        )
    }

    violations = []
    for provider, counts in sorted(daily.items()):
        if len(counts) < 3:
            continue  # not enough history to say what normal is
        median = statistics.median(counts)
        floor = median * VOLUME_FLOOR_RATIO
        actual = today.get(provider, 0)
        if actual < floor:
            violations.append(
                Violation(
                    subject=provider,
                    detail=f"wrote {actual} rows in 24h against a trailing median of {median:.0f}",
                    data={"provider": provider, "actual": actual, "median": median},
                )
            )
    return violations


INVARIANTS: list[Invariant] = [
    Invariant(
        "no_work_for_disabled_models",
        "Queued and running jobs only target enabled models",
        no_work_for_disabled_models,
        remediable=True,
    ),
    Invariant(
        "every_provider_is_progressing",
        "Every provider with enabled models wrote a metric recently",
        every_provider_is_progressing,
    ),
    Invariant(
        "enabled_models_are_being_measured",
        "Every enabled model has a recent successful measurement",
        enabled_models_are_being_measured,
    ),
    Invariant(
        "catalogue_is_fresh",
        "provider_catalog has been written within 48h",
        catalogue_is_fresh,
    ),
    Invariant(
        "dead_letters_are_not_accumulating",
        "Dead letters are not dominated by recent additions",
        dead_letters_are_not_accumulating,
    ),
    Invariant(
        "no_case_duplicate_models",
        "No model is enabled under more than one spelling",
        no_case_duplicate_models,
        remediable=True,
    ),
    Invariant(
        "provider_volume_within_band",
        "Per-provider row volume is within a band of its trailing median",
        provider_volume_within_band,
    ),
]


def evaluate(db: Database, *, now: datetime | None = None, only: set[str] | None = None) -> list[Result]:
    """Run every invariant. A check that raises is a failure, not a skip."""
    now = now or utcnow()
    results = []
    for inv in INVARIANTS:
        if only and inv.name not in only:
            continue
        try:
            violations = inv.check(db, now)
            results.append(Result(inv.name, inv.description, not violations, violations, inv.remediable, now))
        except Exception as exc:  # noqa: BLE001
            # An invariant that cannot run tells us nothing, so it must not
            # read as green. That is the exact failure this module exists for.
            results.append(
                Result(
                    inv.name,
                    inv.description,
                    False,
                    [],
                    inv.remediable,
                    now,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
