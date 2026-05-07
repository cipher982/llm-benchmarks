from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.collection import Collection
from pymongo.database import Database

from llm_bench.scheduler.mongo import errors_collection_name
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import heartbeats_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.queue import scheduled_job_id


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def health_collection(db: Database) -> Collection:
    return db[health_collection_name()]


def dedupe_existing_health_docs(db: Database) -> int:
    coll = health_collection(db)
    removed = 0
    groups = coll.aggregate(
        [
            {"$match": {"provider": {"$exists": True}, "model_id": {"$exists": True}}},
            {"$sort": {"updated_at": -1}},
            {"$group": {"_id": {"provider": "$provider", "model_id": "$model_id"}, "ids": {"$push": "$_id"}}},
            {"$match": {"ids.1": {"$exists": True}}},
        ]
    )
    for group in groups:
        ids = group.get("ids") or []
        stale_ids = ids[1:]
        if not stale_ids:
            continue
        result = coll.delete_many({"_id": {"$in": stale_ids}})
        removed += result.deleted_count
    return removed


def ensure_indexes(db: Database) -> None:
    dedupe_existing_health_docs(db)
    coll = health_collection(db)
    coll.create_index([("provider", 1), ("model_id", 1)], unique=True)
    coll.create_index([("freshness_status", 1), ("updated_at", -1)])
    coll.create_index([("enabled", 1), ("provider", 1)])


def compute_freshness_status(
    *,
    enabled: bool,
    cadence_seconds: int,
    last_success_at: datetime | None,
    now: datetime | None = None,
) -> tuple[str, int | None]:
    now = now or utcnow()
    if not enabled:
        return "disabled", None
    if last_success_at is None:
        return "never_run", None
    staleness_seconds = max(0, int((now - last_success_at).total_seconds()))
    if staleness_seconds > cadence_seconds * 3:
        return "critical", staleness_seconds
    if staleness_seconds > cadence_seconds * 1.5:
        return "stale", staleness_seconds
    return "fresh", staleness_seconds


def _recent_counts(db: Database, *, provider: str, model_id: str, now: datetime) -> tuple[int, int, int]:
    since = now - timedelta(hours=24)
    successes = db[metrics_collection_name()].count_documents(
        {"provider": provider, "model_name": model_id, "run_ts": {"$gte": since}}
    )
    failures = db[errors_collection_name()].count_documents(
        {"provider": provider, "model_name": model_id, "ts": {"$gte": since}}
    )
    deadline_misses = db[errors_collection_name()].count_documents(
        {"provider": provider, "model_name": model_id, "ts": {"$gte": since}, "error_kind": "timeout"}
    )
    return successes, failures, deadline_misses


def refresh_model_health_doc(
    db: Database,
    *,
    provider: str,
    model_id: str,
    enabled: bool,
    cadence_seconds: int,
    deprecated: bool = False,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    effective_enabled = enabled and not deprecated
    existing = health_collection(db).find_one({"provider": provider, "model_id": model_id})
    last_success_at = existing.get("last_success_at") if existing else None
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=effective_enabled,
        cadence_seconds=cadence_seconds,
        last_success_at=last_success_at,
        now=now,
    )
    health_collection(db).update_one(
        {"provider": provider, "model_id": model_id},
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id),
                "provider": provider,
                "model_id": model_id,
                "last_success_at": None,
                "last_attempt_at": None,
                "last_error_at": None,
                "last_error_kind": None,
                "last_error_message": None,
                "consecutive_failures": 0,
                "successes_24h": 0,
                "failures_24h": 0,
                "deadline_misses_24h": 0,
            },
            "$set": {
                "enabled": effective_enabled,
                "cadence_seconds": cadence_seconds,
                "staleness_seconds": staleness_seconds,
                "freshness_status": freshness_status,
                "updated_at": now,
            },
        },
        upsert=True,
    )


def refresh_all_model_docs(db: Database, *, cadence_seconds: int, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or utcnow()
    models = list(
        db[models_collection_name()].find(
            {},
            {"provider": 1, "model_id": 1, "enabled": 1, "deprecated": 1},
        )
    )
    for model in models:
        provider = model.get("provider")
        model_id = model.get("model_id")
        if not provider or not model_id:
            continue
        refresh_model_health_doc(
            db,
            provider=provider,
            model_id=model_id,
            enabled=bool(model.get("enabled", False)),
            deprecated=bool(model.get("deprecated", False)),
            cadence_seconds=cadence_seconds,
            now=now,
        )
    return models


def backfill_from_metrics(db: Database, *, cadence_seconds: int, now: datetime | None = None) -> int:
    now = now or utcnow()
    pipeline = [
        {"$sort": {"run_ts": -1}},
        {
            "$group": {
                "_id": {"provider": "$provider", "model_id": "$model_name"},
                "last_success_at": {"$first": "$run_ts"},
            }
        },
    ]
    updated = 0
    for row in db[metrics_collection_name()].aggregate(pipeline, allowDiskUse=True):
        provider = row["_id"].get("provider")
        model_id = row["_id"].get("model_id")
        last_success_at = row.get("last_success_at")
        if not provider or not model_id or not last_success_at:
            continue
        existing = health_collection(db).find_one({"provider": provider, "model_id": model_id})
        if existing and existing.get("last_success_at"):
            continue
        successes, failures, deadline_misses = _recent_counts(db, provider=provider, model_id=model_id, now=now)
        freshness_status, staleness_seconds = compute_freshness_status(
            enabled=True,
            cadence_seconds=cadence_seconds,
            last_success_at=last_success_at,
            now=now,
        )
        health_collection(db).update_one(
            {"provider": provider, "model_id": model_id},
            {
                "$setOnInsert": {
                    "_id": scheduled_job_id(provider, model_id),
                    "provider": provider,
                    "model_id": model_id,
                    "last_attempt_at": None,
                    "last_error_at": None,
                    "last_error_kind": None,
                    "last_error_message": None,
                    "consecutive_failures": 0,
                },
                "$set": {
                    "enabled": True,
                    "cadence_seconds": cadence_seconds,
                    "last_success_at": last_success_at,
                    "successes_24h": successes,
                    "failures_24h": failures,
                    "deadline_misses_24h": deadline_misses,
                    "staleness_seconds": staleness_seconds,
                    "freshness_status": freshness_status,
                    "updated_at": now,
                },
            },
            upsert=True,
        )
        updated += 1
    return updated


def record_success(
    db: Database,
    *,
    provider: str,
    model_id: str,
    cadence_seconds: int,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    successes, failures, deadline_misses = _recent_counts(db, provider=provider, model_id=model_id, now=now)
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=True,
        cadence_seconds=cadence_seconds,
        last_success_at=now,
        now=now,
    )
    health_collection(db).update_one(
        {"provider": provider, "model_id": model_id},
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id),
                "provider": provider,
                "model_id": model_id,
            },
            "$set": {
                "enabled": True,
                "cadence_seconds": cadence_seconds,
                "last_success_at": now,
                "last_attempt_at": now,
                "last_error_at": None,
                "last_error_kind": None,
                "last_error_message": None,
                "consecutive_failures": 0,
                "successes_24h": successes,
                "failures_24h": failures,
                "deadline_misses_24h": deadline_misses,
                "staleness_seconds": staleness_seconds,
                "freshness_status": freshness_status,
                "updated_at": now,
            },
        },
        upsert=True,
    )


def record_error(
    db: Database,
    *,
    provider: str,
    model_id: str,
    cadence_seconds: int,
    error_kind: str,
    error_message: str,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    existing = health_collection(db).find_one({"provider": provider, "model_id": model_id}) or {}
    last_success_at = existing.get("last_success_at")
    successes, failures, deadline_misses = _recent_counts(db, provider=provider, model_id=model_id, now=now)
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=bool(existing.get("enabled", True)),
        cadence_seconds=cadence_seconds,
        last_success_at=last_success_at,
        now=now,
    )
    health_collection(db).update_one(
        {"provider": provider, "model_id": model_id},
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id),
                "provider": provider,
                "model_id": model_id,
                "last_success_at": None,
            },
            "$set": {
                "enabled": bool(existing.get("enabled", True)),
                "cadence_seconds": cadence_seconds,
                "last_attempt_at": now,
                "last_error_at": now,
                "last_error_kind": error_kind,
                "last_error_message": error_message[:2000],
                "successes_24h": successes,
                "failures_24h": failures,
                "deadline_misses_24h": deadline_misses,
                "staleness_seconds": staleness_seconds,
                "freshness_status": freshness_status,
                "updated_at": now,
            },
            "$inc": {"consecutive_failures": 1},
        },
        upsert=True,
    )


def heartbeat(
    db: Database,
    *,
    component: str,
    details: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    db[heartbeats_collection_name()].update_one(
        {"_id": component},
        {"$set": {"component": component, "details": details or {}, "updated_at": now}},
        upsert=True,
    )
