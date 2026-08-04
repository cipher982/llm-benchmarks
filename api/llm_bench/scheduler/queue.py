from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database

from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.mongo import models_collection_name

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_RETRYABLE_STATUSES = {"success", "failed", "timeout"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def scheduled_job_id(provider: str, model_id: str) -> str:
    return f"{provider}:{model_id}"


def manual_job_id(provider: str, model_id: str, now: datetime) -> str:
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    return f"manual:{provider}:{model_id}:{timestamp}"


def smoke_hang_job_id(provider: str, model_id: str) -> str:
    return f"smoke_hang:{provider}:{model_id}"


def jobs_collection(db: Database) -> Collection:
    return db[jobs_collection_name()]


def ensure_indexes(db: Database) -> None:
    coll = jobs_collection(db)
    coll.create_index([("provider", 1), ("status", 1), ("not_before", 1), ("priority", -1), ("created_at", 1)])
    coll.create_index([("status", 1), ("lease_expires_at", 1)])
    coll.create_index([("job_kind", 1), ("updated_at", -1)])
    coll.create_index([("status", 1), ("last_attempt_error_kind", 1), ("updated_at", -1)])


def _new_job_doc(
    *,
    job_id: str,
    provider: str,
    model_id: str,
    priority: float,
    job_kind: str,
    now: datetime,
    not_before: datetime | None = None,
    max_attempts: int = policies.DEFAULT_MAX_ATTEMPTS,
    deadline_seconds: int = policies.DEFAULT_DEADLINE_SECONDS,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "_id": job_id,
        "provider": provider,
        "model_id": model_id,
        "status": "queued",
        "priority": priority,
        "attempt": 0,
        "max_attempts": max_attempts,
        "deadline_seconds": deadline_seconds,
        "not_before": not_before or now,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "lease_expires_at": None,
        "worker_id": None,
        "last_attempt_error_kind": None,
        "last_attempt_error_message": None,
        "job_kind": job_kind,
    }
    if extra:
        doc.update(extra)
    return doc


def enqueue_scheduled_job(
    db: Database,
    *,
    provider: str,
    model_id: str,
    priority: float,
    not_before: datetime | None = None,
    now: datetime | None = None,
) -> bool:
    now = now or utcnow()
    job_id = scheduled_job_id(provider, model_id)
    doc = _new_job_doc(
        job_id=job_id,
        provider=provider,
        model_id=model_id,
        priority=priority,
        job_kind="scheduled",
        now=now,
        not_before=not_before,
    )
    coll = jobs_collection(db)
    existing = coll.find_one({"_id": job_id}, {"status": 1})
    if existing and existing.get("status") in ACTIVE_STATUSES | {"dead_letter"}:
        return False
    if existing and existing.get("status") not in TERMINAL_RETRYABLE_STATUSES:
        return False
    coll.replace_one({"_id": job_id}, doc, upsert=True)
    return True


def enqueue_smoke_hang_job(
    db: Database,
    *,
    provider: str,
    model_id: str,
    seconds: int,
    priority: float = 10_000,
    deadline_seconds: int | None = None,
    now: datetime | None = None,
) -> bool:
    now = now or utcnow()
    job_id = smoke_hang_job_id(provider, model_id)
    doc = _new_job_doc(
        job_id=job_id,
        provider=provider,
        model_id=model_id,
        priority=priority,
        job_kind="smoke_hang",
        now=now,
        deadline_seconds=deadline_seconds or policies.DEFAULT_DEADLINE_SECONDS,
        extra={"smoke_seconds": seconds},
    )
    coll = jobs_collection(db)
    existing = coll.find_one({"_id": job_id}, {"status": 1})
    if existing and existing.get("status") in ACTIVE_STATUSES:
        return False
    coll.replace_one({"_id": job_id}, doc, upsert=True)
    return True


def enqueue_manual_job(
    db: Database,
    *,
    provider: str,
    model_id: str,
    priority: float = 10_000,
    deadline_seconds: int = policies.DEFAULT_DEADLINE_SECONDS,
    max_attempts: int = policies.DEFAULT_MAX_ATTEMPTS,
    now: datetime | None = None,
) -> str:
    now = now or utcnow()
    job_id = manual_job_id(provider, model_id, now)
    doc = _new_job_doc(
        job_id=job_id,
        provider=provider,
        model_id=model_id,
        priority=priority,
        job_kind="manual",
        now=now,
        deadline_seconds=deadline_seconds,
        max_attempts=max_attempts,
    )
    jobs_collection(db).insert_one(doc)
    return job_id


def claim_next_job(
    db: Database,
    *,
    provider: str,
    worker_id: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    now = now or utcnow()
    coll = jobs_collection(db)
    job = coll.find_one_and_update(
        {
            "provider": provider,
            "status": "queued",
            "not_before": {"$lte": now},
        },
        [
            {
                "$set": {
                    "status": "running",
                    "started_at": now,
                    "updated_at": now,
                    "worker_id": worker_id,
                    "attempt": {"$add": [{"$ifNull": ["$attempt", 0]}, 1]},
                    "lease_expires_at": {
                        "$dateAdd": {
                            "startDate": now,
                            "unit": "second",
                            "amount": {
                                "$add": [
                                    {"$ifNull": ["$deadline_seconds", policies.DEFAULT_DEADLINE_SECONDS]},
                                    policies.LEASE_GRACE_SECONDS,
                                ]
                            },
                        }
                    },
                }
            }
        ],
        sort=[("priority", -1), ("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if job is not None and not is_model_eligible(db, provider=provider, model_id=job.get("model_id")):
        # The catalogue is the authority on what should be benchmarked. Jobs
        # outlive catalogue decisions, so a model disabled after its job was
        # queued would otherwise keep consuming a worker slot forever.
        cancel_job(db, job_id=job["_id"], reason="model no longer enabled", now=now)
        return None
    return job


def is_model_eligible(db: Database, *, provider: str, model_id: Any) -> bool:
    """True when the catalogue still wants this model benchmarked."""
    if not model_id:
        return False
    doc = db[models_collection_name()].find_one(
        {"provider": provider, "model_id": model_id},
        {"enabled": 1, "deprecated": 1},
    )
    return bool(doc and doc.get("enabled") and not doc.get("deprecated"))


def cancel_job(db: Database, *, job_id: Any, reason: str, now: datetime | None = None) -> bool:
    now = now or utcnow()
    result = jobs_collection(db).update_one(
        {"_id": job_id},
        {
            "$set": {
                "status": "cancelled",
                "updated_at": now,
                "finished_at": now,
                "lease_expires_at": None,
                "worker_id": None,
                "cancelled_reason": reason,
            }
        },
    )
    return result.modified_count > 0


def cancel_ineligible_jobs(db: Database, *, now: datetime | None = None) -> int:
    """Cancel queued or running jobs whose model is no longer enabled.

    Demotion has to reach work already in the queue. Without this, disabling a
    model stops new scheduling but leaves its existing jobs cycling.
    """
    now = now or utcnow()
    enabled = {
        (m["provider"], m["model_id"])
        for m in db[models_collection_name()].find(
            {"enabled": True, "deprecated": {"$ne": True}}, {"provider": 1, "model_id": 1}
        )
    }
    cancelled = 0
    for job in jobs_collection(db).find({"status": {"$in": ["queued", "running"]}}, {"provider": 1, "model_id": 1}):
        if (job.get("provider"), job.get("model_id")) not in enabled:
            cancelled += int(cancel_job(db, job_id=job["_id"], reason="model no longer enabled", now=now))
    return cancelled


def mark_success(db: Database, *, job_id: Any, worker_id: str | None = None, now: datetime | None = None) -> bool:
    now = now or utcnow()
    query: dict[str, Any] = {"_id": job_id, "status": "running"}
    if worker_id:
        query["worker_id"] = worker_id
    result = jobs_collection(db).update_one(
        query,
        {
            "$set": {
                "status": "success",
                "updated_at": now,
                "finished_at": now,
                "lease_expires_at": None,
                "worker_id": None,
                "last_attempt_error_kind": None,
                "last_attempt_error_message": None,
            }
        },
    )
    return result.modified_count > 0


def _failure_update(job: dict[str, Any], *, error_kind: str, error_message: str, now: datetime) -> dict[str, Any]:
    attempt = int(job.get("attempt") or 0)
    max_attempts = int(job.get("max_attempts") or policies.DEFAULT_MAX_ATTEMPTS)
    retry = policies.should_retry(error_kind, attempt, max_attempts, error_message)
    status = "queued" if retry else "dead_letter"
    not_before = now + policies.retry_backoff(error_kind, attempt=attempt) if retry else now
    return {
        "$set": {
            "status": status,
            "updated_at": now,
            "finished_at": now if not retry else None,
            "not_before": not_before,
            "lease_expires_at": None,
            "worker_id": None,
            "last_attempt_error_kind": error_kind,
            "last_attempt_error_message": error_message[:2000],
        }
    }


def mark_failure(
    db: Database,
    *,
    job: dict[str, Any],
    error_kind: str,
    error_message: str,
    worker_id: str | None = None,
    now: datetime | None = None,
) -> str | None:
    now = now or utcnow()
    query: dict[str, Any] = {"_id": job["_id"], "status": "running"}
    if worker_id:
        query["worker_id"] = worker_id
    update = _failure_update(job, error_kind=error_kind, error_message=error_message, now=now)
    result = jobs_collection(db).update_one(query, update)
    if result.modified_count == 0:
        return None
    return update["$set"]["status"]


def requeue_retryable_dead_letters(
    db: Database,
    *,
    now: datetime | None = None,
    min_age_seconds: int = policies.DEAD_LETTER_RETRY_AFTER_SECONDS,
) -> list[dict[str, Any]]:
    """Return old recoverable dead letters to the queue."""
    now = now or utcnow()
    cutoff = now - timedelta(seconds=min_age_seconds)
    billing_cutoff = now - timedelta(seconds=policies.BILLING_DEAD_LETTER_RETRY_AFTER_SECONDS)
    coll = jobs_collection(db)
    retryable_kinds = sorted(policies.RETRYABLE_ERROR_KINDS)
    query: dict[str, Any] = {
        "status": "dead_letter",
        "$or": [
            {
                "updated_at": {"$lte": cutoff},
                "$or": [
                    {"last_attempt_error_kind": {"$in": retryable_kinds}},
                    {
                        "last_attempt_error_kind": "unknown",
                        "last_attempt_error_message": {
                            "$regex": "overloaded|model busy|retry later|temporarily unavailable",
                            "$options": "i",
                        },
                    },
                ],
            },
            {
                "updated_at": {"$lte": billing_cutoff},
                "last_attempt_error_kind": "billing",
            },
        ],
    }
    transitioned: list[dict[str, Any]] = []
    for job in coll.find(query):
        if not is_model_eligible(db, provider=job.get("provider"), model_id=job.get("model_id")):
            # Do not resurrect work for a model the catalogue has demoted.
            cancel_job(db, job_id=job["_id"], reason="model no longer enabled", now=now)
            continue
        if int(job.get("dead_letter_requeues") or 0) >= policies.MAX_DEAD_LETTER_REQUEUES:
            continue
        result = coll.update_one(
            {"_id": job["_id"], "status": "dead_letter", "updated_at": job.get("updated_at")},
            {
                "$set": {
                    "status": "queued",
                    "updated_at": now,
                    "finished_at": None,
                    "not_before": now,
                    "lease_expires_at": None,
                    "worker_id": None,
                },
                "$inc": {"dead_letter_requeues": 1},
            },
        )
        if result.modified_count:
            transitioned.append({**job, "transitioned_status": "queued"})
    return transitioned


def expire_orphaned_running(db: Database, *, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or utcnow()
    coll = jobs_collection(db)
    expired = list(coll.find({"status": "running", "lease_expires_at": {"$lte": now}}))
    transitioned: list[dict[str, Any]] = []
    for job in expired:
        update = _failure_update(job, error_kind="timeout", error_message="lease expired", now=now)
        result = coll.update_one({"_id": job["_id"], "status": "running"}, update)
        if result.modified_count > 0:
            transitioned.append({**job, "transitioned_status": update["$set"]["status"]})
    return transitioned
