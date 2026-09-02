from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database

from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.routing import freeze_route_snapshot

ACTIVE_STATUSES = {"queued", "running"}
# States a model can be scheduled out of again. `cancelled` belongs here: it is
# what the sweep writes when a model was not eligible at the time, and
# eligibility changes — a model disabled last week and re-enabled today had a
# cancelled job standing between it and ever being measured again. 76 enabled
# models were in exactly that state, never_run and unschedulable, with nothing
# reporting a fault because the scheduler had simply stopped offering them.
#
# The same shape as the four coverage outages this scheduler has already had:
# any terminal state needs a way back, or it is a ratchet.
TERMINAL_RETRYABLE_STATUSES = {"success", "failed", "timeout", "cancelled"}


# Sample roles whose work is deliberately not published. Kept here rather than
# imported from the runner so the queue does not pull in every provider module.
SAMPLE_ROLE_PROBE = "probe"
SAMPLE_ROLE_SHADOW = "shadow"
NON_PUBLISHING_SAMPLE_ROLES = frozenset({SAMPLE_ROLE_PROBE, SAMPLE_ROLE_SHADOW})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def scheduled_job_id(provider: str, model_id: str, endpoint_tag: str | None = None) -> str:
    """One id per unit of work the scheduler actually measures.

    An endpoint target is `(model, endpoint tag)`, not `(provider, model)`:
    `deepinfra/bf16` and `deepinfra/turbo` serve the same model at different
    speeds. Sharing an id would let one endpoint's run stand in for its
    siblings' — the same collapse that made a model look fresh when only one of
    its lanes had been measured.

    Routes predating endpoint identity keep their two-part id, so nothing in
    flight is orphaned by the change.
    """
    if endpoint_tag:
        return f"{provider}:{model_id}:{endpoint_tag}"
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
    route_snapshot: dict[str, Any] | None = None,
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
    frozen_route = freeze_route_snapshot(provider, model_id, route_snapshot, now=now)
    if frozen_route is not None:
        doc["route_snapshot"] = frozen_route
    return doc


ROUTED_CEILING_STATE_ID = "openrouter:routed_ceiling"
PROVIDER_STATE_COLLECTION = "provider_state"


def _admission_bucket_id(provider: str, at: datetime) -> str:
    return f"{provider}:admissions:{at.astimezone(timezone.utc):%Y-%m-%dT%H}"


def record_admission(db: Database, *, provider: str, now: datetime | None = None) -> None:
    """Count one job entering the lane, in an hourly ledger that nothing overwrites.

    Counting job documents does not work: a target's job id is stable and a
    re-enqueue replaces the document, so a target run twenty times in a day
    counted once — exactly the shape a runaway takes. The ledger is a counter
    per provider per UTC hour in `provider_state`, incremented by every path
    that admits work: scheduled, manual, probe, and dead-letter requeue.
    """
    now = now or utcnow()
    db[PROVIDER_STATE_COLLECTION].update_one(
        {"_id": _admission_bucket_id(provider, now)},
        {
            "$set": {
                "provider": provider,
                "kind": "admissions",
                "hour": now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0),
            },
            "$inc": {"n": 1},
        },
        upsert=True,
    )


def routed_jobs_last_24h(db: Database, *, provider: str, now: datetime | None = None) -> int:
    """Admissions into the lane over the trailing 24 hourly buckets, this hour included."""
    now = now or utcnow()
    ids = [_admission_bucket_id(provider, now - timedelta(hours=h)) for h in range(25)]
    return sum(int(doc.get("n") or 0) for doc in db[PROVIDER_STATE_COLLECTION].find({"_id": {"$in": ids}}, {"n": 1}))


def routed_ceiling_reached(db: Database, *, provider: str, now: datetime | None = None) -> bool:
    """True when the lane may admit nothing more today. Records the hit so it pages.

    The ceiling is a refusal, not a queue: nothing is deferred to "later", the
    next pass simply asks again, and the trailing window opens on its own. A
    hit is written to `provider_state` so `routed_jobs_under_daily_ceiling`
    pages through the invariant path rather than the drop being silent.
    """
    if provider not in policies.ROUTED_LANES:
        return False
    now = now or utcnow()
    ceiling = policies.max_routed_jobs_per_day()
    count = routed_jobs_last_24h(db, provider=provider, now=now)
    if count < ceiling:
        return False
    state = db[PROVIDER_STATE_COLLECTION]
    previous = state.find_one({"_id": ROUTED_CEILING_STATE_ID}, {"last_logged_at": 1})
    last_logged = previous.get("last_logged_at") if previous else None
    if last_logged is not None and last_logged.tzinfo is None:
        last_logged = last_logged.replace(tzinfo=timezone.utc)
    update: dict[str, Any] = {
        "$set": {"provider": provider, "kind": "routed_ceiling", "hit_at": now, "count_24h": count, "ceiling": ceiling},
        "$inc": {"hits": 1},
    }
    # One log line per five minutes, not one per refused job.
    if last_logged is None or now - last_logged >= timedelta(minutes=5):
        update["$set"]["last_logged_at"] = now
        print(
            f"ceiling_hit provider={provider} jobs_24h={count} ceiling={ceiling}: refusing routed admission", flush=True
        )
    state.update_one({"_id": ROUTED_CEILING_STATE_ID}, update, upsert=True)
    return True


def enqueue_scheduled_job(
    db: Database,
    *,
    provider: str,
    model_id: str,
    priority: float,
    not_before: datetime | None = None,
    now: datetime | None = None,
    route_snapshot: dict[str, Any] | None = None,
    endpoint_tag: str | None = None,
    cadence_seconds: int | None = None,
    benchmark_profile_id: str | None = None,
) -> bool:
    now = now or utcnow()
    job_id = scheduled_job_id(provider, model_id, endpoint_tag)
    extra: dict[str, Any] = {}
    if endpoint_tag:
        extra["endpoint_tag"] = endpoint_tag
    if cadence_seconds is not None:
        extra["cadence_seconds"] = cadence_seconds
    if benchmark_profile_id:
        # A published sample under a named profile. The runner reads the id
        # to size the budget; the dashboard keeps non-default rows off the
        # legacy charts and Delivered TPS reads them regardless of profile.
        extra["benchmark_profile_id"] = benchmark_profile_id
    doc = _new_job_doc(
        job_id=job_id,
        provider=provider,
        model_id=model_id,
        priority=priority,
        job_kind="scheduled",
        now=now,
        not_before=not_before,
        route_snapshot=route_snapshot,
        extra=extra,
    )
    coll = jobs_collection(db)
    if endpoint_tag and coll.find_one(
        {"provider": provider, "model_id": model_id, "status": {"$in": sorted(ACTIVE_STATUSES)}},
        {"_id": 1},
    ):
        return False
    existing = coll.find_one({"_id": job_id}, {"status": 1})
    if existing and existing.get("status") in ACTIVE_STATUSES | {"dead_letter"}:
        return False
    if existing and existing.get("status") not in TERMINAL_RETRYABLE_STATUSES:
        return False
    if routed_ceiling_reached(db, provider=provider, now=now):
        return False
    coll.replace_one({"_id": job_id}, doc, upsert=True)
    if provider in policies.ROUTED_LANES:
        record_admission(db, provider=provider, now=now)
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
    route_snapshot: dict[str, Any] | None = None,
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
        route_snapshot=route_snapshot,
    )
    if routed_ceiling_reached(db, provider=provider, now=now):
        raise RuntimeError(
            f"routed admission ceiling reached for {provider}: "
            f"{policies.max_routed_jobs_per_day()} jobs in the trailing 24h"
        )
    jobs_collection(db).insert_one(doc)
    if provider in policies.ROUTED_LANES:
        record_admission(db, provider=provider, now=now)
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
    if job is not None:
        reason = job_ineligibility_reason(db, job)
        if reason:
            # Catalogue decisions outlive queued jobs. Refuse stale work after
            # claiming it so a retired target never reaches a paid provider.
            cancel_job(db, job_id=job["_id"], reason=reason, now=now)
            return None
    return job


def is_model_eligible(db: Database, *, provider: str, model_id: Any, sample_role: str = "published") -> bool:
    """True when the catalogue still wants this work done.

    Published work requires an enabled model.

    A probe requires the opposite: a candidate under admission is deliberately
    not enabled yet, and refusing to run its probe would make probe-before-
    promote impossible. `probing` is what separates "not enabled because we are
    still deciding" from "not enabled because we decided no".

    A shadow sample is neither. It re-measures a model the site already wants
    under a different benchmark profile, so it is legitimate for an enabled
    model and for a candidate — the reasoning profile exists precisely to get
    numbers for enabled models the default profile cannot measure. Requiring
    `probing` here would have cancelled every one of those jobs as ineligible.
    """
    if not model_id:
        return False
    doc = db[models_collection_name()].find_one(
        {"provider": provider, "model_id": model_id},
        {"enabled": 1, "deprecated": 1, "status": 1},
    )
    if not doc or doc.get("deprecated"):
        return False
    if sample_role == SAMPLE_ROLE_PROBE:
        return doc.get("status") == "probing"
    if sample_role in NON_PUBLISHING_SAMPLE_ROLES:
        return bool(doc.get("enabled")) or doc.get("status") == "probing"
    return bool(doc.get("enabled"))


def job_ineligibility_reason(db: Database, job: dict[str, Any]) -> str | None:
    provider = str(job.get("provider") or "")
    model_id = job.get("model_id")
    if not is_model_eligible(
        db,
        provider=provider,
        model_id=model_id,
        sample_role=job.get("sample_role") or "published",
    ):
        return "model no longer enabled"

    endpoint_tag = job.get("endpoint_tag")
    endpoints = db[endpoint_discovery.endpoints_collection_name()]
    if endpoint_tag:
        target = endpoints.find_one(
            {"model_id": model_id, "endpoint_tag": endpoint_tag, "enabled": True},
            {"_id": 1},
        )
        if target is None:
            return "endpoint no longer enabled"
    elif provider == "openrouter" and job.get("job_kind") == "scheduled":
        if endpoints.find_one({"model_id": model_id, "enabled": True}, {"_id": 1}) is not None:
            return "superseded by endpoint rotation"
    return None


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
    """Cancel active jobs superseded by current model and endpoint catalogues."""
    now = now or utcnow()
    cancelled = 0
    for job in jobs_collection(db).find({"status": {"$in": ["queued", "running"]}}):
        reason = job_ineligibility_reason(db, job)
        if reason:
            cancelled += int(cancel_job(db, job_id=job["_id"], reason=reason, now=now))
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
            # Which protocol this verdict was reached under. A terminal failure
            # that describes the measurement rather than the model stops being
            # evidence when the protocol changes, and the sweep needs to be able
            # to tell.
            "last_attempt_protocol_version": policies.MEASUREMENT_PROTOCOL_VERSION,
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


def pin_budget_exhausted_to_reasoning_profile(
    db: Database,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Move `budget_exhausted` dead letters onto the reasoning profile.

    Off unless BENCHMARK_REASONING_PUBLISH is on. For each dead letter whose
    last verdict was `budget_exhausted` under a non-reasoning profile and whose
    target is still in the catalogue: pin the target's health doc to
    `cloud-reasoning-v1` (with the reason and a timestamp, so it can be
    audited and unpinned) and cancel the dead letter so the scheduler can
    create the once-a-day reasoning job in its place. Targets already pinned,
    or that failed under the reasoning profile itself, are left alone: a
    model that cannot answer inside 2048 tokens under the higher ceiling is
    a verdict, not a budget problem.
    """
    if not policies.reasoning_publish_enabled():
        return []
    now = now or utcnow()
    coll = jobs_collection(db)
    health_coll = db[health_collection_name()]
    pinned: list[dict[str, Any]] = []
    for job in coll.find(
        {
            "status": "dead_letter",
            "last_attempt_error_kind": "budget_exhausted",
            "benchmark_profile_id": {"$ne": policies.REASONING_PROFILE_ID},
        }
    ):
        if job_ineligibility_reason(db, job):
            continue
        model_id = job.get("model_id")
        endpoint_tag = job.get("endpoint_tag") or None
        health_coll.update_one(
            {"provider": job.get("provider"), "model_id": model_id, "endpoint_tag": endpoint_tag},
            {
                "$set": {
                    "measurement_profile": policies.REASONING_PROFILE_ID,
                    "measurement_profile_reason": (
                        "budget_exhausted under the default profile: the model answered but spent "
                        "the budget its price allows on thinking; measured once a day at 2048 tokens"
                    ),
                    "measurement_profile_set_at": now,
                    "measurement_profile_from_job": job["_id"],
                    "cadence_seconds": policies.REASONING_CADENCE_SECONDS,
                }
            },
            upsert=True,
        )
        cancel_job(db, job_id=job["_id"], reason="pinned to reasoning profile", now=now)
        pinned.append({"model_id": model_id, "endpoint_tag": endpoint_tag, "job_id": job["_id"]})
    return pinned


def requeue_retryable_dead_letters(
    db: Database,
    *,
    now: datetime | None = None,
    min_age_seconds: int = policies.DEAD_LETTER_RETRY_AFTER_SECONDS,
) -> list[dict[str, Any]]:
    """Return old recoverable dead letters to the queue.

    Recoverable covers two different things. A transient failure recovers with
    time, so it waits for a cutoff. A verdict about the measurement — the model
    could not produce visible output inside the budget — recovers when the
    protocol changes, and waiting does nothing for it. Without the second
    clause, raising the token budget leaves every model that failed under the
    old one permanently dead-lettered, which is how 419 of them came to be
    holding a verdict reached against a budget that no longer existed.
    """
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
            {
                # Stale-protocol verdicts. An absent version predates the field,
                # so the protocol is unknown and the verdict cannot be relied on.
                "last_attempt_error_kind": {"$in": sorted(policies.PROTOCOL_DEPENDENT_ERROR_KINDS)},
                "$or": [
                    {"last_attempt_protocol_version": {"$exists": False}},
                    {"last_attempt_protocol_version": {"$lt": policies.MEASUREMENT_PROTOCOL_VERSION}},
                ],
            },
        ],
    }
    transitioned: list[dict[str, Any]] = []
    for job in coll.find(query):
        reason = job_ineligibility_reason(db, job)
        if reason:
            # Do not resurrect work the current catalogues have superseded.
            cancel_job(db, job_id=job["_id"], reason=reason, now=now)
            continue
        if job.get("endpoint_tag") and coll.find_one(
            {
                "provider": job.get("provider"),
                "model_id": job.get("model_id"),
                "status": {"$in": sorted(ACTIVE_STATUSES)},
                "_id": {"$ne": job["_id"]},
            },
            {"_id": 1},
        ):
            continue
        if int(job.get("dead_letter_requeues") or 0) >= policies.MAX_DEAD_LETTER_REQUEUES:
            continue
        # A requeue is an admission. After a cap reset every billing dead
        # letter is eligible at once; the ceiling makes that a drain at the
        # daily rate rather than a burst — the remainder stays dead-lettered
        # and eligible for the next pass, in insertion order.
        if routed_ceiling_reached(db, provider=str(job.get("provider") or ""), now=now):
            break
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
                    "last_requeued_at": now,
                },
                "$inc": {"dead_letter_requeues": 1},
            },
        )
        if result.modified_count:
            if str(job.get("provider") or "") in policies.ROUTED_LANES:
                record_admission(db, provider=str(job.get("provider")), now=now)
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
