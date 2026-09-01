from __future__ import annotations

import os
import re
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
from llm_bench.scheduler.mongo import published_profile_filter
from llm_bench.scheduler.queue import scheduled_job_id


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def health_collection(db: Database) -> Collection:
    return db[health_collection_name()]


def health_filter(provider: str, model_id: str, endpoint_tag: str | None = None) -> dict[str, Any]:
    """The identity of one health record.

    Freshness has to be per endpoint, not per model. `deepinfra/bf16` and
    `deepinfra/turbo` are separate deployments of the same model, and if they
    share a health document then measuring one marks the other fresh — the
    scheduler stops asking for an endpoint nobody has actually benchmarked.

    Records written before endpoint identity carry no tag and keep matching on
    the two-part key, so existing freshness state survives the change.
    """
    if endpoint_tag:
        return {"provider": provider, "model_id": model_id, "endpoint_tag": endpoint_tag}
    return {"provider": provider, "model_id": model_id, "endpoint_tag": {"$in": [None, ""]}}


def dedupe_existing_health_docs(db: Database) -> int:
    coll = health_collection(db)
    removed = 0
    groups = coll.aggregate(
        [
            {"$match": {"provider": {"$exists": True}, "model_id": {"$exists": True}}},
            {"$sort": {"updated_at": -1}},
            {
                "$group": {
                    # Endpoint records are not duplicates of their model's
                    # record. Grouping without the tag would treat every
                    # endpoint of a model as a redundant copy and delete all but
                    # one, silently collapsing the fleet back to model
                    # granularity.
                    "_id": {
                        "provider": "$provider",
                        "model_id": "$model_id",
                        "endpoint_tag": "$endpoint_tag",
                    },
                    "ids": {"$push": "$_id"},
                }
            },
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
    # Uniqueness is per endpoint. A model served by deepinfra/bf16 and
    # deepinfra/turbo needs one health record each; a unique index on
    # (provider, model_id) alone would reject the second and leave the fleet
    # measuring one deployment while believing it measured both.
    #
    # The narrower index must be dropped, not merely superseded. create_index
    # adds; it does not replace. Leaving the old unique pair in place meant the
    # first endpoint record for an already-known model raised DuplicateKeyError,
    # and because that surfaced inside the scheduler pass it aborted the whole
    # pass -- no jobs enqueued for any provider, every tick, with the loop still
    # reporting healthy.
    for stale in ("provider_1_model_id_1",):
        try:
            coll.drop_index(stale)
        except Exception:  # noqa: BLE001 - absent is the expected steady state
            pass
    coll.create_index([("provider", 1), ("model_id", 1), ("endpoint_tag", 1)], unique=True)
    coll.create_index([("freshness_status", 1), ("updated_at", -1)])
    coll.create_index([("enabled", 1), ("provider", 1)])
    db[metrics_collection_name()].create_index(
        [("provider", 1), ("model_name", 1), ("run_ts", -1)],
        background=True,
    )
    # provider_progress() asks for each provider's newest completion, sorting on
    # gen_ts then run_ts. Without this index Mongo sorted the full metrics
    # collection in memory for every health payload; matching the sort key
    # reduces each lookup to one examined document.
    db[metrics_collection_name()].create_index(
        [("provider", 1), ("gen_ts", -1), ("run_ts", -1)],
        background=True,
    )
    db[errors_collection_name()].create_index(
        [("provider", 1), ("model_name", 1), ("ts", -1), ("error_kind", 1)],
        background=True,
    )


def compute_freshness_status(
    *,
    enabled: bool,
    cadence_seconds: int,
    last_success_at: datetime | None,
    now: datetime | None = None,
) -> tuple[str, int | None]:
    now = now or utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if last_success_at is not None and last_success_at.tzinfo is None:
        last_success_at = last_success_at.replace(tzinfo=timezone.utc)
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


def _recent_counts(
    db: Database, *, provider: str, model_id: str, now: datetime, endpoint_tag: str | None = None
) -> tuple[int, int, int]:
    # Published-profile rows only, on both sides. These counts describe the
    # model's primary health: a model succeeding only at 512 tokens is not
    # being measured for the site, and a model failing only long runs is not
    #
    # An endpoint doc counts only that endpoint's rows. Reusing the model's
    # counts would let a healthy sibling endpoint mask a dead one, which is
    # the same conflation that let one endpoint's run stand in for the model's.
    since = now - timedelta(hours=24)
    scope: dict[str, Any] = {"provider": provider, "model_name": model_id}
    if endpoint_tag:
        scope["route_endpoint_tag"] = endpoint_tag
    successes = db[metrics_collection_name()].count_documents(
        {**scope, "run_ts": {"$gte": since}, **published_profile_filter()}
    )
    failures = db[errors_collection_name()].count_documents(
        {**scope, "ts": {"$gte": since}, **published_profile_filter("profile_id")}
    )
    deadline_misses = db[errors_collection_name()].count_documents(
        {
            **scope,
            "ts": {"$gte": since},
            "error_kind": "timeout",
            **published_profile_filter("profile_id"),
        }
    )
    return successes, failures, deadline_misses


def refresh_model_health_doc(
    db: Database,
    *,
    provider: str,
    model_id: str,
    endpoint_tag: str | None = None,
    enabled: bool,
    cadence_seconds: int,
    deprecated: bool = False,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    effective_enabled = enabled and not deprecated
    existing = health_collection(db).find_one(health_filter(provider, model_id, endpoint_tag))
    last_success_at = existing.get("last_success_at") if existing else None
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=effective_enabled,
        cadence_seconds=cadence_seconds,
        last_success_at=last_success_at,
        now=now,
    )
    health_collection(db).update_one(
        health_filter(provider, model_id, endpoint_tag),
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id, endpoint_tag),
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
    updated = 0
    model_rows = db[models_collection_name()].find(
        {},
        {"provider": 1, "model_id": 1, "enabled": 1, "deprecated": 1},
    )
    for model_row in model_rows:
        provider = model_row.get("provider")
        model_id = model_row.get("model_id")
        if not provider or not model_id:
            continue
        # Backfill walks the model catalogue, so it can only ever seed
        # model-level freshness. Endpoint records earn theirs by being measured.
        existing = health_collection(db).find_one(health_filter(provider, model_id, None))
        if existing and existing.get("last_success_at"):
            continue
        latest = db[metrics_collection_name()].find_one(
            # Backfilled freshness must mean "published series has data", so a
            # long-profile row cannot seed last_success_at for a model whose
            # 64-token series never ran.
            {"provider": provider, "model_name": model_id, **published_profile_filter()},
            {"run_ts": 1},
            sort=[("run_ts", -1)],
        )
        last_success_at = latest.get("run_ts") if latest else None
        if not last_success_at:
            refresh_model_health_doc(
                db,
                provider=provider,
                model_id=model_id,
                enabled=bool(model_row.get("enabled", False)),
                deprecated=bool(model_row.get("deprecated", False)),
                cadence_seconds=cadence_seconds,
                now=now,
            )
            continue
        successes, failures, deadline_misses = _recent_counts(db, provider=provider, model_id=model_id, now=now)
        freshness_status, staleness_seconds = compute_freshness_status(
            enabled=bool(model_row.get("enabled", False)) and not bool(model_row.get("deprecated", False)),
            cadence_seconds=cadence_seconds,
            last_success_at=last_success_at,
            now=now,
        )
        health_collection(db).update_one(
            health_filter(provider, model_id, None),
            {
                "$setOnInsert": {
                    "_id": scheduled_job_id(provider, model_id, None),
                    "last_attempt_at": None,
                    "last_error_at": None,
                    "last_error_kind": None,
                    "last_error_message": None,
                    "consecutive_failures": 0,
                },
                "$set": {
                    "enabled": bool(model_row.get("enabled", False)) and not bool(model_row.get("deprecated", False)),
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
    endpoint_tag: str | None = None,
    cadence_seconds: int,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    successes, failures, deadline_misses = _recent_counts(
        db, provider=provider, model_id=model_id, now=now, endpoint_tag=endpoint_tag
    )
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=True,
        cadence_seconds=cadence_seconds,
        last_success_at=now,
        now=now,
    )
    health_collection(db).update_one(
        health_filter(provider, model_id, endpoint_tag),
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id, endpoint_tag),
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
    endpoint_tag: str | None = None,
    cadence_seconds: int,
    error_kind: str,
    error_message: str,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    existing = health_collection(db).find_one(health_filter(provider, model_id, endpoint_tag)) or {}
    last_success_at = existing.get("last_success_at")
    successes, failures, deadline_misses = _recent_counts(
        db, provider=provider, model_id=model_id, now=now, endpoint_tag=endpoint_tag
    )
    freshness_status, staleness_seconds = compute_freshness_status(
        enabled=bool(existing.get("enabled", True)),
        cadence_seconds=cadence_seconds,
        last_success_at=last_success_at,
        now=now,
    )
    health_collection(db).update_one(
        health_filter(provider, model_id, endpoint_tag),
        {
            "$setOnInsert": {
                "_id": scheduled_job_id(provider, model_id, endpoint_tag),
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


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def provider_progress(
    db: Database,
    *,
    providers: list[str],
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Age of the newest metric for each expected provider, reported separately.

    Answers a different question from `liveness_status`: not "is this process
    wedged" but "is each lane producing". One aggregate query cannot express
    that — a busy OpenAI lane hides dead Together, DeepInfra and Vertex ones.
    """
    now = _as_utc(now or utcnow())
    metrics = db[metrics_collection_name()]
    result: dict[str, dict[str, Any]] = {}
    # One small indexed query per provider rather than an aggregation: the
    # provider list is single digits, and this keeps the same code path working
    # against every Mongo-alike the tests use.
    for provider in sorted(providers):
        latest = metrics.find_one(
            # Published rows only: this feeds the lane-health invariants, and a
            # lane producing nothing but long-profile rows has a dead published
            # series even though its worker is demonstrably alive. Process
            # aliveness is liveness_status's question, not this one.
            {"provider": provider, **published_profile_filter()},
            {"gen_ts": 1, "run_ts": 1},
            sort=[("gen_ts", -1), ("run_ts", -1)],
        )
        stamp = _as_utc((latest or {}).get("gen_ts") or (latest or {}).get("run_ts"))
        result[provider] = {
            "latest_completed_at": stamp.isoformat() if stamp else None,
            "age_seconds": int((now - stamp).total_seconds()) if stamp else None,
        }
    return result


def worker_heartbeat_max_age_seconds() -> int:
    return int(os.getenv("BENCHMARK_WORKER_HEARTBEAT_MAX_AGE_SECONDS", "300"))


def control_heartbeat_max_age_seconds() -> int:
    return int(os.getenv("BENCHMARK_CONTROL_HEARTBEAT_MAX_AGE_SECONDS", "900"))


def stale_worker_lanes(
    db: Database,
    *,
    providers: list[str],
    now: datetime,
    max_age_seconds: int,
) -> list[dict[str, Any]]:
    """Provider lanes whose worker threads have stopped checking in.

    This is the signal the completion-age check cannot give. Workers heartbeat
    on every poll, including an idle one, so a thread that dies goes quiet
    within a poll interval whether or not there was work to do — which is the
    failure that ran for eight days with the process up and `RestartCount=0`.
    Completion age cannot distinguish that from a runner correctly idle because
    nothing is stale yet.
    """
    collection = db[heartbeats_collection_name()]
    stale: list[dict[str, Any]] = []
    for provider in providers:
        docs = collection.find({"_id": {"$regex": f"^worker:{re.escape(provider)}:"}}, {"updated_at": 1})
        freshest = max(
            (age for age in (_as_utc(doc.get("updated_at")) for doc in docs) if age is not None),
            default=None,
        )
        age = int((now - freshest).total_seconds()) if freshest else None
        if age is None or age > max_age_seconds:
            stale.append({"provider": provider, "age_seconds": age})
    return stale


def liveness_status(
    db: Database,
    *,
    providers: list[str] | None = None,
    now: datetime | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check that the scheduler loop and every configured worker lane are alive.

    Process health is answered entirely by heartbeats. Workers write one on
    every poll, including an idle poll, so this catches the original failure
    where worker threads died while the process stayed up. Benchmark completion
    is a product outcome, not process liveness: cadence can make an idle runner
    correct, and auth or billing failures are not repaired by restarting it.
    Provider progress remains in the payload for diagnosis and is enforced by
    the invariant monitor.
    """
    now = _as_utc(now or utcnow())
    scheduler_heartbeat = db[heartbeats_collection_name()].find_one({"_id": "scheduler"}, {"updated_at": 1})
    heartbeat_at = _as_utc((scheduler_heartbeat or {}).get("updated_at"))
    heartbeat_age = int((now - heartbeat_at).total_seconds()) if heartbeat_at else None
    heartbeat_limit = control_heartbeat_max_age_seconds()
    worker_limit = worker_heartbeat_max_age_seconds()
    stale_lanes = (
        stale_worker_lanes(db, providers=providers, now=now, max_age_seconds=worker_limit) if providers else []
    )

    details = {
        "scheduler_heartbeat_at": heartbeat_at.isoformat() if heartbeat_at else None,
        "scheduler_heartbeat_age_seconds": heartbeat_age,
        "worker_heartbeat_max_age_seconds": worker_limit,
        "stale_worker_lanes": stale_lanes,
        "providers": providers or [],
        "provider_progress": provider_progress(db, providers=providers, now=now) if providers else {},
    }
    if stale_lanes:
        details["reason"] = "worker threads have stopped checking in"
        return False, details
    if heartbeat_age is None or heartbeat_age > heartbeat_limit:
        details["reason"] = "scheduler heartbeat is stale"
        return False, details
    details["reason"] = "ok"
    return True, details
