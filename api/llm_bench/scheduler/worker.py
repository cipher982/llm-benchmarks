from __future__ import annotations

import socket
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from datetime import timezone

from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler import runner
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.mongo import route_revocation_generation
from llm_bench.scheduler.routing import cooldown_route_snapshot
from llm_bench.scheduler.runner import run_job_in_child

# Source-provider workers are separate lanes, but routed work shares one
# OpenRouter quota. This process-wide gate keeps those source lanes from
# multiplying concurrent OpenRouter requests beyond the shared budget.
OPENROUTER_GATE = threading.BoundedSemaphore(policies.openrouter_concurrency())


def worker_id(provider: str, slot: int) -> str:
    return f"{socket.gethostname()}:{provider}:{slot}:{uuid.uuid4().hex[:8]}"


def _safe_heartbeat(db, *, component: str, details: dict) -> None:
    try:
        health.heartbeat(db, component=component, details=details)
    except Exception as exc:
        print(f"Worker heartbeat error component={component}: {type(exc).__name__}: {exc}", flush=True)


def acquire_openrouter_slot(timeout_seconds: float) -> bool:
    """Acquire the process-wide OpenRouter quota slot for one attempt."""

    return OPENROUTER_GATE.acquire(timeout=max(0.0, float(timeout_seconds)))


def persist_route_cooldown(db, *, job: dict, result: runner.RunnerResult, now: datetime) -> None:
    """Persist a route failure as cooldown while preserving the source lane."""

    snapshot = job.get("route_snapshot")
    failure_reason = result.fallback_reason or result.error_message
    quota_fallback = result.fallback_reason == "openrouter-quota-unavailable"
    if (not result.route_attempted and not quota_fallback) or not failure_reason or not isinstance(snapshot, dict):
        return
    cooled = cooldown_route_snapshot(snapshot, failure_reason=failure_reason, now=now)
    key = {
        "source_provider": cooled.get("source_provider", job.get("provider")),
        "source_model_id": cooled.get("source_model_id", job.get("model_id")),
        "route_snapshot_at": cooled.get("route_snapshot_at"),
    }
    if job.get("_id") is not None:
        # A retry uses the job's frozen snapshot, not the route-decision
        # collection. Freeze the cooldown onto that job before it is queued
        # again so a transient route failure cannot immediately re-enter OR.
        queue.jobs_collection(db).update_one(
            {
                "_id": job["_id"],
                "status": "running",
                "route_snapshot.state": "active",
            },
            {"$set": {"route_snapshot": cooled}},
        )
    # The job snapshot is the retry safety boundary. Update it first, then
    # mirror the same state into the operator-facing route collection.
    db[route_decisions_collection_name()].update_one(key, {"$set": cooled}, upsert=True)


def apply_route_revocation_guard(db, job: dict) -> dict:
    """Re-check the monotonic revocation generation immediately before dispatch."""

    snapshot = job.get("route_snapshot")
    if not isinstance(snapshot, dict):
        return job
    provider = str(job.get("provider") or "")
    model_id = str(job.get("model_id") or "")
    current = route_revocation_generation(db, provider=provider, model_id=model_id)
    try:
        queued = int(snapshot.get("route_revocation_generation", 0))
    except (TypeError, ValueError):
        queued = -1
    if current <= queued:
        return job
    updated = dict(job)
    revoked = deepcopy(snapshot)
    revoked.update(
        {
            "state": "revoked",
            "transport_provider": "direct",
            "route_policy": "direct",
            "route_revocation_generation": current,
            "route_revocation_reason": "route-revoked-before-dispatch",
        }
    )
    updated["route_snapshot"] = revoked
    updated["route_fallback_reason"] = "route-revoked-before-dispatch"
    return updated


def direct_fallback_job(job: dict, *, reason: str) -> dict:
    """Return a copy that cannot resolve back to OpenRouter."""

    fallback = dict(job)
    snapshot = deepcopy(job.get("route_snapshot") or {})
    snapshot.update(
        {
            "state": "revoked",
            "transport_provider": "direct",
            "route_policy": "direct",
            "route_revocation_reason": reason,
        }
    )
    fallback["route_snapshot"] = snapshot
    fallback["route_fallback_reason"] = reason
    return fallback


def run_worker_loop(
    *,
    provider: str,
    slot: int,
    cadence_seconds: int,
    stop_event: threading.Event,
    idle_sleep_seconds: float = 2.0,
) -> None:
    wid = worker_id(provider, slot)
    _, db_name = mongo_env()
    client = mongo_client()
    db = client[db_name]
    print(f"Worker started provider={provider} slot={slot} worker_id={wid}", flush=True)
    try:
        while not stop_event.is_set():
            try:
                job = queue.claim_next_job(db, provider=provider, worker_id=wid)
            except Exception as exc:
                # PyMongo can raise when Mongo restarts or a socket is reset.
                # Keep this worker alive so the next iteration can reconnect.
                print(
                    f"Worker queue error provider={provider} slot={slot}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                stop_event.wait(idle_sleep_seconds)
                continue
            if not job:
                _safe_heartbeat(
                    db,
                    component=f"worker:{provider}:{slot}",
                    details={"idle": True, "worker_id": wid},
                )
                stop_event.wait(idle_sleep_seconds)
                continue

            job = apply_route_revocation_guard(db, job)

            deadline = float(job.get("deadline_seconds") or policies.DEFAULT_DEADLINE_SECONDS)
            attempt_started = time.monotonic()
            print(
                f"Claimed job {job['_id']} provider={provider} model={job.get('model_id')} "
                f"attempt={job.get('attempt')} deadline={deadline}s",
                flush=True,
            )
            _safe_heartbeat(
                db,
                component=f"worker:{provider}:{slot}",
                details={"idle": False, "worker_id": wid, "job_id": str(job["_id"])},
            )
            try:
                if runner.job_requires_openrouter(job):
                    remaining = max(0.0, deadline - (time.monotonic() - attempt_started))
                    # Keep a bounded direct lane reserve. A quota stall must
                    # not turn a valid source provider into a lost sample.
                    reserve = max(10.0, deadline * 0.25)
                    acquire_wait = max(0.0, remaining - reserve)
                    acquired = acquire_openrouter_slot(acquire_wait)
                    if not acquired:
                        remaining = max(0.0, deadline - (time.monotonic() - attempt_started))
                        if remaining > 0:
                            result = run_job_in_child(
                                direct_fallback_job(job, reason="openrouter-quota-unavailable"),
                                deadline_seconds=remaining,
                            )
                        else:
                            result = runner.RunnerResult(
                                status="timeout",
                                error_kind="timeout",
                                error_message="direct fallback reserve expired before dispatch",
                                route_attempted=True,
                                transport_provider="direct",
                                fallback_reason="openrouter-quota-unavailable",
                            )
                    else:
                        try:
                            remaining = max(0.0, deadline - (time.monotonic() - attempt_started))
                            if remaining <= 0:
                                result = runner.RunnerResult(
                                    status="timeout",
                                    error_kind="timeout",
                                    error_message="job deadline expired before OpenRouter child start",
                                    route_attempted=True,
                                    transport_provider="openrouter",
                                )
                            else:
                                result = run_job_in_child(job, deadline_seconds=remaining)
                        finally:
                            OPENROUTER_GATE.release()
                else:
                    remaining = max(0.0, deadline - (time.monotonic() - attempt_started))
                    if remaining <= 0:
                        result = runner.RunnerResult(
                            status="timeout",
                            error_kind="timeout",
                            error_message="job deadline expired before child start",
                        )
                    else:
                        result = run_job_in_child(job, deadline_seconds=remaining)
                now = datetime.now(timezone.utc)
                model_id = str(job.get("model_id"))
                persist_route_cooldown(db, job=job, result=result, now=now)

                if result.status == "success":
                    if queue.mark_success(db, job_id=job["_id"], worker_id=wid, now=now):
                        # A probe or shadow sample is not evidence that this
                        # model is being measured for the site. Counting it
                        # would make freshness reflect what we tried, not what
                        # we publish.
                        publishes = result.sample_role not in runner.NON_PUBLISHING_ROLES
                        if job.get("job_kind") != "smoke_hang" and publishes:
                            health.record_success(
                                db,
                                provider=provider,
                                model_id=model_id,
                                cadence_seconds=cadence_seconds,
                                now=now,
                            )
                    print(f"Completed job {job['_id']} status=success", flush=True)
                    continue

                error_kind = result.error_kind or "unknown"
                error_message = result.error_message or result.status
                next_status = queue.mark_failure(
                    db,
                    job=job,
                    error_kind=error_kind,
                    error_message=error_message,
                    worker_id=wid,
                    now=now,
                )
                if next_status:
                    health.record_error(
                        db,
                        provider=provider,
                        model_id=model_id,
                        cadence_seconds=cadence_seconds,
                        error_kind=error_kind,
                        error_message=error_message,
                        now=now,
                    )
                print(f"Completed job {job['_id']} status={result.status} next_status={next_status}", flush=True)
            except Exception as exc:
                # A database outage during result bookkeeping must not kill
                # the worker. The lease reaper will recover a claimed job.
                print(
                    f"Worker processing error provider={provider} model={job.get('model_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
    finally:
        client.close()


def start_provider_workers(
    *,
    providers: list[str],
    cadence_seconds: int,
    stop_event: threading.Event,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    for provider in providers:
        for slot in range(policies.provider_concurrency(provider)):
            thread = threading.Thread(
                target=run_worker_loop,
                kwargs={
                    "provider": provider,
                    "slot": slot,
                    "cadence_seconds": cadence_seconds,
                    "stop_event": stop_event,
                },
                name=f"worker-{provider}-{slot}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
    return threads
