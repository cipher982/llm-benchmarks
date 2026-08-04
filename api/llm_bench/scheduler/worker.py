from __future__ import annotations

import socket
import threading
import uuid
from datetime import datetime
from datetime import timezone

from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.runner import run_job_in_child


def worker_id(provider: str, slot: int) -> str:
    return f"{socket.gethostname()}:{provider}:{slot}:{uuid.uuid4().hex[:8]}"


def _safe_heartbeat(db, *, component: str, details: dict) -> None:
    try:
        health.heartbeat(db, component=component, details=details)
    except Exception as exc:
        print(f"Worker heartbeat error component={component}: {type(exc).__name__}: {exc}", flush=True)


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

            deadline = int(job.get("deadline_seconds") or policies.DEFAULT_DEADLINE_SECONDS)
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
                result = run_job_in_child(job, deadline_seconds=deadline)
                now = datetime.now(timezone.utc)
                model_id = str(job.get("model_id"))

                if result.status == "success":
                    if queue.mark_success(db, job_id=job["_id"], worker_id=wid, now=now):
                        if job.get("job_kind") != "smoke_hang":
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
