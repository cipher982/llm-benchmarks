from __future__ import annotations

import threading

from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.runner import log_error_mongo


def run_reaper_pass(*, cadence_seconds: int) -> int:
    _, db_name = mongo_env()
    client = mongo_client()
    try:
        db = client[db_name]
        expired = queue.expire_orphaned_running(db)
        for job in expired:
            log_error_mongo(
                provider=str(job.get("provider")),
                model_id=str(job.get("model_id")),
                stage="timeout",
                message="lease expired",
                exc_type="TimeoutError",
                client=client,
                db_name=db_name,
            )
            health.record_error(
                db,
                provider=str(job.get("provider")),
                model_id=str(job.get("model_id")),
                cadence_seconds=cadence_seconds,
                error_kind="timeout",
                error_message="lease expired",
            )
        health.heartbeat(db, component="reaper", details={"expired": len(expired)})
        if expired:
            print(f"Reaper expired {len(expired)} running jobs", flush=True)
        return len(expired)
    finally:
        client.close()


def run_reaper_loop(
    *,
    cadence_seconds: int,
    stop_event: threading.Event,
    tick_seconds: int = policies.SCHEDULER_TICK_SECONDS,
) -> None:
    print("Reaper loop started", flush=True)
    while not stop_event.is_set():
        try:
            run_reaper_pass(cadence_seconds=cadence_seconds)
        except Exception as exc:
            print(f"Reaper loop error: {type(exc).__name__}: {exc}", flush=True)
        stop_event.wait(tick_seconds)
