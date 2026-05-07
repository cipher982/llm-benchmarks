from __future__ import annotations

import signal
import threading
import time
from datetime import datetime
from datetime import timezone
from typing import Optional

import dotenv
import typer

from llm_bench.models_db import load_provider_models
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.reaper import run_reaper_loop
from llm_bench.scheduler.reaper import run_reaper_pass
from llm_bench.scheduler.runner import PROVIDER_MODULES
from llm_bench.scheduler.worker import start_provider_workers

dotenv.load_dotenv(".env")

app = typer.Typer(add_completion=False)


def _selected_providers(providers: str | None, provider_models: dict[str, list[str]]) -> list[str]:
    excluded = policies.excluded_providers()
    if not providers or providers.strip().lower() == "all":
        return sorted(provider for provider in provider_models if provider not in excluded)
    selected = [provider.strip() for provider in providers.split(",") if provider.strip()]
    return [provider for provider in selected if provider not in excluded]


def _worker_providers(providers: str | None, provider_models: dict[str, list[str]]) -> list[str]:
    if not providers or providers.strip().lower() == "all":
        excluded = policies.excluded_providers()
        return sorted(provider for provider in PROVIDER_MODULES if provider not in excluded)
    return _selected_providers(providers, provider_models)


def _freshness_priority(doc: dict, cadence_seconds: int) -> float:
    status = doc.get("freshness_status")
    if status == "never_run":
        return 1000.0
    staleness = doc.get("staleness_seconds")
    if not isinstance(staleness, (int, float)):
        return 0.0
    multiplier = 2.0 if status == "critical" else 1.0
    return multiplier * float(staleness) / max(1, cadence_seconds)


def scheduler_pass(*, providers: str | None, limit: int, cadence_seconds: int) -> int:
    _, db_name = mongo_env()
    client = mongo_client()
    now = datetime.now(timezone.utc)
    enqueued = 0
    try:
        db = client[db_name]
        provider_models = load_provider_models()
        selected = _selected_providers(providers, provider_models)
        health.refresh_all_model_docs(db, cadence_seconds=cadence_seconds, now=now)
        for provider in selected:
            models = provider_models.get(provider, [])[:limit]
            for model_id in models:
                doc = health.health_collection(db).find_one({"provider": provider, "model_id": model_id})
                if not doc:
                    continue
                if doc.get("freshness_status") not in {"stale", "critical", "never_run"}:
                    continue
                priority = _freshness_priority(doc, cadence_seconds)
                if queue.enqueue_scheduled_job(db, provider=provider, model_id=model_id, priority=priority, now=now):
                    enqueued += 1
        health.heartbeat(
            db,
            component="scheduler",
            details={"providers": selected, "enqueued": enqueued},
            now=now,
        )
        if enqueued:
            print(f"Scheduler enqueued {enqueued} stale jobs", flush=True)
        return enqueued
    finally:
        client.close()


def run_scheduler_loop(
    *,
    providers: str | None,
    limit: int,
    cadence_seconds: int,
    stop_event: threading.Event,
    tick_seconds: int = policies.SCHEDULER_TICK_SECONDS,
) -> None:
    print(f"Scheduler loop started tick={tick_seconds}s cadence={cadence_seconds}s providers={providers}", flush=True)
    while not stop_event.is_set():
        try:
            scheduler_pass(providers=providers, limit=limit, cadence_seconds=cadence_seconds)
        except Exception as exc:
            print(f"Scheduler loop error: {type(exc).__name__}: {exc}", flush=True)
        stop_event.wait(tick_seconds)


@app.command()
def daemon(
    providers: Optional[str] = typer.Option(
        "all",
        "--providers",
        help="Comma-separated list of providers, or 'all' for all enabled providers",
    ),
    limit: int = typer.Option(100, "--limit", help="Limit per-provider models"),
    fresh_minutes: int = typer.Option(
        policies.fresh_minutes(),
        "--fresh-minutes",
        help="Global sampling cadence in minutes",
    ),
    tick_seconds: int = typer.Option(
        policies.SCHEDULER_TICK_SECONDS,
        "--tick-seconds",
        help="Scheduler/reaper tick interval",
    ),
) -> None:
    """Run the Mongo-backed benchmark scheduler daemon."""
    cadence_seconds = policies.cadence_seconds(fresh_minutes)
    _, db_name = mongo_env()
    client = mongo_client()
    stop_event = threading.Event()

    def stop(signum, frame) -> None:
        print(f"Received signal {signum}; stopping scheduler", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        db = client[db_name]
        queue.ensure_indexes(db)
        health.ensure_indexes(db)
        backfilled = health.backfill_from_metrics(db, cadence_seconds=cadence_seconds)
        expired = run_reaper_pass(cadence_seconds=cadence_seconds)
        provider_models = load_provider_models()
        selected = _worker_providers(providers, provider_models)
        print(
            "Scheduler daemon starting " f"providers={selected} backfilled={backfilled} expired={expired}",
            flush=True,
        )
    finally:
        client.close()

    scheduler_thread = threading.Thread(
        target=run_scheduler_loop,
        kwargs={
            "providers": providers,
            "limit": limit,
            "cadence_seconds": cadence_seconds,
            "stop_event": stop_event,
            "tick_seconds": tick_seconds,
        },
        name="scheduler-loop",
        daemon=True,
    )
    reaper_thread = threading.Thread(
        target=run_reaper_loop,
        kwargs={"cadence_seconds": cadence_seconds, "stop_event": stop_event, "tick_seconds": tick_seconds},
        name="reaper-loop",
        daemon=True,
    )
    scheduler_thread.start()
    reaper_thread.start()
    workers = start_provider_workers(providers=selected, cadence_seconds=cadence_seconds, stop_event=stop_event)

    while not stop_event.is_set():
        time.sleep(1)

    scheduler_thread.join(timeout=10)
    reaper_thread.join(timeout=10)
    for worker in workers:
        worker.join(timeout=10)


@app.command("smoke-hang")
def smoke_hang(
    provider: str = typer.Option("openai", "--provider"),
    model: str = typer.Option("fake-hang", "--model"),
    seconds: int = typer.Option(300, "--seconds"),
    deadline_seconds: int = typer.Option(policies.DEFAULT_DEADLINE_SECONDS, "--deadline-seconds"),
) -> None:
    """Enqueue a fake hanging job for scheduler timeout smoke tests."""
    _, db_name = mongo_env()
    client = mongo_client()
    try:
        db = client[db_name]
        queue.ensure_indexes(db)
        created = queue.enqueue_smoke_hang_job(
            db,
            provider=provider,
            model_id=model,
            seconds=seconds,
            deadline_seconds=deadline_seconds,
        )
        status = "enqueued" if created else "already-active"
        print(f"{status} smoke hang job provider={provider} model={model} seconds={seconds}", flush=True)
    finally:
        client.close()


@app.command("enqueue")
def enqueue(
    provider: str = typer.Option(..., "--provider"),
    model: str = typer.Option(..., "--model"),
    priority: float = typer.Option(10_000, "--priority"),
    deadline_seconds: int = typer.Option(policies.DEFAULT_DEADLINE_SECONDS, "--deadline-seconds"),
    max_attempts: int = typer.Option(policies.DEFAULT_MAX_ATTEMPTS, "--max-attempts"),
) -> None:
    """Enqueue a manual benchmark job in bench_jobs."""
    _, db_name = mongo_env()
    client = mongo_client()
    try:
        db = client[db_name]
        queue.ensure_indexes(db)
        job_id = queue.enqueue_manual_job(
            db,
            provider=provider,
            model_id=model,
            priority=priority,
            deadline_seconds=deadline_seconds,
            max_attempts=max_attempts,
        )
        print(f"enqueued manual benchmark job id={job_id} provider={provider} model={model}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    app()
