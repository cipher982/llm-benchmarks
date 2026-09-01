from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from datetime import datetime
from datetime import timezone
from typing import Optional

import dotenv
import typer

from llm_bench.models_db import load_provider_models
from llm_bench.ops import admission
from llm_bench.ops import desired_set
from llm_bench.ops import endpoint_discovery
from llm_bench.ops import endpoint_retirement
from llm_bench.ops import identity
from llm_bench.ops import invariants
from llm_bench.ops import llm_error_classifier
from llm_bench.ops import model_naming
from llm_bench.ops import openrouter_discovery
from llm_bench.ops import reconciler
from llm_bench.ops import route_renewal
from llm_bench.ops import vertex_discovery
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.mongo import route_snapshot
from llm_bench.scheduler.reaper import run_reaper_loop
from llm_bench.scheduler.reaper import run_reaper_pass
from llm_bench.scheduler.routing import endpoint_route_snapshot
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


def _worker_providers(
    providers: str | None,
    provider_models: dict[str, list[str]],
    *,
    probe_providers: set[str] | None = None,
) -> list[str]:
    """Lanes to start, derived from enabled models and probe candidates.

    This used to start every provider in PROVIDER_MODULES, so lanes existed for
    providers with nothing enabled. That makes "this provider wrote no metric"
    ambiguous — idle by design and dead are indistinguishable — which any
    per-provider liveness check depends on being able to tell apart.

    A provider with enabled models or admission candidates but no adapter cannot
    be worked at all, so it is surfaced rather than dropped silently.
    """
    selected = _selected_providers(providers, provider_models)
    requested = (
        None
        if not providers or providers.strip().lower() == "all"
        else {item.strip() for item in providers.split(",") if item.strip()}
    )
    if probe_providers:
        excluded = policies.excluded_providers()
        selected = sorted(
            set(selected)
            | {
                provider
                for provider in probe_providers
                if provider not in excluded and (requested is None or provider in requested)
            }
        )
    # OpenRouter discovery can add the first candidate after a transient
    # catalogue failure. Keep its adapter lane alive under `all` so a later
    # successful refresh never strands probe jobs until a process restart.
    if (not providers or providers.strip().lower() == "all") and "openrouter" in PROVIDER_MODULES:
        selected = sorted(set(selected) | {"openrouter"})
    unroutable = [provider for provider in selected if provider not in PROVIDER_MODULES]
    if unroutable:
        print(
            f"WARNING: enabled models exist for {', '.join(sorted(unroutable))} "
            "but no runner adapter is registered; those models cannot be benchmarked",
            flush=True,
        )
    return [provider for provider in selected if provider in PROVIDER_MODULES]


def _probe_providers(db) -> set[str]:
    """Return provider lanes needed by disabled admission candidates."""

    return set(
        db[models_collection_name()].distinct(
            "provider",
            {"status": admission.CANDIDATE_STATUS},
        )
    )


def _freshness_priority(doc: dict, cadence_seconds: int) -> float:
    status = doc.get("freshness_status")
    if status == "never_run":
        return 1000.0
    staleness = doc.get("staleness_seconds")
    if not isinstance(staleness, (int, float)):
        return 0.0
    multiplier = 2.0 if status == "critical" else 1.0
    return multiplier * float(staleness) / max(1, cadence_seconds)


def _as_utc(value) -> datetime:
    """Mongo hands back naive datetimes; comparing one to an aware `now` raises."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _endpoint_candidates(db, *, provider: str, now: datetime | None = None) -> list[tuple[float, str, str, int]]:
    """Endpoint candidates under the provider-count rotating tier policy.

    A model earns one scheduling opportunity per tier interval. Its endpoint
    rows rotate oldest-first, so provider-family count controls how often the
    model page changes without multiplying every opportunity into a full sweep.
    """
    now = now or datetime.now(timezone.utc)
    enabled_models = set(
        db[models_collection_name()].distinct(
            "model_id",
            {"provider": provider, "enabled": True, "deprecated": {"$ne": True}},
        )
    )
    rows_by_model: dict[str, list[dict]] = {}
    endpoints = db[endpoint_discovery.endpoints_collection_name()].find(
        {"enabled": True, "model_id": {"$in": sorted(enabled_models)}},
        {"model_id": 1, "endpoint_tag": 1, "provider_canonical": 1},
    )
    for row in endpoints:
        model_id = row.get("model_id")
        tag = row.get("endpoint_tag")
        if model_id and tag:
            rows_by_model.setdefault(model_id, []).append(row)

    active_models = set(
        queue.jobs_collection(db).distinct(
            "model_id",
            {
                "provider": provider,
                "status": {"$in": sorted(queue.ACTIVE_STATUSES)},
            },
        )
    )
    health_coll = health.health_collection(db)
    candidates: list[tuple[float, str, str, int]] = []
    for model_id, model_rows in rows_by_model.items():
        provider_count = len(
            {
                row.get("provider_canonical") or endpoint_discovery.provider_canonical(row["endpoint_tag"])
                for row in model_rows
            }
        )
        endpoint_count = len(model_rows)
        model_interval = policies.endpoint_tier_interval_seconds(provider_count)
        endpoint_cadence = model_interval * endpoint_count
        model_last_attempt: datetime | None = None
        endpoint_docs: list[tuple[str, dict]] = []
        for row in model_rows:
            tag = row["endpoint_tag"]
            query = health.health_filter(provider, model_id, tag)
            doc = health_coll.find_one(query)
            if not doc:
                try:
                    health.refresh_model_health_doc(
                        db,
                        provider=provider,
                        model_id=model_id,
                        endpoint_tag=tag,
                        enabled=True,
                        cadence_seconds=endpoint_cadence,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"endpoint health registration failed {model_id}:{tag}: {exc}", flush=True)
                    continue
                doc = health_coll.find_one(query)
            if not doc:
                continue
            health_updates = {}
            if doc.get("cadence_seconds") != endpoint_cadence:
                health_updates["cadence_seconds"] = endpoint_cadence
            if doc.get("endpoint_rotation_policy_version") != policies.ENDPOINT_ROTATION_POLICY_VERSION:
                health_updates.update(
                    {
                        "endpoint_rotation_policy_version": policies.ENDPOINT_ROTATION_POLICY_VERSION,
                        "endpoint_rotation_policy_started_at": now,
                    }
                )
            if health_updates:
                health_coll.update_one(query, {"$set": health_updates})
                doc.update(health_updates)
            endpoint_docs.append((tag, doc))
            attempted_at = doc.get("last_attempt_at") or doc.get("last_success_at")
            if attempted_at is not None:
                attempted_at = _as_utc(attempted_at)
                if model_last_attempt is None or attempted_at > model_last_attempt:
                    model_last_attempt = attempted_at

        if model_id in active_models:
            continue

        if model_last_attempt is not None:
            model_age = max(0.0, (now - model_last_attempt).total_seconds())
            if model_age < model_interval:
                continue
            model_priority = model_age / model_interval
        else:
            model_priority = 1000.0

        for tag, doc in endpoint_docs:
            last_success = doc.get("last_success_at")
            endpoint_priority = (
                1000.0
                if last_success is None
                else max(0.0, (now - _as_utc(last_success)).total_seconds()) / endpoint_cadence
            )
            priority = model_priority + min(endpoint_priority, 1000.0) / 1000.0
            candidates.append((priority, model_id, tag, endpoint_cadence))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates


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
            endpoint_mode = provider == "openrouter" and policies.endpoint_targets_enabled()
            endpoint_model_ids = (
                set(db[endpoint_discovery.endpoints_collection_name()].distinct("model_id", {"enabled": True}))
                if endpoint_mode
                else set()
            )
            # Every model is considered; the cap bounds how much work one pass
            # creates, not which models are allowed to exist.
            #
            # This used to slice the model list — `models[:limit]` — which with
            # 112 DeepInfra models and a limit of 100 meant twelve of them were
            # never scheduled at all. Always the same twelve, because the order
            # is stable, and one more each time a model was added. They had run
            # successfully months earlier and then simply stopped, with no
            # error, no dead letter and nothing disabled. Admission adds models
            # continuously, so the slice was a coverage leak that widened on its
            # own.
            candidates = []
            for model_id in provider_models.get(provider, []):
                if model_id in endpoint_model_ids:
                    continue
                # Model-level freshness only. Endpoint health records live in
                # the same collection and would otherwise satisfy this lookup,
                # letting one endpoint's run stand in for the model's.
                doc = health.health_collection(db).find_one(health.health_filter(provider, model_id, None))
                if not doc:
                    continue
                if doc.get("freshness_status") not in {"stale", "critical", "never_run"}:
                    continue
                candidates.append((_freshness_priority(doc, cadence_seconds), model_id))

            # Stalest first, so a pass that cannot cover everything covers what
            # has waited longest. A model starved by one pass is at the front of
            # the next, which a fixed slice could never do.
            candidates.sort(key=lambda item: -item[0])
            # A dead-lettered job is still stale, but enqueue_scheduled_job
            # deliberately refuses to replace it. Do not let those rows spend
            # the whole per-pass allowance and starve models later in the list.
            # The limit is for jobs actually created, not candidates inspected.
            created_for_provider = 0
            for priority, model_id in candidates:
                if created_for_provider >= limit:
                    break
                if queue.enqueue_scheduled_job(
                    db,
                    provider=provider,
                    model_id=model_id,
                    priority=priority,
                    now=now,
                    route_snapshot=route_snapshot(db, provider=provider, model_id=model_id),
                ):
                    enqueued += 1
                    created_for_provider += 1

            # Endpoint targets are the unit the site publishes. They share the
            # OpenRouter lane, so they get their own bound rather than
            # competing for the model allowance.
            if endpoint_mode:
                endpoint_budget = policies.endpoint_targets_per_pass()
                created_endpoints = 0
                catalogue = {
                    (r["model_id"], r["endpoint_tag"]): r
                    for r in db[endpoint_discovery.endpoints_collection_name()].find(
                        {"enabled": True},
                        {"model_id": 1, "endpoint_tag": 1, "quantization": 1, "provider_name": 1},
                    )
                }
                scheduled_endpoint_models: set[str] = set()
                for priority, model_id, tag, endpoint_cadence in _endpoint_candidates(db, provider=provider, now=now):
                    if model_id in scheduled_endpoint_models:
                        continue
                    if created_endpoints >= endpoint_budget:
                        break
                    if queue.enqueue_scheduled_job(
                        db,
                        provider=provider,
                        model_id=model_id,
                        priority=priority,
                        now=now,
                        route_snapshot=endpoint_route_snapshot(
                            provider,
                            model_id,
                            endpoint_tag=tag,
                            provider_canonical=endpoint_discovery.provider_canonical(tag),
                            quantization=(catalogue.get((model_id, tag)) or {}).get("quantization"),
                            provider_display=(catalogue.get((model_id, tag)) or {}).get("provider_name"),
                            now=now,
                        ),
                        endpoint_tag=tag,
                        cadence_seconds=endpoint_cadence,
                    ):
                        enqueued += 1
                        scheduled_endpoint_models.add(model_id)
                        created_endpoints += 1
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


def run_liveness_watchdog(
    *,
    providers: list[str],
    stop_event: threading.Event,
    startup_grace_seconds: int,
    tick_seconds: int,
    exit_process=os._exit,
) -> None:
    """Restart the container when the scheduler or a worker thread stops."""
    started_at = time.monotonic()
    while not stop_event.wait(max(5, min(tick_seconds, 60))):
        if time.monotonic() - started_at < startup_grace_seconds:
            continue
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                healthy, details = health.liveness_status(
                    client[db_name],
                    providers=providers,
                )
            finally:
                client.close()
        except Exception as exc:
            healthy = False
            details = {"reason": f"watchdog database check failed: {type(exc).__name__}: {exc}"}
        if not healthy:
            print(f"Liveness watchdog stopping daemon: {details}", flush=True)
            exit_process(1)


def run_invariants_loop(
    *,
    stop_event: threading.Event,
    snapshot_interval_seconds: int,
    check_interval_seconds: int,
    cadence_seconds: int,
) -> None:
    """Capture desired-set snapshots and evaluate invariants on a schedule.

    Deliberately never acts on what it finds. Results land in bench_check_runs,
    and the layer above — a Sauron job reading that ledger — is what alerts,
    including when this loop stops producing rows at all. Each layer watches a
    different failure domain: this one watches the pipeline, Sauron watches
    this, and the external dead man watches Sauron.
    """
    last_snapshot = 0.0
    last_check = 0.0
    while not stop_event.wait(30):
        now = time.monotonic()
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                db = client[db_name]
                if now - last_snapshot >= snapshot_interval_seconds:
                    captured = desired_set.capture(db)
                    last_snapshot = now
                    print(f"Desired-set snapshot: {captured.model_count} models", flush=True)
                if now - last_check >= check_interval_seconds:
                    results = invariants.evaluate(db, cadence_seconds=cadence_seconds)
                    last_check = now
                    failed = [r for r in results if not r.ok]
                    if failed:
                        print(
                            "Invariant violations: " + "; ".join(r.summary for r in failed),
                            flush=True,
                        )
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            # Never take the daemon down over a check; the missing ledger rows
            # are themselves the signal that this loop stopped working.
            print(f"Invariant loop error: {type(exc).__name__}: {exc}", flush=True)


def run_admission_loop(*, stop_event: threading.Event, interval_seconds: int) -> None:
    """Register candidates, probe them, and promote what keeps working.

    Paced at roughly the spacing admission requires between samples. Running it
    faster would not promote anything sooner — two calls minutes apart count as
    one observation — it would only spend more.
    """
    while not stop_event.wait(interval_seconds):
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                db = client[db_name]
                report = admission.run_admission_pass(db)
                print(f"Admission pass: {report.summary()}", flush=True)
                for subject in report.promoted:
                    print(f"  promoted {subject}", flush=True)
                for subject, reason in report.rejected:
                    print(f"  rejected {subject}: {reason}", flush=True)
                # Immediately after admission, not on an independent timer. A
                # model admitted here has no readable name until this runs, and
                # a separate loop would leave newly promoted models rendering as
                # raw ids for however long the two schedules happened to differ.
                naming = model_naming.apply_names(db, apply=True)
                if naming.get("to_change"):
                    print(f"Naming pass: renamed {naming['to_change']} ({naming['by_source']})", flush=True)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Admission loop error: {type(exc).__name__}: {exc}", flush=True)


def run_classifier_loop(*, stop_event: threading.Event, interval_seconds: int) -> None:
    """Resolve error fingerprints the deterministic taxonomy could not.

    `classify_error` keys on HTTP status and returns `unknown` for everything
    else, on the understanding that a later pass resolves it. That pass existed
    but had no caller, so `unknown` was terminal rather than provisional and 946
    fingerprints accumulated in it — including models the provider had deleted
    and models that simply could not be measured in 64 tokens, which need
    opposite responses.

    Fingerprints, not rows, so a few hundred calls cover hundreds of thousands
    of errors. Hourly is far more often than new fingerprints appear; the cost
    of a pass with nothing to do is one Mongo query.
    """
    while not stop_event.wait(interval_seconds):
        try:
            stats = asyncio.run(
                llm_error_classifier.classify_unclassified_rollups(
                    max_rollups=int(os.getenv("BENCHMARK_CLASSIFIER_MAX_ROLLUPS", "200"))
                )
            )
            if stats.get("updated"):
                print(f"Classifier resolved {stats['updated']} error fingerprints", flush=True)
            # A verdict nothing reads is not a verdict. Retry policy and the
            # catalogue tools read the job's own field, so the resolved kind is
            # copied onto the rows and jobs that actually decide with it.
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                moved = llm_error_classifier.propagate_classifications(client, db_name)
                if moved["errors_updated"] or moved["jobs_updated"]:
                    print(
                        f"Classifier propagated: {moved['errors_updated']} error rows, "
                        f"{moved['jobs_updated']} dead letters",
                        flush=True,
                    )
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            # Never kill the daemon over this. An unclassified error is a worse
            # report, not a broken benchmark.
            print(f"Classifier loop error: {type(exc).__name__}: {exc}", flush=True)


def run_reconciler_loop(*, stop_event: threading.Event, interval_seconds: int) -> None:
    """Keep chart lines from splitting as the catalogue changes.

    Two providers share a chart line only when their display names match
    character for character. `claude-haiku-4.5` and `claude-haiku-4-5` were the
    same model at three providers drawn as two lines, and nothing noticed for
    months because both lines looked fine on their own.

    Every model that arrives is a new chance to diverge, and admission adds them
    continuously. Running this once by hand fixed the backlog and would have let
    it rebuild from zero — which is how the site got into that state the first
    time.

    Order matters. Resolve identities for endpoints that have none, consolidate
    groups that arrival order split under two names, then unify display names
    inside each group. Unifying before consolidating would leave the two halves
    of a split group with different names and nothing to join them.
    """
    while not stop_event.wait(interval_seconds):
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                db = client[db_name]
                resolved = reconciler.resolve_missing_identities(db, call_llm=identity.call_openrouter)
                if resolved:
                    print(f"Identity resolved {len(resolved)} new endpoints", flush=True)

                merges = reconciler.consolidate(db, dry_run=False)
                for merge in merges:
                    print(f"Consolidated {merge['absorb']} into {merge['keep']}", flush=True)

                # Report only. `model_naming` is the single writer of
                # display_name now, and it already achieves what this pass was
                # for: an endpoint inherits its label from its identity
                # sibling, so a group agrees by construction rather than by
                # being rewritten afterwards.
                #
                # Left writing, the two disagree and thrash. 22 identity groups
                # hold two or more endpoints from one provider; `model_naming`
                # splits those to keep names unique within a provider, and this
                # pass merges them back to the group's majority name. Every
                # admission cycle would split, every reconciler cycle would
                # merge, forever, with the chart label alternating.
                for change in reconciler.unify_display_names(db, dry_run=True):
                    print(
                        f"Naming drift: {change['provider']}/{change['model_id']} "
                        f"differs from its identity group ({change['to']})",
                        flush=True,
                    )
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            # Naming drift is a display problem. It must never be able to stop
            # the machine from measuring.
            print(f"Reconciler loop error: {type(exc).__name__}: {exc}", flush=True)


def run_vertex_discovery_loop(*, stop_event: threading.Event, interval_seconds: int) -> None:
    """Keep Vertex's catalogue current from inside the process that has its keys.

    Every other provider is discovered by a Sauron job. Sauron holds no GCP
    credentials, so Vertex had no discovery at all and
    `discovery_completed_recently` was permanently red for it. This daemon
    already authenticates to Vertex in order to benchmark it.
    """
    while not stop_event.wait(interval_seconds):
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                result = vertex_discovery.refresh_catalog(client[db_name])
                print(f"Vertex discovery: {result}", flush=True)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Vertex discovery loop error: {type(exc).__name__}: {exc}", flush=True)


def run_openrouter_discovery_loop(*, stop_event: threading.Event, interval_seconds: int) -> None:
    """Keep the OpenRouter-native catalogue flowing into admission."""

    while not stop_event.wait(interval_seconds):
        try:
            _, db_name = mongo_env()
            client = mongo_client()
            try:
                db = client[db_name]
                result = openrouter_discovery.refresh_catalog(db)
                print(f"OpenRouter discovery: {result}", flush=True)
                # The endpoint catalogue is what the site will schedule from:
                # one row per (model, endpoint tag), because a model is served
                # by several deployments at different speeds and quantizations.
                enabled_ids = [
                    str(row["model_id"])
                    for row in db[models_collection_name()].find(
                        {"enabled": True, "provider": "openrouter"}, {"model_id": 1}
                    )
                    if row.get("model_id")
                ]
                endpoint_result = endpoint_discovery.refresh_endpoints(db, model_ids=enabled_ids)
                print(f"Endpoint discovery: {endpoint_result}", flush=True)

                # Retirement runs with discovery because both decide what the
                # catalogue contains, and both need the same ordering: restore
                # first, then retire. Restoring afterwards would re-enable an
                # endpoint this very pass had just retired under the current
                # protocol.
                restored = endpoint_retirement.restore_stale_protocol_retirements(db)
                if restored:
                    print(f"Endpoint retirement: restored {len(restored)} on protocol change", flush=True)
                retired = endpoint_retirement.retire_unmeasurable_endpoints(db)
                if retired:
                    print(f"Endpoint retirement: retired {len(retired)} unmeasurable endpoints", flush=True)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            print(f"OpenRouter discovery loop error: {type(exc).__name__}: {exc}", flush=True)


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
        desired_set.ensure_indexes(db)
        backfilled = health.backfill_from_metrics(db, cadence_seconds=cadence_seconds)
        expired = run_reaper_pass(cadence_seconds=cadence_seconds)
        discovery_result = openrouter_discovery.refresh_catalog(db)
        admission_report = admission.run_admission_pass(db)
        provider_models = load_provider_models()
        selected = _worker_providers(providers, provider_models, probe_providers=_probe_providers(db))
        print(
            "Scheduler daemon starting "
            f"providers={selected} backfilled={backfilled} expired={expired} "
            f"openrouter_discovery={discovery_result} admission={admission_report.summary()}",
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
    liveness_thread = threading.Thread(
        target=run_liveness_watchdog,
        kwargs={
            "providers": selected,
            "stop_event": stop_event,
            "startup_grace_seconds": int(os.getenv("BENCHMARK_LIVENESS_STARTUP_GRACE_SECONDS", "600")),
            "tick_seconds": tick_seconds,
        },
        name="liveness-watchdog",
        daemon=True,
    )
    invariants_thread = threading.Thread(
        target=run_invariants_loop,
        kwargs={
            "stop_event": stop_event,
            "snapshot_interval_seconds": int(os.getenv("BENCHMARK_SNAPSHOT_INTERVAL_SECONDS", "3600")),
            "check_interval_seconds": int(os.getenv("BENCHMARK_INVARIANT_INTERVAL_SECONDS", "900")),
            "cadence_seconds": cadence_seconds,
        },
        name="invariants-loop",
        daemon=True,
    )
    admission_thread = threading.Thread(
        target=run_admission_loop,
        kwargs={
            "stop_event": stop_event,
            "interval_seconds": int(os.getenv("BENCHMARK_ADMISSION_INTERVAL_SECONDS", "7200")),
        },
        name="admission-loop",
        daemon=True,
    )
    classifier_thread = threading.Thread(
        target=run_classifier_loop,
        kwargs={
            "stop_event": stop_event,
            "interval_seconds": int(os.getenv("BENCHMARK_CLASSIFIER_INTERVAL_SECONDS", "3600")),
        },
        name="classifier-loop",
        daemon=True,
    )
    reconciler_thread = threading.Thread(
        target=run_reconciler_loop,
        kwargs={
            "stop_event": stop_event,
            # Naming drift only matters when the catalogue changes, and each pass
            # costs LLM calls, so this is deliberately slow.
            "interval_seconds": int(os.getenv("BENCHMARK_RECONCILER_INTERVAL_SECONDS", "21600")),
        },
        name="reconciler-loop",
        daemon=True,
    )
    route_renewal_thread = threading.Thread(
        target=route_renewal.run_renewal_loop,
        kwargs={
            "stop_event": stop_event,
            # Well inside the renewal window, so a route is renewed at most an
            # hour after it becomes due rather than a window later.
            "interval_seconds": int(os.getenv("BENCHMARK_ROUTE_RENEW_INTERVAL_SECONDS", "3600")),
        },
        name="route-renewal-loop",
        daemon=True,
    )
    vertex_discovery_thread = threading.Thread(
        target=run_vertex_discovery_loop,
        kwargs={
            "stop_event": stop_event,
            # Well inside the 48h the discovery invariant allows, so one failed
            # pass is not a violation.
            "interval_seconds": int(os.getenv("BENCHMARK_VERTEX_DISCOVERY_INTERVAL_SECONDS", "43200")),
        },
        name="vertex-discovery-loop",
        daemon=True,
    )
    openrouter_discovery_thread = threading.Thread(
        target=run_openrouter_discovery_loop,
        kwargs={
            "stop_event": stop_event,
            "interval_seconds": int(os.getenv("BENCHMARK_OPENROUTER_DISCOVERY_INTERVAL_SECONDS", "21600")),
        },
        name="openrouter-discovery-loop",
        daemon=True,
    )
    scheduler_thread.start()
    reaper_thread.start()
    liveness_thread.start()
    invariants_thread.start()
    admission_thread.start()
    classifier_thread.start()
    vertex_discovery_thread.start()
    route_renewal_thread.start()
    reconciler_thread.start()
    openrouter_discovery_thread.start()
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
