from __future__ import annotations

import os

from pymongo import MongoClient


def mongo_env() -> tuple[str, str]:
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")
    return uri, db_name


def mongo_client() -> MongoClient:
    uri, _ = mongo_env()
    return MongoClient(uri)


def collection_name(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)


# The profile whose rows the site publishes. Lives here rather than in the
# runner so freshness/coverage readers can filter without importing the runner
# (which would be a circular import); runner.DEFAULT_PROFILE_ID re-exports it.
PUBLISHED_PROFILE_ID = "cloud-default-v1"


def published_profile_filter(field: str = "benchmark_profile_id") -> dict:
    """Query fragment matching only rows measured under the published profile.

    Rows written before profiles existed carry no profile field at all, so
    absence must count as published — otherwise every historical row vanishes
    from freshness the moment this filter lands. Long-profile rows share
    metrics_cloud_v2 by design (the regression needs them queryable next to
    default rows), which is exactly why every published-progress reader has to
    say which series it means: a model succeeding only at 512 tokens must not
    look measured while its published 64-token series is dead.

    Returns an `$or`; callers whose query already uses `$or` must wrap both in
    an `$and`.
    """
    return {"$or": [{field: {"$exists": False}}, {field: PUBLISHED_PROFILE_ID}]}


def jobs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_BENCH_JOBS", "bench_jobs")


def health_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODEL_HEALTH", "bench_model_health")


def metrics_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_CLOUD", "metrics_cloud_v2")


def probe_metrics_collection_name() -> str:
    """Where non-published samples go.

    A separate collection rather than a flag on the published one. The dashboard
    reads metrics_cloud_v2 from a dozen places, and a probe row is only excluded
    from the site if every one of them remembers to filter — which is the silent
    fallback shape this codebase keeps being bitten by. Routing by collection
    makes exclusion structural instead of a thing each query must remember.
    """
    return collection_name("MONGODB_COLLECTION_CLOUD_PROBE", "metrics_cloud_probe")


def route_decisions_collection_name() -> str:
    """Collection containing reviewed, versioned route decisions.

    The scheduler reads this collection when it creates a job. A missing row
    is intentionally equivalent to no route evidence and therefore selects
    the direct adapter.
    """
    return collection_name("MONGODB_COLLECTION_ROUTE_DECISIONS", "bench_route_decisions")


def route_revocations_collection_name() -> str:
    """Collection containing monotonic route revocation generations."""

    return collection_name("MONGODB_COLLECTION_ROUTE_REVOCATIONS", "bench_route_revocations")


def route_reconciliation_collection_name() -> str:
    """Collection containing immutable reconciliation run manifests."""

    return collection_name("MONGODB_COLLECTION_ROUTE_RECONCILIATIONS", "bench_route_reconciliations")


def route_audit_collection_name() -> str:
    """Collection containing historical per-row route decisions."""

    return collection_name("MONGODB_COLLECTION_ROUTE_AUDIT", "bench_route_decision_audit")


def route_snapshot(db, *, provider: str, model_id: str) -> dict | None:
    """Return the newest reviewed route decision for one source row."""
    return db[route_decisions_collection_name()].find_one(
        {"source_provider": provider, "source_model_id": model_id, "state": "active"},
        sort=[("route_snapshot_at", -1), ("updated_at", -1)],
    )


def route_revocation_generation(db, *, provider: str, model_id: str) -> int:
    """Read the newest revocation generation, defaulting safely to zero."""

    row = db[route_revocations_collection_name()].find_one(
        {"source_provider": provider, "source_model_id": model_id},
        sort=[("generation", -1)],
    )
    if not row:
        return 0
    try:
        return max(0, int(row.get("generation", 0)))
    except (TypeError, ValueError):
        return 0


def errors_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERRORS", "errors_cloud")


def error_rollups_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")


def models_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODELS", "models")


def heartbeats_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_SCHEDULER_HEARTBEATS", "bench_scheduler_heartbeats")
