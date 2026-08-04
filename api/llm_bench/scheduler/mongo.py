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


def errors_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERRORS", "errors_cloud")


def error_rollups_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")


def models_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODELS", "models")


def heartbeats_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_SCHEDULER_HEARTBEATS", "bench_scheduler_heartbeats")
