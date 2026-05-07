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


def errors_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERRORS", "errors_cloud")


def error_rollups_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")


def models_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODELS", "models")


def heartbeats_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_SCHEDULER_HEARTBEATS", "bench_scheduler_heartbeats")
