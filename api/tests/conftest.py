"""Hermetic defaults for the test suite.

Several modules call `dotenv.load_dotenv()` at import scope, so importing one of
them fills any unset environment variable from the developer's `.env`. That file
sets `MONGODB_COLLECTION_CLOUD=metrics_cloud_staging`, so a test that wrote to
`metrics_cloud_v2` and a check that read the collection name resolved to
different collections — but only when the polluting module happened to be
imported first, which made it look like flakiness.

Pinning the names here makes the suite independent of both the developer's .env
and test import order.
"""

import pytest

COLLECTION_ENV = {
    "MONGODB_DB": "llm-bench-test",
    "MONGODB_COLLECTION_CLOUD": "metrics_cloud_v2",
    "MONGODB_COLLECTION_ERRORS": "errors_cloud",
    "MONGODB_COLLECTION_MODELS": "models",
    "MONGODB_COLLECTION_BENCH_JOBS": "bench_jobs",
    "MONGODB_COLLECTION_MODEL_HEALTH": "bench_model_health",
    "MONGODB_COLLECTION_ERROR_ROLLUPS": "error_rollups",
    "MONGODB_COLLECTION_SCHEDULER_HEARTBEATS": "bench_scheduler_heartbeats",
}


@pytest.fixture(autouse=True)
def _pinned_collection_names(monkeypatch):
    for key, value in COLLECTION_ENV.items():
        monkeypatch.setenv(key, value)
