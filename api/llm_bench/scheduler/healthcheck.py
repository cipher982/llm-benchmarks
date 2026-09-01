"""Container healthcheck for the Mongo-backed direct-provider runner."""

from __future__ import annotations

import json
import os
import sys

from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env


def _providers() -> list[str] | None:
    """Lanes to hold to account, minus the ones this host runs no worker for.

    Liveness now fails a lane with no heartbeat, and a provider this host does
    not run has none by definition. Left unfiltered, naming a retired provider
    here restarts the container forever — the same list still named eight
    providers after they were consolidated onto OpenRouter. Filtering through
    the exclusion policy ties the check to the set that decides which workers
    start, instead of to a second list that has to be remembered.
    """
    raw = os.getenv("BENCHMARK_LIVENESS_PROVIDERS", "")
    excluded = policies.excluded_providers()
    providers = [item.strip() for item in raw.split(",") if item.strip() and item.strip() not in excluded]
    return providers or None


def main() -> int:
    try:
        _, db_name = mongo_env()
        client = mongo_client()
        try:
            db = client[db_name]
            db.command("ping")
            healthy, details = health.liveness_status(
                db,
                providers=_providers(),
            )
        finally:
            client.close()
    except Exception as exc:
        print(json.dumps({"healthy": False, "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1

    print(json.dumps({"healthy": healthy, **details}))
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
