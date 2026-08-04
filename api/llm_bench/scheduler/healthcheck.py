"""Container healthcheck for the Mongo-backed direct-provider runner."""

from __future__ import annotations

import json
import os
import sys

from llm_bench.scheduler import health
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env


def _providers() -> list[str] | None:
    raw = os.getenv("BENCHMARK_LIVENESS_PROVIDERS", "")
    providers = [item.strip() for item in raw.split(",") if item.strip()]
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
                max_idle_seconds=int(os.getenv("BENCHMARK_LIVENESS_MAX_IDLE_SECONDS", "900")),
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
