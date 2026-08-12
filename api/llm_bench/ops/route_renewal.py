"""Periodic renewal of OpenRouter route evidence.

Route snapshots carry a bounded evidence window (`expires_at`, 72h by default):
the paired canary that approved the route goes stale, so the runner fails
closed to the direct lane when the window passes. The renewal half of that
design is this loop: a route inside `renew_ahead_hours` of expiry gets one
small routed probe call. Success extends the window; failure cools the route
down and a later pass retries. Cooldowned routes whose cooldown window has
passed get the same probe: success restores them to active with a fresh
window, and enough consecutive probe failures revoke the route permanently
(fail-closed, like hard_model) until a human re-promotes it with new evidence.

These probes only mutate the route document. They write no metrics rows: they
verify the path, they do not measure it.
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Callable

from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.routing import OPENROUTER_TRANSPORT
from llm_bench.scheduler.routing import RouteDecision
from llm_bench.scheduler.routing import cooldown_route_snapshot
from llm_bench.scheduler.routing import recover_route_snapshot

PROBE_MAX_TOKENS = 64


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def renew_ahead_hours() -> float:
    """Renew a route when its evidence window drops below this many hours."""
    return float(os.getenv("BENCHMARK_ROUTE_RENEW_AHEAD_HOURS", "24"))


def expires_hours() -> float:
    """Hours of fresh evidence granted by one successful probe."""
    return float(os.getenv("BENCHMARK_ROUTE_RENEW_EXPIRES_HOURS", "72"))


def max_per_pass() -> int:
    """Blast radius: bound the probes one pass fires, never the population."""
    return int(os.getenv("BENCHMARK_ROUTE_RENEW_MAX_PER_PASS", "8"))


def max_failures() -> int:
    """Consecutive probe failures before a route is permanently revoked."""
    return int(os.getenv("BENCHMARK_ROUTE_RENEW_MAX_FAILURES", "12"))


def probe_timeout_seconds() -> float:
    return float(os.getenv("BENCHMARK_ROUTE_RENEW_TIMEOUT_SECONDS", "30"))


def probe_route(decision: RouteDecision, *, now: datetime) -> None:
    """One routed call. Raises on any failure or unverifiable provider.

    Mirrors the runner's routed validation: the OpenRouter response must carry
    verified provider metadata and the observed provider slug must match the
    route's pinned provider, or the probe is a failure.
    """
    from llm_bench.scheduler import runner

    run_config = {"query": runner.QUERY_TEXT, "max_tokens": PROBE_MAX_TOKENS}
    runner._generate_and_validate(  # noqa: SLF001 - reuse the runner's exact routed validation
        decision,
        run_ts=now.isoformat(),
        run_config=run_config,
        max_tokens=PROBE_MAX_TOKENS,
        timeout_seconds=probe_timeout_seconds(),
    )


def renew_pass(
    db: Any,
    *,
    now: datetime | None = None,
    probe: Callable[[RouteDecision, datetime], None] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Renew due routes and recover cooled routes, oldest evidence first.

    Returns a report of what happened per pass. Every route is handled in
    isolation: one probe failure never blocks the rest of the pass.
    """
    now = now or utcnow()
    probe = probe or probe_route
    collection = db[route_decisions_collection_name()]
    ahead = renew_ahead_hours()
    renew_hours = expires_hours()
    failures_allowed = max_failures()
    cap = limit if limit is not None else max_per_pass()
    cutoff = now + timedelta(hours=ahead)

    candidates = list(
        collection.find(
            {
                "state": {"$in": ["active", "cooldown"]},
                "$or": [
                    {"state": "active", "expires_at": {"$lte": cutoff.isoformat()}},
                    {"state": "cooldown", "cooldown_until": {"$lte": now.timestamp()}},
                ],
            }
        )
        .sort([("expires_at", 1)])
        .limit(cap)
    )

    report = {"considered": len(candidates), "renewed": 0, "cooled": 0, "revoked": 0, "skipped": 0}
    for route in candidates:
        # The probe must be able to route even when the stored evidence is
        # expired or the route is cooling down; that is exactly what this pass
        # re-verifies. Resolve a copy with fresh evidence to derive the probe
        # call, and let the probe outcome write the real expiry.
        probe_snapshot = dict(route)
        probe_snapshot["state"] = "active"
        probe_snapshot["expires_at"] = (now + timedelta(hours=1)).isoformat()
        probe_snapshot["recheck_at"] = probe_snapshot["expires_at"]
        decision = RouteDecision.from_snapshot(
            str(route.get("source_provider") or ""),
            str(route.get("source_model_id") or ""),
            probe_snapshot,
            now=now,
        )
        if decision.transport_provider != OPENROUTER_TRANSPORT:
            report["skipped"] += 1
            continue
        stamp = now.strftime("%Y%m%dT%H%M%S")
        probe_id = f"renewal:{route.get('source_provider')}:{route.get('source_model_id')}:{stamp}"
        try:
            probe(decision, now=now)
        except Exception as exc:  # noqa: BLE001 - one bad route must not block the pass
            updated = cooldown_route_snapshot(route, failure_reason=f"{type(exc).__name__}: {exc}", now=now)
            failures = int(updated.get("route_health_failure_count", 0))
            if failures >= failures_allowed:
                updated.update(
                    {
                        "state": "revoked",
                        "route_health_state": "revoked",
                        "route_revocation_reason": "renewal-probe-failed",
                        "route_revoked_at": now.isoformat(),
                    }
                )
                report["revoked"] += 1
            else:
                report["cooled"] += 1
            collection.update_one({"_id": route["_id"]}, {"$set": updated})
            continue

        if route.get("state") == "cooldown":
            updated = recover_route_snapshot(
                route,
                recovery_probe_id=probe_id,
                recovery_probe_passed=True,
                now=now,
            )
        else:
            updated = dict(route)
            updated["state"] = "active"
        fresh = (now + timedelta(hours=renew_hours)).isoformat()
        updated.update(
            {
                "expires_at": fresh,
                "recheck_at": fresh,
                "last_renewed_at": now.isoformat(),
                "last_renewal_probe_id": probe_id,
                "last_renewal_probe_ok": True,
                "updated_at": now.isoformat(),
            }
        )
        collection.update_one({"_id": route["_id"]}, {"$set": updated})
        report["renewed"] += 1

    return report


def run_renewal_loop(*, stop_event: Any, interval_seconds: int) -> None:
    """Daemon loop: renew route evidence on a cadence, never die on failure.

    A pass with nothing to do is one Mongo query. The probes only run for
    routes inside their renewal window, so steady-state call volume is a tiny
    routed call per route per day.
    """
    while not stop_event.wait(interval_seconds):
        try:
            client = mongo_client()
            try:
                _, db_name = mongo_env()
                db = client[db_name]
                report = renew_pass(db)
                if report["renewed"] or report["cooled"] or report["revoked"]:
                    print(f"Route renewal: {report}", flush=True)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001 - never kill the daemon over a renewal pass
            print(f"Route renewal loop error: {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    """Run one renewal pass from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="max routes to probe this pass")
    args = parser.parse_args()
    client = mongo_client()
    try:
        _, db_name = mongo_env()
        report = renew_pass(client[db_name], limit=args.limit)
    finally:
        client.close()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
