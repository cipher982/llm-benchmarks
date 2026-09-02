"""OpenRouter key headroom: know the cap is coming before it lands.

The OpenRouter key carries a monthly spend limit. On 2026-08-20 the limit was
reached and every routed call answered 403 "Key limit exceeded (monthly
limit)" until the counter reset on 2026-09-01. The runner produced almost no
rows for eleven days and nothing said why: the errors were classified as
`auth` at the time, the liveness watchdog saw a healthy process, and the
per-provider progress check was satisfied by the direct OpenAI and Vertex
lanes. Silence looked like a quiet fleet.

The key itself can say how much is left. `GET /api/v1/auth/key` returns the
limit, the remaining balance and usage over the last day, week and month, so
the daemon records that once an hour and an invariant fails while the balance
is exhausted or on course to be exhausted within a few days at the current
burn. That turns an eleven-day outage into a page a few days before the cap,
with the number that decides the spend question printed on it.

One GET per pass; no benchmark call is made here.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
STATE_ID = "openrouter:key_limit"
PROVIDER_STATE_COLLECTION = "provider_state"

# A doc older than this is not evidence about now; the check must say it
# could not look rather than pass on a stale reading.
MAX_STATE_AGE = timedelta(hours=3)

# Fail while the balance would run out inside this many days at the current
# burn. Three days is enough to raise the limit or trim the population before
# the cap, and short enough that a normal month never trips it.
MIN_HEADROOM_DAYS = 3.0


def fetch_key_status(
    *, api_key: str | None = None, base_url: str | None = None, timeout: float = 20.0
) -> dict[str, Any]:
    """The `data` object from `/auth/key`."""
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    base = (base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    req = urllib.request.Request(
        f"{base}/auth/key",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"unexpected /auth/key payload: {str(payload)[:200]}")
    return data


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def headroom(state: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """Days of spend left at the current burn, from a recorded key status.

    Burn is the larger of today's usage and the weekly average, so a reading
    taken at 00:30 UTC (when `usage_daily` is near zero) does not report a
    month of headroom that a full day would contradict.
    """
    limit = _number(state.get("limit"))
    remaining = _number(state.get("limit_remaining"))
    daily = _number(state.get("usage_daily")) or 0.0
    weekly = _number(state.get("usage_weekly")) or 0.0
    burn_per_day = max(daily, weekly / 7.0)

    if limit is None or remaining is None:
        # An unlimited key. Nothing to run out of; report burn only.
        return {"limited": False, "burn_per_day": burn_per_day, "days_left": None, "exhausted": False}

    exhausted = remaining <= 0
    days_left = None if burn_per_day <= 0 else remaining / burn_per_day
    return {
        "limited": True,
        "limit": limit,
        "remaining": remaining,
        "burn_per_day": burn_per_day,
        "days_left": days_left,
        "exhausted": exhausted,
    }


def record_key_status(db: Database, status: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Upsert the key's limit and usage into `provider_state`. Never stores the key."""
    checked_at = now or datetime.now(timezone.utc)
    doc = {
        "provider": "openrouter",
        "kind": "key_limit",
        "checked_at": checked_at,
        "limit": _number(status.get("limit")),
        "limit_remaining": _number(status.get("limit_remaining")),
        "limit_reset": status.get("limit_reset"),
        "usage_daily": _number(status.get("usage_daily")),
        "usage_weekly": _number(status.get("usage_weekly")),
        "usage_monthly": _number(status.get("usage_monthly")),
        "usage_total": _number(status.get("usage")),
        "label": status.get("label"),
    }
    doc.update({f"headroom_{k}": v for k, v in headroom(doc, now=checked_at).items()})
    db[PROVIDER_STATE_COLLECTION].update_one({"_id": STATE_ID}, {"$set": doc}, upsert=True)
    return doc


def load_key_state(db: Database) -> dict[str, Any] | None:
    return db[PROVIDER_STATE_COLLECTION].find_one({"_id": STATE_ID})


def refresh(db: Database, *, now: datetime | None = None) -> dict[str, Any]:
    """One pass: fetch, record, return the stored doc."""
    return record_key_status(db, fetch_key_status(), now=now)


def run_key_limit_loop(*, stop_event: Any, interval_seconds: int) -> None:
    """Daemon loop. One GET per interval; never dies on a failed pass."""
    # Record once at start-up so a fresh container is not blind for an interval.
    first = True
    while first or not stop_event.wait(interval_seconds):
        first = False
        try:
            client = mongo_client()
            try:
                _, db_name = mongo_env()
                doc = refresh(client[db_name])
                if doc.get("headroom_limited"):
                    days = doc.get("headroom_days_left")
                    print(
                        "OpenRouter key: "
                        f"${doc.get('limit_remaining'):.2f} of ${doc.get('limit'):.2f} left, "
                        f"burn ${doc.get('headroom_burn_per_day'):.2f}/day, "
                        f"{'no burn yet' if days is None else f'{days:.1f} days'} of headroom",
                        flush=True,
                    )
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001 - never kill the daemon over a status read
            print(f"OpenRouter key status error: {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    """Run one pass from the command line and print the stored doc."""
    client = mongo_client()
    try:
        _, db_name = mongo_env()
        doc = refresh(client[db_name])
    finally:
        client.close()
    printable = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in doc.items()}
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
