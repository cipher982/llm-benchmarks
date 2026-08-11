"""Periodic long-generation samples for the throughput regression.

Every model is measured at 64 output tokens (`cloud-default-v1`). Roughly every
few hours each model also gets one 512-token run under `cloud-long-v1`, so a
downstream estimator can regress total generate_time on generated tokens: the
slope is steady-state tokens/sec, the intercept is the floor latency. One long
sample per window is all that regression needs.

Long rows are written to the published collection, `metrics_cloud_v2`, with
`benchmark_profile_id: "cloud-long-v1"` — they must be queryable alongside
default rows, which is the whole point. The dashboard's transform already drops
any row whose profile is not the published one, so they are invisible to
current charts by construction rather than by every query remembering to
filter.

Selection is time-based and starvation-free, not a modulo counter: a model is
due when it has no long attempt or long success within the window, candidates
are served oldest-first, and the per-pass cap bounds how much work one pass
creates, never which models are allowed to exist. A failed long attempt is
recorded as `long_profile_state` on the health doc — it delays the next try by
one window and touches nothing else, so a model that fails only long runs
simply keeps failing them quietly.

Out of scope on purpose: Bedrock (separate EC2 runner), models the reasoning
shadow profile already measures at 2048 tokens, and anything listed in
`BENCHMARK_LONG_PROFILE_EXCLUDE`.

Spend bound for a later audit: volume is at most one 512-token generation per
eligible model per window — fleet_size / BENCHMARK_LONG_PROFILE_HOURS runs per
hour (≈300 models / 6h ≈ 50 runs/hour, ≈1,200/day), additionally rate-limited
by the per-pass cap. Failed attempts count against the window, so a broken
model costs one attempt per window, not one per pass.
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from pymongo.database import Database

from llm_bench.ops import reasoning_shadow
from llm_bench.scheduler import health
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import probe_metrics_collection_name

PROFILE_ID = "cloud-long-v1"
SAMPLE_ROLE = "long"
JOB_KIND = "long_profile"

# A model the shadow pipeline measured this recently is already covered at a
# larger budget; giving it long runs too would double-spend on the expensive
# models specifically.
SHADOW_COVERAGE_WINDOW = timedelta(days=7)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def interval_hours() -> float:
    """Hours between long samples per model. 0 disables long runs entirely."""
    return float(os.getenv("BENCHMARK_LONG_PROFILE_HOURS", "6"))


def enabled() -> bool:
    return interval_hours() > 0


def deadline_seconds() -> int:
    """Long runs generate 8x the tokens; the default deadline would kill them."""
    return int(os.getenv("BENCHMARK_LONG_PROFILE_TIMEOUT_SECONDS", "180"))


def excluded_keys() -> frozenset[str]:
    """`provider/model_id` keys exempted from long runs, e.g. expensive models."""
    raw = os.getenv("BENCHMARK_LONG_PROFILE_EXCLUDE", "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def max_jobs_per_pass() -> int:
    """Blast radius: bound the work one pass creates, never the population."""
    return int(os.getenv("BENCHMARK_MAX_LONG_JOBS", "25"))


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _shadow_measured(db: Database, *, now: datetime) -> set[tuple[str, str]]:
    """Models the reasoning shadow pipeline owns.

    Both signals, because each alone lags: `find_unmeasurable` is the live
    trigger and recent probe rows are the evidence a shadow sample landed.
    """
    measured = set(reasoning_shadow.find_unmeasurable(db))
    for row in db[probe_metrics_collection_name()].find(
        {
            "benchmark_profile_id": reasoning_shadow.PROFILE_ID,
            "run_ts": {"$gte": now - SHADOW_COVERAGE_WINDOW},
        },
        {"provider": 1, "model_name": 1},
    ):
        measured.add((row.get("provider"), row.get("model_name")))
    return measured


def _last_long_activity(db: Database, *, provider: str, model_id: str) -> datetime | None:
    """Newest long attempt or long success, whichever is later.

    Attempts count as activity so a failing model retries once per window
    instead of on every pass; successes are read from the metrics rows as well
    so the window survives a lost health write.
    """
    doc = health.health_collection(db).find_one(
        {"provider": provider, "model_id": model_id},
        {"long_profile_state": 1},
    )
    attempt_at = ((doc or {}).get("long_profile_state") or {}).get("last_attempt_at")
    row = db[metrics_collection_name()].find_one(
        {"provider": provider, "model_name": model_id, "benchmark_profile_id": PROFILE_ID},
        {"run_ts": 1},
        sort=[("run_ts", -1)],
    )
    success_at = (row or {}).get("run_ts")
    stamps = [stamp for stamp in (_as_utc(attempt_at), _as_utc(success_at)) if stamp is not None]
    return max(stamps) if stamps else None


def _pending_long_job(db: Database, *, provider: str, model_id: str) -> bool:
    return bool(
        db[queue.jobs_collection_name()].count_documents(
            {
                "provider": provider,
                "model_id": model_id,
                "benchmark_profile_id": PROFILE_ID,
                "status": {"$in": ["queued", "running"]},
            }
        )
    )


def find_due(db: Database, *, now: datetime) -> list[tuple[datetime, str, str]]:
    """Models owed a long sample, oldest activity first."""
    if not enabled():
        return []
    window = timedelta(hours=interval_hours())
    excluded = excluded_keys()
    shadowed = _shadow_measured(db, now=now)
    due: list[tuple[datetime, str, str]] = []
    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1},
    ):
        provider, model_id = doc["provider"], doc["model_id"]
        if f"{provider}/{model_id}" in excluded:
            continue
        if (provider, model_id) in shadowed:
            continue
        last = _last_long_activity(db, provider=provider, model_id=model_id)
        if last is not None and now - last < window:
            continue
        if _pending_long_job(db, provider=provider, model_id=model_id):
            continue
        due.append((last or _EPOCH, provider, model_id))
    # Stalest first, so a pass that cannot cover everything covers what has
    # waited longest; a model missed by one pass is at the front of the next.
    due.sort(key=lambda item: item[0])
    return due


def _job_id(provider: str, model_id: str, now: datetime) -> str:
    return f"long:{provider}:{model_id}:{now.strftime('%Y%m%dT%H%M%S')}"


def enqueue_long_samples(db: Database, *, now: datetime | None = None, limit: int | None = None) -> list[str]:
    """Queue one long sample per due model, capped per pass."""
    now = now or utcnow()
    limit = max_jobs_per_pass() if limit is None else limit
    enqueued: list[str] = []
    for _, provider, model_id in find_due(db, now=now)[:limit]:
        job = queue._new_job_doc(
            job_id=_job_id(provider, model_id, now),
            provider=provider,
            model_id=model_id,
            priority=1.0,
            job_kind=JOB_KIND,
            now=now,
            deadline_seconds=deadline_seconds(),
            max_attempts=1,
            extra={"sample_role": SAMPLE_ROLE, "benchmark_profile_id": PROFILE_ID},
        )
        db[queue.jobs_collection_name()].replace_one({"_id": job["_id"]}, job, upsert=True)
        enqueued.append(f"{provider}/{model_id}")
    return enqueued
