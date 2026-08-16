"""Measure the models the default profile cannot measure.

A reasoning model spends the whole 64-token budget on hidden reasoning and emits
nothing visible, so it fails validation every time and has no data at all. That
is not the model being broken; it is the benchmark profile being unable to
measure it, which is why the failure has its own class. This applies to newly
discovered admission candidates as well as already-enabled models.

Raising the budget for everything would change what every published number
means, and the series is already hard enough to reason about: 975 of 1,182
Together rows over 30 days came from multi-attempt retries at a cap other than
the nominal 64. So nothing here touches the published profile.

Instead these models are measured under `cloud-reasoning-v1` as shadow samples.
Shadow rows go to the probe collection, never reach the site, and never count as
freshness. The result is real numbers for models that currently have none, and a
decision about whether they belong on the public chart can then be made by
looking at them rather than by argument.

What is still open, precisely: whether a reasoning model measured at 2048 tokens
belongs on the same axis as a non-reasoning model measured at 64. That is a
publication question. Collecting the measurement is not.
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import probe_metrics_collection_name

PROFILE_ID = "cloud-reasoning-v1"
SAMPLE_ROLE = "shadow"

# The signal that the default profile could not measure a model. Set by the
# validator and classified in the taxonomy; not a guess about the model's name.
TRIGGER_ERROR_KIND = "budget_exhausted"

# One sample per model per interval. This is diagnostic data for a decision, not
# a published series, so it does not need the cadence of a real benchmark.
SAMPLE_INTERVAL = timedelta(hours=int(os.getenv("BENCHMARK_SHADOW_INTERVAL_HOURS", "6")))

# Blast radius. A profile change that suddenly matched hundreds of models must
# not turn into an unbounded bill; reasoning models are the expensive ones.
MAX_SHADOW_JOBS_PER_RUN = int(os.getenv("BENCHMARK_MAX_SHADOW_JOBS", "15"))

# Reasoning generations are long. The default deadline would kill them mid-run
# and record a timeout that says nothing about the model.
DEADLINE_SECONDS = int(os.getenv("BENCHMARK_SHADOW_DEADLINE_SECONDS", "300"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def find_unmeasurable(db: Database) -> list[tuple[str, str]]:
    """Enabled models whose newest attempt exhausted the budget.

    Driven by what the runner recorded, not by a list of model names. A model
    that starts fitting inside the default budget stops qualifying on its own.
    """
    jobs = db[queue.jobs_collection_name()]
    subjects = []
    for doc in db[models_collection_name()].find(
        {
            "deprecated": {"$ne": True},
            "$or": [
                {"enabled": True},
                # Native OpenRouter candidates are deliberately disabled while
                # they are being admitted, but still need the larger diagnostic
                # profile when the published probe exhausts its budget.
                {"provider": "openrouter", "status": "probing"},
            ],
        },
        {"provider": 1, "model_id": 1},
    ):
        job = jobs.find_one(
            {
                "provider": doc["provider"],
                "model_id": doc["model_id"],
                # Do not let a successful shadow sample hide the default
                # profile failure that caused this diagnostic run.
                "sample_role": {"$ne": SAMPLE_ROLE},
            },
            {"last_attempt_error_kind": 1},
            sort=[("updated_at", -1)],
        )
        if job and job.get("last_attempt_error_kind") == TRIGGER_ERROR_KIND:
            subjects.append((doc["provider"], doc["model_id"]))
    return subjects


def _recent_shadow(db: Database, *, provider: str, model_id: str, now: datetime) -> bool:
    """True when this model already has a recent shadow sample or pending job."""
    pending = db[queue.jobs_collection_name()].count_documents(
        {
            "provider": provider,
            "model_id": model_id,
            "sample_role": SAMPLE_ROLE,
            "status": {"$in": ["queued", "running"]},
        }
    )
    if pending:
        return True
    return bool(
        db[probe_metrics_collection_name()].find_one(
            {
                "provider": provider,
                "model_name": model_id,
                "benchmark_profile_id": PROFILE_ID,
                "run_ts": {"$gte": now - SAMPLE_INTERVAL},
            },
            {"_id": 1},
        )
    )


def _job_id(provider: str, model_id: str, now: datetime) -> str:
    return f"shadow:{provider}:{model_id}:{now.strftime('%Y%m%dT%H%M%S')}"


def enqueue_shadow_samples(
    db: Database, *, now: datetime | None = None, limit: int = MAX_SHADOW_JOBS_PER_RUN
) -> list[str]:
    """Queue one shadow sample per unmeasurable model that is due one."""
    now = now or utcnow()
    enqueued: list[str] = []
    for provider, model_id in find_unmeasurable(db):
        if len(enqueued) >= limit:
            break
        if _recent_shadow(db, provider=provider, model_id=model_id, now=now):
            continue
        job = queue._new_job_doc(
            job_id=_job_id(provider, model_id, now),
            provider=provider,
            model_id=model_id,
            priority=1.0,
            job_kind="shadow",
            now=now,
            deadline_seconds=DEADLINE_SECONDS,
            max_attempts=1,
            extra={"sample_role": SAMPLE_ROLE, "benchmark_profile_id": PROFILE_ID},
        )
        db[queue.jobs_collection_name()].replace_one({"_id": job["_id"]}, job, upsert=True)
        enqueued.append(f"{provider}/{model_id}")
    return enqueued


def summarize(db: Database) -> dict[str, Any]:
    """What the shadow profile has learned so far.

    The point of the exercise: numbers for models that otherwise have none, so
    the publication question can be answered by looking.
    """
    rows = list(
        db[probe_metrics_collection_name()].find(
            {"benchmark_profile_id": PROFILE_ID},
            {"provider": 1, "model_name": 1, "tokens_per_second": 1, "visible_tokens_per_second": 1},
        )
    )
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(f"{row['provider']}/{row['model_name']}", []).append(row)
    return {
        "profile": PROFILE_ID,
        "models_measured": len(by_model),
        "samples": len(rows),
        "per_model": {
            subject: {
                "samples": len(items),
                "tokens_per_second": round(sum(i.get("tokens_per_second") or 0 for i in items) / len(items), 2),
                "visible_tokens_per_second": round(
                    sum(i.get("visible_tokens_per_second") or 0 for i in items) / len(items), 2
                ),
            }
            for subject, items in sorted(by_model.items())
        },
    }
