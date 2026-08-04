"""Admit models to the site by measuring them, not by reading their name.

A provider's `/models` endpoint does not say what is benchmarkable. Together
lists dedicated-endpoint-only models with normal pricing and `type: "chat"`, and
reports `running: false` for all 274 including working ones. Fireworks and
Cerebras list models that 404. Two passes of name-pattern filtering still let
through `veo`, `kling`, `vidu`, `ideogram` and `parakeet`.

So a real call is the classifier. An endpoint that returns text tokens at a
measurable rate belongs on a site about token throughput, whatever kind of model
it is — a guard model's latency is a real question. An endpoint that cannot
return text will fail its probe without anyone maintaining a brand list.

One success is not admission. It says this account, adapter and moment worked;
it cannot see flapping availability, an alias whose weights moved, or a model
that answers once and then rate-limits forever. So a candidate needs successes
in separate collection windows before it is published, and it is published as
provisional until it has kept working.

Probe samples never reach the public collection and never count as freshness —
see `scheduler/runner.py`. Without that separation admission would contaminate
the series it exists to fill.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import probe_metrics_collection_name

__all__ = [
    "models_collection_name",
    "probe_metrics_collection_name",
    "run_admission_pass",
    "find_candidates",
    "evaluate_candidates",
]

# Successes required before publishing, and how far apart they must be. Two
# back-to-back calls are highly correlated — they mostly prove the endpoint was
# warm, which is the thing least likely to still be true tomorrow.
REQUIRED_SUCCESSES = 2
WINDOW = timedelta(hours=2)

# A candidate that cannot produce them within this long is not merely slow.
ADMISSION_DEADLINE = timedelta(days=3)

# Blast radius. A provider that suddenly lists 10,000 IDs, or a discovery bug
# that widens the diff, must not turn into an unbounded spend.
MAX_NEW_CANDIDATES_PER_RUN = int(os.getenv("BENCHMARK_MAX_NEW_CANDIDATES", "25"))
MAX_PROBES_PER_RUN = int(os.getenv("BENCHMARK_MAX_PROBES_PER_RUN", "60"))

CANDIDATE_STATUS = "probing"
PROMOTED_STATUS = "probation"
REJECTED_STATUS = "rejected"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass
class AdmissionReport:
    registered: list[str]
    probes_enqueued: list[str]
    promoted: list[str]
    rejected: list[tuple[str, str]]

    def summary(self) -> str:
        return (
            f"registered={len(self.registered)} probes={len(self.probes_enqueued)} "
            f"promoted={len(self.promoted)} rejected={len(self.rejected)}"
        )


def find_candidates(db: Database, *, limit: int = MAX_NEW_CANDIDATES_PER_RUN) -> list[dict[str, Any]]:
    """Catalogue rows the site has never had an opinion about.

    Ordered so that a model several providers already serve is probed first: a
    model measured at one provider and available at three is where the site is
    least useful today, and filling it is what turns a two-provider chart line
    into a five-provider one.
    """
    known = {
        (doc["provider"], doc["model_id"])
        for doc in db[models_collection_name()].find({}, {"provider": 1, "model_id": 1})
    }
    catalogue = list(db.provider_catalog.find({}, {"provider": 1, "model_id": 1, "name": 1}))

    # How many distinct providers serve each rough base name, from the catalogue.
    spread: dict[str, set[str]] = {}
    for row in catalogue:
        spread.setdefault(_base_name(row["model_id"]), set()).add(row["provider"])

    candidates = [row for row in catalogue if (row["provider"], row["model_id"]) not in known]
    candidates.sort(key=lambda row: (-len(spread.get(_base_name(row["model_id"]), ())), row["provider"]))
    return candidates[:limit]


def _base_name(model_id: str) -> str:
    """Rough base name, only for prioritising which candidate to probe first.

    Deliberately crude — it decides probe order, never identity. Grouping for
    display is Phase 2's job and uses evidence, not string munging.
    """
    text = str(model_id).lower()
    text = text.replace("accounts/fireworks/models/", "")
    if text.startswith("us."):
        text = text[3:]
    text = text.rsplit("/", 1)[-1]
    for suffix in ("-turbo", "-fp8", "-instruct", "-it", "-chat", "-hf", "-versatile", "-latest", "-v1:0"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return "".join(ch for ch in text if ch.isalnum())


def register_candidates(
    db: Database, *, now: datetime | None = None, limit: int = MAX_NEW_CANDIDATES_PER_RUN
) -> list[str]:
    """Record candidates as probing. Not enabled, so nothing publishes yet."""
    now = now or utcnow()
    registered = []
    for row in find_candidates(db, limit=limit):
        db[models_collection_name()].update_one(
            {"provider": row["provider"], "model_id": row["model_id"]},
            {
                "$setOnInsert": {
                    "provider": row["provider"],
                    "model_id": row["model_id"],
                    "display_name": row.get("name"),
                    "enabled": False,
                    "status": CANDIDATE_STATUS,
                    "admission_started_at": now,
                    "created_at": now,
                    "source": "provider_catalog",
                }
            },
            upsert=True,
        )
        registered.append(f"{row['provider']}/{row['model_id']}")
    return registered


def _probe_job_id(provider: str, model_id: str, now: datetime) -> str:
    return f"probe:{provider}:{model_id}:{now.strftime('%Y%m%dT%H%M%S')}"


def enqueue_probes(db: Database, *, now: datetime | None = None, limit: int = MAX_PROBES_PER_RUN) -> list[str]:
    """Queue one probe per candidate that is due another sample."""
    now = now or utcnow()
    enqueued = []
    for doc in db[models_collection_name()].find(
        {"status": CANDIDATE_STATUS},
        {"provider": 1, "model_id": 1},
    ):
        if len(enqueued) >= limit:
            break
        provider, model_id = doc["provider"], doc["model_id"]
        if not _needs_probe(db, provider=provider, model_id=model_id, now=now):
            continue
        job = queue._new_job_doc(
            job_id=_probe_job_id(provider, model_id, now),
            provider=provider,
            model_id=model_id,
            priority=1.0,
            job_kind="probe",
            now=now,
            # A candidate that hangs must not hold a worker slot as long as a
            # published model is allowed to.
            deadline_seconds=90,
            max_attempts=2,
            extra={"sample_role": "probe"},
        )
        db[queue.jobs_collection_name()].replace_one({"_id": job["_id"]}, job, upsert=True)
        enqueued.append(f"{provider}/{model_id}")
    return enqueued


def _needs_probe(db: Database, *, provider: str, model_id: str, now: datetime) -> bool:
    """True when this candidate has neither enough evidence nor pending work."""
    pending = db[queue.jobs_collection_name()].count_documents(
        {"provider": provider, "model_id": model_id, "status": {"$in": ["queued", "running"]}}
    )
    if pending:
        return False
    successes = _probe_successes(db, provider=provider, model_id=model_id)
    if len(successes) >= REQUIRED_SUCCESSES:
        return False
    # Space samples out. Back-to-back successes mostly prove the endpoint was
    # warm, which is the least durable thing about it.
    if successes and (now - max(successes)) < WINDOW:
        return False
    return True


def _probe_successes(db: Database, *, provider: str, model_id: str) -> list[datetime]:
    stamps = [
        _as_utc(row.get("run_ts"))
        for row in db[probe_metrics_collection_name()].find(
            {"provider": provider, "model_name": model_id},
            {"run_ts": 1},
        )
    ]
    return sorted(s for s in stamps if s is not None)


def _windows_covered(stamps: list[datetime]) -> int:
    """How many separated observations there are, not how many samples."""
    if not stamps:
        return 0
    count, last = 1, stamps[0]
    for stamp in stamps[1:]:
        if stamp - last >= WINDOW:
            count += 1
            last = stamp
    return count


def evaluate_candidates(db: Database, *, now: datetime | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """Promote candidates with enough spaced evidence; reject the exhausted."""
    now = now or utcnow()
    promoted, rejected = [], []
    for doc in db[models_collection_name()].find(
        {"status": CANDIDATE_STATUS},
        {"provider": 1, "model_id": 1, "admission_started_at": 1},
    ):
        provider, model_id = doc["provider"], doc["model_id"]
        subject = f"{provider}/{model_id}"
        successes = _probe_successes(db, provider=provider, model_id=model_id)

        if _windows_covered(successes) >= REQUIRED_SUCCESSES:
            db[models_collection_name()].update_one(
                {"provider": provider, "model_id": model_id},
                {
                    "$set": {
                        "enabled": True,
                        "status": PROMOTED_STATUS,
                        "promoted_at": now,
                        "admission_evidence": {
                            "successes": len(successes),
                            "windows": _windows_covered(successes),
                            "first_success": successes[0],
                            "last_success": successes[-1],
                        },
                    }
                },
            )
            promoted.append(subject)
            continue

        started = _as_utc(doc.get("admission_started_at")) or now
        if now - started > ADMISSION_DEADLINE:
            reason = (
                f"no probe success in {ADMISSION_DEADLINE.days}d "
                f"({len(successes)} sample(s), {_windows_covered(successes)} window(s))"
            )
            db[models_collection_name()].update_one(
                {"provider": provider, "model_id": model_id},
                {
                    "$set": {
                        "enabled": False,
                        "status": REJECTED_STATUS,
                        "disabled_class": "hard_model",
                        "disabled_reason": reason,
                        "disabled_at": now,
                    }
                },
            )
            rejected.append((subject, reason))
    return promoted, rejected


def run_admission_pass(db: Database, *, now: datetime | None = None) -> AdmissionReport:
    """One full cycle. Safe to run repeatedly; every step is bounded."""
    now = now or utcnow()
    registered = register_candidates(db, now=now)
    probes = enqueue_probes(db, now=now)
    promoted, rejected = evaluate_candidates(db, now=now)
    return AdmissionReport(
        registered=registered,
        probes_enqueued=probes,
        promoted=promoted,
        rejected=rejected,
    )
