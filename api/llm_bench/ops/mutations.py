"""Reversible, bounded catalogue mutations.

`disabled_reason` plus a timestamp is an audit hint, not a rollback mechanism.
It records that something changed and roughly why, but not what the value was
before, so a confidently wrong agent can demote fifty models and leave no way to
put them back. That is the gap this closes: every change captures its prior
value, and a batch can be inverted as a unit.

The caps matter as much as the reversibility. Enabling a model spends money and
publishes a claim, and setting `enabled:false` afterwards reverses neither. A
provider that suddenly lists 10,000 IDs, a discovery bug that widens a diff, or
a prompt regression should hit a wall well before it hits the catalogue — so an
over-limit batch applies *nothing* rather than the first N changes. Half a
migration is worse than none, because it is the state nobody designed for.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.scheduler.mongo import collection_name
from llm_bench.scheduler.mongo import models_collection_name

# Blast radius for one batch. Chosen to be comfortably above any legitimate
# routine change and well below "the catalogue".
MAX_CHANGES_PER_BATCH = int(os.getenv("BENCHMARK_MAX_CHANGES_PER_BATCH", "40"))
MAX_CHANGES_PER_PROVIDER = int(os.getenv("BENCHMARK_MAX_CHANGES_PER_PROVIDER", "25"))

# One switch that stops every mutation while read-only monitoring continues, so
# a misbehaving agent can be stopped without taking the site down.
KILL_SWITCH_ENV = "BENCHMARK_MUTATIONS_DISABLED"


def batches_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MUTATION_BATCHES", "bench_mutation_batches")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def mutations_disabled() -> bool:
    return os.getenv(KILL_SWITCH_ENV, "").strip().lower() in {"1", "true", "yes"}


class MutationRefused(Exception):
    """The batch was not applied. Nothing changed."""


@dataclass
class Change:
    provider: str
    model_id: str
    set_fields: dict[str, Any]
    before: dict[str, Any] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass
class MutationBatch:
    """A set of catalogue changes that succeed or fail together.

    Staged first, checked against the caps, then applied. Nothing touches the
    catalogue until `apply` decides the whole batch is allowed.
    """

    db: Database
    reason: str
    actor: str
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    changes: list[Change] = field(default_factory=list)

    def set_model_fields(self, *, provider: str, model_id: str, **fields: Any) -> None:
        """Stage a change. Does not touch the database."""
        if not fields:
            return
        self.changes.append(Change(provider=provider, model_id=model_id, set_fields=fields))

    def has_room_for(self, provider: str) -> bool:
        """Whether one more change to this provider would still fit.

        The caps bound how much a single pass may change. A caller that stages
        everything it wants and then applies gets an all-or-nothing refusal,
        which for a recurring pass is not a safety limit but a deadlock: the
        backlog only grows, so the batch is refused again every time.

        Admission hit exactly that. 43 candidates had earned promotion and 44
        more had earned rejection; the pass staged 87 changes, exceeded the cap
        of 40, applied none, and logged the same refusal every two hours. No
        model could ever be promoted again, and each new one made it worse.

        Asking first lets a pass fill one batch, apply it, and leave the rest
        for the next run. The blast radius per pass is unchanged; the difference
        is that work drains instead of jamming.
        """
        if len(self.changes) >= MAX_CHANGES_PER_BATCH:
            return False
        for_provider = sum(1 for change in self.changes if change.provider == provider)
        return for_provider < MAX_CHANGES_PER_PROVIDER

    def _check_caps(self) -> None:
        if mutations_disabled():
            raise MutationRefused(f"{KILL_SWITCH_ENV} is set; refusing to mutate")
        if len(self.changes) > MAX_CHANGES_PER_BATCH:
            raise MutationRefused(
                f"batch of {len(self.changes)} changes exceeds the cap of {MAX_CHANGES_PER_BATCH}; "
                "nothing was applied"
            )
        per_provider: dict[str, int] = {}
        for change in self.changes:
            per_provider[change.provider] = per_provider.get(change.provider, 0) + 1
        for provider, count in sorted(per_provider.items()):
            if count > MAX_CHANGES_PER_PROVIDER:
                raise MutationRefused(
                    f"{count} changes to {provider} exceeds the per-provider cap of "
                    f"{MAX_CHANGES_PER_PROVIDER}; nothing was applied"
                )

    def apply(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Record the batch, then apply every change, isolating failures.

        The record is written before the first mutation, not after the last.
        Writing it last meant a failure partway through left earlier changes
        applied with no record of them — unrevertible, which is the one thing
        this class exists to guarantee. A DuplicateKeyError did exactly that in
        production, and the damage was invisible because the audit trail is what
        was missing.

        A failing change no longer aborts the ones behind it. One bad row used
        to take out an entire admission pass, so a single case-variant duplicate
        blocked every unrelated promotion in the same batch.

        The returned count is what actually changed. It used to report
        `len(changes)` unconditionally, so a change matching no document
        reported as applied.
        """
        now = now or utcnow()
        self._check_caps()
        if not self.changes:
            return {"batch_id": self.batch_id, "applied": 0, "failed": []}

        models = self.db[models_collection_name()]
        for change in self.changes:
            # Capture only the fields being written. A full document copy would
            # make the inverse restore unrelated concurrent edits too.
            existing = models.find_one(
                {"provider": change.provider, "model_id": change.model_id},
                {key: 1 for key in change.set_fields},
            )
            change.before = {key: (existing or {}).get(key) for key in change.set_fields}

        self.db[batches_collection_name()].insert_one(
            {
                "_id": self.batch_id,
                "applied_at": now,
                "reason": self.reason,
                "actor": self.actor,
                "reverted_at": None,
                # Until every change lands this reads `applying`. A record stuck
                # in that state is a crash mid-batch, and it is still revertible
                # because the before-images are already here.
                "status": "applying",
                "changes": [
                    {
                        "provider": c.provider,
                        "model_id": c.model_id,
                        "after": c.set_fields,
                        "before": c.before,
                    }
                    for c in self.changes
                ],
            }
        )

        applied = 0
        failed: list[dict[str, str]] = []
        for change in self.changes:
            try:
                result = models.update_one(
                    {"provider": change.provider, "model_id": change.model_id},
                    {"$set": {**change.set_fields, "mutation_batch_id": self.batch_id}},
                )
            except Exception as exc:  # noqa: BLE001
                failed.append({"subject": change.subject, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if result.matched_count == 0:
                failed.append({"subject": change.subject, "error": "no document matched"})
            else:
                applied += 1

        self.db[batches_collection_name()].update_one(
            {"_id": self.batch_id},
            {"$set": {"status": "applied", "applied": applied, "failed": failed}},
        )
        return {"batch_id": self.batch_id, "applied": applied, "failed": failed}


def revert(db: Database, *, batch_id: str, now: datetime | None = None) -> int:
    """Restore every field a batch changed to the value it held before.

    Fields that did not exist beforehand are unset rather than written as null,
    so reverting leaves the document shaped as it was rather than merely
    equivalent.
    """
    now = now or utcnow()
    batch = db[batches_collection_name()].find_one({"_id": batch_id})
    if batch is None:
        raise MutationRefused(f"no mutation batch {batch_id}")
    if batch.get("reverted_at"):
        raise MutationRefused(f"batch {batch_id} was already reverted at {batch['reverted_at']}")

    models = db[models_collection_name()]
    for change in batch["changes"]:
        before = change.get("before") or {}
        to_set = {key: value for key, value in before.items() if value is not None}
        to_unset = {key: "" for key, value in before.items() if value is None}
        update: dict[str, Any] = {}
        if to_set:
            update["$set"] = to_set
        if to_unset:
            update["$unset"] = to_unset
        if update:
            models.update_one({"provider": change["provider"], "model_id": change["model_id"]}, update)

    db[batches_collection_name()].update_one(
        {"_id": batch_id},
        {"$set": {"reverted_at": now}},
    )
    return len(batch["changes"])


def recent_batches(db: Database, *, limit: int = 20) -> list[dict[str, Any]]:
    return list(db[batches_collection_name()].find(sort=[("applied_at", -1)], limit=limit))
