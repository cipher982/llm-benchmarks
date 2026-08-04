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
        """Capture before-images, apply every change, record the batch."""
        now = now or utcnow()
        self._check_caps()
        if not self.changes:
            return {"batch_id": self.batch_id, "applied": 0}

        models = self.db[models_collection_name()]
        for change in self.changes:
            # Capture only the fields being written. A full document copy would
            # make the inverse restore unrelated concurrent edits too.
            existing = models.find_one(
                {"provider": change.provider, "model_id": change.model_id},
                {key: 1 for key in change.set_fields},
            )
            change.before = {key: (existing or {}).get(key) for key in change.set_fields}

        for change in self.changes:
            models.update_one(
                {"provider": change.provider, "model_id": change.model_id},
                {"$set": {**change.set_fields, "mutation_batch_id": self.batch_id}},
            )

        self.db[batches_collection_name()].insert_one(
            {
                "_id": self.batch_id,
                "applied_at": now,
                "reason": self.reason,
                "actor": self.actor,
                "reverted_at": None,
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
        return {"batch_id": self.batch_id, "applied": len(self.changes)}


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
