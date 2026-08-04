"""Retire models the providers have actually stopped serving.

The dangerous half of keeping a catalogue in sync. Adding a model that turns out
to be wrong costs a chart line; removing one that is still live loses a series
the site has been building for months, and the loss is quiet.

So absence is only counted across *complete* discovery runs. A run that errored,
stopped short of pagination, or never happened proves nothing about what the
provider offers, and the ledger exists precisely so those can be told apart —
before it, a filter change and a provider deletion looked identical.

Three consecutive complete runs, not three calendar days. Polling daily with
jitter means "three days" can be two observations, and a deprecation resting on
two observations is a coin flip on one bad night.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.ops import identity
from llm_bench.ops import invariants
from llm_bench.ops import mutations
from llm_bench.scheduler.mongo import models_collection_name

# Consecutive complete runs a model must be missing from before it is retired.
REQUIRED_ABSENT_RUNS = int(os.getenv("BENCHMARK_ABSENT_RUNS_TO_DEPRECATE", "3"))

# Providers with no discovery authority. Absence from a catalogue that is never
# read is not evidence, so their models are never retired this way.
UNCOVERED_PROVIDERS = frozenset({"bedrock", "vertex"})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass
class Retirement:
    provider: str
    model_id: str
    reason: str

    @property
    def subject(self) -> str:
        return f"{self.provider}/{self.model_id}"


def complete_run_times(db: Database, *, provider: str, limit: int = REQUIRED_ABSENT_RUNS) -> list[datetime]:
    """When this provider's last N fully-successful catalogue reads finished."""
    runs = db[invariants.discovery_runs_collection_name()].find(
        {"provider": provider, "status": "completed", "pagination_complete": True},
        {"finished_at": 1},
        sort=[("finished_at", -1)],
        limit=limit,
    )
    return [stamp for stamp in (_as_utc(r.get("finished_at")) for r in runs) if stamp is not None]


def find_retirements(db: Database, *, now: datetime | None = None) -> list[Retirement]:
    """Enabled models absent from every one of the last N complete runs."""
    now = now or utcnow()
    seen: dict[tuple[str, str], datetime] = {}
    for row in db.provider_catalog.find({}, {"provider": 1, "model_id": 1, "last_seen_at": 1}):
        stamp = _as_utc(row.get("last_seen_at"))
        if stamp is not None:
            seen[(row["provider"], row["model_id"])] = stamp

    cutoffs: dict[str, datetime | None] = {}
    retirements: list[Retirement] = []

    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1},
    ):
        provider, model_id = doc["provider"], doc["model_id"]
        if provider in UNCOVERED_PROVIDERS:
            continue

        if provider not in cutoffs:
            runs = complete_run_times(db, provider=provider)
            # Not enough complete history to judge absence yet.
            cutoffs[provider] = runs[-1] if len(runs) >= REQUIRED_ABSENT_RUNS else None
        cutoff = cutoffs[provider]
        if cutoff is None:
            continue

        last_seen = seen.get((provider, model_id))
        if last_seen is None or last_seen < cutoff:
            retirements.append(
                Retirement(
                    provider=provider,
                    model_id=model_id,
                    reason=(
                        f"absent from the last {REQUIRED_ABSENT_RUNS} complete discovery runs "
                        f"(last seen {last_seen.isoformat() if last_seen else 'never'})"
                    ),
                )
            )
    return retirements


def retire(db: Database, *, now: datetime | None = None, dry_run: bool = True) -> list[Retirement]:
    """Deprecate models the provider has stopped listing.

    Applies as one bounded batch, so a discovery regression that empties a
    provider's catalogue hits the cap and retires nothing. That is the failure
    this guards against: the run looks successful, and every model vanishes.
    """
    now = now or utcnow()
    retirements = find_retirements(db, now=now)
    if dry_run or not retirements:
        return retirements

    batch = mutations.MutationBatch(db=db, reason="absent from provider catalogue", actor="reconciler")
    for item in retirements:
        batch.set_model_fields(
            provider=item.provider,
            model_id=item.model_id,
            enabled=False,
            deprecated=True,
            disabled_class="provider_retired",
            disabled_reason=item.reason,
            disabled_at=now,
        )
    batch.apply(now=now)
    return retirements


def summarize(db: Database, *, now: datetime | None = None) -> dict[str, Any]:
    """What the reconciler would do, without doing any of it."""
    now = now or utcnow()
    retirements = find_retirements(db, now=now)
    by_provider: dict[str, int] = {}
    for item in retirements:
        by_provider[item.provider] = by_provider.get(item.provider, 0) + 1
    return {
        "retirement_count": len(retirements),
        "by_provider": by_provider,
        "subjects": [item.subject for item in retirements[:50]],
    }


# --------------------------------------------------------------------------
# The nightly pass
# --------------------------------------------------------------------------


def resolve_missing_identities(
    db: Database,
    *,
    call_llm: Any,
    limit: int = 40,
    now: datetime | None = None,
) -> list[str]:
    """Give every enabled endpoint a stored identity, cheapest-first.

    Only endpoints without a current relation are resolved, so this costs one
    call per genuinely new endpoint rather than one per model per night.
    """
    now = now or utcnow()
    known = {(r["provider"], r["model_id"]) for r in identity.current_identities(db)}
    resolved = []
    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1, "display_name": 1},
    ):
        if len(resolved) >= limit:
            break
        key = (doc["provider"], doc["model_id"])
        if key in known:
            continue
        try:
            record = identity.match_endpoint(
                db,
                provider=doc["provider"],
                model_id=doc["model_id"],
                name=doc.get("display_name"),
                call_llm=call_llm,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001
            # One endpoint failing to resolve must not stop the rest. It simply
            # stays ungrouped, which is the safe direction.
            print(f"identity resolution failed for {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        resolved.append(f"{doc['provider']}/{doc['model_id']}" + ("" if record["resolved"] else " (unresolved)"))
    return resolved


def grouping_divergence(db: Database) -> dict[str, Any]:
    """Compare derived identity against what the site groups by today.

    Run before letting derived keys drive display. A disagreement is not
    automatically an error — the hand-built table has known false merges, and
    finding them is the point — but every one should be explainable before the
    mapping changes under a live chart.
    """
    identities = {(r["provider"], r["model_id"]): r.get("canonical_key") for r in identity.current_identities(db)}

    derived: dict[str, set[str]] = {}
    current: dict[str, set[str]] = {}
    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1, "display_name": 1},
    ):
        endpoint = f"{doc['provider']}/{doc['model_id']}"
        key = identities.get((doc["provider"], doc["model_id"]))
        if key:
            derived.setdefault(key, set()).add(endpoint)
        shown = doc.get("display_name")
        if shown:
            current.setdefault(shown, set()).add(endpoint)

    current_sets = {frozenset(v) for v in current.values()}
    derived_sets = {frozenset(v) for v in derived.values()}
    return {
        "endpoints_with_identity": len(identities),
        "derived_groups": len(derived),
        "current_groups": len(current),
        "agreeing_groups": len(current_sets & derived_sets),
        "derived_only": sorted(
            ("|".join(sorted(s)) for s in derived_sets - current_sets),
        )[:25],
        "current_only": sorted(
            ("|".join(sorted(s)) for s in current_sets - derived_sets),
        )[:25],
    }


def unify_display_names(db: Database, *, now: datetime | None = None, dry_run: bool = True) -> list[dict[str, Any]]:
    """Give every endpoint in a derived group the same display name.

    The publication pipeline groups by (providerCanonical, display_name), so two
    providers land on one chart line exactly when their display names match
    character for character. In production `claude-haiku-4.5` and
    `claude-haiku-4-5` were the same model at three providers, split into two
    lines by a single hyphen.

    So the derived identity does not need to replace the mapping code to be
    useful — it only has to say which endpoints should share a name. The name
    itself is the one most of the group already uses, so existing slugs and URLs
    keep working and only the outlier moves.
    """
    now = now or utcnow()
    current = {
        (r["provider"], r["model_id"]): r.get("canonical_key")
        for r in identity.current_identities(db)
        if r.get("canonical_key")
    }

    groups: dict[str, list[dict[str, Any]]] = {}
    for doc in db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1, "display_name": 1},
    ):
        key = current.get((doc["provider"], doc["model_id"]))
        if key:
            groups.setdefault(key, []).append(doc)

    changes = []
    for key, members in sorted(groups.items()):
        names = [m.get("display_name") for m in members if m.get("display_name")]
        if len(members) < 2 or len(set(names)) < 2:
            continue
        # Only unify across providers. The chart exists to compare providers, so
        # merging two endpoints at one provider buys nothing and risks real harm:
        # DeepSeek-V3.1 and DeepSeek-V3.1-Terminus are separate checkpoints at
        # DeepInfra that share a derived key, and collapsing them would hide one
        # behind the other rather than put two providers on a line.
        if len({m["provider"] for m in members}) < 2:
            continue
        # Majority wins, so the group keeps the name most of the site already
        # publishes. Ties break on the shortest, which is the least decorated.
        winner = sorted(set(names), key=lambda n: (-names.count(n), len(n), n))[0]
        # Endpoints already published under the winning name, by provider. A
        # rename that collides with one of them does not add a provider to the
        # line — it merges two of that provider's deployments into a single row
        # and averages them. DeepInfra serves both Llama-3.3-70B-Instruct and
        # its Turbo build; renaming the second onto the first hides one.
        taken = {
            doc["provider"]
            for doc in db[models_collection_name()].find(
                {"enabled": True, "deprecated": {"$ne": True}, "display_name": winner},
                {"provider": 1},
            )
        }
        for member in members:
            if member.get("display_name") != winner and member["provider"] not in taken:
                changes.append(
                    {
                        "provider": member["provider"],
                        "model_id": member["model_id"],
                        "from": member.get("display_name"),
                        "to": winner,
                        "canonical_key": key,
                    }
                )

    if dry_run or not changes:
        return changes

    batch = mutations.MutationBatch(db=db, reason="unify display names within a derived group", actor="reconciler")
    for change in changes:
        batch.set_model_fields(
            provider=change["provider"],
            model_id=change["model_id"],
            display_name=change["to"],
            identity_key=change["canonical_key"],
        )
    batch.apply(now=now)
    return changes


def consolidate(db: Database, *, dry_run: bool = True, now: datetime | None = None) -> list[dict[str, Any]]:
    """Rejoin groups that are the same model under two names.

    Groups form one endpoint at a time, so arrival order decides the name and
    the same model can end up split — Llama 3.3 70B became `llama-3.3-70b` and
    `llama-3.3-70b-instruct`, turning a four-provider line into two. Consolidation
    also caught `minimax-m2p7` against `minimax-m2.7`, which no rule anyone would
    write anticipates.
    """
    return identity.consolidate_groups(
        db,
        call_llm=lambda prompt: identity.call_openrouter(prompt, model=identity.CONSOLIDATION_MODEL, max_tokens=8000),
        now=now,
        dry_run=dry_run,
    )
