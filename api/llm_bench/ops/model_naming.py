"""Human-readable model names, derived rather than maintained.

The site used to carry a hand-written display-name list. At ~60 models that was
tractable; at 388 it rotted visibly — `claude-opus-4.6` beside
`claude-opus-4-7`, `gpt-4` twice, and one row that simply said `undefined`.

The names were never missing. OpenRouter publishes `"{vendor}: {model}"` on
every catalogue entry and we have stored it all along;
`openrouter_discovery._display_name` preferred the dated `canonical_slug` over
it and called the slug a display name, and admission copied that onto the model
row. So the leaderboard rendered `aion-labs/aion-2.0-20260223` while
`AionLabs: Aion 2.0` sat unread in the same document.

What this module does NOT do is decide chart identity. Grouping on a
presentation string is what split `claude-haiku-4.5` from `claude-haiku-4-5`
into two lines for one model, and a label sourced from a third party can change
under us at any time. Identity lives in `bench_model_identity.canonical_key`;
this module only decides what a row is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.ops import identity
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name

CATALOGUE_COLLECTION = "openrouter_catalog"

# How far behind the newest catalogue row a document may be and still count as
# current. The collection is append-only — discovery upserts and never removes —
# so a row last seen in April 2026 is still present and must not be read as a
# live answer about a model's name.
CATALOGUE_FRESHNESS = timedelta(hours=36)

# Where a label came from, most to least authoritative. Recorded on the row so a
# later reader can tell a sourced name from a fallback without re-deriving it.
SOURCE_CATALOGUE = "openrouter_catalogue"
SOURCE_IDENTITY_SIBLING = "openrouter_catalogue_via_identity"
SOURCE_EXISTING = "existing_curated"
SOURCE_FALLBACK = "canonical_id_fallback"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_catalogue_name(name: str | None) -> tuple[str | None, str | None]:
    """Split OpenRouter's `"{vendor}: {model}"` into its two halves.

    Only 577 of 619 catalogue rows contain the separator, so the unsplit case is
    ordinary rather than exceptional: the whole string is the model label and the
    vendor is unknown. Splits on the *first* separator, because model names
    contain colons (`GPT-4o: Extended`) far more often than vendors do.
    """
    if not name:
        return None, None
    text = str(name).strip()
    if not text:
        return None, None
    vendor, separator, model = text.partition(": ")
    if not separator:
        return None, text
    model = model.strip()
    vendor = vendor.strip()
    return (vendor or None), (model or text)


@dataclass(frozen=True)
class CatalogueLabel:
    model_id: str
    label: str
    vendor: str | None
    org: str | None


def catalogue_labels(db: Database, *, now: datetime | None = None) -> dict[str, CatalogueLabel]:
    """Labels from catalogue rows that are current and serve text.

    Freshness is measured against the newest row rather than wall-clock, so a
    discovery outage degrades to "no new labels" instead of "every label is
    stale" — the second would silently rewrite the whole site to fallbacks.
    """
    del now  # freshness is relative to the catalogue, not to the caller's clock
    collection = db[CATALOGUE_COLLECTION]
    anchor = _last_complete_catalogue_read(db)
    if anchor is None:
        # Never having read the whole catalogue is not a licence to treat a
        # partial read as authoritative. Falling back to the newest row here
        # reproduced exactly the omission-to-fallback failure this anchor
        # exists to prevent, on any database whose only runs had failed.
        # Returning nothing is safe: existing readable names are kept as
        # SOURCE_EXISTING rather than overwritten.
        return {}
    cutoff = anchor - CATALOGUE_FRESHNESS

    labels: dict[str, CatalogueLabel] = {}
    for doc in collection.find(
        {"last_seen_at": {"$gte": cutoff}},
        {"openrouter_id": 1, "name": 1, "org": 1, "output_modalities": 1},
    ):
        model_id = doc.get("openrouter_id")
        if not model_id:
            continue
        modalities = doc.get("output_modalities") or []
        if modalities and "text" not in modalities:
            continue
        vendor, label = parse_catalogue_name(doc.get("name"))
        if not label:
            continue
        labels[str(model_id)] = CatalogueLabel(
            model_id=str(model_id),
            label=label,
            vendor=vendor,
            org=(str(doc["org"]) if doc.get("org") else None),
        )
    return labels


def _last_complete_catalogue_read(db: Database) -> datetime | None:
    """When we last saw the whole catalogue, not merely part of it.

    `refresh_catalog` writes every row it received *before* recording the run,
    and marks the run failed when pagination came back short. So a truncated
    read still bumps `last_seen_at` on the subset it returned. Those names are
    real and safe to use — the damage is to the window: anchoring freshness to
    the newest row lets a partial read move the cutoff forward and push every
    model it happened to omit out of scope and onto a fallback name.

    Anchoring to the last complete read means a run of partial refreshes leaves
    the existing labels alone instead of eroding them.
    """
    run = db[_discovery_runs_collection()].find_one(
        {"provider": "openrouter", "status": "completed"},
        sort=[("started_at", -1)],
    )
    started = (run or {}).get("started_at")
    return started if isinstance(started, datetime) else None


def _discovery_runs_collection() -> str:
    from llm_bench.ops.openrouter_discovery import discovery_runs_collection_name

    return discovery_runs_collection_name()


def _looks_like_a_raw_id(value: str | None, model_id: str) -> bool:
    """Is this 'name' actually just the identifier wearing a name's clothes?

    Both spellings are in production: the id itself, and the dated canonical
    slug that discovery mistook for a display name. A slash is the reliable tell
    for either, because no human-facing label contains one.
    """
    if not value:
        return True
    text = str(value).strip()
    if not text or text.lower() == "undefined":
        return True
    return "/" in text or text == model_id


@dataclass(frozen=True)
class Proposal:
    provider: str
    model_id: str
    label: str
    source: str
    vendor: str | None = None
    current: str | None = None

    @property
    def changed(self) -> bool:
        return self.label != (self.current or "")


# The longest window any chart renders. A model disabled yesterday is still on
# the site until its last row falls out of this.
PUBLICATION_WINDOW = timedelta(days=30)


def renderable_endpoints(db: Database, *, now: datetime | None = None) -> set[tuple[str, str]]:
    """(provider, model_id) the site can still draw, enabled or not.

    Naming only enabled models was too narrow: retiring the OpenAI rows from the
    OpenRouter lane left 28 of them on the leaderboard under raw ids, because
    they still had rows inside the two-day window and no longer qualified to be
    named. 289 disabled models are inside the 30-day window at any time.

    Keyed by endpoint, not by model id. Matching on the id alone let a recent
    OpenRouter row drag an unrelated, long-disabled Bedrock or Vertex row with
    the same id into the pass, mutating something nothing renders.
    """
    now = now or utcnow()
    return {
        (str(row["_id"]["provider"]), str(row["_id"]["model_name"]))
        for row in db[metrics_collection_name()].aggregate(
            [
                {"$match": {"run_ts": {"$gte": now - PUBLICATION_WINDOW}}},
                {"$group": {"_id": {"provider": "$provider", "model_name": "$model_name"}}},
            ]
        )
        if row.get("_id", {}).get("provider") and row["_id"].get("model_name")
    }


def _identity_siblings(db: Database) -> dict[tuple[str, str], str]:
    """Endpoint -> canonical_key, for the endpoints identity has resolved."""
    return {
        (str(row["provider"]), str(row["model_id"])): str(row["canonical_key"])
        for row in identity.current_identities(db)
        if row.get("canonical_key") and row.get("provider") and row.get("model_id")
    }


def plan(db: Database) -> list[Proposal]:
    """What every enabled model should be called, and on whose authority.

    Order of preference is deliberate. A direct lane has no presentation feed of
    its own — Vertex discovery writes `name = model_id` and the Bedrock
    catalogue is a list of ids — so the only way it gets a real name without a
    human writing one is to inherit it from an OpenRouter sibling that identity
    has already proven is the same model.
    """
    labels = catalogue_labels(db)
    identities = _identity_siblings(db)

    # canonical_key -> the best catalogue label available anywhere in that group
    group_label: dict[str, CatalogueLabel] = {}
    for (provider, model_id), key in identities.items():
        if provider != "openrouter":
            continue
        label = labels.get(model_id)
        if label and key not in group_label:
            group_label[key] = label

    proposals: list[Proposal] = []
    renderable = renderable_endpoints(db)
    for row in db[models_collection_name()].find(
        {"$or": [{"enabled": True}, {"model_id": {"$in": sorted({m for _, m in renderable})}}]},
        {"provider": 1, "model_id": 1, "display_name": 1, "enabled": 1},
    ):
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        if not provider or not model_id:
            continue
        if not row.get("enabled") and (provider, model_id) not in renderable:
            continue
        current = row.get("display_name")

        direct = labels.get(model_id) if provider == "openrouter" else None
        sibling = group_label.get(identities.get((provider, model_id), ""))

        if direct:
            proposals.append(
                Proposal(provider, model_id, direct.label, SOURCE_CATALOGUE, direct.org or direct.vendor, current)
            )
        elif sibling:
            proposals.append(
                Proposal(
                    provider, model_id, sibling.label, SOURCE_IDENTITY_SIBLING, sibling.org or sibling.vendor, current
                )
            )
        elif not _looks_like_a_raw_id(current, model_id):
            # A curated name that predates this module. Keeping it is not
            # reintroducing the hand list — it is declining to overwrite one
            # human-readable string with a worse machine-readable one.
            proposals.append(Proposal(provider, model_id, str(current), SOURCE_EXISTING, None, current))
        else:
            proposals.append(Proposal(provider, model_id, fallback_label(model_id), SOURCE_FALLBACK, None, current))

    return _disambiguate(proposals)


def fallback_label(model_id: str) -> str:
    """The least-bad name for a model nothing can name.

    Drops the org prefix and the vendor's serving suffix, so a direct-only row
    reads as `nova-lite` rather than `amazon.nova-lite-v1:0`. Marked as a
    fallback on the row, so this is visible as an absence of data rather than
    passing for a real name.
    """
    text = model_id.split("/")[-1]
    text = text.split(":")[0]
    return text or model_id


def _qualifier(model_id: str) -> str:
    """A distinguishing suffix that will not be mistaken for a raw id next pass.

    The qualifier used to be the model id verbatim, which contains a slash — so
    the following pass read the whole label as machine data and re-derived it.
    `v/other` became `alpha-2 (v/other)` and then `other`: a visible rename and
    a wasted mutation batch, on every model that needed qualifying.
    """
    return model_id.replace("/", " ").strip()


def _disambiguate(proposals: list[Proposal]) -> list[Proposal]:
    """Two rows on one provider must not answer to the same name.

    Stripping the vendor prefix is not collision-safe — the catalogue has nine
    duplicate parsed labels across 18 rows, including `Reka Edge` from two
    different organisations. Publication groups by name within a provider, so a
    collision would average two unrelated deployments into one series.
    """
    seen: dict[tuple[str, str], list[Proposal]] = {}
    for proposal in proposals:
        seen.setdefault((proposal.provider, proposal.label.casefold()), []).append(proposal)

    resolved: list[Proposal] = []
    for group in seen.values():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        # Re-deriving from the id is the first resort, not a parenthetical.
        # OpenAI's two `gpt-4` rows are `gpt-4` and `gpt-4-turbo`; the ids
        # already name them apart, and `gpt-4` / `gpt-4-turbo` reads better than
        # `gpt-4 (gpt-4)` / `gpt-4 (gpt-4-turbo)`.
        from_ids = [fallback_label(p.model_id) for p in group]
        if len(set(from_ids)) == len(group):
            for proposal, label in zip(group, from_ids):
                resolved.append(
                    Proposal(
                        proposal.provider, proposal.model_id, label, proposal.source, proposal.vendor, proposal.current
                    )
                )
            continue
        # Otherwise qualify with whatever actually differs. The two `Reka Edge`
        # rows are `reka/reka-edge` and `other/reka-edge`, so the last path
        # segment distinguishes nothing. Vendor usually does; the full id always
        # does, being unique per provider by construction.
        vendors = [p.vendor for p in group]
        use_vendor = all(vendors) and len(set(vendors)) == len(vendors)
        for proposal in group:
            qualifier = proposal.vendor if use_vendor else _qualifier(proposal.model_id)
            resolved.append(
                Proposal(
                    proposal.provider,
                    proposal.model_id,
                    f"{proposal.label} ({qualifier})",
                    proposal.source,
                    proposal.vendor,
                    proposal.current,
                )
            )

    # Resolving each group in isolation is not enough: a rewritten label can
    # land on a name some other group already holds. `gpt-4`/`gpt-4-turbo`
    # collide and re-derive from their ids — straight onto an existing
    # `gpt-4-turbo` row that never collided in the first place. Settle globally,
    # and bound it so a pathological catalogue cannot spin here.
    final: list[Proposal] = []
    taken: set[tuple[str, str]] = set()
    for proposal in sorted(resolved, key=lambda p: (p.provider, p.model_id)):
        label = proposal.label
        attempt = 0
        # Unbounded on purpose, and guaranteed to terminate: model ids are
        # unique per provider, so the first qualified form is already unique
        # unless it collides with a *literal* label someone else holds, and each
        # further attempt adds a distinct counter. A fixed cap would have let the
        # loop fall through and append a name that was still taken.
        while (proposal.provider, label.casefold()) in taken:
            attempt += 1
            suffix = _qualifier(proposal.model_id) if attempt == 1 else f"{_qualifier(proposal.model_id)} {attempt}"
            label = f"{proposal.label} ({suffix})"
        taken.add((proposal.provider, label.casefold()))
        final.append(
            Proposal(proposal.provider, proposal.model_id, label, proposal.source, proposal.vendor, proposal.current)
        )
    return final


def audit(db: Database) -> dict[str, Any]:
    """Report only. Classifies every enabled model without changing anything."""
    proposals = plan(db)
    by_source: dict[str, int] = {}
    for proposal in proposals:
        by_source[proposal.source] = by_source.get(proposal.source, 0) + 1

    raw_now = sum(1 for p in proposals if _looks_like_a_raw_id(p.current, p.model_id))
    raw_after = sum(1 for p in proposals if _looks_like_a_raw_id(p.label, p.model_id))
    return {
        "models": len(proposals),
        "by_source": by_source,
        "changed": sum(1 for p in proposals if p.changed),
        "raw_looking_before": raw_now,
        "raw_looking_after": raw_after,
        "catalogue_labels_available": len(catalogue_labels(db)),
        "identity_resolved_endpoints": len(_identity_siblings(db)),
    }


def _drain(db: Database, staged: list[Proposal], *, reason: str, actor: str) -> list[str]:
    """Apply in cap-sized batches, oldest first.

    The caps refuse an over-large batch outright rather than applying a prefix,
    so staging 388 changes at once would apply none of them.
    """
    from llm_bench.ops.mutations import MutationBatch

    applied: list[str] = []
    pending = list(staged)
    while pending:
        batch = MutationBatch(db=db, reason=reason, actor=actor)
        remainder: list[Proposal] = []
        for proposal in pending:
            if batch.has_room_for(proposal.provider):
                batch.set_model_fields(
                    provider=proposal.provider,
                    model_id=proposal.model_id,
                    display_name=proposal.label,
                    display_name_source=proposal.source,
                    display_vendor=proposal.vendor,
                )
            else:
                remainder.append(proposal)
        if not batch.changes:
            raise RuntimeError(f"no room for any of {len(pending)} remaining changes; caps misconfigured")
        batch.apply()
        applied.append(batch.batch_id)
        pending = remainder
    return applied


def apply_names(db: Database, *, apply: bool, actor: str = "model_naming") -> dict[str, Any]:
    """Write the planned labels, reversibly."""
    proposals = [p for p in plan(db) if p.changed]
    report = audit(db)
    report["to_change"] = len(proposals)
    if not apply or not proposals:
        report["applied"] = False
        return report
    report["batches"] = _drain(
        db,
        proposals,
        reason="Display names derived from the OpenRouter catalogue rather than a hand-maintained list",
        actor=actor,
    )
    report["applied"] = True
    return report
