"""Make OpenRouter the transport, and stop judging models we never called.

Two things had accumulated in the catalogue at once.

The site was running nine providers. Three of them are direct on purpose —
OpenAI, Vertex and Bedrock are the owner's real consumption and their numbers
mean something specific. The rest were a maintenance tax: separate keys, separate
rate limits, separate outages, and 55 of their 69 enabled rows were the same
model already enabled natively on OpenRouter, published twice.

Meanwhile 45 live models sat disabled behind the reason "OpenRouter duplicates
direct providers, adds latency. Use for discovery only" — the previous policy,
now exactly inverted. A further 18 live models were disabled by an operator for
having "never succeeded and no recorded errors", which is indistinguishable from
never having been scheduled: 37 of those 38 rows had no job row at any point.
Silence was read as failure, and the models were removed for it.

Run with --apply; prints the plan and changes nothing otherwise.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from llm_bench.ops.mutations import MutationBatch
from llm_bench.scheduler.mongo import models_collection_name

ACTOR = "openrouter_consolidation"

# Direct on purpose: the owner consumes these accounts, so the measurement is of
# something real rather than of a marketplace's routing.
DIRECT_PROVIDERS = ("openai", "vertex", "bedrock")

REASON_ROUTED = (
    "Consolidated onto OpenRouter transport: this model is served by OpenRouter, "
    "and maintaining a separate direct provider integration for it is upkeep with "
    "no measurement benefit."
)
REASON_POLICY_REVERSED = (
    "Re-enabled: disabled under the earlier 'OpenRouter duplicates direct providers' "
    "policy, which is now inverted — OpenRouter is the transport."
)
REASON_JUDGED_UNTESTED = (
    "Re-enabled: disabled for having no successes, but the model had no job row at "
    "all, so the silence was never evidence about the model."
)


def _catalog_ids(db: Database) -> set[str]:
    return {
        str(doc["openrouter_id"])
        for doc in db["openrouter_catalog"].find({}, {"openrouter_id": 1})
        if doc.get("openrouter_id")
    }


def _never_scheduled(db: Database, model_id: str) -> bool:
    return db["bench_jobs"].count_documents({"model_id": model_id}, limit=1) == 0


def plan(db: Database) -> dict[str, list[dict[str, Any]]]:
    """What to change, and why, without touching anything."""
    models = db[models_collection_name()]
    live = _catalog_ids(db)

    reenable: list[dict[str, Any]] = []
    for row in models.find({"provider": "openrouter", "enabled": False}):
        model_id = str(row.get("model_id"))
        if model_id not in live:
            continue  # OpenRouter no longer lists it; leave it disabled.
        reason = str(row.get("disabled_reason") or "")
        if "Phase 1 cleanup" in reason:
            reenable.append({"model_id": model_id, "reason": REASON_POLICY_REVERSED})
        elif "AI Operator" in reason and _never_scheduled(db, model_id):
            reenable.append({"model_id": model_id, "reason": REASON_JUDGED_UNTESTED})

    retire: list[dict[str, Any]] = []
    for row in models.find(
        {"enabled": True, "provider": {"$nin": [*DIRECT_PROVIDERS, "openrouter"]}},
        {"provider": 1, "model_id": 1},
    ):
        retire.append({"provider": str(row["provider"]), "model_id": str(row["model_id"])})

    return {"reenable": reenable, "retire": retire}


def _drain(db: Database, staged: list[tuple[str, str, dict[str, Any]]], *, reason: str) -> list[str]:
    """Apply staged changes in cap-sized batches, oldest first.

    The caps refuse an over-large batch outright rather than applying a prefix,
    so a caller that stages everything gets nothing. Filling one batch at a time
    and applying it drains the work instead of jamming on it.
    """
    applied: list[str] = []
    pending = list(staged)
    while pending:
        batch = MutationBatch(db=db, reason=reason, actor=ACTOR)
        remainder: list[tuple[str, str, dict[str, Any]]] = []
        for provider, model_id, fields in pending:
            if batch.has_room_for(provider):
                batch.set_model_fields(provider=provider, model_id=model_id, **fields)
            else:
                remainder.append((provider, model_id, fields))
        if not batch.changes:
            raise RuntimeError(f"no room for any of {len(pending)} remaining changes; caps misconfigured")
        batch.apply()
        applied.append(batch.batch_id)
        print(f"  applied batch {batch.batch_id} ({len(batch.changes)} changes)")
        pending = remainder
    return applied


def run(db: Database, *, apply: bool) -> dict[str, Any]:
    work = plan(db)
    print(f"re-enable {len(work['reenable'])} OpenRouter models judged without evidence or under a reversed policy")
    for item in work["reenable"]:
        print(f"    + openrouter/{item['model_id']}")
    print(f"retire {len(work['retire'])} direct rows onto OpenRouter transport")
    for item in work["retire"]:
        print(f"    - {item['provider']}/{item['model_id']}")

    if not apply:
        print("\ndry run; pass --apply to make these changes")
        return {"applied": False, **{k: len(v) for k, v in work.items()}}

    batches: list[str] = []
    if work["reenable"]:
        staged = [
            (
                "openrouter",
                item["model_id"],
                {"enabled": True, "disabled_reason": None, "disabled_at": None, "reenabled_reason": item["reason"]},
            )
            for item in work["reenable"]
        ]
        batches += _drain(db, staged, reason=REASON_POLICY_REVERSED)
    if work["retire"]:
        staged = [
            (item["provider"], item["model_id"], {"enabled": False, "disabled_reason": REASON_ROUTED})
            for item in work["retire"]
        ]
        batches += _drain(db, staged, reason=REASON_ROUTED)

    print(f"\n{len(batches)} batches applied; revert any with llm_bench.ops.mutations.revert")
    return {"applied": True, "batches": batches, **{k: len(v) for k, v in work.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="make the changes (default is a dry run)")
    args = parser.parse_args()

    uri = os.environ["MONGODB_URI"]
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    client = MongoClient(uri)
    try:
        run(client[db_name], apply=args.apply)
    finally:
        client.close()


if __name__ == "__main__":
    main()
