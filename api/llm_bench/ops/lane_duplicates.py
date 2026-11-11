"""One measurement per model per serving host.

The site publishes a row per (model, host). A direct lane and an OpenRouter row
that reach the same model on the same host are one measurement published twice:
OpenAI's API called directly and openai/* called through OpenRouter both measure
OpenAI serving the model, and the router hop is the same for every row on the
site. On 2026-11-11 the catalogue held 24 direct OpenAI rows (dead since
2026-10-13 for want of a key) beside OpenRouter rows for the same models, three
Vertex Gemini rows beside OpenRouter's Google-served Gemini, Bedrock Claude and
Nova beside OpenRouter's Bedrock-served endpoints, and 18 OpenRouter `~*-latest`
aliases that are pointers to models already measured under their own ids.

OpenRouter is the transport, so where both exist the OpenRouter row is kept and
the direct row is retired. A direct row with no same-host OpenRouter equivalent
is not a duplicate and stays: Bedrock serving Llama is a host of its own.

`duplicates` is what the invariant reports; `run --apply` resolves it.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from llm_bench.ops.mutations import MutationBatch
from llm_bench.scheduler.mongo import models_collection_name

ACTOR = "lane_duplicates"

# OpenRouter endpoint providers that are the same host as a direct lane. Azure
# is OpenAI's own serving partner and the only host OpenRouter lists for the
# GPT-6 family; for our purposes it is OpenAI serving OpenAI's model.
SAME_HOST_ENDPOINTS: dict[str, frozenset[str]] = {
    "openai": frozenset({"openai", "azure"}),
    "vertex": frozenset({"google-vertex", "google-ai-studio"}),
    "bedrock": frozenset({"amazon-bedrock", "claude-on-aws"}),
}

# An OpenRouter row with no admitted endpoint is routed by OpenRouter, which
# sends a first-party model to its first party. These vendor prefixes are served
# by the direct lane's own host.
FIRST_PARTY_VENDORS: dict[str, frozenset[str]] = {
    "openai": frozenset({"openai"}),
    "vertex": frozenset({"google"}),
    "bedrock": frozenset({"amazon"}),
}

REASON_RETIRED = (
    "Retired direct lane: OpenRouter measures this model on the same host, and one "
    "model on one host is one measurement. Publishing both double-counts it "
    "(lane_duplicates, 2026-11-11)."
)
REASON_ROUTED_BACK = (
    "Re-enabled: the direct lane that owned this model was retired, so OpenRouter is "
    "now its only measurement (lane_duplicates, 2026-11-11)."
)
REASON_ALIAS = (
    "Disabled: an OpenRouter ~*-latest alias points at a model already measured "
    "under its own id, so measuring the alias double-counts that model."
)

# Disabled for this reason only because the direct lane owned the model. Any
# other disabled reason is evidence about the model and is left alone.
_RETIRED_FOR_DIRECT_LANE = "Retired from the OpenRouter lane: this vendor has a direct lane"


def is_alias(model_id: str) -> bool:
    return model_id.startswith("~")


def openrouter_id_for(provider: str, model_id: str) -> str | None:
    """The OpenRouter id for a direct-lane model, or None when there is no mapping."""
    if provider == "openai":
        return f"openai/{model_id}"
    if provider == "vertex":
        return f"google/{model_id}"
    if provider != "bedrock":
        return None
    bare = re.sub(r"^(us|eu|apac|global)\.", "", model_id)
    vendor, _, name = bare.partition(".")
    name = re.sub(r":\d+$", "", name)
    if vendor == "amazon":
        return f"amazon/{name}"
    if vendor == "anthropic":
        name = re.sub(r"-v\d+$", "", name)
        name = re.sub(r"-\d{8}$", "", name)
        name = re.sub(r"(?<=\d)-(?=\d)", ".", name)
        return f"anthropic/{name}"
    return None


def _same_host(db: Database, provider: str, openrouter_id: str) -> str | None:
    """The OpenRouter host that duplicates `provider` for this model, if any."""
    tags = [
        str(doc.get("provider_canonical") or "")
        for doc in db["bench_endpoints"].find({"model_id": openrouter_id, "enabled": True}, {"provider_canonical": 1})
    ]
    for tag in tags:
        if tag in SAME_HOST_ENDPOINTS.get(provider, ()):
            return tag
    if not tags and openrouter_id.split("/", 1)[0] in FIRST_PARTY_VENDORS.get(provider, ()):
        return "openrouter-default"
    return None


def duplicates(db: Database) -> list[dict[str, Any]]:
    """Enabled pairs that measure one model on one host, plus enabled aliases."""
    models = db[models_collection_name()]
    out: list[dict[str, Any]] = []
    for row in models.find({"enabled": True, "provider": {"$in": list(SAME_HOST_ENDPOINTS)}}):
        openrouter_id = openrouter_id_for(row["provider"], row["model_id"])
        if not openrouter_id:
            continue
        if not models.find_one({"provider": "openrouter", "model_id": openrouter_id, "enabled": True}, {"_id": 1}):
            continue
        host = _same_host(db, row["provider"], openrouter_id)
        if host:
            out.append(
                {
                    "kind": "direct",
                    "provider": row["provider"],
                    "model_id": row["model_id"],
                    "openrouter_id": openrouter_id,
                    "host": host,
                }
            )
    for row in models.find({"provider": "openrouter", "enabled": True, "model_id": {"$regex": "^~"}}, {"model_id": 1}):
        out.append({"kind": "alias", "provider": "openrouter", "model_id": row["model_id"]})
    return out


def plan(db: Database) -> dict[str, list[dict[str, Any]]]:
    """Duplicates to retire, and OpenRouter rows to bring back so nothing is lost.

    A direct row whose OpenRouter twin was disabled only because the direct lane
    owned it is also a duplicate: re-enable the twin and retire the direct row.
    """
    models = db[models_collection_name()]
    catalogue = {str(d["openrouter_id"]) for d in db["openrouter_catalog"].find({}, {"openrouter_id": 1})}
    found = duplicates(db)
    retire = [d for d in found if d["kind"] == "direct"]
    aliases = [d for d in found if d["kind"] == "alias"]
    reenable: list[dict[str, Any]] = []
    already = {(d["provider"], d["model_id"]) for d in retire}
    for row in models.find({"enabled": True, "provider": {"$in": list(SAME_HOST_ENDPOINTS)}}):
        if (row["provider"], row["model_id"]) in already:
            continue
        openrouter_id = openrouter_id_for(row["provider"], row["model_id"])
        if not openrouter_id or openrouter_id not in catalogue:
            continue
        twin = models.find_one({"provider": "openrouter", "model_id": openrouter_id})
        if (
            not twin
            or twin.get("enabled")
            or not str(twin.get("disabled_reason") or "").startswith(_RETIRED_FOR_DIRECT_LANE)
        ):
            continue
        host = _same_host(db, row["provider"], openrouter_id)
        if not host:
            continue
        reenable.append({"provider": "openrouter", "model_id": openrouter_id})
        retire.append(
            {
                "kind": "direct",
                "provider": row["provider"],
                "model_id": row["model_id"],
                "openrouter_id": openrouter_id,
                "host": host,
            }
        )
    return {"retire": retire, "reenable": reenable, "aliases": aliases}


def _drain(db: Database, staged: list[tuple[str, str, dict[str, Any]]], *, reason: str) -> list[str]:
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
        pending = remainder
    return applied


def run(db: Database, *, apply: bool) -> dict[str, Any]:
    work = plan(db)
    print(f"retire {len(work['retire'])} direct rows OpenRouter measures on the same host")
    for item in work["retire"]:
        print(f"    - {item['provider']}/{item['model_id']}  ==  openrouter/{item['openrouter_id']} via {item['host']}")
    print(f"re-enable {len(work['reenable'])} OpenRouter rows the retired direct lane had owned")
    for item in work["reenable"]:
        print(f"    + openrouter/{item['model_id']}")
    print(f"disable {len(work['aliases'])} OpenRouter ~*-latest aliases")
    for item in work["aliases"]:
        print(f"    - openrouter/{item['model_id']}")
    if not apply:
        print("\ndry run; pass --apply to make these changes")
        return {"applied": False, **{k: len(v) for k, v in work.items()}}

    batches: list[str] = []
    # Bring the OpenRouter twins back before retiring the direct rows, so no
    # model is ever between lanes.
    if work["reenable"]:
        batches += _drain(
            db,
            [
                (
                    "openrouter",
                    d["model_id"],
                    {"enabled": True, "disabled_reason": None, "reenabled_reason": REASON_ROUTED_BACK},
                )
                for d in work["reenable"]
            ],
            reason=REASON_ROUTED_BACK,
        )
    if work["retire"]:
        batches += _drain(
            db,
            [
                (d["provider"], d["model_id"], {"enabled": False, "disabled_reason": REASON_RETIRED})
                for d in work["retire"]
            ],
            reason=REASON_RETIRED,
        )
    if work["aliases"]:
        batches += _drain(
            db,
            [
                ("openrouter", d["model_id"], {"enabled": False, "disabled_reason": REASON_ALIAS})
                for d in work["aliases"]
            ],
            reason=REASON_ALIAS,
        )
    print(f"\n{len(batches)} batches applied; revert any with llm_bench.ops.mutations.revert")
    return {"applied": True, "batches": batches, **{k: len(v) for k, v in work.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="make the changes (default is a dry run)")
    args = parser.parse_args()
    client = MongoClient(os.environ["MONGODB_URI"])
    try:
        run(client[os.getenv("MONGODB_DB", "llm-bench")], apply=args.apply)
    finally:
        client.close()


if __name__ == "__main__":
    main()
