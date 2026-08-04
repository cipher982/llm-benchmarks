"""What model is this, and which endpoints are the same model.

`modelMapping.ts` is 377 hand-maintained lines and already wrong: `Meta-Llama-3-8B`
and `Meta-Llama-3-8B-Instruct` both map to `llama-3-8b`, which are different
models. Hand-maintenance is what this epic retires.

An endpoint is placed by showing a model the groups that already exist and
asking which one it belongs to, or whether it is new. There is no attribute
schema. The first version decomposed IDs into developer/family/version/params
and assembled a key from them, which only works for names that decompose that
way — Anthropic's tiers did not, so the prompt grew a list of how each vendor
names things. That is the same table in a different file. Matching against real
groups needs no list, and a vendor with a convention nobody anticipated simply
forms its own group.

What stays in code is chart policy, not identity: only unify names across
providers, never rename onto a name a provider already publishes, and treat an
unmatched endpoint as its own line. Those are statements about what a
comparison chart should show, and they do not change when a vendor invents a
naming scheme.

Quantization is deliberately not identity. Measured on 2026-08-04: 1% of enabled
models declare it, splitting on it would affect one chart line, and on that line
provider infrastructure accounts for a 12x spread it does not explain.

The governing asymmetry is that a false merge is worse than a missed merge. A
wrong merge silently reports one provider as faster than another when the rows
are not comparable; a missed merge shows two lines, which is visible and
self-correcting. So an uncertain endpoint starts its own group, and that is the
correct outcome under the no-review-queue rule rather than a dodge around it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Callable

import httpx
from pymongo.database import Database

from llm_bench.scheduler.mongo import collection_name

# Bumped when the grouping function or the prompt changes, so a stored relation
# records which policy produced it and old rows can be re-derived rather than
# trusted blindly.
#
# v3: endpoints are matched against groups that already exist instead of being
# decomposed into developer/family/version/params. The decomposition only fitted
# names that decompose that way, so Anthropic's tiers had to be listed in the
# prompt — the hand-maintained table this epic retires, moved into a string.
POLICY_VERSION = 3


def identity_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODEL_IDENTITY", "bench_model_identity")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slug(value: str) -> str:
    """Stable form of a group name, so casing or spacing cannot split a group."""
    return re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower()).strip("-")


# Tokens that appear across unrelated models and so retrieve nothing useful.
# "Instruct" is the longest token in most IDs and the least distinctive.
GENERIC_TOKENS = frozenset(
    {
        "instruct",
        "chat",
        "turbo",
        "fast",
        "base",
        "preview",
        "latest",
        "versatile",
        "thinking",
        "reasoning",
        "exp",
        "beta",
        "free",
        "fp8",
        "bf16",
    }
)


def _search_tokens(model_id: str, *, limit: int = 2) -> list[str]:
    """The most distinctive chunks of an ID, used only to retrieve neighbours."""
    tail = str(model_id).rsplit("/", 1)[-1]
    tokens = [t for t in re.split(r"[-_.]", tail) if len(t) > 2 and t.lower() not in GENERIC_TOKENS]
    return tokens[:limit]


def current_identities(db: Database) -> list[dict[str, Any]]:
    """The newest relation per endpoint, which is what grouping should read."""
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in db[identity_collection_name()].find(sort=[("effective_from", 1)]):
        newest[(row["provider"], row["model_id"])] = row
    return list(newest.values())


# --------------------------------------------------------------------------
# The model call
# --------------------------------------------------------------------------

# Personal-funded OpenRouter, per the standing provider routing. Identity is
# cheap structured extraction, so a small fast model is the right tier; the
# expensive part of getting this wrong is a false merge, and the guard against
# that is the null-rather-than-guess rule, not model size.
DEFAULT_MODEL = os.getenv("BENCHMARK_IDENTITY_MODEL", "openai/gpt-5.6-luna")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(prompt: str, *, model: str | None = None, timeout: float = 45.0) -> dict[str, Any]:
    """Ask a model to extract identity attributes. Returns parsed JSON."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model or DEFAULT_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    # Some models still fence JSON even when asked not to.
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


# --------------------------------------------------------------------------
# Matching, which replaces the fixed attribute taxonomy
# --------------------------------------------------------------------------

MATCH_PROMPT = """Decide whether a provider's endpoint serves the same model as
any group that already exists.

Endpoint: {provider} / {model_id}
Provider's display name: {name}

Existing groups, with the endpoints already in them:
{candidates}

Answer with JSON:
  {{"group": "<exact id of an existing group>"}}       if it is the same model
  {{"group": null, "name": "<short lowercase name>"}}  if it is a new model

Same model means the same weights, served by someone. Serving differences —
turbo, fp8, a provider's own suffix, a date stamp — do not make it a different
model. These do:
- a different size (8b is not 70b)
- a different tier in a family, whatever the vendor calls them
- base weights versus instruction-tuned weights
- a different version or generation

If you are not sure it is the same model, return a new group. Two lines on a
chart is a visible, fixable mistake. Silently merging two different models
reports a speed difference that does not exist.
"""


def candidate_groups(db: Database, *, model_id: str, limit: int = 12) -> dict[str, list[str]]:
    """Existing groups whose members look related to this endpoint."""
    tokens = _search_tokens(model_id)
    if not tokens:
        return {}
    pattern = "|".join(re.escape(t) for t in tokens)
    groups: dict[str, list[str]] = {}
    for row in db[identity_collection_name()].find(
        {"canonical_key": {"$ne": None}, "model_id": {"$regex": pattern, "$options": "i"}},
        {"provider": 1, "model_id": 1, "canonical_key": 1},
    ):
        members = groups.setdefault(row["canonical_key"], [])
        entry = f"{row['provider']}/{row['model_id']}"
        if entry not in members:
            members.append(entry)
        if len(groups) >= limit:
            break
    return groups


def build_match_prompt(*, provider: str, model_id: str, name: str | None, candidates: dict[str, list[str]]) -> str:
    rendered = (
        "\n".join(f"  {key}: {', '.join(members)}" for key, members in sorted(candidates.items()))
        if candidates
        else "  (no existing groups look related)"
    )
    return MATCH_PROMPT.format(provider=provider, model_id=model_id, name=name or "(none)", candidates=rendered)


def match_endpoint(
    db: Database,
    *,
    provider: str,
    model_id: str,
    name: str | None,
    call_llm: Callable[[str], dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Place an endpoint in an existing group, or start a new one.

    No taxonomy. The earlier version asked for developer/family/version/params
    and assembled a key from them, which meant maintaining a list of how each
    vendor names things — Anthropic's tiers had to be spelled out in the prompt
    the moment Claude broke the schema. That is the hand-maintained table this
    epic exists to retire, moved into a string.

    Matching against groups that already exist needs no such list. A vendor with
    a naming convention nobody anticipated simply forms its own group.
    """
    now = now or utcnow()
    candidates = candidate_groups(db, model_id=model_id)
    answer = call_llm(build_match_prompt(provider=provider, model_id=model_id, name=name, candidates=candidates))

    chosen = answer.get("group")
    if chosen and chosen in candidates:
        key, basis = chosen, "matched an existing group"
    elif chosen:
        # It named a group that does not exist. Treating that as a match would
        # invent a merge target, so it starts its own group instead.
        key, basis = _slug(str(chosen)), "named a group that did not exist; started its own"
    else:
        proposed = answer.get("name")
        key = _slug(str(proposed)) if proposed else None
        basis = "new group" if key else "declined to name it"

    record = {
        "provider": provider,
        "model_id": model_id,
        "canonical_key": key,
        "policy_version": POLICY_VERSION,
        "effective_from": now,
        "evidence": {
            "candidates_offered": sorted(candidates),
            "basis": basis,
            "provider_display_name": name,
        },
        "resolved": key is not None,
    }
    db[identity_collection_name()].insert_one(dict(record))
    return record
