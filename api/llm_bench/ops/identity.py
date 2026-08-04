"""What model is this, and which endpoints are the same model.

`modelMapping.ts` is 377 hand-maintained lines and already wrong: `Meta-Llama-3-8B`
and `Meta-Llama-3-8B-Instruct` both map to `llama-3-8b`, which are different
models. Hand-maintenance is what this epic retires.

Placing an endpoint is one question with no scaffolding around it: here is every
group that exists, does this belong to one of them or is it new. The whole list
goes in the prompt. No attribute schema, no candidate filtering, no stopword
list deciding which groups the model is allowed to consider.

Two earlier versions got this wrong the same way. The first decomposed IDs into
developer/family/version/params, which only works for names that decompose that
way — Anthropic's tiers did not, so the prompt grew a list of how each vendor
names things, which is the 377-line table in a different file. The second kept
the question but pre-filtered the candidates by shared tokens, using a
hand-maintained list of "generic" words to ignore. Both were me deciding the
taxonomy and leaving the model to fill in the blanks.

What stays in code is chart policy, not identity: only unify names across
providers, never rename onto a name a provider already publishes, and let an
unmatched endpoint stand alone. Those say what a comparison chart should show,
and they do not change when a vendor invents a naming scheme.

Quantization is deliberately not identity. Measured on 2026-08-04: 1% of enabled
models declare it, splitting on it would affect one chart line, and on that line
provider infrastructure accounts for a 12x spread it does not explain.

The governing asymmetry is that a false merge is worse than a missed merge. A
wrong merge silently reports one provider as faster than another when the rows
are not comparable; a missed merge shows two lines, which is visible and
self-correcting. So an uncertain endpoint starts its own group.
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

# v4: the prompt carries every existing group. v3 pre-filtered candidates by
# shared tokens against a hand-maintained stopword list, which is a smaller
# version of the taxonomy v2 was replaced for.
POLICY_VERSION = 4


def identity_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODEL_IDENTITY", "bench_model_identity")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slug(value: str) -> str:
    """Stable form of a group name, so casing or spacing cannot split a group."""
    return re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower()).strip("-")


def current_identities(db: Database) -> list[dict[str, Any]]:
    """The newest relation per endpoint, which is what grouping should read."""
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in db[identity_collection_name()].find(sort=[("effective_from", 1)]):
        newest[(row["provider"], row["model_id"])] = row
    return list(newest.values())


def existing_groups(db: Database) -> dict[str, list[str]]:
    """Every group there is, with its members.

    All of them, every time. A few hundred short strings is nothing next to what
    a wrong merge costs, and any rule for trimming the list is a rule about which
    models resemble each other — the judgment being delegated in the first place.
    """
    groups: dict[str, list[str]] = {}
    for row in current_identities(db):
        key = row.get("canonical_key")
        if not key:
            continue
        entry = f"{row['provider']}/{row['model_id']}"
        members = groups.setdefault(key, [])
        if entry not in members:
            members.append(entry)
    return groups


MATCH_PROMPT = """Here are the model groups this benchmark site already tracks,
each with the provider endpoints in it:

{groups}

New endpoint: {provider} / {model_id}
Provider's display name: {name}

Does this endpoint serve the same model as one of the groups above, or is it a
model the list does not have yet?

Answer with JSON:
  {{"group": "<exact id from the list>"}}          if it is the same model
  {{"group": null, "name": "<short name>"}}        if it is new

Same model means the same weights, served by someone else. How a provider
serves it — turbo, fp8, a date stamp, their own suffix — does not make it a
different model. A different size, a different tier in a family, base versus
instruction-tuned weights, or a different version are different models.

If you are not certain it is the same model, say it is new. Two lines on a chart
is a visible mistake someone can fix. Merging two different models reports a
speed difference that does not exist, and nobody can see that it happened.
"""


def build_match_prompt(*, provider: str, model_id: str, name: str | None, groups: dict[str, list[str]]) -> str:
    rendered = (
        "\n".join(f"  {key}: {', '.join(sorted(members))}" for key, members in sorted(groups.items()))
        if groups
        else "  (the list is empty — this is the first endpoint)"
    )
    return MATCH_PROMPT.format(groups=rendered, provider=provider, model_id=model_id, name=name or "(none)")


def match_endpoint(
    db: Database,
    *,
    provider: str,
    model_id: str,
    name: str | None,
    call_llm: Callable[[str], dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Place an endpoint in an existing group, or start a new one."""
    now = now or utcnow()
    groups = existing_groups(db)
    answer = call_llm(build_match_prompt(provider=provider, model_id=model_id, name=name, groups=groups))

    chosen = answer.get("group")
    if chosen and chosen in groups:
        key, basis = chosen, "matched an existing group"
    elif chosen:
        # It named a group that is not on the list. Treating that as a match
        # would invent a merge target, so it starts its own group instead.
        key, basis = _slug(str(chosen)), "named a group not on the list; started its own"
    else:
        proposed = answer.get("name")
        if proposed:
            key, basis = _slug(str(proposed)), "new group"
        else:
            # It would not name the model. Rather than leave the endpoint with
            # no group — which would keep it out of the list forever, so a
            # second provider serving the same thing could never match it — key
            # it to its own ID. That is a group of one that others can join
            # later. The judgment stays the model's; only the label is ours.
            key = _slug(str(model_id).rsplit("/", 1)[-1])
            basis = "unnamed; keyed to its own id so it can be matched later"

    record = {
        "provider": provider,
        "model_id": model_id,
        "canonical_key": key,
        "policy_version": POLICY_VERSION,
        "effective_from": now,
        "evidence": {
            "groups_offered": len(groups),
            "basis": basis,
            "provider_display_name": name,
        },
        "resolved": key is not None,
    }
    db[identity_collection_name()].insert_one(dict(record))
    return record


# --------------------------------------------------------------------------
# The model call
# --------------------------------------------------------------------------

# Personal-funded OpenRouter, per the standing provider routing. This is one
# short question against a list, so the cheapest capable model is the right one.
DEFAULT_MODEL = os.getenv("BENCHMARK_IDENTITY_MODEL", "deepseek/deepseek-v4-flash-0731")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Generous on purpose. This model reasons before answering, and a cap that fits
# the answer but not the reasoning returns empty content — the same failure that
# currently blocks 19 models on the site.
MAX_TOKENS = 4000


def call_openrouter(prompt: str, *, model: str | None = None, timeout: float = 90.0) -> dict[str, Any]:
    """Ask a model to place an endpoint. Returns parsed JSON."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model or DEFAULT_MODEL,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = (response.json()["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("model returned no content; the token budget was probably spent on reasoning")
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------

CONSOLIDATE_PROMPT = """Here are the model groups this benchmark site tracks,
each with the provider endpoints in it:

{groups}

Some of these are the same model under two names, because groups are created one
endpoint at a time and whichever arrived first chose the name. For example
"llama-3.3-70b" and "llama-3.3-70b-instruct" were the same model split in two.

Which groups should be one group?

Answer with JSON:
  {{"merges": [{{"keep": "<group id>", "absorb": ["<group id>", ...]}}, ...]}}

Use group ids exactly as written above. Return {{"merges": []}} if none should
be merged.

Only merge groups that are the same weights. Different sizes, different tiers,
different versions, base versus instruction-tuned, and fine-tunes by other
people are all different models and must stay separate. If you are not certain
two groups are the same model, leave them apart.
"""


def build_consolidate_prompt(groups: dict[str, list[str]]) -> str:
    rendered = "\n".join(f"  {key}: {', '.join(sorted(members))}" for key, members in sorted(groups.items()))
    return CONSOLIDATE_PROMPT.format(groups=rendered)


def consolidate_groups(
    db: Database,
    *,
    call_llm: Callable[[str], dict[str, Any]],
    max_merges: int = 25,
    now: datetime | None = None,
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Merge groups that are the same model under two names.

    Groups are created one endpoint at a time, so the same model can end up
    under two names depending on which endpoint arrived first — Llama 3.3 70B
    split into `llama-3.3-70b` and `llama-3.3-70b-instruct` exactly that way.

    The alternative was to make naming deterministic, which is what the
    attribute schema tried and why it needed a vendor taxonomy. Asking the same
    question about the group list instead keeps the judgment where it belongs
    and needs no rules.
    """
    now = now or utcnow()
    groups = existing_groups(db)
    if len(groups) < 2:
        return []

    answer = call_llm(build_consolidate_prompt(groups))
    applied: list[dict[str, Any]] = []

    for merge in (answer.get("merges") or [])[:max_merges]:
        keep = merge.get("keep")
        absorb = [a for a in (merge.get("absorb") or []) if a in groups and a != keep]
        # Both sides must be groups that actually exist; inventing either would
        # move endpoints into a name nothing else uses.
        if keep not in groups or not absorb:
            continue
        applied.append({"keep": keep, "absorb": absorb, "endpoints": sum(len(groups[a]) for a in absorb)})
        if dry_run:
            continue
        for key in absorb:
            for row in current_identities(db):
                if row.get("canonical_key") != key:
                    continue
                db[identity_collection_name()].insert_one(
                    {
                        "provider": row["provider"],
                        "model_id": row["model_id"],
                        "canonical_key": keep,
                        "policy_version": POLICY_VERSION,
                        "effective_from": now,
                        "evidence": {"basis": f"consolidated from {key}", "groups_offered": len(groups)},
                        "resolved": True,
                    }
                )
    return applied
