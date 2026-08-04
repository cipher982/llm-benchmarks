"""What model is this, and which endpoints are the same model.

`modelMapping.ts` is 377 hand-maintained lines and already wrong: `Meta-Llama-3-8B`
and `Meta-Llama-3-8B-Instruct` both map to `llama-3-8b`, which are different
models. Hand-maintenance is what this epic retires.

The split: the *judgment* — what does this string denote — goes to a model,
where fuzzy world knowledge belongs. The *policy* — what counts as the same
model — stays in code, where it can be read, versioned and tested. Asking one
model to do both is how you get groupings that reshuffle when a new endpoint
appears.

Two properties follow from doing it this way. It is idempotent, because each
endpoint is normalised independently and cannot perturb an existing group. And
it is auditable, because grouping is a pure function you can run over stored
attributes without calling anything.

Quantization is deliberately not part of the key. Measured on 2026-08-04: 1% of
enabled models declare it, splitting on it would affect one chart line, and on
that line provider infrastructure accounts for a 12x spread it does not explain.
It is display annotation, not identity.

The governing asymmetry is that a false merge is worse than a missed merge. A
wrong merge silently reports one provider as faster than another when the rows
are not comparable; a missed merge shows two lines, which is visible and
self-correcting. So ambiguity leaves an endpoint on its own rather than guessing,
and that is the correct outcome under the no-review-queue rule rather than a
dodge around it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Callable

import httpx
from pymongo.database import Database

from llm_bench.scheduler.mongo import collection_name

# Bumped when the grouping function changes, so a stored relation records which
# policy produced it and old rows can be re-derived rather than trusted blindly.
POLICY_VERSION = 1


def identity_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_MODEL_IDENTITY", "bench_model_identity")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Attributes:
    """What an endpoint denotes. Everything here is evidence, not policy."""

    developer: str | None = None
    family: str | None = None
    version: str | None = None
    params: str | None = None
    role: str | None = None
    # Carried for display only. Never part of the key.
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether there is enough to group on at all."""
        return bool(self.developer and self.family)


def canonical_key(attrs: Attributes) -> str | None:
    """The grouping policy, as a pure function.

    Returns None when the attributes are too thin to group, which keeps the
    endpoint on its own rather than merging it into whatever it most resembles.
    """
    if not attrs.complete:
        return None
    parts = [attrs.developer, attrs.family]
    if attrs.version:
        parts.append(attrs.version)
    if attrs.params:
        parts.append(attrs.params)
    # Base and instruct-tuned weights are different models. The existing table
    # merges them, which is the concrete bug that motivated replacing it.
    parts.append(attrs.role or "base")
    return "-".join(_slug(p) for p in parts if p)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower()).strip("-")


def group_by_identity(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group endpoints by canonical key. Ungroupable endpoints stand alone.

    Pure and offline: it reads stored attributes and calls nothing, so a
    grouping can be recomputed and diffed without spending anything.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        attrs = row.get("attributes")
        key = canonical_key(attrs) if isinstance(attrs, Attributes) else None
        # An endpoint we cannot place gets a key of its own rather than joining
        # a group on a guess.
        groups.setdefault(key or f"unresolved:{row['provider']}/{row['model_id']}", []).append(row)
    return groups


def sibling_context(db: Database, *, model_id: str, limit: int = 8) -> list[dict[str, Any]]:
    """Endpoints at other providers whose IDs look related.

    Identity is a relation, not a property of one string. A base and an
    instruct-tuned sibling disambiguate each other; a lone ID cannot. This is
    the cheap version of that context — the rows the decision should see.
    """
    tokens = _search_tokens(model_id)
    if not tokens:
        return []
    pattern = "|".join(re.escape(t) for t in tokens)
    return list(
        db.provider_catalog.find(
            {"model_id": {"$regex": pattern, "$options": "i"}},
            {"_id": 0, "provider": 1, "model_id": 1, "name": 1},
        ).limit(limit)
    )


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


PROMPT = """You are identifying which model a provider's endpoint ID refers to.

Endpoint: {provider} / {model_id}
Provider's display name: {name}

Other endpoints with similar IDs, for disambiguation:
{siblings}

Return ONLY a JSON object with these fields:
  developer  the organisation that trained it, lowercase (meta, openai, mistralai, deepseek, qwen, google, anthropic)
  family     the model family, lowercase (llama, gpt, mixtral, deepseek-v3, qwen3)
  version    version number as written, or null (3.1, 4, 2.5)
  params     parameter count as written, or null (8b, 70b, 480b-a35b)
  role       one of: base, instruct, chat, reasoning, guard, code, or null if unclear

Rules:
- Report only what the evidence supports. Use null rather than guessing.
- "instruct" and the base model are DIFFERENT models. Do not conflate them.
- Ignore serving hints like turbo, fast, fp8 — they are not identity.
- If you cannot tell what the model is, return nulls. A missing answer is
  correct and expected; a confident wrong one merges unrelated series.
"""


def build_prompt(*, provider: str, model_id: str, name: str | None, siblings: list[dict[str, Any]]) -> str:
    rendered = "\n".join(f"  - {s['provider']} / {s['model_id']}" for s in siblings) if siblings else "  (none found)"
    return PROMPT.format(provider=provider, model_id=model_id, name=name or "(none)", siblings=rendered)


def attributes_from_response(payload: dict[str, Any]) -> Attributes:
    """Parse a model's answer, keeping only fields it actually filled in."""

    def clean(key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None if text not in {"null", "none", "unknown", ""} else None

    return Attributes(
        developer=clean("developer"),
        family=clean("family"),
        version=clean("version"),
        params=clean("params"),
        role=clean("role"),
    )


def resolve_endpoint(
    db: Database,
    *,
    provider: str,
    model_id: str,
    name: str | None,
    call_llm: Callable[[str], dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Derive and store one endpoint's identity, effective-dated.

    Relations are appended rather than overwritten. The dashboard applies
    today's mapping to old metric rows, so without effective dates a rename
    silently rewrites what past measurements claim to be about.
    """
    now = now or utcnow()
    siblings = sibling_context(db, model_id=model_id)
    prompt = build_prompt(provider=provider, model_id=model_id, name=name, siblings=siblings)
    attrs = attributes_from_response(call_llm(prompt))
    key = canonical_key(attrs)

    record = {
        "provider": provider,
        "model_id": model_id,
        "canonical_key": key,
        "attributes": {
            "developer": attrs.developer,
            "family": attrs.family,
            "version": attrs.version,
            "params": attrs.params,
            "role": attrs.role,
        },
        "policy_version": POLICY_VERSION,
        "effective_from": now,
        "evidence": {
            "sibling_count": len(siblings),
            "siblings": [f"{s['provider']}/{s['model_id']}" for s in siblings],
            "provider_display_name": name,
        },
        # Deliberately absent: any self-reported confidence score. It is not
        # calibrated probability and must not gate publication. What gates a
        # merge is whether the attributes are complete enough to key on.
        "resolved": key is not None,
    }
    db[identity_collection_name()].insert_one(dict(record))
    return record


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
