"""Ask every endpoint for the least reasoning it will accept.

Nearly every model now reasons by default. Sent no reasoning setting, 7 of 10
reasoning models probed on 2026-11-11 spent a 600-token budget thinking and
returned no visible text; reasoning was 76% of the generated tokens on a week of
published rows. Speed is measured on visible tokens, so that thinking is pure
cost, and it is the cause of `budget_exhausted`.

No single setting works everywhere (probe of 10 models x 5 settings):

- `none` disables reasoning on Qwen, DeepSeek, Kimi, GPT-6.1 and Gemini, and is
  refused as "Reasoning is mandatory" by GLM-5.3, GPT-5, Fable-5.2 and o3.
- `minimal` gives 0 reasoning tokens on GLM-5.3 and GPT-5, but turns reasoning
  ON for Qwen, DeepSeek and Kimi, which map it to their thinking mode.
- o3 refuses `minimal` too ("Unsupported value ... reasoning.effort") and
  accepts `low`.

So the ladder is none -> minimal -> low, tried in that order, and the first rung
an endpoint accepts is stored so later runs start there. A refusal is an HTTP
400 that names reasoning; any other failure ends the ladder. If every rung is
refused the request omits the setting (the provider default), which is what
every request did before. Stored floors expire after `FLOOR_TTL_DAYS`, so an
endpoint that starts accepting a lower rung is found again.

Every run still stops reading at the 64th visible token, so the published series
keeps one run length and the cl100k yardstick it has always had. Budget only
bounds what a provider that bills past a disconnect can charge: `none` (or no
reasoning parameter) gets `SHORT_BUDGET`, a reasoning rung gets at most
`REASONING_BUDGET_CAP`. If a short budget ends before the 64th visible token, the
endpoint is flagged `needs_headroom` and gets the price-clamped budget until the
flag expires.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler.mongo import collection_name

LADDER: tuple[str, ...] = ("none", "minimal", "low")
SHORT_BUDGET = int(os.getenv("BENCHMARK_SHORT_BUDGET_TOKENS", "128"))
REASONING_BUDGET_CAP = int(os.getenv("BENCHMARK_REASONING_BUDGET_CAP_TOKENS", "1024"))
FLOOR_TTL_DAYS = float(os.getenv("BENCHMARK_REASONING_FLOOR_TTL_DAYS", "14"))
OMITTED = "omitted"

_HTTP_400 = re.compile(r"Error code: 400|BadRequestError")
# The two refusal shapes seen on 2026-11-11: OpenRouter's own "Reasoning is
# mandatory for this endpoint" for `none`, and an upstream's
# `param: reasoning.effort` / "Unsupported value" for a rung it lacks.
_NAMES_REASONING = re.compile(r"reasoning is mandatory|reasoning\.effort|reasoning_effort", re.I)


def floors_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_REASONING_FLOORS", "bench_reasoning_floors")


def floor_key(model_id: str, endpoint_tag: str | None) -> str:
    return f"{model_id}|{endpoint_tag or ''}"


@dataclass(frozen=True)
class ReasoningPlan:
    """What to send, and how much budget to give, for one endpoint."""

    supported: bool
    rungs: tuple[str | None, ...]
    needs_headroom: bool = False

    @property
    def first(self) -> str | None:
        return self.rungs[0]


LEGACY_PLAN = ReasoningPlan(supported=False, rungs=(None,), needs_headroom=True)


def _supported_parameters(db: Any, model_id: str, endpoint_tag: str | None) -> list[str] | None:
    if endpoint_tag:
        doc = db[endpoint_discovery.endpoints_collection_name()].find_one(
            {"model_id": model_id, "endpoint_tag": endpoint_tag}, {"supported_parameters": 1}
        )
    else:
        doc = db["openrouter_catalog"].find_one({"openrouter_id": model_id}, {"supported_parameters": 1})
    if not doc:
        return None
    params = doc.get("supported_parameters")
    return list(params) if isinstance(params, list) else None


def _fresh(value: Any, now: datetime) -> bool:
    if not isinstance(value, datetime):
        return False
    stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return now - stamp < timedelta(days=FLOOR_TTL_DAYS)


def plan_for(
    db: Any,
    model_id: str,
    endpoint_tag: str | None,
    *,
    skip: tuple[str, ...] = (),
    now: datetime | None = None,
) -> ReasoningPlan:
    """The rungs to try, starting from the stored floor, and the headroom flag.

    `skip` drops rungs already refused earlier in the same job (the fallback
    attempt of a routed job), so a job never asks twice for a refused rung.
    """
    now = now or datetime.now(timezone.utc)
    stored = db[floors_collection_name()].find_one({"_id": floor_key(model_id, endpoint_tag)}) or {}
    needs_headroom = bool(stored.get("needs_headroom")) and _fresh(stored.get("needs_headroom_at"), now)
    params = _supported_parameters(db, model_id, endpoint_tag)
    if params is None or "reasoning" not in params:
        # Sending a parameter the endpoint does not list fails the pinned
        # request outright (`require_parameters`), so do not send one.
        return ReasoningPlan(supported=False, rungs=(None,), needs_headroom=needs_headroom)
    floor = stored.get("floor") if _fresh(stored.get("learned_at"), now) else None
    if floor == OMITTED:
        rungs: tuple[str | None, ...] = (None,)
    else:
        start = LADDER.index(floor) if floor in LADDER else 0
        rungs = (*[rung for rung in LADDER[start:] if rung not in skip], None)
    return ReasoningPlan(supported=True, rungs=rungs, needs_headroom=needs_headroom)


def is_rejection(message: str) -> bool:
    """An HTTP 400 that refuses the reasoning setting, and nothing else."""
    text = message or ""
    return bool(_HTTP_400.search(text) and _NAMES_REASONING.search(text))


def budget_for(base_budget: int, effort: str | None, plan: ReasoningPlan) -> int:
    """max_tokens for one attempt. Never above the price-clamped `base_budget`."""
    if plan.needs_headroom:
        return base_budget
    if effort == "none" or not plan.supported:
        return min(base_budget, max(SHORT_BUDGET, 64))
    if effort in ("minimal", "low"):
        return min(base_budget, REASONING_BUDGET_CAP)
    return base_budget


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_floor(db: Any, model_id: str, endpoint_tag: str | None, effort: str | None, *, rejected: list[str]) -> None:
    """Store the rung that worked, and the rungs that were refused on the way."""
    update: dict[str, Any] = {
        "$set": {
            "model_id": model_id,
            "endpoint_tag": endpoint_tag,
            "floor": effort or OMITTED,
            "learned_at": _now(),
        }
    }
    if rejected:
        update["$addToSet"] = {"rejected": {"$each": rejected}}
    db[floors_collection_name()].update_one({"_id": floor_key(model_id, endpoint_tag)}, update, upsert=True)


def record_needs_headroom(db: Any, model_id: str, endpoint_tag: str | None, *, effort: str | None) -> None:
    """The short budget ran out before the 64th visible token: give it room until the flag expires."""
    db[floors_collection_name()].update_one(
        {"_id": floor_key(model_id, endpoint_tag)},
        {
            "$set": {
                "model_id": model_id,
                "endpoint_tag": endpoint_tag,
                "needs_headroom": True,
                "needs_headroom_at": _now(),
                "needs_headroom_effort": effort or OMITTED,
            }
        },
        upsert=True,
    )
