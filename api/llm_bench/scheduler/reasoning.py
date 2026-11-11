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
- o3 refuses `minimal` too and accepts `low`.

So the ladder is none -> minimal -> low, tried in that order, and the first rung
an endpoint accepts is stored so later runs start there. A rejection is an HTTP
400 that bills nothing. If every rung is rejected the request omits the setting
(the provider default), which is what every request did before.

Budget follows from the rung. An endpoint at `none` (or one that does not take
the parameter) answers with almost no thinking, so it gets a short budget and
the stream is read to the end, which makes the provider's own usage count
available instead of a tokenizer estimate. If that short budget ever runs out
before any visible text, the endpoint is flagged `needs_headroom` and gets the
price-clamped budget with the early stop at the 64th visible token from then on.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any

from llm_bench.scheduler.mongo import collection_name

LADDER: tuple[str, ...] = ("none", "minimal", "low")
SHORT_BUDGET = int(os.getenv("BENCHMARK_SHORT_BUDGET_TOKENS", "128"))
REASONING_BUDGET_CAP = int(os.getenv("BENCHMARK_REASONING_BUDGET_CAP_TOKENS", "1024"))
OMITTED = "omitted"

# What a rung rejection looks like. OpenRouter says "Reasoning is mandatory" for
# `none`; an upstream that lacks a rung (o3 and `minimal`) answers a plain 400
# "Provider returned error". Any other failure is not about the rung.
_REJECTION = re.compile(r"Error code: 400|BadRequestError|reasoning is mandatory|unsupported.*reasoning", re.I)


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


def _supported_parameters(db: Any, model_id: str, endpoint_tag: str | None) -> list[str] | None:
    if endpoint_tag:
        doc = db["bench_endpoints"].find_one(
            {"model_id": model_id, "endpoint_tag": endpoint_tag}, {"supported_parameters": 1}
        )
    else:
        doc = db["openrouter_catalog"].find_one({"openrouter_id": model_id}, {"supported_parameters": 1})
    if not doc:
        return None
    params = doc.get("supported_parameters")
    return list(params) if isinstance(params, list) else None


def plan_for(db: Any, model_id: str, endpoint_tag: str | None) -> ReasoningPlan:
    """The rungs to try, starting from the stored floor, and the headroom flag."""
    stored = db[floors_collection_name()].find_one({"_id": floor_key(model_id, endpoint_tag)}) or {}
    needs_headroom = bool(stored.get("needs_headroom"))
    params = _supported_parameters(db, model_id, endpoint_tag)
    if params is None or "reasoning" not in params:
        # Sending a parameter the endpoint does not list fails the pinned
        # request outright (`require_parameters`), so do not send one.
        return ReasoningPlan(supported=False, rungs=(None,), needs_headroom=needs_headroom)
    floor = stored.get("floor")
    if floor == OMITTED:
        return ReasoningPlan(supported=True, rungs=(None,), needs_headroom=needs_headroom)
    start = LADDER.index(floor) if floor in LADDER else 0
    return ReasoningPlan(supported=True, rungs=(*LADDER[start:], None), needs_headroom=needs_headroom)


def is_rejection(message: str) -> bool:
    return bool(_REJECTION.search(message or ""))


def budget_for(base_budget: int, effort: str | None, plan: ReasoningPlan) -> tuple[int, bool]:
    """(max_tokens, read_to_end) for one attempt.

    `base_budget` is the price-clamped profile budget, already floored at the
    64-token mark. The short budget never raises it.
    """
    short = (effort == "none" or not plan.supported) and not plan.needs_headroom
    if short:
        return min(base_budget, max(SHORT_BUDGET, 64)), True
    if effort in ("minimal", "low"):
        return min(base_budget, REASONING_BUDGET_CAP), False
    return base_budget, False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_floor(db: Any, model_id: str, endpoint_tag: str | None, effort: str | None, *, rejected: list[str]) -> None:
    """Store the rung that worked, and the rungs that were refused on the way."""
    db[floors_collection_name()].update_one(
        {"_id": floor_key(model_id, endpoint_tag)},
        {
            "$set": {
                "model_id": model_id,
                "endpoint_tag": endpoint_tag,
                "floor": effort or OMITTED,
                "learned_at": _now(),
            },
            "$addToSet": {"rejected": {"$each": rejected}} if rejected else {"rejected": {"$each": []}},
        },
        upsert=True,
    )


def record_needs_headroom(db: Any, model_id: str, endpoint_tag: str | None, *, effort: str | None) -> None:
    """The short budget ran out before any visible text: give it room from now on."""
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
