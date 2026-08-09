#!/usr/bin/env python3
"""Pinned-OpenRouter-vs-direct overhead sweep.

Same calibration protocol as calib_sweep.py (randomized complete blocks, lanes,
pre-launch cap, budget-dependent timeout, censoring), applied to the SAME model
served two ways, so the only difference is the OpenRouter hop:

  - direct   : provider API directly (deepinfra, fireworks)
  - pinned   : openrouter API, provider pinned via `provider.order`

Fit wall = a + N/lambda per arm. If lambda (throughput) is preserved and only
`a` (fixed latency/overhead) grows, OpenRouter costs latency but not speed.

Model: deepseek-v4-flash-0731 across deepinfra and fireworks routes.

USAGE: python overhead_sweep.py [--smoke]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time

from openai import AsyncOpenAI

PROBE = "Tell a long and happy story about the history of the world."
BUDGETS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
REPS = 8
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# (label, base_url, api_key_env, model_id, extra_body)
ARMS = [
    (
        "direct/deepinfra",
        "https://api.deepinfra.com/v1/openai",
        "DEEPINFRA_API_KEY",
        "deepseek-ai/deepseek-v4-flash-0731",
        None,
    ),
    (
        "pinned/deepinfra",
        OPENROUTER_BASE,
        "OPENROUTER_API_KEY",
        "deepseek/deepseek-v4-flash-0731",
        {"provider": {"order": ["deepinfra"]}},
    ),
    (
        "direct/fireworks",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        "accounts/fireworks/models/deepseek-v4-flash-0731",
        None,
    ),
    (
        "pinned/fireworks",
        OPENROUTER_BASE,
        "OPENROUTER_API_KEY",
        "deepseek/deepseek-v4-flash-0731",
        {"provider": {"order": ["fireworks"]}},
    ),
]

CAP_DOLLARS = float(os.getenv("OVERHEAD_CAP_DOLLARS", "12"))
MAX_CALLS = int(os.getenv("OVERHEAD_MAX_CALLS", "800"))
OUT = os.getenv("OVERHEAD_OUT", "/tmp/overhead_results.jsonl")
MIN_USABLE_TOKENS = 4
PER_ARM_CONCURRENCY = 4
# pessimistic (all cheap)
PRICE = 5.0e-7


def timeout_for(budget: int) -> float:
    return max(20.0, budget * 0.15 + 15.0)


class Ledger:
    def __init__(self):
        self.spent = 0.0
        self.reserved = 0.0
        self.launched = 0
        self.lock = asyncio.Lock()


def _reserve(budget: int) -> float:
    return PRICE * budget * 2.5 + PRICE * len(PROBE.split())


async def run_one(sem, client, arm, budget, block, pos, ledger, out_f):
    label, base_url, _, model, extra = arm
    async with sem:
        reserve = _reserve(budget)
        async with ledger.lock:
            if ledger.spent + ledger.reserved + reserve > CAP_DOLLARS:
                return label, budget, None
            if ledger.launched >= MAX_CALLS:
                return label, budget, None
            ledger.reserved += reserve
            ledger.launched += 1

        attempt = {
            "label": label,
            "base_url": base_url,
            "model": model,
            "extra_body": extra,
            "block": block,
            "pos": pos,
            "requested_max_tokens": budget,
            "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "lane": label.split("/")[1],
        }
        t0 = time.perf_counter()
        exc = None
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": PROBE}],
            "max_tokens": budget,
            "timeout": timeout_for(budget),
            "stream": False,
        }
        if extra is not None:
            kwargs["extra_body"] = extra
        try:
            r = await client.chat.completions.create(**kwargs)
            usage = getattr(r, "usage", None)
            n = int(getattr(usage, "completion_tokens", 0) or 0)
            fr = r.choices[0].finish_reason if r.choices else None
            wall = round(time.perf_counter() - t0, 4)
            reported = getattr(r, "model", None)
        except Exception as e:  # noqa: BLE001
            exc = e
            n, fr, wall, reported = None, None, round(time.perf_counter() - t0, 4), None

        if exc is not None:
            attempt.update(
                {
                    "wall_seconds": wall,
                    "n_tokens": None,
                    "finish": None,
                    "model_reported": None,
                    "censored": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            cost = reserve
        else:
            attempt.update({"wall_seconds": wall, "n_tokens": n, "finish": fr, "model_reported": reported})
            if n is None:
                attempt["censored"] = "missing usage"
            elif n < MIN_USABLE_TOKENS:
                attempt["censored"] = f"too few tokens ({n})"
            else:
                attempt["censored"] = None
            cost = PRICE * (n or 0) + PRICE * len(PROBE.split())

        async with ledger.lock:
            ledger.reserved -= reserve
            ledger.spent += cost
        out_f.write(json.dumps(attempt) + "\n")
        out_f.flush()
        return label, budget, attempt


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    clients = {}
    for label, base_url, key_env, *_ in ARMS:
        clients[label] = AsyncOpenAI(api_key=os.environ.get(key_env), base_url=base_url, max_retries=0)
    sems = {a[0]: asyncio.Semaphore(PER_ARM_CONCURRENCY) for a in ARMS}
    ledger = Ledger()

    if args.smoke:
        with open("/tmp/overhead_smoke.jsonl", "w") as out_f:
            for i, arm in enumerate(ARMS):
                await run_one(sems[arm[0]], clients[arm[0]], arm, 16, block=0, pos=i, ledger=ledger, out_f=out_f)
        print(f"SMOKE done -> /tmp/overhead_smoke.jsonl; spent=${ledger.spent:.4f}", flush=True)
        return 0

    import random

    rng = random.Random(20260809)
    expected = len(ARMS) * len(BUDGETS) * REPS
    print(
        f"sweep: {len(ARMS)} arms x {len(BUDGETS)} budgets x {REPS} reps = {expected} cells; " f"cap=${CAP_DOLLARS}",
        flush=True,
    )
    blocks = []
    for _ in range(REPS):
        perm = [(a, b) for a in ARMS for b in BUDGETS]
        rng.shuffle(perm)
        blocks.append(perm)

    rows = 0
    with open(OUT, "w") as out_f:
        for block_idx, perm in enumerate(blocks):
            tasks = [
                asyncio.create_task(
                    run_one(sems[a[0]], clients[a[0]], a, b, block=block_idx, pos=pos, ledger=ledger, out_f=out_f)
                )
                for pos, (a, b) in enumerate(perm)
            ]
            results = await asyncio.gather(*tasks)
            rows += sum(1 for r in results if r is not None and r[2] is not None)
            if (ledger.spent + ledger.reserved) >= CAP_DOLLARS:
                print("cap reached; stopping", flush=True)
                break
    print(
        f"done. launched={ledger.launched} spent=${ledger.spent:.4f} rows={rows} expected={expected} -> {OUT}",
        flush=True,
    )
    return 2 if rows == 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
