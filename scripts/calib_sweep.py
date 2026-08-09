#!/usr/bin/env python3
"""Throughput saturation calibration sweep (v2 protocol, sol-reviewed plan).

Per (provider, route, model), measure aggregate output throughput vs actual
generated tokens N (fit T = a + N / lambda). Randomized complete blocks per rep
(each rep runs to completion before the next), provider lanes, hard pre-launch
spend cap with worst-case charging on unknown outcomes.

Isolated: runs standalone in the bench container, writes only to the OUT file,
touches no Mongo, no scheduler. Every request carries its token budget and SDK
retries are disabled.

USAGE: python calib_sweep.py [--smoke]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
import time

from openai import AsyncOpenAI

PROBE = "Tell a long and happy story about the history of the world."
BUDGETS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
REPS = 10

# (label, provider, model, kind, base_url)
MODELS = [
    ("openai/luna", "openai", "gpt-5.6-luna", "responses", None),
    ("openai/terra", "openai", "gpt-5.6-terra", "responses", None),
    ("openai/sol", "openai", "gpt-5.6-sol", "responses", None),
    (
        "openrouter/dsv4-flash",
        "openrouter",
        "deepseek/deepseek-v4-flash-0731",
        "chat",
        "https://openrouter.ai/api/v1",
    ),
]
LANE_CONCURRENCY = {"openai": 4, "openrouter": 4}
# Pessimistic per-token $, used only to bound launching and worst-case charging.
PRICE_OUT = {"openai": 1.0e-5, "openrouter": 5.0e-7}
PRICE_IN = {"openai": 1.0e-5, "openrouter": 5.0e-7}
CAP_DOLLARS = float(os.getenv("CALIB_CAP_DOLLARS", "12"))
MAX_CALLS = int(os.getenv("CALIB_MAX_CALLS", "600"))
OUT = os.getenv("CALIB_OUT", "/tmp/calib_results.jsonl")
MIN_USABLE_TOKENS = 4


def timeout_for(budget: int) -> float:
    return max(20.0, budget * 0.15 + 15.0)


class Ledger:
    def __init__(self):
        self.spent = 0.0
        self.reserved = 0.0
        self.launched = 0
        self.lock = asyncio.Lock()


def _reserve(provider: str, budget: int) -> float:
    return PRICE_OUT[provider] * budget * 2.5 + PRICE_IN[provider] * len(PROBE.split())


def _prereq_responses(rdetail):
    if rdetail is None:
        return None
    rt = getattr(rdetail, "reasoning_tokens", None)
    return int(rt) if rt is not None else None


async def _responses(client, model, budget, timeout, t0):
    r = await client.responses.create(model=model, input=PROBE, max_output_tokens=budget, timeout=timeout, stream=False)
    usage = getattr(r, "usage", None)
    n = int(getattr(usage, "output_tokens", 0) or 0)
    rt = _prereq_responses(getattr(usage, "output_tokens_details", None))
    inc = getattr(r, "incomplete_details", None)
    finish = getattr(r, "status", None)
    if inc is not None and getattr(inc, "reason", None):
        finish = f"{finish}/{inc.reason}"
    return {
        "wall_seconds": round(time.perf_counter() - t0, 4),
        "n_tokens": n,
        "reasoning_tokens": rt,
        "finish": finish,
        "model_reported": getattr(r, "model", None),
    }


async def _chat(client, model, budget, timeout, t0):
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROBE}],
        max_tokens=budget,
        timeout=timeout,
        stream=False,
    )
    usage = getattr(r, "usage", None)
    n = int(getattr(usage, "completion_tokens", 0) or 0)
    fr = r.choices[0].finish_reason if r.choices else None
    return {
        "wall_seconds": round(time.perf_counter() - t0, 4),
        "n_tokens": n,
        "reasoning_tokens": None,
        "finish": fr,
        "model_reported": getattr(r, "model", None),
    }


async def run_one(sem, client, model_def, budget, block, pos, ledger, out_f):
    label, provider, model, kind, base_url = model_def
    async with sem:
        reserve = _reserve(provider, budget)
        async with ledger.lock:
            if ledger.spent + ledger.reserved + reserve > CAP_DOLLARS:
                return label, budget, None
            if ledger.launched >= MAX_CALLS:
                return label, budget, None
            ledger.reserved += reserve
            ledger.launched += 1

        attempt = {
            "label": label,
            "provider": provider,
            "model": model,
            "kind": kind,
            "base_url": base_url,
            "block": block,
            "pos": pos,
            "requested_max_tokens": budget,
            "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "lane": provider,
        }
        t0 = time.perf_counter()
        exc = None
        try:
            if kind == "responses":
                res = await _responses(client, model, budget, timeout_for(budget), t0)
            else:
                res = await _chat(client, model, budget, timeout_for(budget), t0)
        except Exception as e:  # noqa: BLE001
            exc = e

        if exc is not None:
            # Unknown outcome: the provider may still bill. Charge the reserve.
            attempt.update(
                {
                    "wall_seconds": round(time.perf_counter() - t0, 4),
                    "n_tokens": None,
                    "reasoning_tokens": None,
                    "finish": None,
                    "model_reported": None,
                    "censored": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            n = None
        else:
            attempt.update(res)
            n = res["n_tokens"]
            if n is None:
                attempt["censored"] = "missing usage"
            elif n < MIN_USABLE_TOKENS:
                attempt["censored"] = f"too few tokens ({n})"
            else:
                attempt["censored"] = None

        cost = (
            reserve if exc is not None else (PRICE_OUT[provider] * (n or 0) + PRICE_IN[provider] * len(PROBE.split()))
        )
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

    clients = {
        "openai": AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0),
        "openrouter": AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
        ),
    }
    sems = {p: asyncio.Semaphore(c) for p, c in LANE_CONCURRENCY.items()}
    ledger = Ledger()
    rng = random.Random(20260809)

    if args.smoke:
        cells = [(m, 16) for m in MODELS]
        with open("/tmp/calib_smoke.jsonl", "w") as out_f:
            for i, (m, b) in enumerate(cells):
                await run_one(sems[m[1]], clients[m[1]], m, b, block=0, pos=i, ledger=ledger, out_f=out_f)
        print(f"SMOKE done -> /tmp/calib_smoke.jsonl; spent=${ledger.spent:.4f}", flush=True)
        return 0

    expected = len(MODELS) * len(BUDGETS) * REPS
    print(
        f"sweep: {len(MODELS)} models x {len(BUDGETS)} budgets x {REPS} reps = {expected} cells; "
        f"lanes={LANE_CONCURRENCY}; cap=${CAP_DOLLARS}",
        flush=True,
    )

    blocks = []
    for _ in range(REPS):
        perm = [(m, b) for m in MODELS for b in BUDGETS]
        rng.shuffle(perm)
        blocks.append(perm)

    rows = 0
    with open(OUT, "w") as out_f:
        for block_idx, perm in enumerate(blocks):
            tasks = [
                asyncio.create_task(
                    run_one(sems[m[1]], clients[m[1]], m, b, block=block_idx, pos=pos, ledger=ledger, out_f=out_f)
                )
                for pos, (m, b) in enumerate(perm)
            ]
            results = await asyncio.gather(*tasks)  # surfaces exceptions (no return_exceptions)
            rows += sum(1 for r in results if r is not None and r[2] is not None)
            if (ledger.spent + ledger.reserved) >= CAP_DOLLARS:
                print("cap reached mid-sweep; stopping", flush=True)
                break

    print(
        f"done. launched={ledger.launched} spent=${ledger.spent:.4f} rows={rows} expected={expected} -> {OUT}",
        flush=True,
    )
    if rows == 0:
        print("FATAL: no rows written", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
