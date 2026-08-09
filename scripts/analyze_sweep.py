#!/usr/bin/env python3
"""Analyze the throughput calibration sweep: fit T=a+N/lambda per model.

Usage: python analyze_sweep.py RESULTS.jsonl
For each model (non-censored rows only):
  - per-budget mean tps + std (consistency signal)
  - robust linear fit wall = a + b*N  (b = 1/lambda, so lambda = 1/b)
  - N_90 = 9*lambda*a  (budget where tps reaches 90% of lambda)
  - leave-one-budget-out stability of lambda
  - extrapolation check: refit excluding the 2 highest budgets, predict held-out
"""

import json
import statistics
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("numpy required")
    sys.exit(2)


def load(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = X @ beta
    ss = float(y.shape[0] * y.std() ** 2) or 1e-9
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss
    return beta, r2, pred


def main(path):
    rows = load(path)
    by = defaultdict(list)
    model_order = []
    for r in rows:
        if r.get("censored") or r.get("n_tokens") is None:
            continue
        key = r["label"]
        if key not in by:
            model_order.append(key)
        by[key].append(r)

    print(f"{'model':26s} {'n':>4s} {'lambda':>9s} {'a(s)':>7s} {'N90':>6s} {'tps64':>7s} {'r2':>5s}  N90-consistency")
    out = {}
    for key in model_order:
        rs = by[key]
        # per-budget stats
        per_b = defaultdict(list)
        for r in rs:
            per_b[r["requested_max_tokens"]].append(r["n_tokens"] / r["wall_seconds"])
        budgets = sorted(per_b)
        means = [(b, statistics.mean(per_b[b])) for b in budgets]
        stds = [(b, statistics.pstdev(per_b[b])) for b in budgets]

        x = np.array([r["n_tokens"] for r in rs], float)
        y = np.array([r["wall_seconds"] for r in rs], float)
        fit = ols(x, y)
        if fit is None:
            continue
        beta, r2, _ = fit
        a = float(beta[0])
        lam = 1.0 / float(beta[1])
        n90 = 9 * lam * a
        # tps at 64 from data
        t64 = next((m for b, m in means if b == 64), None)
        # leave-one-budget-out lambda stability
        lamos = []
        for leave in budgets:
            sub = [r for r in rs if r["requested_max_tokens"] != leave]
            if len(set(r["requested_max_tokens"] for r in sub)) < 2:
                continue
            s = ols(np.array([r["n_tokens"] for r in sub], float), np.array([r["wall_seconds"] for r in sub], float))
            if s and s[1][1] > 0:
                lamos.append(1.0 / s[1][1])
        loo_cv = (statistics.pstdev(lamos) / statistics.mean(lamos) * 100) if lamos else float("nan")

        # extrapolation: fit without the top 2 budgets, predict them
        hi = sorted(budgets)[-2:]
        sub = [r for r in rs if r["requested_max_tokens"] not in set(hi) and r["requested_max_tokens"] < min(hi)]
        # simpler: exclude > 512
        sub = [r for r in rs if r["requested_max_tokens"] <= 512]
        if len(set(r["requested_max_tokens"] for r in sub)) >= 2:
            sf = ols(np.array([r["n_tokens"] for r in sub], float), np.array([r["wall_seconds"] for r in sub], float))
            ext = None
            if sf:
                pa, plam = float(sf[0]), 1.0 / float(sf[1])
                # predict mean wall at 768,1024
                preds = [(b, pa + b / plam) for b in hi]
                actuals = [(b, stats_mean_tps(per_b[b])) for b in hi]
                ext = preds, actuals, plam
        else:
            ext = None

        print(
            f"{key:26s} {len(rs):>4d} {lam:9.2f} {a:7.3f} {n90:6.0f} "
            f"{t64:7.2f} {r2:5.3f}  LOO-CV {loo_cv:6.1f}%  slp={1/beta[1]:.0f}"
        )
        out[key] = {
            "lambda": lam,
            "a": a,
            "n90": n90,
            "tps64": t64,
            "r2": r2,
            "loo_cv": loo_cv,
            "extrap": ext,
            "per_budget": dict(means),
            "budget_std": dict(stds),
        }
    return out


def stats_mean_tps(list_):
    return statistics.mean(list_)


if __name__ == "__main__":
    main(sys.argv[1])
