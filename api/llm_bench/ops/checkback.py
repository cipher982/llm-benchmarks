"""Did the automation do its job without anyone helping it?

The invariants and the external dead man already answer "is the site healthy"
continuously. They cannot answer this, because every loop added on 2026-08-05
had never run unattended before that date — and three of them turned out to have
never run at all, having been written, tested and called from nowhere.

So this asks the one question no continuous check covers: since some cutoff, did
each autonomous loop actually produce an effect? A loop that runs and decides
nothing looks identical to a loop that is not running, and both look identical
to a healthy quiet period. The difference is whether there was work available.

Run it directly:
    python -m llm_bench.ops.checkback --since-hours 48
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from llm_bench.ops import admission
from llm_bench.ops import identity
from llm_bench.ops import reasoning_shadow
from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.mongo import probe_metrics_collection_name


def _finding(name: str, ok: bool, detail: str, evidence: Any = None) -> dict[str, Any]:
    return {"check": name, "ok": ok, "detail": detail, "evidence": evidence}


def admission_promoted_something(db, since: datetime) -> dict[str, Any]:
    """Zero promotions is only a fault when candidates were waiting.

    The deadlock this catches applied nothing for as long as admission had
    existed, while 43 models sat eligible. Nothing raised; the loop logged a
    refusal every two hours and no check looked at the outcome.
    """
    M = db[models_collection_name()]
    promoted = M.count_documents({"promoted_at": {"$gte": since}})
    pending = M.count_documents({"status": admission.CANDIDATE_STATUS})
    if promoted:
        return _finding("admission promotes unattended", True, f"{promoted} promoted since cutoff")
    if not pending:
        return _finding("admission promotes unattended", True, "nothing promoted, but no candidates were waiting")
    return _finding(
        "admission promotes unattended",
        False,
        f"nothing promoted while {pending} candidates wait — the decision pool is not draining",
        {"pending": pending},
    )


def reconciler_ran(db, since: datetime) -> dict[str, Any]:
    """Naming drift is what split a three-provider model into two chart lines.

    The reconciler had no caller at all until 2026-08-05, so this is the loop
    most likely to be silently absent.
    """
    relations = db[identity.identity_collection_name()].count_documents({"effective_from": {"$gte": since}})
    if relations:
        return _finding("reconciler runs unattended", True, f"{relations} identity relations written since cutoff")
    # No new endpoints means nothing to resolve, which is legitimate quiet.
    new_models = db[models_collection_name()].count_documents({"created_at": {"$gte": since}})
    if not new_models:
        return _finding("reconciler runs unattended", True, "no identity work, and no new models arrived to need any")
    return _finding(
        "reconciler runs unattended",
        False,
        f"{new_models} new models arrived and no identity relation was written — the loop is not running",
        {"new_models": new_models},
    )


def shadow_is_accumulating(db, since: datetime) -> dict[str, Any]:
    """The reasoning-model chart needs a distribution, not one point per model."""
    samples = db[probe_metrics_collection_name()].count_documents(
        {"benchmark_profile_id": reasoning_shadow.PROFILE_ID, "run_ts": {"$gte": since}}
    )
    summary = reasoning_shadow.summarize(db)
    enough = summary["models_measured"] >= 5 and summary["samples"] >= 40
    return _finding(
        "shadow profile accumulating",
        bool(samples),
        f"{samples} samples since cutoff; {summary['samples']} total across "
        f"{summary['models_measured']} models"
        + ("; enough to build the reasoning chart" if enough else "; not yet enough for a chart"),
        {"ready_for_chart": enough},
    )


def measurement_period_still_holds(db, since: datetime) -> dict[str, Any]:
    """The staleness horizon is derived from a number measured once.

    45 minutes was the real rotation over a 220-model catalogue. The catalogue
    grows, so this is the threshold most likely to drift into lying — a horizon
    that is too tight reports healthy models as starved, which is how a check
    stops being read.
    """
    from llm_bench.ops import invariants

    runs: dict[tuple[str, str], list[datetime]] = {}
    for row in db[metrics_collection_name()].find(
        {"run_ts": {"$gte": since}}, {"provider": 1, "model_name": 1, "run_ts": 1}
    ):
        runs.setdefault((row["provider"], row["model_name"]), []).append(row["run_ts"])

    gaps: list[float] = []
    for stamps in runs.values():
        stamps.sort()
        gaps.extend((b - a).total_seconds() for a, b in zip(stamps, stamps[1:]))
    if not gaps:
        return _finding("measurement period still holds", False, "no measurements in the window at all")

    gaps.sort()
    p95 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.95))] / 60
    configured = invariants.MODEL_MEASUREMENT_PERIOD.total_seconds() / 60
    ok = p95 <= configured * 1.5
    return _finding(
        "measurement period still holds",
        ok,
        f"observed p95 gap {p95:.1f} min against a configured {configured:.0f} min"
        + ("" if ok else " — raise BENCHMARK_MODEL_PERIOD_MINUTES or the staleness check will cry wolf"),
        {"observed_p95_minutes": round(p95, 1), "configured_minutes": configured},
    )


def decisions_are_draining(db, since: datetime) -> dict[str, Any]:
    """Anything past its own policy deadline means a pool jammed again."""
    from llm_bench.ops import invariants

    ctx = invariants.Context(db=db, now=datetime.now(timezone.utc))
    violations = invariants.pending_work_is_being_decided(ctx)
    return _finding(
        "decision pools draining",
        not violations,
        f"{len(violations)} item(s) past their decision deadline",
        [v.subject for v in violations[:10]],
    )


def dead_letters_are_classified(db, since: datetime) -> dict[str, Any]:
    """`unknown` was terminal until the classifier was wired up."""
    J = db[jobs_collection_name()]
    unknown = J.count_documents({"status": "dead_letter", "last_attempt_error_kind": "unknown"})
    total = J.count_documents({"status": "dead_letter"})
    share = (unknown / total * 100) if total else 0
    return _finding(
        "dead letters classified",
        share < 25,
        f"{unknown} of {total} dead letters still unknown ({share:.0f}%)",
    )


CHECKS = (
    admission_promoted_something,
    reconciler_ran,
    shadow_is_accumulating,
    decisions_are_draining,
    measurement_period_still_holds,
    dead_letters_are_classified,
)


def run(*, since_hours: int = 48) -> dict[str, Any]:
    _, db_name = mongo_env()
    client = mongo_client()
    try:
        db = client[db_name]
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        findings = [check(db, since) for check in CHECKS]
        return {
            "since_hours": since_hours,
            "failing": [f["check"] for f in findings if not f["ok"]],
            "findings": findings,
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Did the automation work unattended?")
    parser.add_argument("--since-hours", type=int, default=48)
    args = parser.parse_args()
    report = run(since_hours=args.since_hours)
    for finding in report["findings"]:
        print(f"{'ok  ' if finding['ok'] else 'FAIL'}  {finding['check']}: {finding['detail']}")
    print()
    print(json.dumps({"failing": report["failing"]}, indent=1))


if __name__ == "__main__":
    main()
