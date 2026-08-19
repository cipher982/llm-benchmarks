"""Continuous assertions over production state.

Every failure found during the 2026-08-04 recovery was two components that
individually worked, disagreeing — or a component reporting success while doing
nothing. The runner had 29 passing unit tests throughout an eight-day outage.
Isolated tests do not see that class of failure; assertions over live state do.

The first version of this module evaluated against live catalogue state, which
made it possible for a remediation to satisfy the very check that triggered it.
Checks that judge coverage now read an immutable `desired_set` snapshot taken
before the window they judge, so an action cannot change the verdict on the
window that produced it. Checks that judge agreement between two live components
(queue vs catalogue, spellings within the catalogue) still read live state,
because there the disagreement itself is the fault and remediating it does not
move a denominator.

Every run is recorded with its inputs, thresholds and results, so a later agent
can tell a check that passed from a check that never ran.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from pymongo.database import Database

from llm_bench.ops import desired_set as desired_set_module
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import collection_name
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import published_profile_filter

# Thresholds are versioned so a check run records which rules produced it.
THRESHOLD_VERSION = 4

# Discovery runs daily; two missed runs is a signal, one is a blip.
DISCOVERY_MAX_AGE = timedelta(hours=48)
# A lane that has written nothing this long is not merely between jobs.
#
# This was a hard 2h constant, which is the same mistake model_measurement_period
# below documents twice: a horizon pinned to one era's scheduling policy goes
# permanently red the moment the policy moves. It was already firing for vertex
# (3 enabled models, 3h cadence) before FRESH_MINUTES changed at all.
#
# Derived instead. A lane cannot write faster than its targets come around, so
# two turns is the narrowest horizon that can distinguish a stalled lane from a
# lane that is simply waiting. Floored at 2h so a fast cadence keeps the
# original sensitivity.
PROVIDER_PROGRESS_FLOOR = timedelta(hours=2)


def provider_progress_max_age() -> timedelta:
    return max(PROVIDER_PROGRESS_FLOOR, timedelta(seconds=policies.cadence_seconds() * 2))


# How often a model's turn actually comes around. This is not the invariant
# loop's cadence, which is how often we *look* — an unrelated number that the
# first version of this check multiplied by mistake, giving a one-hour horizon
# and reporting thirteen perfectly healthy models as starved.
#
# Derived from the scheduling policy, not measured and pinned. It was a
# separate env var defaulting to the 45 minutes observed on clifford in August
# 2026 — correct until FRESH_MINUTES moved and this did not, at which point the
# check asks whether models were measured within 3h while the scheduler
# deliberately waits 4.5h before rescheduling them. Permanently red, which is
# the failure this file's own history warns about twice.
#
# A model is rescheduled once its last success passes 1.5x the freshness
# horizon (health.compute_freshness_status), so that is the period, whatever
# FRESH_MINUTES is set to.
def model_measurement_period() -> timedelta:
    return timedelta(seconds=policies.cadence_seconds() * 1.5)


# Four turns missed in a row. Wide enough that ordinary jitter and a slow
# provider lane never fire it, narrow enough that a model which has genuinely
# stopped is named within a few hours.
MODEL_STALENESS_MULTIPLIER = 4
# Work that has sat unclaimed this long means the lane is not draining.
MAX_QUEUE_AGE = timedelta(hours=6)
# A terminal reason older than this must be re-probed rather than trusted.
TERMINAL_REASON_MAX_AGE = timedelta(days=7)
# Share of the desired set that may vanish between snapshots before it reads as
# the detector shrinking its own denominator rather than a legitimate change.
MAX_DESIRED_SET_SHRINK = 0.10


def check_runs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_CHECK_RUNS", "bench_check_runs")


def discovery_runs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_DISCOVERY_RUNS", "bench_discovery_runs")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CannotEvaluate(Exception):
    """The inputs a check needs are missing.

    Raised rather than returning no violations, because "I could not look" and
    "I looked and everything was fine" are the two states this module exists to
    keep apart.
    """


@dataclass
class Violation:
    """One offending record, with enough context to act on it."""

    subject: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Inputs shared by every check in one evaluation."""

    db: Database
    now: datetime
    cadence_seconds: int = 1800
    _snapshot: Any = None
    _snapshot_loaded: bool = False

    @property
    def desired(self):
        """The settled snapshot, or CannotEvaluate if none exists yet."""
        if not self._snapshot_loaded:
            self._snapshot = desired_set_module.for_evaluation(self.db, now=self.now)
            self._snapshot_loaded = True
        if self._snapshot is None:
            raise CannotEvaluate(
                "no desired_set snapshot older than the settling window; "
                "coverage cannot be judged against a denominator that does not exist"
            )
        return self._snapshot


@dataclass
class Result:
    name: str
    description: str
    ok: bool
    violations: list[Violation]
    remediable: bool
    checked_at: datetime
    error: str | None = None
    evaluated: bool = True

    @property
    def summary(self) -> str:
        if not self.evaluated:
            return f"{self.name}: COULD NOT EVALUATE — {self.error}"
        if self.error:
            return f"{self.name}: ERRORED — {self.error}"
        if self.ok:
            return f"{self.name}: ok"
        return f"{self.name}: {len(self.violations)} violation(s)"


@dataclass
class Invariant:
    name: str
    description: str
    check: Callable[[Context], list[Violation]]
    # True when a violation can be corrected without evidence beyond the
    # violation itself. Anything that spends money, changes what is published,
    # or removes a model from the desired set is deliberately not in this class.
    remediable: bool = False


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Agreement checks — live state on both sides, no denominator to move
# --------------------------------------------------------------------------


def no_work_for_disabled_models(ctx: Context) -> list[Violation]:
    """Queue and catalogue must agree about what should be benchmarked.

    On 2026-08-04, 223 of 252 queued or running jobs (88%) targeted models that
    had been disabled hours earlier. Demotion stopped new scheduling but never
    reached work already queued, and the dead-letter sweep kept resurrecting it.

    Reads live catalogue on purpose: the fault is the disagreement, and
    cancelling the job resolves it without touching either side's contents.
    """
    eligible = set(desired_set_module.capture_view(ctx.db))
    # Candidates under admission are deliberately not enabled — that is what
    # probing means. Their probe work is legitimate and must not read as a
    # queue/catalogue disagreement.
    probing = {
        (doc["provider"], doc["model_id"])
        for doc in ctx.db[models_collection_name()].find({"status": "probing"}, {"provider": 1, "model_id": 1})
    }
    violations = []
    for job in ctx.db[jobs_collection_name()].find(
        {"status": {"$in": ["queued", "running"]}},
        {"provider": 1, "model_id": 1, "status": 1, "sample_role": 1},
    ):
        key = (job.get("provider"), job.get("model_id"))
        if (job.get("sample_role") or "published") != "published":
            if key in probing:
                continue
        if key not in eligible:
            violations.append(
                Violation(
                    subject=f"{job.get('provider')}/{job.get('model_id')}",
                    detail=f"job {job['_id']} is {job.get('status')} but the model is not enabled",
                    data={"job_id": job["_id"], "provider": job.get("provider")},
                )
            )
    return violations


def no_case_duplicate_models(ctx: Context) -> list[Violation]:
    """The unique index on (provider, model_id) is case-sensitive.

    Qwen/Qwen2.5-7B-Instruct-Turbo and qwen/qwen2.5-7b-instruct-turbo are
    separate documents and can both be enabled, which benchmarks the model twice
    and shows it twice on the site.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for provider, model_id in desired_set_module.capture_view(ctx.db):
        seen.setdefault((provider, model_id.lower()), []).append(model_id)
    return [
        Violation(
            subject=f"{provider}/{lowered}",
            detail=f"enabled under {len(ids)} spellings: {', '.join(sorted(ids))}",
            data={"provider": provider, "model_ids": sorted(ids)},
        )
        for (provider, lowered), ids in sorted(seen.items())
        if len(ids) > 1
    ]


def no_job_is_stuck_in_queue(ctx: Context) -> list[Violation]:
    """Runnable work that is never claimed means a lane is not draining.

    Replaces the earlier dead-letter-growth check, which counted a pile rather
    than an outcome: a stable 1,409 dead letters is compatible with every model
    being measured and with none of them being measured.

    Measured from `not_before`, not `created_at`. A job in exponential backoff
    legitimately carries a months-old creation date while being retried on
    schedule; the first version of this check read six of those as stalls
    against production. Only work whose start time has already passed is owed a
    worker, which is also the exact condition `claim_next_job` selects on.
    """
    cutoff = ctx.now - MAX_QUEUE_AGE
    stale = ctx.db[jobs_collection_name()].aggregate(
        [
            {"$match": {"status": "queued", "not_before": {"$lt": cutoff}}},
            {"$group": {"_id": "$provider", "n": {"$sum": 1}, "oldest": {"$min": "$not_before"}}},
        ]
    )
    return [
        Violation(
            subject=row["_id"] or "unknown",
            detail=(
                f"{row['n']} job(s) runnable since {_as_utc(row['oldest']).isoformat()} "
                f"and still unclaimed after {MAX_QUEUE_AGE}"
            ),
            data={"provider": row["_id"], "count": row["n"]},
        )
        for row in stale
    ]


# --------------------------------------------------------------------------
# Coverage checks — judged against a settled snapshot
# --------------------------------------------------------------------------


def every_provider_is_progressing(ctx: Context) -> list[Violation]:
    """Each provider lane must be producing, not just the fleet as a whole.

    liveness_status is deliberately aggregate because it drives process exit;
    this is where a single dead lane is supposed to be caught.
    """
    desired = ctx.desired
    cutoff = ctx.now - provider_progress_max_age()
    # Published rows only. A lane emitting nothing but long-profile rows has a
    # dead published series; counting those rows here would hide exactly the
    # stall this check exists to catch. (Process liveness deliberately does
    # count them — see health.liveness_status.)
    recent = set(
        ctx.db[metrics_collection_name()].distinct(
            "provider", {"run_ts": {"$gte": cutoff}, **published_profile_filter()}
        )
    )
    return [
        Violation(
            subject=provider,
            detail=f"was expected to be benchmarking but wrote no metric since {cutoff.isoformat()}",
            data={"provider": provider, "expected_since": desired.captured_at.isoformat()},
        )
        for provider in sorted(set(desired.providers) - recent)
    ]


def pending_work_is_being_decided(ctx: Context) -> list[Violation]:
    """Nothing sits in a pending state past the deadline its own policy sets.

    Four separate outages on this site were the same shape: a guard that was
    correct in isolation became a permanent stop, and the symptom was silence.
    `max_attempts` with no way back removed a model on every transient failure
    and decayed coverage to 11.7%. A `[:limit]` slice meant twelve DeepInfra
    models were never scheduled, always the same twelve. A mutation cap refused
    an 87-change batch and applied nothing, every two hours, so 43 models that
    had earned promotion could never reach the site. A duplicate key aborted a
    batch mid-flight and one bad row blocked every other decision in it.

    None of them raised anything a person saw, and every unit test passed
    throughout, because each component worked. What they have in common is not
    the refusal — two of them dropped work without any refusal at all — it is
    that items entered a pool and never left it.

    So the check is on the pool, not the guard: admission says a candidate is
    decided within `ADMISSION_DEADLINE`, and a candidate older than that is
    proof the decider is not deciding, whatever the reason. A saturated bound, a
    silent truncation and a crash mid-pass are indistinguishable from here, and
    they need the same response.
    """
    from llm_bench.ops import admission

    # Generous slack over the policy deadline. One missed pass is not a jam;
    # a candidate a full day past its own deadline is.
    cutoff = ctx.now - (admission.ADMISSION_DEADLINE + timedelta(days=1))
    stuck = [
        doc
        for doc in ctx.db[models_collection_name()].find(
            {"status": admission.CANDIDATE_STATUS, "admission_started_at": {"$lt": cutoff}},
            {"provider": 1, "model_id": 1, "admission_started_at": 1},
        )
    ]
    return [
        Violation(
            subject=f"{doc['provider']}/{doc['model_id']}",
            detail=(
                f"still awaiting an admission decision, started "
                f"{_as_utc(doc.get('admission_started_at'))}; the deadline is "
                f"{admission.ADMISSION_DEADLINE.days}d, so the decider is not draining its pool"
            ),
            data={"provider": doc["provider"], "model_id": doc["model_id"]},
        )
        for doc in stuck
    ]


def endpoint_targets_are_being_measured(ctx: Context) -> list[Violation]:
    """No endpoint sits unmeasured past the cadence its own health doc sets.

    The same shape as `pending_work_is_being_decided`, one identity down. Three
    defects shipped on the night endpoint scheduling went live, and all three
    were invisible to every existing check because each one looked like health:

    - Endpoint completions were credited to the model's health doc, so endpoint
      docs stayed `never_run`, held maximum staleness priority forever, and the
      same 16 of 693 were re-run every 30 seconds while 677 were never measured.
    - A run that fell back to unpinned routing was recorded as an endpoint
      success, so 469 endpoints read `fresh` with zero rows carrying their tag.
    - A 9-provider slug allowlist could not verify 550 of 693 endpoints, which
      were served as unpinned fallbacks under an endpoint label.

    Counters could not see any of them: the scheduler enqueued, the workers
    completed, the rows were written. What separates all three from a healthy
    fleet is item age in an unmeasured state, which is what this reads.

    `last_success_at` is the only field trusted here, and deliberately not
    `freshness_status`: two of the three defects wrote a status that was wrong.
    A cross-check against rows actually carrying the endpoint's tag is what
    catches a success recorded for a run that measured something else.
    """
    violations: list[Violation] = []
    for doc in ctx.db[health_collection_name()].find(
        {"provider": "openrouter", "endpoint_tag": {"$ne": None}, "enabled": True},
        {
            "model_id": 1,
            "endpoint_tag": 1,
            "last_success_at": 1,
            "last_attempt_at": 1,
            "last_error_kind": 1,
            "last_error_message": 1,
            "cadence_seconds": 1,
        },
    ):
        # Skip endpoints whose parent model is explicitly disabled/quarantined in the catalogue
        parent_model = ctx.db[models_collection_name()].find_one({"model_id": doc["model_id"]})
        if parent_model is not None and not parent_model.get("enabled"):
            continue
        # The endpoint's own cadence, not a global one. The deadline this check
        # enforces has to be the deadline that endpoint's policy set, or it
        # measures the check's assumptions instead of the fleet.
        cadence = timedelta(seconds=int(doc.get("cadence_seconds") or policies.cadence_seconds()))
        # Generous slack. One missed pass is not starvation; an endpoint that
        # has gone three of its own cycles unmeasured is not being served.
        cutoff = ctx.now - (cadence * 3)
        last = _as_utc(doc.get("last_success_at"))
        last_attempt = _as_utc(doc.get("last_attempt_at"))
        if last is not None and last >= cutoff:
            continue
        if (
            last is None
            and last_attempt is not None
            and last_attempt >= cutoff
            and doc.get("last_error_kind")
            in (
                "timeout",
                "transient_provider",
                "network",
                "rate_limit",
            )
        ):
            continue
        # An endpoint the published profile cannot measure is a
        # measurement-design question, not a scheduling fault, and is reported
        # by models_measurable_by_the_published_profile instead. Scheduling it
        # harder only spends money.
        #
        # The verdict is read from the health doc first and from the endpoint's
        # own job second. Those disagreed for eight endpoints: the runs failed
        # before completions were credited per endpoint, so the job holds
        # `budget_exhausted` while the health doc holds nothing at all. Trusting
        # only the health doc would report them as starving and send someone to
        # fix a scheduler that is working.
        verdict = doc.get("last_error_kind")
        doc_msg = str(doc.get("last_error_message") or "").lower()
        job = ctx.db[jobs_collection_name()].find_one(
            {"provider": "openrouter", "model_id": doc["model_id"], "endpoint_tag": doc["endpoint_tag"]},
            {"last_attempt_error_kind": 1, "last_attempt_error_message": 1},
        )
        job_verdict = (job or {}).get("last_attempt_error_kind")
        job_msg = str((job or {}).get("last_attempt_error_message") or "").lower()
        if verdict in ("budget_exhausted", "hard_model", "hard_capability") or job_verdict in (
            "budget_exhausted",
            "hard_model",
            "hard_capability",
        ):
            continue
        if "visible output text is empty" in doc_msg or "visible output text is empty" in job_msg:
            continue
        subject = f"{doc['model_id']}:{doc['endpoint_tag']}"
        detail = (
            f"never measured; cadence is {int(cadence.total_seconds())}s"
            if last is None
            else f"last measured {last.isoformat()}, more than three cadences ago"
        )
        violations.append(
            Violation(
                subject=subject,
                detail=detail,
                data={"model_id": doc["model_id"], "endpoint_tag": doc["endpoint_tag"]},
            )
        )
    return violations


def endpoint_freshness_is_backed_by_rows(ctx: Context) -> list[Violation]:
    """An endpoint marked measured has rows that actually carry its tag.

    This is the check that catches a success credited to the wrong identity,
    which no age-based check can see: 469 endpoints read `fresh` with
    `successes_24h: 0` because a run that fell back to unpinned routing was
    recorded against the endpoint it failed to pin. The row was written
    honestly -- it carried no endpoint tag, because none was measured -- so the
    health record and the evidence disagreed, and only the health record was
    ever read.

    Comparing the two is cheap and is the only thing that would have caught it.
    """
    # Only recently-credited endpoints: an old record predates the identity
    # fixes and is covered by the starvation check above.
    cutoff = ctx.now - timedelta(seconds=policies.cadence_seconds() * 3)
    violations: list[Violation] = []
    for doc in ctx.db[health_collection_name()].find(
        {
            "provider": "openrouter",
            "endpoint_tag": {"$ne": None},
            "last_success_at": {"$ne": None, "$gte": cutoff},
        },
        {"model_id": 1, "endpoint_tag": 1, "last_success_at": 1},
    ):
        rows = ctx.db[metrics_collection_name()].count_documents(
            {"model_name": doc["model_id"], "route_endpoint_tag": doc["endpoint_tag"]},
            limit=1,
        )
        if rows:
            continue
        violations.append(
            Violation(
                subject=f"{doc['model_id']}:{doc['endpoint_tag']}",
                detail=(
                    f"marked measured at {_as_utc(doc.get('last_success_at'))} but no row carries "
                    "this endpoint tag; a run that measured something else was credited here"
                ),
                data={"model_id": doc["model_id"], "endpoint_tag": doc["endpoint_tag"]},
            )
        )
    return violations


def _unmeasurable_by_profile(ctx: Context) -> set[tuple[str, str]]:
    """Models whose newest attempt exhausted the published profile's budget.

    Read from what the runner recorded, not from a list of model names. A model
    that starts fitting inside the budget stops qualifying on its own.
    """
    return {
        (job["provider"], job["model_id"])
        for job in ctx.db[jobs_collection_name()].find(
            {"last_attempt_error_kind": "budget_exhausted"},
            {"provider": 1, "model_id": 1},
        )
    }


def models_measurable_by_the_published_profile(ctx: Context) -> list[Violation]:
    """Every enabled model can actually be measured by the profile we publish.

    Ten enabled models cannot. They answer correctly and quickly; the 64-token
    budget is spent on hidden reasoning before they emit anything visible, so
    they produce no publishable row however often they are scheduled.

    This is reported separately from starvation because the two need opposite
    responses. A starved model is a scheduling fault and someone should fix the
    scheduler. A model the profile cannot measure is a measurement-design
    question, and scheduling it harder only spends money.

    It is a violation rather than a silent exemption because the site claims to
    measure these models and does not. The shadow profile is collecting real
    numbers for them; the open question is whether those numbers may be
    published alongside 64-token rows.
    """
    unmeasurable = _unmeasurable_by_profile(ctx)
    if not unmeasurable:
        return []
    enabled = set(desired_set_module.capture_view(ctx.db))
    return [
        Violation(
            subject=f"{provider}/{model_id}",
            detail="enabled, but the published 64-token profile cannot produce a visible-token measurement",
            data={"provider": provider, "model_id": model_id},
        )
        for provider, model_id in sorted(unmeasurable & enabled)
    ]


def desired_models_are_being_measured(ctx: Context) -> list[Violation]:
    """Every model the system intended to measure has recent data.

    The denominator is the snapshot, not the live catalogue, so disabling a
    starved model does not retroactively make this pass. A model that has since
    been removed from the catalogue is still reported, tagged with that fact, so
    the shrinkage is visible in the same result rather than hidden by it.

    A model the published profile cannot measure is not starved. A reasoning
    model spends the whole 64-token budget on hidden reasoning and emits nothing
    visible, so it will never produce a published row no matter how often it is
    scheduled. Counting those here would make this check permanently red for a
    reason no amount of scheduling can fix, and a check that can never go green
    is one people stop reading — which is how the eight-day outage stayed
    invisible. They are reported by `models_measurable_by_the_published_profile`
    instead, where the number means something that can be acted on.
    """
    desired = ctx.desired
    horizon = ctx.now - model_measurement_period() * MODEL_STALENESS_MULTIPLIER
    # Published rows only: a model succeeding only at the long profile is
    # starved for the series the site actually shows, which is the exact
    # condition this check must report.
    fresh = set(
        ctx.db[metrics_collection_name()].distinct(
            "model_name", {"run_ts": {"$gte": horizon}, **published_profile_filter()}
        )
    )
    still_enabled = set(desired_set_module.capture_view(ctx.db))
    unmeasurable = _unmeasurable_by_profile(ctx)

    violations = []
    for provider, model_id in desired.models:
        if model_id in fresh or (provider, model_id) in unmeasurable:
            continue
        removed = (provider, model_id) not in still_enabled
        violations.append(
            Violation(
                subject=f"{provider}/{model_id}",
                detail=(
                    "no successful measurement within the staleness horizon"
                    + (" and has since been removed from the catalogue" if removed else "")
                ),
                data={
                    "provider": provider,
                    "model_id": model_id,
                    "removed_since_snapshot": removed,
                },
            )
        )
    return violations


def desired_set_is_not_silently_shrinking(ctx: Context) -> list[Violation]:
    """Catch the detector making itself green by measuring less.

    Individually reasonable demotions can add up to a system that reports
    perfect coverage of a catalogue it quietly emptied. Coverage decayed to
    11.7% in exactly that shape, with no single incident to point at.
    """
    desired = ctx.desired
    if not desired.models:
        return []
    delta = desired_set_module.drift(ctx.db, since=desired, now=ctx.now)
    removed = delta["removed"]
    limit = max(1, int(len(desired.models) * MAX_DESIRED_SET_SHRINK))
    if len(removed) <= limit:
        return []
    return [
        Violation(
            subject="desired_set",
            detail=(
                f"{len(removed)} of {len(desired.models)} models left the catalogue since "
                f"{desired.captured_at.isoformat()} (limit {limit})"
            ),
            data={"removed": removed[:50], "removed_count": len(removed), "limit": limit},
        )
    ]


# --------------------------------------------------------------------------
# Evidence checks — is the input the rest of this depends on real
# --------------------------------------------------------------------------


def discovery_completed_recently(ctx: Context) -> list[Violation]:
    """A completed provider sync, not a row timestamp, is the observable event.

    `provider_catalog.last_seen_at` cannot express this. Requiring every row to
    be fresh fires forever on genuinely retired models; requiring the newest row
    to be fresh passes on a partial one-row response. Deprecation decisions are
    downstream of this, so a partial run must never look like a complete one.
    """
    runs = ctx.db[discovery_runs_collection_name()]
    if runs.estimated_document_count() == 0:
        raise CannotEvaluate(
            "no discovery run ledger exists; provider absence cannot be distinguished " "from a failed or partial sync"
        )
    cutoff = ctx.now - DISCOVERY_MAX_AGE
    latest_by_provider: dict[str, dict[str, Any]] = {}
    for run in runs.find({"status": "completed"}, sort=[("finished_at", -1)]):
        latest_by_provider.setdefault(run.get("provider"), run)

    violations = []
    for provider in sorted(ctx.desired.providers):
        run = latest_by_provider.get(provider)
        if run is None:
            violations.append(
                Violation(
                    subject=provider,
                    detail="no completed discovery run on record",
                    data={"provider": provider},
                )
            )
            continue
        finished = _as_utc(run.get("finished_at"))
        if finished is None or finished < cutoff:
            violations.append(
                Violation(
                    subject=provider,
                    detail=f"last completed discovery run {finished.isoformat() if finished else 'unknown'}",
                    data={"provider": provider},
                )
            )
        elif not run.get("pagination_complete", False):
            violations.append(
                Violation(
                    subject=provider,
                    detail="most recent run completed without exhausting pagination",
                    data={"provider": provider, "run_id": run.get("_id")},
                )
            )
    return violations


# Classes where the provider may start serving the model again on its own.
RECOVERABLE_DISABLED_CLASSES = {
    "billing",
    "auth",
    "rate_limit",
    "timeout",
    "transient_provider",
    "quota",
}
# Classes that are settled: the model is gone, unsuitable, or superseded.
PERMANENT_DISABLED_CLASSES = {
    "duplicate_spelling",
    "provider_retired",
    "hard_model",
    "unsuitable",
    "deprecated",
}
# Legacy rows carry free-text reasons written by the old operator rather than a
# class. These markers are what a recoverable condition looks like in prose.
RECOVERABLE_REASON_MARKERS = (
    "billing",
    "payment",
    "credit",
    "insufficient",
    "quota",
    "rate limit",
    "402",
    "401",
    "auth",
    "api key",
)


def _is_recoverable_reason(doc: dict[str, Any]) -> bool:
    """Whether this model could come back without anyone changing the catalogue."""
    klass = doc.get("disabled_class")
    if klass:
        return klass in RECOVERABLE_DISABLED_CLASSES
    reason = (doc.get("disabled_reason") or "").lower()
    return any(marker in reason for marker in RECOVERABLE_REASON_MARKERS)


def terminal_reasons_are_current(ctx: Context) -> list[Violation]:
    """A *recoverable* terminal reason must expire, or it suppresses forever.

    DeepInfra's 402 cleared when the balance returned; a `billing` label written
    once and trusted forever would have kept the whole provider dead. So a
    reason in a recoverable class needs a recheck date.

    Scope matters as much as the rule. The first version asked this of every
    disabled model and fired on 466 of them — Anyscale models whose public API
    shut down in 2024, Claude 2, provider-retired checkpoints. Those are
    correctly dead and re-probing them would spend money to learn nothing. Only
    live providers and recoverable classes are in scope.
    """
    cutoff = ctx.now - TERMINAL_REASON_MAX_AGE
    live_providers = {provider for provider, _ in desired_set_module.capture_view(ctx.db)}
    violations = []
    for doc in ctx.db[models_collection_name()].find(
        {
            "enabled": False,
            "deprecated": {"$ne": True},
            "provider": {"$in": sorted(live_providers)},
            "disabled_reason": {"$exists": True, "$ne": None},
        },
        {
            "provider": 1,
            "model_id": 1,
            "disabled_reason": 1,
            "disabled_class": 1,
            "disabled_at": 1,
            "recheck_after": 1,
        },
    ):
        if not _is_recoverable_reason(doc):
            continue
        recheck = _as_utc(doc.get("recheck_after"))
        disabled_at = _as_utc(doc.get("disabled_at"))
        if recheck is not None:
            if recheck < ctx.now:
                violations.append(
                    Violation(
                        subject=f"{doc.get('provider')}/{doc.get('model_id')}",
                        detail=f"recheck was due {recheck.isoformat()} and has not happened",
                        data={"provider": doc.get("provider"), "model_id": doc.get("model_id")},
                    )
                )
            continue
        if disabled_at is None or disabled_at < cutoff:
            violations.append(
                Violation(
                    subject=f"{doc.get('provider')}/{doc.get('model_id')}",
                    detail=(
                        f"disabled_reason {doc.get('disabled_reason')!r} carries no recheck_after "
                        "and is older than the trust window"
                    ),
                    data={"provider": doc.get("provider"), "model_id": doc.get("model_id")},
                )
            )
    return violations


INVARIANTS: list[Invariant] = [
    Invariant(
        "no_work_for_disabled_models",
        "Queued and running jobs only target enabled models",
        no_work_for_disabled_models,
        remediable=True,
    ),
    Invariant(
        "no_case_duplicate_models",
        "No model is enabled under more than one spelling",
        no_case_duplicate_models,
        remediable=True,
    ),
    Invariant(
        "no_job_is_stuck_in_queue",
        "No job has sat queued past the drain window",
        no_job_is_stuck_in_queue,
    ),
    Invariant(
        "every_provider_is_progressing",
        "Every provider in the settled desired set wrote a metric recently",
        every_provider_is_progressing,
    ),
    Invariant(
        "desired_models_are_being_measured",
        "Every model in the settled desired set has a recent successful measurement",
        desired_models_are_being_measured,
    ),
    Invariant(
        "pending_work_is_being_decided",
        "Nothing waits in a pending pool past the deadline its own policy sets",
        pending_work_is_being_decided,
    ),
    Invariant(
        "endpoint_targets_are_being_measured",
        "No endpoint sits unmeasured past three of its own cadences",
        endpoint_targets_are_being_measured,
    ),
    Invariant(
        "endpoint_freshness_is_backed_by_rows",
        "Every endpoint marked measured has a row carrying its tag",
        endpoint_freshness_is_backed_by_rows,
    ),
    Invariant(
        "models_measurable_by_the_published_profile",
        "Every enabled model can produce a publishable measurement under the published profile",
        models_measurable_by_the_published_profile,
    ),
    Invariant(
        "desired_set_is_not_silently_shrinking",
        "The catalogue is not being emptied faster than the shrink limit",
        desired_set_is_not_silently_shrinking,
    ),
    Invariant(
        "discovery_completed_recently",
        "Every expected provider has a recent complete discovery run",
        discovery_completed_recently,
    ),
    Invariant(
        "terminal_reasons_are_current",
        "No model is suppressed by a terminal reason that never expires",
        terminal_reasons_are_current,
    ),
]


def evaluate(
    db: Database,
    *,
    now: datetime | None = None,
    only: set[str] | None = None,
    cadence_seconds: int = 1800,
    record: bool = True,
) -> list[Result]:
    """Run every invariant and record the run.

    A check that raises reads as failed, never as skipped. A check that cannot
    reach its inputs reads as unevaluated — also not a pass, but distinguished,
    because "the denominator is missing" needs a different fix from "the
    denominator is wrong".
    """
    now = now or utcnow()
    ctx = Context(db=db, now=now, cadence_seconds=cadence_seconds)
    results = []
    for inv in INVARIANTS:
        if only and inv.name not in only:
            continue
        try:
            violations = inv.check(ctx)
            results.append(Result(inv.name, inv.description, not violations, violations, inv.remediable, now))
        except CannotEvaluate as exc:
            results.append(
                Result(inv.name, inv.description, False, [], inv.remediable, now, error=str(exc), evaluated=False)
            )
        except Exception as exc:  # noqa: BLE001
            # An invariant that cannot run tells us nothing, so it must not
            # read as green. That is the exact failure this module exists for.
            results.append(
                Result(
                    inv.name,
                    inv.description,
                    False,
                    [],
                    inv.remediable,
                    now,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    if record:
        record_check_run(db, results, now=now, cadence_seconds=cadence_seconds)
    return results


def record_check_run(
    db: Database,
    results: list[Result],
    *,
    now: datetime,
    cadence_seconds: int,
) -> None:
    """Append an immutable record of one evaluation.

    Without this, a period with no alerts is indistinguishable from a period
    where nothing ran — which is how a disabled Sauron job passed for three
    months as a live data source.
    """
    db[check_runs_collection_name()].insert_one(
        {
            "checked_at": now,
            "threshold_version": THRESHOLD_VERSION,
            "cadence_seconds": cadence_seconds,
            "results": [
                {
                    "name": r.name,
                    "ok": r.ok,
                    "evaluated": r.evaluated,
                    "violation_count": len(r.violations),
                    "error": r.error,
                    "subjects": [v.subject for v in r.violations[:50]],
                }
                for r in results
            ],
        }
    )
