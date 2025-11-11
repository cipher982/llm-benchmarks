from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List

from .collector import CatalogState, LifecycleSnapshot


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    MONITOR = "monitor"
    FAILING = "failing"
    STALE = "stale"
    LIKELY_DEPRECATED = "likely_deprecated"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    NEVER_SUCCEEDED = "never_succeeded"


@dataclass(slots=True)
class LifecycleDecision:
    status: LifecycleStatus
    confidence: str
    reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


def classify_snapshot(snapshot: LifecycleSnapshot, *, now: datetime) -> LifecycleDecision:
    metadata = snapshot.metadata
    successes = snapshot.successes
    errors = snapshot.errors

    reasons: List[str] = []
    actions: List[str] = []

    if metadata.deprecated:
        reasons.append("Model already flagged as deprecated in catalog metadata.")
        if metadata.deprecation_date:
            reasons.append(f"Deprecated date recorded: {metadata.deprecation_date.date().isoformat()}.")
        return LifecycleDecision(
            status=LifecycleStatus.DEPRECATED,
            confidence="high",
            reasons=reasons,
        )

    if not metadata.enabled:
        reasons.append("Model currently disabled in catalog (enabled=false).")
        return LifecycleDecision(
            status=LifecycleStatus.DISABLED,
            confidence="medium",
            reasons=reasons,
        )

    days_since_success = successes.age_days(now)
    days_since_error = errors.age_days(now)
    has_recent_success = successes.successes_30d > 0
    hard_errors_recent = errors.hard_failures_7d > 0 or errors.hard_failures_30d >= 3
    continuous_failures = (
        errors.errors_7d >= 5 and (days_since_error is not None) and (days_since_success is None or days_since_error < days_since_success)
    )

    if has_recent_success and not hard_errors_recent:
        reasons.append("Successful benchmarks observed within last 30 days.")
        if errors.errors_7d:
            reasons.append(f"{errors.errors_7d} soft error(s) occurred in last 7 days; monitor but no action required.")
            confidence = "medium"
        else:
            confidence = "high"
        return LifecycleDecision(
            status=LifecycleStatus.ACTIVE,
            confidence=confidence,
            reasons=reasons,
        )

    if hard_errors_recent and not has_recent_success:
        reasons.append("Hard failures (404/invalid/unsupported) recorded without any recent successes.")
        if errors.hard_failures_7d:
            reasons.append(f"{errors.hard_failures_7d} hard failure(s) in last 7 days.")
        if errors.hard_failures_30d and errors.hard_failures_30d != errors.hard_failures_7d:
            reasons.append(f"{errors.hard_failures_30d} hard failure(s) in last 30 days.")
        if snapshot.catalog_state == CatalogState.MISSING:
            reasons.append("Provider catalog no longer lists this model.")
            actions.append("mark_deprecated")
        actions.append("disable_scheduler_runs")
        return LifecycleDecision(
            status=LifecycleStatus.LIKELY_DEPRECATED,
            confidence="high" if snapshot.catalog_state != CatalogState.UNKNOWN else "medium",
            reasons=reasons,
            recommended_actions=actions,
        )

    if successes.last_success is None:
        reasons.append("No successful benchmarks recorded for this model yet.")
        if errors.errors_30d:
            reasons.append(f"Encountered {errors.errors_30d} error(s) in last 30 days without a single success.")
            actions.append("validate_credentials_or_model_id")
        if snapshot.catalog_state == CatalogState.MISSING:
            actions.append("remove_from_catalog")
            reasons.append("Provider catalog lookup could not find this model.")
        return LifecycleDecision(
            status=LifecycleStatus.NEVER_SUCCEEDED,
            confidence="medium",
            reasons=reasons,
            recommended_actions=actions,
        )

    if days_since_success is not None and days_since_success > 60:
        reasons.append(f"Last success {int(days_since_success)} days ago; exceeds 60-day freshness budget.")
        if snapshot.catalog_state == CatalogState.MISSING:
            reasons.append("Model missing from provider catalog.")
        if errors.errors_30d:
            reasons.append(f"{errors.errors_30d} error(s) observed in last 30 days.")
        actions.append("investigate_provider_catalog")
        return LifecycleDecision(
            status=LifecycleStatus.STALE,
            confidence="medium",
            reasons=reasons,
            recommended_actions=actions,
        )

    if continuous_failures:
        reasons.append("Multiple consecutive failures observed after the last successful benchmark.")
        if errors.errors_7d:
            reasons.append(f"{errors.errors_7d} failure(s) in last 7 days, no newer successes.")
        actions.append("inspect_error_logs")
        return LifecycleDecision(
            status=LifecycleStatus.FAILING,
            confidence="medium",
            reasons=reasons,
            recommended_actions=actions,
        )

    reasons.append("Model has mixed signals; continue monitoring.")
    if errors.errors_7d:
        reasons.append(f"Observed {errors.errors_7d} error(s) in last 7 days.")
    if days_since_success is not None:
        reasons.append(f"Last success {int(days_since_success)} days ago.")
    if snapshot.catalog_state == CatalogState.MISSING:
        actions.append("verify_catalog_entry")

    return LifecycleDecision(
        status=LifecycleStatus.MONITOR,
        confidence="medium",
        reasons=reasons,
        recommended_actions=actions,
    )
