"""Fail-closed route decisions for source-provider benchmark jobs.

The route snapshot is evidence attached to a queued job.  It is deliberately
separate from the source provider and model identity, so a route can never
rename a source row or silently turn an unknown route into OpenRouter work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Mapping

ROUTE_DECISION_VERSION = "or-route-v1"
DIRECT_TRANSPORT = "direct"
OPENROUTER_TRANSPORT = "openrouter"
DIRECT_POLICY = "direct"
PINNED_PROVIDER_POLICY = "pinned-provider"


def _timestamp_is_expired(value: Any, *, now: datetime | None) -> bool:
    if not value:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return True
    elif isinstance(value, datetime):
        parsed = value
    else:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= now


@dataclass(frozen=True)
class RouteDecision:
    """The transport selected for one source model and one queued job."""

    source_provider: str
    source_model_id: str
    transport_provider: str = DIRECT_TRANSPORT
    transport_model_id: str | None = None
    route_model_id: str | None = None
    route_provider_slug: str | None = None
    observed_provider: str | None = None
    observed_provider_slug: str | None = None
    route_policy: str = DIRECT_POLICY
    route_snapshot_at: str | None = None
    route_probe_id: str | None = None
    route_decision_version: str = ROUTE_DECISION_VERSION
    state: str = "direct"
    reason: str = "missing-route-snapshot"

    @classmethod
    def direct(cls, source_provider: str, source_model_id: str, *, reason: str) -> "RouteDecision":
        return cls(
            source_provider=source_provider,
            source_model_id=source_model_id,
            transport_model_id=source_model_id,
            reason=reason,
        )

    @classmethod
    def from_snapshot(
        cls,
        source_provider: str,
        source_model_id: str,
        snapshot: Mapping[str, Any] | None,
        *,
        now: datetime | None = None,
    ) -> "RouteDecision":
        """Resolve a snapshot, returning direct for every invalid case."""

        if not isinstance(snapshot, Mapping):
            return cls.direct(source_provider, source_model_id, reason="missing-route-snapshot")

        if (
            snapshot.get("source_provider", source_provider) != source_provider
            or snapshot.get("source_model_id", source_model_id) != source_model_id
        ):
            return cls.direct(source_provider, source_model_id, reason="route-source-mismatch")

        if snapshot.get("route_decision_version") != ROUTE_DECISION_VERSION:
            return cls.direct(source_provider, source_model_id, reason="route-decision-version-mismatch")
        if source_provider == "bedrock":
            return cls.direct(source_provider, source_model_id, reason="bedrock-out-of-scope")
        if snapshot.get("state") != "active":
            return cls.direct(source_provider, source_model_id, reason="route-state-not-active")
        if snapshot.get("transport_provider") != OPENROUTER_TRANSPORT:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-transport")
        if snapshot.get("route_policy") != PINNED_PROVIDER_POLICY:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-policy")
        if _timestamp_is_expired(snapshot.get("expires_at") or snapshot.get("recheck_at"), now=now):
            return cls.direct(source_provider, source_model_id, reason="route-evidence-expired")

        required = (
            "route_model_id",
            "route_provider_slug",
            "observed_provider_slug",
            "route_snapshot_at",
            "route_probe_id",
        )
        if any(not snapshot.get(key) for key in required):
            return cls.direct(source_provider, source_model_id, reason="incomplete-route-evidence")
        if snapshot.get("provider_metadata_verified") is not True:
            return cls.direct(source_provider, source_model_id, reason="unverified-provider-metadata")
        if snapshot.get("observed_provider_slug") != snapshot.get("route_provider_slug"):
            return cls.direct(source_provider, source_model_id, reason="observed-provider-mismatch")

        route_model_id = str(snapshot["route_model_id"])
        if route_model_id.count("/") != 1:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-model-id")

        return cls(
            source_provider=source_provider,
            source_model_id=source_model_id,
            transport_provider=OPENROUTER_TRANSPORT,
            transport_model_id=route_model_id,
            route_model_id=route_model_id,
            route_provider_slug=str(snapshot["route_provider_slug"]),
            observed_provider=str(snapshot.get("observed_provider") or ""),
            observed_provider_slug=str(snapshot["observed_provider_slug"]),
            route_policy=PINNED_PROVIDER_POLICY,
            route_snapshot_at=snapshot.get("route_snapshot_at"),
            route_probe_id=snapshot.get("route_probe_id"),
            route_decision_version=ROUTE_DECISION_VERSION,
            state="active",
            reason="active-pinned-route",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metric_fields(self) -> dict[str, Any]:
        """Return additive provenance fields for a metric document."""

        fields = self.as_dict()
        fields.pop("reason", None)
        fields["route_state"] = fields.pop("state")
        fields["transport_attempt"] = "route" if self.transport_provider == OPENROUTER_TRANSPORT else DIRECT_TRANSPORT
        fields["route_reason"] = self.reason
        return fields


def freeze_route_snapshot(
    source_provider: str,
    source_model_id: str,
    snapshot: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Copy a route snapshot before placing it on a job document.

    A queued job must not retain a caller-owned mutable mapping.  Invalid or
    incomplete evidence is preserved as direct evidence, never promoted.
    """

    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise TypeError("route_snapshot must be a mapping")
    frozen = deepcopy(dict(snapshot))
    frozen.setdefault("source_provider", source_provider)
    frozen.setdefault("source_model_id", source_model_id)
    if now is not None and "queued_at" not in frozen:
        frozen["queued_at"] = now.isoformat()
    return frozen


def resolve_job_route(job: Mapping[str, Any], *, now: datetime | None = None) -> RouteDecision:
    """Resolve the immutable route snapshot attached to a queued job."""

    return RouteDecision.from_snapshot(
        str(job.get("provider") or ""),
        str(job.get("model_id") or ""),
        job.get("route_snapshot"),
        now=now,
    )
