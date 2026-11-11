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
# The site measures one named OpenRouter endpoint and publishes it as that
# endpoint. What matters is only that the request reached the endpoint it
# named, which `allow_fallbacks: False` and the observed-slug check already
# enforce on every run. It is the only policy that routes anywhere: the older
# paired-canary and marketplace (or-served) policies were retired with their
# direct lanes.
PINNED_ENDPOINT_POLICY = "pinned-endpoint"


@dataclass(frozen=True)
class RouteDecision:
    """The transport selected for one source model and one queued job."""

    source_provider: str
    source_model_id: str
    transport_provider: str = DIRECT_TRANSPORT
    transport_model_id: str | None = None
    route_model_id: str | None = None
    route_provider_slug: str | None = None
    # The exact endpoint to pin. `route_provider_slug` is the provider family
    # and stays the verification key, because OpenRouter's response reports a
    # display name and never echoes the tag: `deepinfra/fp8` and
    # `deepinfra/turbo` are indistinguishable once a generation completes.
    # Pinning the family instead of the tag would load-balance the variants,
    # which is the non-determinism this exists to remove.
    route_endpoint_tag: str | None = None
    # What that endpoint is actually running. An fp4 deployment is faster than a
    # bf16 one because it is a smaller artifact, not because the provider is
    # quicker, so the row cannot publish the speed without it.
    route_endpoint_quantization: str | None = None
    # The display name OpenRouter itself reported for this endpoint at
    # discovery. It is the only key that can confirm a pin for an arbitrary
    # provider: the response echoes a display name, never the tag, and a
    # hand-written slug allowlist covered 9 of the fleet's 65 providers.
    route_provider_display: str | None = None
    observed_provider: str | None = None
    observed_provider_slug: str | None = None
    route_policy: str = DIRECT_POLICY
    route_snapshot_at: str | None = None
    route_reasoning_exclude: bool = False
    profile_hash: str | None = None
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
        if snapshot.get("state") != "active":
            return cls.direct(source_provider, source_model_id, reason="route-state-not-active")
        if snapshot.get("transport_provider") != OPENROUTER_TRANSPORT:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-transport")
        if snapshot.get("route_policy") != PINNED_ENDPOINT_POLICY:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-policy")
        tag = snapshot.get("route_endpoint_tag")
        if not tag:
            return cls.direct(source_provider, source_model_id, reason="endpoint-route-has-no-tag")
        return cls(
            source_provider=source_provider,
            source_model_id=source_model_id,
            transport_provider=OPENROUTER_TRANSPORT,
            transport_model_id=str(snapshot.get("route_model_id") or source_model_id),
            route_model_id=str(snapshot.get("route_model_id") or source_model_id),
            route_provider_slug=str(snapshot.get("route_provider_slug") or ""),
            route_endpoint_tag=str(tag),
            route_endpoint_quantization=(
                str(snapshot["route_endpoint_quantization"]) if snapshot.get("route_endpoint_quantization") else None
            ),
            route_provider_display=(
                str(snapshot["route_provider_display"]) if snapshot.get("route_provider_display") else None
            ),
            route_policy=PINNED_ENDPOINT_POLICY,
            route_snapshot_at=snapshot.get("route_snapshot_at"),
            route_reasoning_exclude=bool(snapshot.get("route_reasoning_exclude")),
            state="active",
            reason="active-endpoint-route",
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


def cooldown_route_snapshot(
    snapshot: Mapping[str, Any],
    *,
    failure_reason: str,
    now: datetime | None = None,
    cooldown_seconds: int = 300,
) -> dict[str, Any]:
    """Return a direct-by-resolution cooldown snapshot after a route failure."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("route_snapshot must be a mapping")
    if cooldown_seconds < 1:
        raise ValueError("cooldown_seconds must be positive")
    now = now or datetime.now(timezone.utc)
    updated = deepcopy(dict(snapshot))
    failures = int(updated.get("route_health_failure_count", 0)) + 1
    updated.update(
        {
            "state": "cooldown",
            "route_health_state": "cooldown",
            "route_health_failure_count": failures,
            "route_health_failure_reason": str(failure_reason),
            "cooldown_until": (now.timestamp() + cooldown_seconds),
            "updated_at": now.isoformat(),
        }
    )
    return updated


def resolve_job_route(
    job: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> RouteDecision:
    """Resolve the immutable route snapshot attached to a queued job."""

    return RouteDecision.from_snapshot(
        str(job.get("provider") or ""),
        str(job.get("model_id") or ""),
        job.get("route_snapshot"),
        now=now,
    )


def endpoint_route_snapshot(
    source_provider: str,
    source_model_id: str,
    *,
    endpoint_tag: str,
    provider_canonical: str,
    quantization: str | None = None,
    provider_display: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """A route that pins one named endpoint, built from the catalogue.

    There is no canary and no expiry here on purpose. Those exist to prove a
    routed lane still matches the direct lane it replaced, and to force a
    recheck before that proof goes stale. An endpoint pin makes no such claim:
    it asserts only that the request went to the endpoint named on the row,
    which `allow_fallbacks: False` enforces on every single call rather than
    once at promotion time. Evidence that has to be re-earned on a timer is what
    left promoted routes silently reverting to direct after 72 hours.

    `route_provider_slug` is the provider family, not the tag, because that is
    the granularity the response can actually confirm.
    """

    now = now or datetime.now(timezone.utc)
    return {
        "source_provider": source_provider,
        "source_model_id": source_model_id,
        "route_decision_version": ROUTE_DECISION_VERSION,
        "transport_provider": OPENROUTER_TRANSPORT,
        "route_model_id": source_model_id,
        "route_provider_slug": provider_canonical,
        "route_endpoint_tag": endpoint_tag,
        "route_endpoint_quantization": quantization,
        "route_provider_display": provider_display,
        "route_policy": PINNED_ENDPOINT_POLICY,
        "route_snapshot_at": now.isoformat(),
        "state": "active",
    }
