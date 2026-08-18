"""Fail-closed route decisions for source-provider benchmark jobs.

The route snapshot is evidence attached to a queued job.  It is deliberately
separate from the source provider and model identity, so a route can never
rename a source row or silently turn an unknown route into OpenRouter work.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from math import isfinite
from typing import Any
from typing import Mapping
from urllib.parse import urlparse

ROUTE_DECISION_VERSION = "or-route-v1"
DIRECT_TRANSPORT = "direct"
OPENROUTER_TRANSPORT = "openrouter"
DIRECT_POLICY = "direct"
PINNED_PROVIDER_POLICY = "pinned-provider"
# OpenRouter serves the model from whichever upstream it picks (marketplace
# policy 2026-08-12): the route records who actually served (`observed`), which
# becomes part of the published row. No pin is required or attempted.
OR_SERVED_POLICY = "or-served"
# The site measures one named OpenRouter endpoint and publishes it as that
# endpoint. This is deliberately not `pinned-provider`: that policy exists to
# prove a routed lane performs like the direct lane it replaced, and gates on
# paired direct-vs-routed canary statistics. Groq, Cerebras and Fireworks have
# no direct lane left to be compared against, so the evidence that gate demands
# cannot be produced -- and it answers a migration question the site is no
# longer asking. What matters here is only that the request reached the
# endpoint it named, which `allow_fallbacks: False` and the observed-slug check
# already enforce on every run.
PINNED_ENDPOINT_POLICY = "pinned-endpoint"
MIN_PROMOTION_TPS_CI95 = 0.8
MAX_PROMOTION_TTFT_CI95 = 1.5
MAX_PROMOTION_COST_CI95 = 1.10

# Direct lanes are reserved for providers consumed directly (site policy
# 2026-08-12): the site publishes real numbers for these, and OpenRouter-served
# lanes for everything else. Resolution refuses to route these even if a stale
# route document exists, so the policy is enforced at runtime, not at
# promotion time.
DIRECT_PROVIDERS = frozenset({"bedrock", "openai", "vertex"})


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
    route_probe_id: str | None = None
    route_decision_version: str = ROUTE_DECISION_VERSION
    route_canary_id: str | None = None
    route_canary_state: str | None = None
    route_canary_successes: int | None = None
    route_canary_required_successes: int | None = None
    route_canary_cost_status: str | None = None
    route_canary_evidence_uri: str | None = None
    route_canary_evidence_sha256: str | None = None
    route_canary_promotion_gate: str | None = None
    route_expires_at: str | None = None
    route_revocation_generation: int = 0
    route_reasoning_exclude: bool = False
    profile_hash: str | None = None
    direct_effective_request_hash: str | None = None
    routed_effective_request_hash: str | None = None
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
        require_promotion_evidence: bool = True,
        current_revocation_generation: int | None = None,
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
        if source_provider in DIRECT_PROVIDERS:
            return cls.direct(source_provider, source_model_id, reason=f"{source_provider}-kept-direct")
        try:
            snapshot_generation = int(snapshot.get("route_revocation_generation", 0))
        except (TypeError, ValueError):
            return cls.direct(source_provider, source_model_id, reason="invalid-route-revocation-generation")
        if snapshot_generation < 0:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-revocation-generation")
        if current_revocation_generation is not None and current_revocation_generation > snapshot_generation:
            return cls.direct(source_provider, source_model_id, reason="route-revoked")
        if snapshot.get("state") != "active":
            return cls.direct(source_provider, source_model_id, reason="route-state-not-active")
        if snapshot.get("transport_provider") != OPENROUTER_TRANSPORT:
            return cls.direct(source_provider, source_model_id, reason="invalid-route-transport")
        route_policy = snapshot.get("route_policy")
        if route_policy not in (PINNED_PROVIDER_POLICY, OR_SERVED_POLICY, PINNED_ENDPOINT_POLICY):
            return cls.direct(source_provider, source_model_id, reason="invalid-route-policy")
        if route_policy == PINNED_ENDPOINT_POLICY:
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
                    str(snapshot["route_endpoint_quantization"])
                    if snapshot.get("route_endpoint_quantization")
                    else None
                ),
                route_provider_display=(
                    str(snapshot["route_provider_display"]) if snapshot.get("route_provider_display") else None
                ),
                route_policy=PINNED_ENDPOINT_POLICY,
                route_snapshot_at=snapshot.get("route_snapshot_at"),
                route_revocation_generation=snapshot_generation,
                route_reasoning_exclude=bool(snapshot.get("route_reasoning_exclude")),
                state="active",
                reason="active-endpoint-route",
            )
        canary_state = snapshot.get("canary_state")
        canary_id = snapshot.get("canary_id")
        try:
            canary_successes = int(snapshot.get("canary_successes", 0))
            canary_required_successes = int(snapshot.get("canary_required_successes", 0))
        except (TypeError, ValueError):
            return cls.direct(source_provider, source_model_id, reason="invalid-canary-evidence")
        if (
            canary_state != "passed"
            or not canary_id
            or canary_required_successes < 1
            or canary_successes < canary_required_successes
        ):
            return cls.direct(source_provider, source_model_id, reason="canary-not-passed")
        expiry = snapshot.get("expires_at") or snapshot.get("recheck_at")
        if not expiry:
            return cls.direct(source_provider, source_model_id, reason="route-expiry-missing")
        if _timestamp_is_expired(expiry, now=now):
            return cls.direct(source_provider, source_model_id, reason="route-evidence-expired")

        if require_promotion_evidence:
            if snapshot.get("canary_promotion_gate") != "passed":
                return cls.direct(source_provider, source_model_id, reason="canary-promotion-gate-not-passed")
            evidence_uri = snapshot.get("canary_evidence_uri")
            evidence_sha256 = snapshot.get("canary_evidence_sha256")
            if (
                not isinstance(evidence_uri, str)
                or urlparse(evidence_uri).scheme != "s3"
                or not urlparse(evidence_uri).netloc
                or not urlparse(evidence_uri).path.strip("/")
                or not isinstance(evidence_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", evidence_sha256) is None
            ):
                return cls.direct(source_provider, source_model_id, reason="canary-evidence-not-immutable")
            # Pinned routes use paired direct-vs-routed statistics. Marketplace
            # routes deliberately use routed-only evidence: OpenRouter is the
            # product lane and may choose whichever upstream serves the model.
            if route_policy != OR_SERVED_POLICY:
                if snapshot.get("canary_cost_status") != "verified":
                    return cls.direct(source_provider, source_model_id, reason="canary-cost-unverified")
                try:
                    tps_lower = float(snapshot["canary_tps_ci95_lower"])
                    cost_upper = float(snapshot["canary_cost_ci95_upper"])
                except (KeyError, TypeError, ValueError):
                    return cls.direct(source_provider, source_model_id, reason="canary-statistical-evidence-missing")
                # The TTFT bound applies unless the canary explicitly recorded that
                # the direct lane measured TTFT on zero pairs; a missing value with
                # no waiver flag remains fail-closed.
                ttft_waived = snapshot.get("canary_ttft_waived_direct_unmeasured") is True
                ttft_upper: float | None = None
                if not ttft_waived:
                    try:
                        ttft_upper = float(snapshot["canary_ttft_ci95_upper"])
                    except (KeyError, TypeError, ValueError):
                        return cls.direct(
                            source_provider, source_model_id, reason="canary-statistical-evidence-missing"
                        )
                checked = (tps_lower, cost_upper) if ttft_waived else (tps_lower, ttft_upper, cost_upper)
                if not all(isfinite(value) for value in checked):
                    reason = "canary-statistical-evidence-nonfinite"
                    return cls.direct(source_provider, source_model_id, reason=reason)
                if tps_lower < MIN_PROMOTION_TPS_CI95:
                    return cls.direct(source_provider, source_model_id, reason="canary-tps-bound-failed")
                if ttft_upper is not None and ttft_upper > MAX_PROMOTION_TTFT_CI95:
                    return cls.direct(source_provider, source_model_id, reason="canary-ttft-bound-failed")
                if cost_upper > MAX_PROMOTION_COST_CI95:
                    return cls.direct(source_provider, source_model_id, reason="canary-cost-bound-failed")

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
        if (
            require_promotion_evidence
            and route_policy != OR_SERVED_POLICY
            and any(
                not snapshot.get(key)
                for key in ("profile_hash", "direct_effective_request_hash", "routed_effective_request_hash")
            )
        ):
            return cls.direct(source_provider, source_model_id, reason="request-evidence-missing")

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
            route_endpoint_tag=(str(snapshot["route_endpoint_tag"]) if snapshot.get("route_endpoint_tag") else None),
            route_endpoint_quantization=(
                str(snapshot["route_endpoint_quantization"]) if snapshot.get("route_endpoint_quantization") else None
            ),
            route_provider_display=(
                str(snapshot["route_provider_display"]) if snapshot.get("route_provider_display") else None
            ),
            observed_provider=str(snapshot.get("observed_provider") or ""),
            observed_provider_slug=str(snapshot["observed_provider_slug"]),
            route_policy=str(route_policy),
            route_snapshot_at=snapshot.get("route_snapshot_at"),
            route_probe_id=snapshot.get("route_probe_id"),
            route_decision_version=ROUTE_DECISION_VERSION,
            route_canary_id=str(canary_id),
            route_canary_state=str(canary_state),
            route_canary_successes=canary_successes,
            route_canary_required_successes=canary_required_successes,
            route_canary_cost_status=snapshot.get("canary_cost_status"),
            route_canary_evidence_uri=snapshot.get("canary_evidence_uri"),
            route_canary_evidence_sha256=snapshot.get("canary_evidence_sha256"),
            route_canary_promotion_gate=snapshot.get("canary_promotion_gate"),
            route_expires_at=str(expiry),
            route_revocation_generation=snapshot_generation,
            route_reasoning_exclude=snapshot.get("route_reasoning_exclude") is True,
            profile_hash=snapshot.get("profile_hash"),
            direct_effective_request_hash=snapshot.get("direct_effective_request_hash"),
            routed_effective_request_hash=snapshot.get("routed_effective_request_hash"),
            state="active",
            reason=("active-or-served-route" if route_policy == OR_SERVED_POLICY else "active-pinned-route"),
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


def recover_route_snapshot(
    snapshot: Mapping[str, Any],
    *,
    recovery_probe_id: str,
    recovery_probe_passed: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Re-enable a cooled route only after an explicit recovery probe."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("route_snapshot must be a mapping")
    if not recovery_probe_id or recovery_probe_passed is not True:
        raise ValueError("recovery_probe_id is required")
    now = now or datetime.now(timezone.utc)
    cooldown_until = snapshot.get("cooldown_until")
    if cooldown_until is not None and float(cooldown_until) > now.timestamp():
        raise ValueError("route cooldown is still active")
    recovered = deepcopy(dict(snapshot))
    recovered.update(
        {
            "state": "active",
            "route_health_state": "recovered",
            "route_recovery_probe_id": str(recovery_probe_id),
            "route_recovered_at": now.isoformat(),
        }
    )
    decision = RouteDecision.from_snapshot(
        str(recovered.get("source_provider") or ""),
        str(recovered.get("source_model_id") or ""),
        recovered,
        now=now,
    )
    if decision.transport_provider != OPENROUTER_TRANSPORT:
        raise ValueError(f"recovery evidence is not activatable: {decision.reason}")
    return recovered


def resolve_job_route(
    job: Mapping[str, Any],
    *,
    now: datetime | None = None,
    current_revocation_generation: int | None = None,
) -> RouteDecision:
    """Resolve the immutable route snapshot attached to a queued job."""

    return RouteDecision.from_snapshot(
        str(job.get("provider") or ""),
        str(job.get("model_id") or ""),
        job.get("route_snapshot"),
        now=now,
        current_revocation_generation=current_revocation_generation,
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
        "route_revocation_generation": 0,
        "state": "active",
    }
