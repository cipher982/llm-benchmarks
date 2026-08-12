"""Route renewal pass: window logic, cooldown recovery, revocation, bounds.

The renewal pass is the other half of the route evidence expiry design: routes
inside their renewal window get one routed probe, success extends the window,
failure cools them down, and enough consecutive failures revoke permanently.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
from llm_bench.ops.route_renewal import renew_pass

NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)


def _db():
    return mongomock.MongoClient()["llm-bench"]


def _route(
    provider: str = "deepinfra",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    *,
    state: str = "active",
    expires_at: datetime | None = None,
    cooldown_until: float | None = None,
    failure_count: int = 0,
    canary_state: str = "passed",
):
    return {
        "_id": f"{provider}:{model}",
        "source_provider": provider,
        "source_model_id": model,
        "transport_provider": "openrouter",
        "route_model_id": model,
        "route_provider_slug": provider,
        "route_policy": "pinned-provider",
        "route_decision_version": "or-route-v1",
        "route_revocation_generation": 0,
        "canary_state": canary_state,
        "canary_id": "canary:test",
        "canary_successes": 30,
        "canary_required_successes": 29,
        "canary_promotion_gate": "passed",
        "canary_cost_status": "verified",
        "canary_evidence_uri": "s3://artifacts/llm-benchmarks/openrouter-consolidation/v4/derived/canaries/test.json",
        "canary_evidence_sha256": "a" * 64,
        "canary_tps_ci95_lower": 0.9,
        "canary_cost_ci95_upper": 1.0,
        "canary_ttft_ci95_upper": 1.2,
        "canary_ttft_waived_direct_unmeasured": False,
        "provider_metadata_verified": True,
        "observed_provider": provider,
        "observed_provider_slug": provider,
        "route_snapshot_at": "2026-08-10T23:51:48.678006+00:00",
        "route_probe_id": "coverage:test",
        "profile_hash": "profile-hash",
        "direct_effective_request_hash": "direct-hash",
        "routed_effective_request_hash": "routed-hash",
        "state": state,
        "expires_at": (expires_at or (NOW + timedelta(hours=2))).isoformat(),
        "cooldown_until": cooldown_until,
        "route_health_failure_count": failure_count,
    }


class Probe:
    """Records probe calls; raises when `ok` is False."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, decision, *, now):
        self.calls.append((decision.source_provider, decision.source_model_id))
        if not self.ok:
            raise RuntimeError("probe boom")


def test_renews_due_active_route():
    db = _db()
    db.bench_route_decisions.insert_one(_route())
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["renewed"] == 1
    doc = db.bench_route_decisions.find_one()
    fresh = NOW + timedelta(hours=72)
    assert doc["state"] == "active"
    assert datetime.fromisoformat(doc["expires_at"]) == fresh
    assert doc["last_renewed_at"] == NOW.isoformat()
    assert probe.calls == [("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct")]


def test_skips_route_far_from_expiry():
    db = _db()
    db.bench_route_decisions.insert_one(_route(expires_at=NOW + timedelta(hours=100)))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["considered"] == 0
    assert report["renewed"] == 0
    assert probe.calls == []


def test_renews_already_expired_active_route():
    db = _db()
    db.bench_route_decisions.insert_one(_route(expires_at=NOW - timedelta(hours=1)))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["renewed"] == 1
    doc = db.bench_route_decisions.find_one()
    assert datetime.fromisoformat(doc["expires_at"]) == NOW + timedelta(hours=72)


def test_recovers_cooled_route_after_window():
    db = _db()
    db.bench_route_decisions.insert_one(_route(state="cooldown", cooldown_until=NOW.timestamp() - 10, failure_count=1))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["renewed"] == 1
    doc = db.bench_route_decisions.find_one()
    assert doc["state"] == "active"
    assert doc["route_recovery_probe_id"].startswith("renewal:")
    assert datetime.fromisoformat(doc["expires_at"]) == NOW + timedelta(hours=72)


def test_leaves_cooling_route_alone():
    db = _db()
    db.bench_route_decisions.insert_one(
        _route(state="cooldown", cooldown_until=NOW.timestamp() + 3600, failure_count=1)
    )
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["considered"] == 0
    assert probe.calls == []


def test_failed_probe_cools_active_route():
    db = _db()
    db.bench_route_decisions.insert_one(_route())
    probe = Probe(ok=False)
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["cooled"] == 1
    doc = db.bench_route_decisions.find_one()
    assert doc["state"] == "cooldown"
    assert doc["route_health_failure_count"] == 1
    assert doc["cooldown_until"] > NOW.timestamp()


def test_repeated_failures_revoke_route():
    db = _db()
    db.bench_route_decisions.insert_one(_route(failure_count=11))
    probe = Probe(ok=False)
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["revoked"] == 1
    doc = db.bench_route_decisions.find_one()
    assert doc["state"] == "revoked"
    assert doc["route_revocation_reason"] == "renewal-probe-failed"
    assert doc["route_revoked_at"] == NOW.isoformat()


def test_bound_and_oldest_first():
    db = _db()
    for i in range(5):
        db.bench_route_decisions.insert_one(_route(model=f"vendor/model-{i}", expires_at=NOW + timedelta(hours=1 + i)))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe, limit=3)

    assert report["considered"] == 3
    assert report["renewed"] == 3
    assert [model for _, model in probe.calls] == ["vendor/model-0", "vendor/model-1", "vendor/model-2"]


def test_skips_route_that_cannot_activate():
    db = _db()
    db.bench_route_decisions.insert_one(_route(canary_state="failed"))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["skipped"] == 1
    assert report["renewed"] == 0
    assert probe.calls == []


def test_one_failure_does_not_block_rest_of_pass():
    db = _db()
    db.bench_route_decisions.insert_one(_route(model="vendor/broken", canary_state="failed"))
    db.bench_route_decisions.insert_one(_route(model="vendor/healthy"))
    probe = Probe()
    report = renew_pass(db, now=NOW, probe=probe)

    assert report["skipped"] == 1
    assert report["renewed"] == 1
    assert [model for _, model in probe.calls] == ["vendor/healthy"]
