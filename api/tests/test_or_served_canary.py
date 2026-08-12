"""Routed-only canary and or-served promotion: marketplace lane evidence.

Site policy 2026-08-12: non-direct providers are measured as OpenRouter
serves them. Evidence is serving reliability + verified, stable provider
metadata; no direct lane and no parity statistics.
"""

from pathlib import Path

import pytest
from llm_bench.scheduler.routing import DIRECT_TRANSPORT
from llm_bench.scheduler.routing import RouteDecision

import scripts.openrouter_or_served_canary as osc
from scripts.openrouter_promote_route import promote_or_served_route


def _candidate(provider: str = "deepinfra", model: str = "Qwen/Qwen3-235B-A22B") -> dict:
    return {
        "source_provider": provider,
        "source_model_id": model,
        "state": "candidate",
        "transport_provider": "openrouter",
        "route_model_id": "qwen/qwen3-235b-a22b",
        "route_provider_slug": "alibaba",
        "observed_provider_slug": "alibaba",
        "observed_provider": "Alibaba",
        "provider_metadata_verified": True,
        "route_policy": "or-served",
        "route_decision_version": "or-route-v1",
        "route_revocation_generation": 0,
        "route_snapshot_at": "2026-08-12T00:00:00+00:00",
        "route_probe_id": "coverage:test",
        "audit_decision": "route-or",
        "canary_required_successes": 12,
    }


def _ok_attempt(observed: str = "alibaba", verified: bool = True) -> dict:
    return {
        "status": "success",
        "metrics": {
            "observed_provider_slug": observed,
            "observed_provider": "Alibaba",
            "provider_metadata_verified": verified,
            "visible_output_tokens": 20,
            "output_tokens": 64,
            "generated_output_tokens": 64,
            "tokens_per_second": 30.0,
            "generate_time": 2.0,
            "time_to_first_token": 0.5,
            "finish_reason": "length",
            "response_status": "complete",
        },
    }


def _canary(candidate: dict, *, state: str = "passed") -> dict:
    return {
        "mode": "report-only-routed-canary",
        "canary_id": f"or-served:{candidate['source_provider']}:{candidate['source_model_id']}:20260812T000000Z",
        "source_provider": candidate["source_provider"],
        "source_model_id": candidate["source_model_id"],
        "route_model_id": candidate["route_model_id"],
        "observed_provider_slug": "alibaba",
        "observed_provider": "Alibaba",
        "evaluation": {
            "mode": "report-only-routed-canary",
            "canary_state": state,
            "promotion_valid": state == "passed",
            "successful": 12,
            "required": 12,
            "observed_provider_slug": "alibaba",
            "provider_metadata_verified": True,
            "route_policy": "or-served",
        },
    }


def test_routed_canary_passes_on_stable_verified_serving(monkeypatch):
    report = {"decisions": [_candidate()]}
    seen: list[str] = []

    def fake_attempt(decision, *, max_tokens, deadline_seconds):
        seen.append(decision.route_model_id)
        return _ok_attempt()

    monkeypatch.setattr(osc, "_attempt", fake_attempt)
    result = osc.run_routed_canary(
        report,
        provider="deepinfra",
        model_id="Qwen/Qwen3-235B-A22B",
        attempts_count=12,
        max_tokens=64,
        deadline_seconds=300,
        min_success_rate=0.95,
    )
    ev = result["evaluation"]
    assert ev["canary_state"] == "passed"
    assert ev["promotion_valid"] is True
    assert ev["observed_provider_slug"] == "alibaba"
    assert result["route_policy"] == "or-served"
    assert len(seen) == 12


def test_routed_canary_fails_on_unstable_observed(monkeypatch):
    report = {"decisions": [_candidate()]}
    counter = [0]

    def fake_attempt(decision, *, max_tokens, deadline_seconds):
        counter[0] += 1
        return _ok_attempt(observed="alibaba" if counter[0] % 2 else "novita")

    monkeypatch.setattr(osc, "_attempt", fake_attempt)
    result = osc.run_routed_canary(
        report,
        provider="deepinfra",
        model_id="Qwen/Qwen3-235B-A22B",
        attempts_count=12,
        max_tokens=64,
        deadline_seconds=300,
        min_success_rate=0.95,
    )
    assert result["evaluation"]["canary_state"] == "failed"
    assert any("not stable" in r for r in result["evaluation"]["reasons"])


def test_routed_canary_fails_when_metadata_unverified(monkeypatch):
    report = {"decisions": [_candidate()]}
    monkeypatch.setattr(
        osc,
        "_attempt",
        lambda decision, *, max_tokens, deadline_seconds: _ok_attempt(verified=False),
    )
    result = osc.run_routed_canary(
        report,
        provider="deepinfra",
        model_id="Qwen/Qwen3-235B-A22B",
        attempts_count=12,
        max_tokens=64,
        deadline_seconds=300,
        min_success_rate=0.95,
    )
    assert result["evaluation"]["canary_state"] == "failed"


def test_promote_or_served_builds_active_route(tmp_path: Path):
    candidate = _candidate()
    evidence = tmp_path / "canary.json"
    evidence.write_text("{}\n", encoding="utf-8")
    route = promote_or_served_route(
        candidate,
        _canary(candidate),
        evidence_path=evidence,
        evidence_uri="s3://artifacts/llm-benchmarks/openrouter-consolidation/v5/canaries/test.json",
        expires_hours=72,
        revocation_generation=0,
    )
    assert route["state"] == "active"
    assert route["route_policy"] == "or-served"
    assert route["observed_provider_slug"] == "alibaba"
    decision = RouteDecision.from_snapshot("deepinfra", "Qwen/Qwen3-235B-A22B", route, require_promotion_evidence=False)
    assert decision.transport_provider != DIRECT_TRANSPORT


def test_promote_or_served_rejects_paired_evidence(tmp_path: Path):
    candidate = _candidate()
    evidence = tmp_path / "c.json"
    evidence.write_text("{}\n", encoding="utf-8")
    paired = _canary(candidate)
    paired["evaluation"]["mode"] = "report-only-paired-canary"
    with pytest.raises(ValueError, match="routed-only"):
        promote_or_served_route(
            candidate,
            paired,
            evidence_path=evidence,
            evidence_uri="s3://x/y.json",
            expires_hours=72,
            revocation_generation=0,
        )


def test_promote_or_served_rejects_failed_canary(tmp_path: Path):
    candidate = _candidate()
    evidence = tmp_path / "c.json"
    evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        promote_or_served_route(
            candidate,
            _canary(candidate, state="failed"),
            evidence_path=evidence,
            evidence_uri="s3://x/y.json",
            expires_hours=72,
            revocation_generation=0,
        )
