from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.scheduler.routing import DIRECT_TRANSPORT
from llm_bench.scheduler.routing import RouteDecision

from scripts.openrouter_promote_route import promote_decision
from scripts.openrouter_route_decisions import apply_decisions
from scripts.openrouter_route_decisions import materialize_report


def audit_row(source_key: str, *, decision: str = "keep-direct") -> dict:
    provider, model_id = source_key.split("/", 1)
    row = {
        "source_key": source_key,
        "provider": provider,
        "model_id": model_id,
        "decision": decision,
        "reason_class": "verified-pinned-route" if decision == "route-or" else "unknown",
    }
    if decision == "route-or":
        row.update(
            {
                "or_model_id": "qwen/qwen3-32b",
                "route_provider_slug": "deepinfra",
                "observed_provider": "DeepInfra",
                "evidence": {
                    "probe": {
                        "route_provider_slug": "deepinfra",
                        "observed_provider_slug": "deepinfra",
                        "provider_metadata_verified": True,
                    }
                },
            }
        )
    return row


def test_materializer_keeps_availability_candidates_fail_closed():
    report = materialize_report(
        {
            "generated_at": "2026-08-09T12:00:00+00:00",
            "rows": [audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or"), audit_row("openai/gpt-4o")],
        },
        audit_path="audit.json",
    )

    candidate, direct = report["decisions"]
    assert report["counts"] == {"candidate": 1, "direct": 1}
    assert candidate["state"] == "candidate"
    assert candidate["canary_state"] == "availability_passed"
    assert candidate["canary_successes"] == 0
    assert candidate["route_probe_id"].startswith("coverage:")
    assert direct["transport_provider"] == "direct"


def test_materializer_proves_exact_241_row_conservative_decision_set():
    rows = [audit_row(f"provider-{index % 7}/model-{index:03d}") for index in range(241)]
    report = materialize_report(
        {"generated_at": "2026-08-09T12:00:00+00:00", "rows": rows},
        audit_path="frozen-241.json",
        expected_source_count=241,
    )

    keys = {(item["source_provider"], item["source_model_id"]) for item in report["decisions"]}
    assert report["source_count"] == 241
    assert len(keys) == 241
    assert all(
        RouteDecision.from_snapshot(item["source_provider"], item["source_model_id"], item).transport_provider
        == DIRECT_TRANSPORT
        for item in report["decisions"]
    )


def test_incomplete_route_evidence_stays_direct():
    row = audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or")
    row["evidence"]["probe"]["provider_metadata_verified"] = False
    report = materialize_report({"rows": [row]}, audit_path="audit.json")

    assert report["decisions"][0]["state"] == "direct"
    assert report["decisions"][0]["audit_reason"] == "incomplete-audit-evidence"


def test_apply_preserves_passed_canary():
    report = materialize_report(
        {"generated_at": "2026-08-09T12:00:00+00:00", "rows": [audit_row("openai/gpt-4o")]},
        audit_path="audit.json",
    )
    client = mongomock.MongoClient()
    collection = client["llm-bench"]["bench_route_decisions"]
    collection.insert_one(
        {
            "source_provider": "openai",
            "source_model_id": "gpt-4o",
            "route_snapshot_at": "2026-08-09T12:00:00+00:00",
            "canary_state": "passed",
            "canary_id": "canary-1",
        }
    )

    result = apply_decisions(report, client=client, db_name="llm-bench")

    assert result == {"inserted_or_updated": 0, "preserved_passed": 1}
    assert collection.find_one({"source_provider": "openai"})["canary_id"] == "canary-1"


def test_promotion_requires_passing_costed_canary_and_sets_expiry(tmp_path):
    report = materialize_report(
        {
            "generated_at": "2026-08-09T12:00:00+00:00",
            "rows": [audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or")],
        },
        audit_path="audit.json",
    )
    evidence = tmp_path / "canary.json"
    evidence.write_text('{"canary":"passed"}\n', encoding="utf-8")
    pair = {
        "order": ["direct", "openrouter"],
        "attempts": {
            "direct": {
                "status": "success",
                "effective_request_hash": "c" * 64,
                "metrics": {
                    "tokens_per_second": 100,
                    "time_to_first_token": 1.0,
                    "input_tokens": 10,
                    "output_tokens": 64,
                },
            },
            "openrouter": {
                "status": "success",
                "effective_request_hash": "d" * 64,
                "effective_request": {"protocol_version": 1},
                "metrics": {
                    "tokens_per_second": 95,
                    "time_to_first_token": 1.1,
                    "input_tokens": 10,
                    "output_tokens": 64,
                    "provider_metadata_verified": True,
                    "observed_provider_slug": "deepinfra",
                },
            },
        },
    }
    pairs = []
    for index in range(30):
        current = dict(pair)
        current["pair_index"] = index + 1
        current["order"] = ["direct", "openrouter"] if index < 15 else ["openrouter", "direct"]
        pairs.append(current)
    canary = {
        "canary_id": "canary-1",
        "source_provider": "deepinfra",
        "source_model_id": "Qwen/Qwen3-32B",
        "route_model_id": "qwen/qwen3-32b",
        "seed": 0,
        "profile_hash": "b" * 64,
        "pricing": {
            "direct": {"input_per_token": 1e-6, "output_per_token": 1e-6},
            "openrouter": {"input_per_token": 1e-6, "output_per_token": 1e-6},
        },
        "pairs": pairs,
        "evaluation": {
            "canary_state": "passed",
            "promotion_valid": True,
            "cost_status": "verified",
            "successful_pairs": 30,
            "required_pairs": 30,
            "route_tps_ratio_ci95": [0.9, 1.1],
            "route_ttft_ratio_ci95": [0.7, 1.2],
            "route_cost_ratio_ci95": [0.9, 1.05],
            "route_error_delta": 0.0,
            "route_metadata_verified": 30,
        },
    }

    route = promote_decision(
        report["decisions"][0],
        canary,
        evidence_path=evidence,
        evidence_uri="s3://artifacts/test/canary.json",
        now=datetime(2026, 8, 10, tzinfo=timezone.utc),
        revocation_generation=0,
    )

    assert route["state"] == "active"
    assert route["canary_promotion_gate"] == "passed"
    assert route["canary_cost_status"] == "verified"
    assert route["expires_at"] == "2026-08-11T00:00:00+00:00"
    assert route["canary_required_successes"] == 29


def test_promotion_rejects_self_declared_one_pair_verdict(tmp_path):
    report = materialize_report(
        {
            "generated_at": "2026-08-09T12:00:00+00:00",
            "rows": [audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or")],
        },
        audit_path="audit.json",
    )
    evidence = tmp_path / "fake.json"
    evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="30 raw"):
        promote_decision(
            report["decisions"][0],
            {
                "canary_id": "fake",
                "source_provider": "deepinfra",
                "source_model_id": "Qwen/Qwen3-32B",
                "route_model_id": "qwen/qwen3-32b",
                "pairs": [{}],
                "evaluation": {"required_pairs": 1, "canary_state": "passed", "promotion_valid": True},
            },
            evidence_path=evidence,
        )


def _valid_route_snapshot(*, provider: str, model_id: str) -> dict:
    """A route document that would otherwise resolve active: full canary evidence."""
    return {
        "source_provider": provider,
        "source_model_id": model_id,
        "transport_provider": "openrouter",
        "route_model_id": "vendor/model",
        "route_provider_slug": provider,
        "route_policy": "pinned-provider",
        "route_decision_version": "or-route-v1",
        "route_revocation_generation": 0,
        "canary_state": "passed",
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
        "provider_metadata_verified": True,
        "observed_provider": "Vendor",
        "observed_provider_slug": provider,
        "route_snapshot_at": "2026-08-10T23:51:48.678006+00:00",
        "route_probe_id": "coverage:test",
        "profile_hash": "profile-hash",
        "direct_effective_request_hash": "direct-hash",
        "routed_effective_request_hash": "routed-hash",
        "state": "active",
        "expires_at": "2026-08-15T02:34:24.795384+00:00",
    }


@pytest.mark.parametrize("provider", ["openai", "vertex", "bedrock"])
def test_direct_provider_route_docs_resolve_direct(provider):
    """Site policy keeps openai/vertex/bedrock direct: even a fully valid route
    document for those sources must resolve to the direct lane."""
    snapshot = _valid_route_snapshot(provider=provider, model_id="some-model")
    decision = RouteDecision.from_snapshot(provider, "some-model", snapshot)
    assert decision.transport_provider == DIRECT_TRANSPORT
    assert decision.reason == f"{provider}-kept-direct"


def test_other_provider_route_doc_still_resolves_active():
    snapshot = _valid_route_snapshot(provider="deepinfra", model_id="some-model")
    decision = RouteDecision.from_snapshot("deepinfra", "some-model", snapshot)
    assert decision.transport_provider != DIRECT_TRANSPORT
    assert decision.reason == "active-pinned-route"


def test_or_served_route_resolves_active():
    """Marketplace policy: an or-served route records who actually served and
    does not pin to the source provider."""
    snapshot = _valid_route_snapshot(provider="deepinfra", model_id="Qwen/Qwen3-235B-A22B")
    snapshot["route_policy"] = "or-served"
    snapshot["route_provider_slug"] = "alibaba"
    snapshot["observed_provider_slug"] = "alibaba"
    snapshot["observed_provider"] = "Alibaba"
    decision = RouteDecision.from_snapshot("deepinfra", "Qwen/Qwen3-235B-A22B", snapshot)
    assert decision.transport_provider != DIRECT_TRANSPORT
    assert decision.route_policy == "or-served"
    assert decision.reason == "active-pinned-route"


def test_or_served_route_still_requires_matching_observed_evidence():
    """An or-served route must still name the provider that actually served."""
    snapshot = _valid_route_snapshot(provider="deepinfra", model_id="Qwen/Qwen3-235B-A22B")
    snapshot["route_policy"] = "or-served"
    snapshot["route_provider_slug"] = "alibaba"
    snapshot["observed_provider_slug"] = "novita"
    decision = RouteDecision.from_snapshot("deepinfra", "Qwen/Qwen3-235B-A22B", snapshot)
    assert decision.transport_provider == DIRECT_TRANSPORT
