import mongomock
import pytest

from scripts.openrouter_reconcile import apply_reconciliation
from scripts.openrouter_reconcile import reconcile
from scripts.openrouter_revoke_route import revoke_route


def test_reconciliation_requires_finalized_decisions():
    report = {"generated_at": "2026-08-10T00:00:00+00:00"}
    decisions = {
        "finalized": True,
        "decisions": [{"source_provider": "openai", "source_model_id": "gpt-4o", "state": "candidate"}],
    }

    with pytest.raises(ValueError, match="route candidates"):
        reconcile(
            report,
            decisions,
            source_snapshot_hash="a",
            catalog_snapshot_hash="b",
            alias_rule_version="or-alias-v1",
            profile_hash="c",
            expected_source_count=1,
        )


def test_reconciliation_is_idempotent_by_run_id():
    report = {"generated_at": "2026-08-10T00:00:00+00:00"}
    decisions = {
        "finalized": True,
        "decisions": [
            {
                "source_provider": "openai",
                "source_model_id": "gpt-4o",
                "state": "direct",
                "transport_provider": "direct",
                "terminal_state": "direct-unknown",
            }
        ],
    }
    artifact = reconcile(
        report,
        decisions,
        source_snapshot_hash="a",
        catalog_snapshot_hash="b",
        alias_rule_version="or-alias-v1",
        profile_hash="c",
        expected_source_count=1,
    )
    client = mongomock.MongoClient()
    apply_reconciliation(artifact, client=client, db_name="llm-bench")
    apply_reconciliation(artifact, client=client, db_name="llm-bench")

    assert client["llm-bench"]["bench_route_reconciliations"].count_documents({}) == 1
    assert client["llm-bench"]["bench_route_decision_audit"].count_documents({}) == 1


def test_route_revocation_generation_increments():
    db = mongomock.MongoClient()["llm-bench"]
    first = revoke_route(db, provider="openai", model_id="gpt-4o", reason="canary regression")
    second = revoke_route(db, provider="openai", model_id="gpt-4o", reason="operator request")

    assert first["generation"] == 1
    assert second["generation"] == 2


def test_reconciliation_reports_delta_against_previous_run():
    report = {"generated_at": "2026-08-10T00:00:00+00:00"}
    previous = {
        "decisions": [
            {
                "source_provider": "openai",
                "source_model_id": "old",
                "terminal_state": "route-approved",
            },
            {
                "source_provider": "openai",
                "source_model_id": "same",
                "terminal_state": "direct-unknown",
            },
        ]
    }
    current = {
        "finalized": True,
        "decisions": [
            {
                "source_provider": "openai",
                "source_model_id": "same",
                "state": "direct",
                "transport_provider": "direct",
                "terminal_state": "direct-incompatible",
            },
            {
                "source_provider": "openai",
                "source_model_id": "new",
                "state": "direct",
                "transport_provider": "direct",
                "terminal_state": "direct-unknown",
            },
        ],
    }
    artifact = reconcile(
        report,
        current,
        source_snapshot_hash="a",
        catalog_snapshot_hash="b",
        alias_rule_version="or-alias-v1",
        profile_hash="c",
        expected_source_count=2,
        previous=previous,
        previous_snapshot_hash="previous-hash",
    )
    assert artifact["delta"] == {
        "baseline": False,
        "new": ["openai/new"],
        "changed": ["openai/same"],
        "stale": [],
        "removed": ["openai/old"],
    }


def test_reconciliation_rejects_stale_approved_profile():
    with pytest.raises(ValueError, match="profile hash"):
        reconcile(
            {"generated_at": "2026-08-10T00:00:00+00:00"},
            {
                "finalized": True,
                "decisions": [
                    {
                        "source_provider": "openai",
                        "source_model_id": "gpt-4o-mini",
                        "state": "active",
                        "transport_provider": "openrouter",
                        "terminal_state": "route-approved",
                        "profile_hash": "old",
                    }
                ],
            },
            source_snapshot_hash="a",
            catalog_snapshot_hash="b",
            alias_rule_version="or-alias-v1",
            profile_hash="new",
            expected_source_count=1,
        )
