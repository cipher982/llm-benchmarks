from scripts.openrouter_finalize_decisions import finalize


def test_unpromoted_candidates_become_direct_unknown():
    result = finalize(
        {
            "decisions": [
                {
                    "source_provider": "deepinfra",
                    "source_model_id": "Qwen3-32B",
                    "state": "candidate",
                    "transport_provider": "openrouter",
                    "route_model_id": "deepinfra/qwen3-32b",
                    "route_provider_slug": "deepinfra",
                    "observed_provider_slug": "deepinfra",
                    "route_probe_id": "probe:test",
                }
            ]
        },
        [],
    )

    row = result["decisions"][0]
    assert row["state"] == "direct"
    assert row["transport_provider"] == "direct"
    assert row["terminal_state"] == "direct-unknown"


def test_promoted_route_is_terminal_route_approved():
    result = finalize(
        {
            "decisions": [
                {
                    "source_provider": "deepinfra",
                    "source_model_id": "Qwen3-32B",
                    "state": "candidate",
                    "transport_provider": "openrouter",
                    "route_model_id": "deepinfra/qwen3-32b",
                    "route_provider_slug": "deepinfra",
                    "observed_provider_slug": "deepinfra",
                    "route_probe_id": "probe:test",
                }
            ]
        },
        [
            {
                "source_provider": "deepinfra",
                "source_model_id": "Qwen3-32B",
                "state": "active",
                "transport_provider": "openrouter",
                "route_model_id": "deepinfra/qwen3-32b",
                "route_decision_version": "or-route-v1",
                "route_policy": "pinned-provider",
                "route_provider_slug": "deepinfra",
                "observed_provider_slug": "deepinfra",
                "observed_provider": "DeepInfra",
                "provider_metadata_verified": True,
                "route_snapshot_at": "2026-08-10T00:00:00+00:00",
                "route_probe_id": "probe:test",
                "route_revocation_generation": 0,
                "canary_id": "canary:test",
                "canary_state": "passed",
                "canary_successes": 30,
                "canary_required_successes": 29,
                "canary_promotion_gate": "passed",
                "canary_cost_status": "verified",
                "canary_evidence_uri": "s3://artifacts/test/canary.json",
                "canary_evidence_sha256": "a" * 64,
                "canary_tps_ci95_lower": 0.9,
                "canary_ttft_ci95_upper": 1.1,
                "canary_cost_ci95_upper": 1.0,
                "profile_hash": "b" * 64,
                "direct_effective_request_hash": "c" * 64,
                "routed_effective_request_hash": "d" * 64,
                "expires_at": "2099-08-10T00:00:00+00:00",
            }
        ],
    )

    assert result["terminal_counts"] == {"route-approved": 1}
    assert result["decisions"][0]["state"] == "active"


def test_failed_canary_becomes_terminal_direct_canary_failed():
    result = finalize(
        {
            "decisions": [
                {
                    "source_provider": "deepinfra",
                    "source_model_id": "Qwen3-32B",
                    "state": "candidate",
                    "transport_provider": "openrouter",
                    "route_model_id": "deepinfra/qwen3-32b",
                    "route_provider_slug": "deepinfra",
                    "observed_provider_slug": "deepinfra",
                    "route_probe_id": "probe:test",
                }
            ]
        },
        [],
        [
            {
                "source_provider": "deepinfra",
                "source_model_id": "Qwen3-32B",
                "evaluation": {
                    "canary_state": "failed",
                    "promotion_valid": False,
                    "failure_reason": "missing-local-credential:OPENAI_API_KEY",
                },
            }
        ],
        canary_paths=["/tmp/canary.json"],
    )

    row = result["decisions"][0]
    assert row["terminal_state"] == "direct-canary-failed"
    assert row["canary_artifact_path"] == "/tmp/canary.json"
