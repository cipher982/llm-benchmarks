from scripts.openrouter_coverage_audit import catalog_scope
from scripts.openrouter_coverage_audit import decide
from scripts.openrouter_coverage_audit import diagnostic_stem


def source(provider: str = "deepinfra", model_id: str = "Qwen/Qwen3-32B") -> dict:
    return {
        "provider": provider,
        "model_id": model_id,
        "display_name": model_id,
        "enabled": True,
        "deprecated": False,
    }


def catalog_row(model_id: str = "qwen/qwen3-32b") -> dict:
    return {
        "id": model_id,
        "canonical_slug": model_id,
        "name": "Qwen 3 32B",
        "description": "official model card",
        "links": {"details": "/api/v1/models/qwen/qwen3-32b/endpoints"},
    }


def endpoint(provider_name: str = "DeepInfra") -> dict:
    return {
        "data": {
            "endpoints": [
                {
                    "provider_name": provider_name,
                    "provider_slug": "deepinfra",
                    "supported_parameters": ["max_tokens", "temperature"],
                }
            ]
        }
    }


def probe(*, observed_provider_slug: str = "deepinfra", status: str = "success") -> dict:
    return {
        "status": status,
        "route_provider_slug": "deepinfra",
        "observed_provider": "DeepInfra",
        "observed_provider_slug": observed_provider_slug,
        "provider_metadata_verified": status == "success",
        "usable_output": status == "success",
    }


def reviewed_aliases() -> dict:
    return {
        "schema_version": 1,
        "rule_version": "or-alias-v1",
        "evidence_manifest": {
            "source-card": {"uri": "s3://test/source.json", "sha256": "a" * 64},
            "or-model-record": {"uri": "s3://test/catalog.json", "sha256": "b" * 64},
        },
        "review_receipts": {
            "sol": {
                "run_id": "hatch_test_sol",
                "uri": "hatch://test/sol",
                "sha256": "c" * 64,
                "verdict": "approved",
                "reviewed_count": 1,
            },
            "grok": {
                "run_id": "hatch_test_grok",
                "uri": "hatch://test/grok",
                "sha256": "d" * 64,
                "verdict": "approved",
                "reviewed_count": 1,
            },
        },
        "aliases": [
            {
                "source_key": "deepinfra/Qwen/Qwen3-32B",
                "target_or_model_id": "qwen/qwen3-32b",
                "evidence_refs": ["source-card", "or-model-record"],
                "reviewers": ["sol", "grok"],
                "rule_version": "or-alias-v1",
            }
        ],
    }


def test_catalog_and_endpoint_evidence_without_probe_stays_direct():
    result = decide(
        source(),
        catalog=[catalog_row()],
        aliases=reviewed_aliases(),
        endpoint_payload=endpoint(),
        probe=None,
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "needs-pinned-probe"


def test_route_requires_matching_observed_provider_and_usable_output():
    result = decide(
        source(),
        catalog=[catalog_row()],
        aliases=reviewed_aliases(),
        endpoint_payload=endpoint(),
        probe=probe(),
    )

    assert result["decision"] == "route-or"
    assert result["reason_class"] == "verified-pinned-route"


def test_observed_provider_mismatch_stays_direct():
    result = decide(
        source(),
        catalog=[catalog_row()],
        aliases=reviewed_aliases(),
        endpoint_payload=endpoint(),
        probe=probe(observed_provider_slug="together"),
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "observed-provider-mismatch"


def test_ambiguous_slug_candidates_stay_direct():
    result = decide(
        source(provider="anthropic", model_id="shared-model"),
        catalog=[{"id": "anthropic/shared-model"}, {"id": "openai/shared-model"}],
        aliases={},
        endpoint_payload=None,
        probe=None,
        catalog_meta={"scope": "global"},
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "ambiguous-model-id"


def test_unique_slug_is_diagnostic_only():
    result = decide(
        {"provider": "deepinfra", "model_id": "qwen3-32b"},
        catalog=[catalog_row()],
        aliases={},
        endpoint_payload=endpoint(),
        probe=probe(),
        catalog_meta={"scope": "global"},
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "no-exact-or-ambiguous-model-id"
    assert result["evidence"]["unique_slug_candidates"] == ["qwen/qwen3-32b"]


def test_incomplete_catalog_cannot_prove_no_match():
    result = decide(
        {"provider": "deepinfra", "model_id": "absent-model"},
        catalog=[catalog_row()],
        aliases={},
        endpoint_payload=None,
        probe=None,
        catalog_meta={"scope": "public-discovery", "problems": ["catalog-scope-not-global"]},
    )

    assert result["reason_class"] == "catalog-evidence-incomplete"


def test_exact_identity_can_use_canonical_official_or_record():
    result = decide(
        {"provider": "deepinfra", "model_id": "qwen/qwen3-32b"},
        catalog=[catalog_row()],
        aliases={},
        endpoint_payload=endpoint(),
        probe=probe(),
        catalog_meta={"scope": "global", "snapshot_sha256": "a" * 64},
    )

    assert result["decision"] == "route-or"
    assert result["evidence"]["identity"]["method"] == "exact-id-canonical-or-record"


def test_diagnostic_stem_preserves_version_dots_and_bedrock_orgs():
    assert diagnostic_stem("Qwen/Qwen2.5-72B-Instruct") == "qwen2.5-72b-instruct"
    assert diagnostic_stem("us.anthropic.claude-opus-4-1-20250805-v1:0") == "claude-opus-4-1"


def test_catalog_repeat_evidence_must_match_observed_count():
    scope, problems = catalog_scope(
        {
            "total_count": 2,
            "catalog_scope": "global",
            "stable_repeated_count": True,
            "stable_repeated_counts": [2, 3],
        },
        observed_count=2,
    )
    assert scope == "incomplete"
    assert "catalog-repeat-evidence-missing" in problems


def test_provider_match_accepts_slug_carried_in_endpoint_tag():
    from scripts.openrouter_coverage_audit import provider_matches

    endpoint = {"provider_name": "Google", "tag": "google-vertex/global", "status": 0}
    assert provider_matches("vertex", endpoint) is True
    assert provider_matches("deepinfra", endpoint) is False
    studio = {"provider_name": "Google AI Studio", "tag": "google-ai-studio", "status": 0}
    assert provider_matches("vertex", studio) is False
