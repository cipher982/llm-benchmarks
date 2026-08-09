from scripts.openrouter_coverage_audit import decide


def source(provider: str = "deepinfra", model_id: str = "Qwen/Qwen3-32B") -> dict:
    return {
        "provider": provider,
        "model_id": model_id,
        "display_name": model_id,
        "enabled": True,
        "deprecated": False,
    }


def endpoint(provider_name: str = "DeepInfra") -> dict:
    return {
        "data": {
            "endpoints": [
                {
                    "provider_name": provider_name,
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


def test_catalog_and_endpoint_evidence_without_probe_stays_direct():
    result = decide(
        source(),
        catalog=[{"id": "qwen/qwen3-32b"}],
        aliases={},
        endpoint_payload=endpoint(),
        probe=None,
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "needs-pinned-probe"


def test_route_requires_matching_observed_provider_and_usable_output():
    result = decide(
        source(),
        catalog=[{"id": "qwen/qwen3-32b"}],
        aliases={},
        endpoint_payload=endpoint(),
        probe=probe(),
    )

    assert result["decision"] == "route-or"
    assert result["reason_class"] == "verified-pinned-route"


def test_observed_provider_mismatch_stays_direct():
    result = decide(
        source(),
        catalog=[{"id": "qwen/qwen3-32b"}],
        aliases={},
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
    )

    assert result["decision"] == "keep-direct"
    assert result["reason_class"] == "ambiguous-model-id"
