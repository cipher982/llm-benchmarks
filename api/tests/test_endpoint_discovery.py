"""Endpoint discovery — the catalogue the site will actually schedule from.

The cases here are the ones two adversarial reviews and a live API sweep said
would break the design: base slugs that are not endpoints, quantization that
makes endpoints incomparable, degraded endpoints admitted by presence, and a
partial discovery pass mass-retiring healthy targets.
"""

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery as ed


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def endpoint(tag, *, status=0, quant="fp8", **extra):
    return {
        "tag": tag,
        "provider_name": tag.split("/")[0].title(),
        "status": status,
        "quantization": quant,
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "pricing": {"completion": "0.0000006", "prompt": "0.00000015"},
        **extra,
    }


class TestIdentity:
    def test_provider_canonical_strips_the_variant(self):
        """The family is the verification key; the tag is the identity key."""
        assert ed.provider_canonical("deepinfra/fp8") == "deepinfra"
        assert ed.provider_canonical("google-vertex/us-east5") == "google-vertex"
        assert ed.provider_canonical("groq") == "groq"

    def test_variants_of_one_provider_are_distinct_endpoints(self):
        """`only:["deepinfra"]` still load-balances bf16 against turbo."""
        a = ed.endpoint_doc("m", endpoint("deepinfra/bf16", quant="bf16"), now=1)
        b = ed.endpoint_doc("m", endpoint("deepinfra/turbo", quant="bf16"), now=1)
        assert a["endpoint_tag"] != b["endpoint_tag"]
        assert a["provider_canonical"] == b["provider_canonical"] == "deepinfra"


class TestQuantization:
    def test_unknown_is_preserved_not_defaulted_to_a_real_value(self):
        """Groq reports `unknown`; grouping it with fp8 would invent a fact."""
        assert ed.quantization_of({"quantization": None}) == "unknown"
        assert ed.quantization_of({}) == "unknown"
        assert ed.quantization_of({"quantization": "  FP8 "}) == "fp8"

    def test_one_model_can_span_fp4_to_bf16(self):
        """gpt-oss-120b really is served at both. They are not one axis."""
        quants = {
            ed.endpoint_doc("openai/gpt-oss-120b", endpoint(t, quant=q), now=1)["quantization"]
            for t, q in [("coreweave/fp4", "fp4"), ("deepinfra/bf16", "bf16"), ("groq", None)]
        }
        assert quants == {"fp4", "bf16", "unknown"}


class TestAdmission:
    def test_degraded_endpoints_are_refused(self):
        """Four of gpt-oss-120b's twenty endpoints were at -2."""
        ok, reason = ed.is_admissible(endpoint("siliconflow/fp8", status=-2))
        assert not ok
        assert reason == "openrouter-endpoint-status--2"

    def test_direct_lane_endpoints_are_refused(self):
        """Bedrock and Vertex are measured on our own credentials."""
        for tag in ("amazon-bedrock", "amazon-bedrock/eu-west-1", "google-vertex/global", "openai"):
            ok, reason = ed.is_admissible(endpoint(tag))
            assert not ok, tag
            assert reason == "served-by-a-direct-lane"

    def test_ordinary_endpoint_is_admitted(self):
        ok, reason = ed.is_admissible(endpoint("groq", quant=None))
        assert ok and reason is None


class TestRefresh:
    def test_admitted_endpoints_land_with_identity_and_price(self, db):
        ed.refresh_endpoints(
            db,
            model_ids=["openai/gpt-oss-120b"],
            fetcher=lambda m: [endpoint("groq", quant=None), endpoint("deepinfra/bf16", quant="bf16")],
        )
        rows = {r["endpoint_tag"]: r for r in db[ed.endpoints_collection_name()].find({})}
        assert set(rows) == {"groq", "deepinfra/bf16"}
        assert rows["groq"]["quantization"] == "unknown"
        assert rows["deepinfra/bf16"]["quantization"] == "bf16"
        # Endpoint-level price, not the model's.
        assert rows["groq"]["completion_price_per_token"] == pytest.approx(6e-7)
        assert all(r["enabled"] for r in rows.values())

    def test_a_failed_model_read_retires_nothing(self, db):
        """A rate-limited pass is not evidence that a deployment disappeared."""

        def boom(model_id):
            raise TimeoutError("rate limited")

        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq")])
        record = ed.refresh_endpoints(db, model_ids=["m"], fetcher=boom)

        row = db[ed.endpoints_collection_name()].find_one({"endpoint_tag": "groq"})
        assert row["enabled"] is True
        assert row["missing_passes"] == 0
        assert record["status"] == "partial"

    def test_retirement_needs_repeated_complete_absences(self, db):
        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq"), endpoint("novita/fp8")])

        col = db[ed.endpoints_collection_name()]
        for expected_misses in range(1, ed.MISSING_PASSES_BEFORE_RETIREMENT):
            ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq")])
            row = col.find_one({"endpoint_tag": "novita/fp8"})
            assert row["enabled"] is True, "retired too early"
            assert row["missing_passes"] == expected_misses

        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq")])
        row = col.find_one({"endpoint_tag": "novita/fp8"})
        assert row["enabled"] is False
        assert "consecutive" in row["disabled_reason"]
        assert col.find_one({"endpoint_tag": "groq"})["enabled"] is True

    def test_a_returning_endpoint_resets_its_absence_count(self, db):
        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq"), endpoint("novita/fp8")])
        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq")])
        ed.refresh_endpoints(db, model_ids=["m"], fetcher=lambda m: [endpoint("groq"), endpoint("novita/fp8")])

        row = db[ed.endpoints_collection_name()].find_one({"endpoint_tag": "novita/fp8"})
        assert row["missing_passes"] == 0
        assert row["enabled"] is True

    def test_routers_are_never_endpoint_targets(self, db):
        called = []
        ed.refresh_endpoints(db, model_ids=["openrouter/auto-beta"], fetcher=lambda m: called.append(m) or [])
        assert called == []
        assert db[ed.endpoints_collection_name()].count_documents({}) == 0
