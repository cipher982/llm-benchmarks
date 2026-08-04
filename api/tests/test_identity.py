from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import identity
from llm_bench.ops.identity import Attributes

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"ident-{request.node.name}"]


def attrs(**kwargs):
    return Attributes(**kwargs)


class TestGroupingPolicy:
    def test_the_same_model_at_two_providers_shares_a_key(self, db):
        a = attrs(developer="meta", family="llama", version="3.3", params="70b", role="instruct")
        b = attrs(developer="meta", family="llama", version="3.3", params="70b", role="instruct")
        assert identity.canonical_key(a) == identity.canonical_key(b)

    def test_base_and_instruct_are_different_models(self):
        """The concrete bug in the 377-line table this replaces.

        Meta-Llama-3-8B and Meta-Llama-3-8B-Instruct both map to llama-3-8b
        today. They have different weights and different throughput.
        """
        base = attrs(developer="meta", family="llama", version="3", params="8b", role="base")
        instruct = attrs(developer="meta", family="llama", version="3", params="8b", role="instruct")
        assert identity.canonical_key(base) != identity.canonical_key(instruct)

    def test_serving_hints_do_not_split_a_group(self):
        """Turbo and FP8 are annotation, not identity — measured, not assumed."""
        plain = attrs(developer="meta", family="llama", version="3.3", params="70b", role="instruct")
        turbo = Attributes(
            developer="meta",
            family="llama",
            version="3.3",
            params="70b",
            role="instruct",
            annotations={"serving": "turbo", "quantization": "fp8"},
        )
        assert identity.canonical_key(plain) == identity.canonical_key(turbo)

    def test_different_sizes_are_different_models(self):
        small = attrs(developer="meta", family="llama", version="3.3", params="8b", role="instruct")
        large = attrs(developer="meta", family="llama", version="3.3", params="70b", role="instruct")
        assert identity.canonical_key(small) != identity.canonical_key(large)

    def test_thin_attributes_produce_no_key(self):
        """A missed merge is recoverable; a false merge is silent and is not."""
        assert identity.canonical_key(attrs(family="llama")) is None
        assert identity.canonical_key(attrs()) is None

    def test_grouping_is_a_pure_function_of_stored_attributes(self, db):
        rows = [
            {"provider": "groq", "model_id": "a", "attributes": attrs(developer="meta", family="llama", params="8b")},
            {
                "provider": "together",
                "model_id": "b",
                "attributes": attrs(developer="meta", family="llama", params="8b"),
            },
            {"provider": "openai", "model_id": "c", "attributes": attrs(developer="openai", family="gpt")},
        ]
        groups = identity.group_by_identity(rows)
        assert len(groups) == 2
        assert max(len(v) for v in groups.values()) == 2

    def test_unresolved_endpoints_never_share_a_group(self, db):
        rows = [
            {"provider": "groq", "model_id": "mystery-1", "attributes": attrs()},
            {"provider": "together", "model_id": "mystery-2", "attributes": attrs()},
        ]
        groups = identity.group_by_identity(rows)
        assert len(groups) == 2

    def test_a_new_endpoint_cannot_perturb_an_existing_group(self, db):
        """Idempotence: the reason normalisation is per-endpoint.

        Asking a model to group a whole list means one new arrival can reshuffle
        everything, which makes the site's history unstable.
        """
        existing = [
            {"provider": "groq", "model_id": "a", "attributes": attrs(developer="meta", family="llama", params="8b")},
        ]
        before = set(identity.group_by_identity(existing))
        after = set(
            identity.group_by_identity(
                existing
                + [{"provider": "x", "model_id": "n", "attributes": attrs(developer="mistralai", family="mixtral")}]
            )
        )
        assert before <= after


class TestResponseParsing:
    def test_nulls_and_placeholders_become_none(self):
        parsed = identity.attributes_from_response(
            {"developer": "meta", "family": "llama", "version": "null", "params": "unknown", "role": ""}
        )
        assert parsed.developer == "meta"
        assert parsed.version is None
        assert parsed.params is None
        assert parsed.role is None

    def test_a_refusal_to_guess_produces_no_key(self):
        parsed = identity.attributes_from_response({"developer": None, "family": None})
        assert identity.canonical_key(parsed) is None


class TestRetrievedContext:
    def test_siblings_at_other_providers_are_offered_for_disambiguation(self, db):
        for provider in ("groq", "together", "deepinfra"):
            db.provider_catalog.insert_one({"provider": provider, "model_id": f"{provider}/llama-3.3-70b-instruct"})
        db.provider_catalog.insert_one({"provider": "openai", "model_id": "gpt-4o"})

        siblings = identity.sibling_context(db, model_id="meta-llama/Llama-3.3-70B-Instruct")

        assert len(siblings) == 3
        assert all("llama" in s["model_id"].lower() for s in siblings)

    def test_the_prompt_carries_the_siblings(self, db):
        prompt = identity.build_prompt(
            provider="together",
            model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            name="Llama 3.3 70B",
            siblings=[{"provider": "groq", "model_id": "llama-3.3-70b-versatile"}],
        )
        assert "llama-3.3-70b-versatile" in prompt
        assert "DIFFERENT models" in prompt


class TestStoredRelations:
    def test_a_resolution_records_its_evidence_and_policy_version(self, db):
        db.provider_catalog.insert_one({"provider": "groq", "model_id": "llama-3.3-70b-versatile"})

        record = identity.resolve_endpoint(
            db,
            provider="together",
            model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            name="Llama 3.3 70B",
            call_llm=lambda _p: {
                "developer": "meta",
                "family": "llama",
                "version": "3.3",
                "params": "70b",
                "role": "instruct",
            },
            now=NOW,
        )

        assert record["canonical_key"] == "meta-llama-3.3-70b-instruct"
        assert record["policy_version"] == identity.POLICY_VERSION
        assert record["evidence"]["sibling_count"] == 1
        assert record["resolved"] is True

    def test_no_confidence_score_is_stored(self, db):
        """Self-reported confidence is not calibrated and must not gate anything."""
        record = identity.resolve_endpoint(
            db,
            provider="groq",
            model_id="m",
            name=None,
            call_llm=lambda _p: {"developer": "meta", "family": "llama", "confidence": 0.99},
            now=NOW,
        )
        assert "confidence" not in record
        assert "confidence" not in record["attributes"]

    def test_relations_are_appended_so_history_survives_a_rename(self, db):
        """The dashboard maps old metric rows with today's mapping.

        Without effective dates, re-resolving an endpoint silently rewrites what
        past measurements claim to be about.
        """
        identity.resolve_endpoint(
            db,
            provider="groq",
            model_id="m",
            name=None,
            call_llm=lambda _p: {"developer": "meta", "family": "llama", "params": "8b"},
            now=NOW - timedelta(days=30),
        )
        identity.resolve_endpoint(
            db,
            provider="groq",
            model_id="m",
            name=None,
            call_llm=lambda _p: {"developer": "meta", "family": "llama", "params": "8b", "role": "instruct"},
            now=NOW,
        )

        assert db.bench_model_identity.count_documents({"model_id": "m"}) == 2
        current = identity.current_identities(db)
        assert len(current) == 1
        assert current[0]["canonical_key"] == "meta-llama-8b-instruct"

    def test_an_unresolvable_endpoint_is_recorded_as_unresolved(self, db):
        record = identity.resolve_endpoint(
            db,
            provider="groq",
            model_id="some-internal-codename",
            name=None,
            call_llm=lambda _p: {"developer": None, "family": None},
            now=NOW,
        )
        assert record["resolved"] is False
        assert record["canonical_key"] is None


class TestTiersAreDistinctModels:
    """Caught by the divergence report on the first production run.

    claude-haiku-4.5 and claude-sonnet-4.5 collapsed into one key, because the
    schema had no place for the tier and for Anthropic the tier is the model.
    That is a false merge — the exact failure the whole design exists to avoid,
    and worse than the base/instruct bug it was replacing.
    """

    def test_haiku_and_sonnet_do_not_share_a_key(self):
        haiku = Attributes(developer="anthropic", family="claude-haiku", version="4.5", role="chat")
        sonnet = Attributes(developer="anthropic", family="claude-sonnet", version="4.5", role="chat")
        assert identity.canonical_key(haiku) != identity.canonical_key(sonnet)

    def test_a_bare_family_would_have_merged_them(self):
        """Why the prompt now demands the tier: without it the key is identical."""
        a = Attributes(developer="anthropic", family="claude", version="4.5", role="chat")
        b = Attributes(developer="anthropic", family="claude", version="4.5", role="chat")
        assert identity.canonical_key(a) == identity.canonical_key(b)

    def test_the_prompt_names_the_failure(self):
        prompt = identity.build_prompt(provider="anthropic", model_id="claude-haiku-4-5", name=None, siblings=[])
        assert "claude-haiku" in prompt
        assert "Haiku is not Claude Sonnet" in prompt
