from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import identity

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"ident-{request.node.name}"]


class TestMatchingWithoutATaxonomy:
    """The attribute schema needed a hand-maintained vendor list to work.

    developer/family/version/params only decomposes names that decompose that
    way. Anthropic's tiers did not, so the prompt grew a list — claude-haiku,
    claude-sonnet, gemini-flash, nova-pro — which is the 377-line table moved
    into a string. Matching against existing groups needs no such list.
    """

    def _existing(self, db, provider, model_id, key):
        db.bench_model_identity.insert_one(
            {"provider": provider, "model_id": model_id, "canonical_key": key, "effective_from": NOW}
        )

    def test_an_endpoint_joins_a_group_the_model_picks(self, db):
        self._existing(db, "anthropic", "claude-haiku-4-5", "claude-haiku-4.5")
        self._existing(db, "anthropic", "claude-sonnet-4-5", "claude-sonnet-4.5")

        record = identity.match_endpoint(
            db,
            provider="bedrock",
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            name=None,
            call_llm=lambda _p: {"group": "claude-haiku-4.5"},
            now=NOW,
        )

        assert record["canonical_key"] == "claude-haiku-4.5"
        assert record["evidence"]["basis"] == "matched an existing group"

    def test_both_tiers_are_offered_so_the_choice_is_the_models_to_make(self, db):
        """No vendor list decides this — the candidates are shown and it picks."""
        self._existing(db, "anthropic", "claude-haiku-4-5", "claude-haiku-4.5")
        self._existing(db, "anthropic", "claude-sonnet-4-5", "claude-sonnet-4.5")

        seen = {}
        identity.match_endpoint(
            db,
            provider="bedrock",
            model_id="us.anthropic.claude-sonnet-4-5-v1:0",
            name=None,
            call_llm=lambda p: seen.update(prompt=p) or {"group": "claude-sonnet-4.5"},
            now=NOW,
        )
        assert "claude-haiku-4.5" in seen["prompt"]
        assert "claude-sonnet-4.5" in seen["prompt"]

    def test_an_unfamiliar_vendor_forms_its_own_group(self, db):
        record = identity.match_endpoint(
            db,
            provider="deepinfra",
            model_id="thinkingmachines/Inkling",
            name=None,
            call_llm=lambda _p: {"group": None, "name": "Inkling"},
            now=NOW,
        )
        assert record["canonical_key"] == "inkling"
        assert record["evidence"]["basis"] == "new group"

    def test_a_hallucinated_group_does_not_become_a_merge_target(self, db):
        """Naming a group that does not exist must not invent one to merge into."""
        self._existing(db, "anthropic", "claude-haiku-4-5", "claude-haiku-4.5")

        record = identity.match_endpoint(
            db,
            provider="groq",
            model_id="something-else",
            name=None,
            call_llm=lambda _p: {"group": "a-group-that-does-not-exist"},
            now=NOW,
        )

        assert record["canonical_key"] == "a-group-that-does-not-exist"
        assert "did not exist" in record["evidence"]["basis"]

    def test_declining_to_name_leaves_the_endpoint_unresolved(self, db):
        record = identity.match_endpoint(
            db,
            provider="groq",
            model_id="internal-codename",
            name=None,
            call_llm=lambda _p: {"group": None, "name": None},
            now=NOW,
        )
        assert record["resolved"] is False

    def test_the_prompt_carries_no_vendor_list(self):
        """The regression that motivated this: a taxonomy hidden in a string."""
        prompt = identity.build_match_prompt(provider="x", model_id="y", name=None, candidates={})
        for vendor_specific in ("claude-haiku", "gemini-flash", "nova-pro", "mixtral"):
            assert vendor_specific not in prompt
