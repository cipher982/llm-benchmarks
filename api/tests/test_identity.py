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
        assert "not on the list" in record["evidence"]["basis"]

    def test_an_unnamed_model_is_still_keyed_so_it_can_be_matched_later(self, db):
        """No group at all keeps an endpoint out of the list forever.

        A second provider serving the same model would never see it as an
        option, so a merge that should happen never can. Keying it to its own
        ID gives a group of one that others can join.
        """
        record = identity.match_endpoint(
            db,
            provider="deepinfra",
            model_id="thinkingmachines/Inkling",
            name=None,
            call_llm=lambda _p: {"group": None, "name": None},
            now=NOW,
        )
        assert record["canonical_key"] == "inkling"
        assert record["resolved"] is True

    def test_a_later_provider_can_join_a_self_keyed_group(self, db):
        self._existing(db, "deepinfra", "thinkingmachines/Inkling", "inkling")

        seen = {}
        record = identity.match_endpoint(
            db,
            provider="together",
            model_id="thinkingmachines/Inkling",
            name=None,
            call_llm=lambda p: seen.update(prompt=p) or {"group": "inkling"},
            now=NOW,
        )
        assert "inkling" in seen["prompt"]
        assert record["canonical_key"] == "inkling"

    def test_the_prompt_carries_no_vendor_list(self):
        """The regression that motivated this: a taxonomy hidden in a string."""
        prompt = identity.build_match_prompt(provider="x", model_id="y", name=None, groups={})
        for vendor_specific in ("claude-haiku", "gemini-flash", "nova-pro", "mixtral"):
            assert vendor_specific not in prompt

    def test_every_group_is_offered_not_a_filtered_subset(self, db):
        """No rule may decide which groups the model is allowed to consider.

        Any such rule is a claim about which models resemble each other, which
        is the judgment being delegated. v3 filtered by shared tokens against a
        stopword list and was a smaller version of the taxonomy it replaced.
        """
        for i in range(30):
            self._existing(db, "p", f"m{i}", f"group-{i}")

        seen = {}
        identity.match_endpoint(
            db,
            provider="groq",
            model_id="totally-unrelated-name",
            name=None,
            call_llm=lambda p: seen.update(prompt=p) or {"group": None, "name": "new"},
            now=NOW,
        )

        for i in range(30):
            assert f"group-{i}" in seen["prompt"]


class TestConsolidation:
    """Groups form one endpoint at a time, so the same model can get two names.

    Llama 3.3 70B split into llama-3.3-70b (bedrock, groq) and
    llama-3.3-70b-instruct (deepinfra, together) purely on arrival order,
    turning a four-provider line into two two-provider lines.
    """

    def _existing(self, db, provider, model_id, key):
        db.bench_model_identity.insert_one(
            {"provider": provider, "model_id": model_id, "canonical_key": key, "effective_from": NOW}
        )

    def _split_llama(self, db):
        self._existing(db, "bedrock", "us.meta.llama3-3-70b-instruct-v1:0", "llama-3.3-70b")
        self._existing(db, "groq", "llama-3.3-70b-versatile", "llama-3.3-70b")
        self._existing(db, "deepinfra", "meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct")
        self._existing(db, "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "llama-3.3-70b-instruct")

    def test_a_split_group_is_rejoined(self, db):
        self._split_llama(db)

        identity.consolidate_groups(
            db,
            call_llm=lambda _p: {"merges": [{"keep": "llama-3.3-70b", "absorb": ["llama-3.3-70b-instruct"]}]},
            now=NOW,
            dry_run=False,
        )

        groups = identity.existing_groups(db)
        assert "llama-3.3-70b-instruct" not in groups
        assert len({m.split("/")[0] for m in groups["llama-3.3-70b"]}) == 4

    def test_dry_run_reports_without_merging(self, db):
        self._split_llama(db)

        merges = identity.consolidate_groups(
            db,
            call_llm=lambda _p: {"merges": [{"keep": "llama-3.3-70b", "absorb": ["llama-3.3-70b-instruct"]}]},
            now=NOW,
            dry_run=True,
        )

        assert merges[0]["endpoints"] == 2
        assert "llama-3.3-70b-instruct" in identity.existing_groups(db)

    def test_a_merge_naming_a_group_that_does_not_exist_is_ignored(self, db):
        """Inventing either side would move endpoints onto a name nothing uses."""
        self._split_llama(db)

        merges = identity.consolidate_groups(
            db,
            call_llm=lambda _p: {"merges": [{"keep": "invented", "absorb": ["llama-3.3-70b"]}]},
            now=NOW,
            dry_run=False,
        )

        assert merges == []
        assert "llama-3.3-70b" in identity.existing_groups(db)

    def test_declining_to_merge_is_respected(self, db):
        self._split_llama(db)
        assert identity.consolidate_groups(db, call_llm=lambda _p: {"merges": []}, now=NOW) == []

    def test_the_prompt_offers_every_group(self, db):
        self._split_llama(db)
        self._existing(db, "openai", "gpt-4o", "gpt-4o")

        seen = {}
        identity.consolidate_groups(db, call_llm=lambda p: seen.update(prompt=p) or {"merges": []}, now=NOW)

        for key in ("llama-3.3-70b", "llama-3.3-70b-instruct", "gpt-4o"):
            assert key in seen["prompt"]
