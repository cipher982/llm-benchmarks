import mongomock
import pytest
from llm_bench.ops import invariants
from llm_bench.ops import lane_duplicates as dup
from llm_bench.scheduler import policies

OLD_REASON = (
    "Retired from the OpenRouter lane: this vendor has a direct lane on a subsidised key, "
    "so paying a marketplace to measure the same model is both a duplicate row on the site "
    "and a bill we do not need to pay."
)


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _model(db, provider, model_id, enabled=True, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": enabled, **extra})


def _endpoint(db, model_id, host, enabled=True):
    db.bench_endpoints.insert_one(
        {"model_id": model_id, "endpoint_tag": host, "provider_canonical": host, "enabled": enabled}
    )


def _found(db):
    return {(d["provider"], d["model_id"]) for d in dup.duplicates(db)}


class TestIdMapping:
    @pytest.mark.parametrize(
        "provider,model_id,expected",
        [
            ("openai", "gpt-6.1-sol", "openai/gpt-6.1-sol"),
            ("vertex", "gemini-2.5-pro", "google/gemini-2.5-pro"),
            ("bedrock", "amazon.nova-lite-v1:0", "amazon/nova-lite-v1"),
            ("bedrock", "us.anthropic.claude-opus-4-5-20251101-v1:0", "anthropic/claude-opus-4.5"),
            ("bedrock", "us.anthropic.claude-opus-4-6-v1", "anthropic/claude-opus-4.6"),
            ("bedrock", "us.anthropic.claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"),
            ("bedrock", "us.meta.llama3-3-70b-instruct-v1:0", None),
        ],
    )
    def test_direct_ids_map_to_openrouter(self, provider, model_id, expected):
        assert dup.openrouter_id_for(provider, model_id) == expected


class TestDuplicates:
    def test_direct_openai_beside_openrouter_openai_is_a_duplicate(self, db):
        _model(db, "openai", "gpt-6.1-sol")
        _model(db, "openrouter", "openai/gpt-6.1-sol")
        _endpoint(db, "openai/gpt-6.1-sol", "azure")

        assert _found(db) == {("openai", "gpt-6.1-sol")}

    def test_unpinned_first_party_row_is_the_same_host(self, db):
        _model(db, "bedrock", "amazon.nova-pro-v1:0")
        _model(db, "openrouter", "amazon/nova-pro-v1")

        assert _found(db) == {("bedrock", "amazon.nova-pro-v1:0")}

    def test_bedrock_claude_is_a_duplicate_only_when_openrouter_serves_it_from_aws(self, db):
        _model(db, "bedrock", "us.anthropic.claude-opus-4-6-v1")
        _model(db, "bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        _model(db, "openrouter", "anthropic/claude-opus-4.6")
        _model(db, "openrouter", "anthropic/claude-haiku-4.5")
        for host in ("anthropic", "claude-on-aws"):
            _endpoint(db, "anthropic/claude-opus-4.6", host)
        _endpoint(db, "anthropic/claude-haiku-4.5", "anthropic")

        assert _found(db) == {("bedrock", "us.anthropic.claude-opus-4-6-v1")}

    def test_a_different_host_is_not_a_duplicate(self, db):
        """Bedrock serving Llama and Groq serving Llama are two measurements."""
        _model(db, "bedrock", "us.meta.llama3-3-70b-instruct-v1:0")
        _model(db, "openrouter", "meta-llama/llama-3.3-70b-instruct")
        _endpoint(db, "meta-llama/llama-3.3-70b-instruct", "groq")

        assert _found(db) == set()

    def test_a_disabled_side_is_not_a_duplicate(self, db):
        _model(db, "openai", "gpt-4o", enabled=False)
        _model(db, "openrouter", "openai/gpt-4o")

        assert _found(db) == set()

    def test_aliases_are_duplicates(self, db):
        _model(db, "openrouter", "~openai/gpt-sol-latest")

        assert _found(db) == {("openrouter", "~openai/gpt-sol-latest")}

    def test_the_invariant_reports_them(self, db):
        _model(db, "openrouter", "~x-ai/grok-latest")
        ctx = invariants.Context(db=db, now=invariants.utcnow())

        assert [v.subject for v in invariants.no_model_measured_twice_on_one_host(ctx)] == [
            "openrouter/~x-ai/grok-latest"
        ]
        assert "no_model_measured_twice_on_one_host" in {i.name for i in invariants.INVARIANTS}


class TestApply:
    def test_retires_direct_and_brings_back_the_openrouter_twin(self, db):
        db.openrouter_catalog.insert_one({"openrouter_id": "openai/gpt-4o"})
        _model(db, "openai", "gpt-4o")
        _model(db, "openrouter", "openai/gpt-4o", enabled=False, disabled_reason=OLD_REASON)

        dup.run(db, apply=True)

        assert db.models.find_one({"provider": "openai"})["enabled"] is False
        assert db.models.find_one({"provider": "openrouter"})["enabled"] is True
        assert dup.duplicates(db) == []

    def test_a_twin_disabled_for_failing_is_left_alone_and_direct_is_kept(self, db):
        db.openrouter_catalog.insert_one({"openrouter_id": "openai/gpt-4o"})
        _model(db, "openai", "gpt-4o")
        _model(db, "openrouter", "openai/gpt-4o", enabled=False, disabled_reason="2 definitive failure(s)")

        dup.run(db, apply=True)

        assert db.models.find_one({"provider": "openai"})["enabled"] is True
        assert db.models.find_one({"provider": "openrouter"})["enabled"] is False

    def test_aliases_are_disabled_and_dry_run_changes_nothing(self, db):
        _model(db, "openrouter", "~z-ai/glm-latest")

        dup.run(db, apply=False)
        assert db.models.find_one({})["enabled"] is True
        dup.run(db, apply=True)
        assert db.models.find_one({})["enabled"] is False


def test_policy_never_admits_an_alias():
    assert policies.openrouter_model_allowed("~openai/gpt-sol-latest") is False
    assert policies.openrouter_model_allowed("openai/gpt-6.1-sol") is True
