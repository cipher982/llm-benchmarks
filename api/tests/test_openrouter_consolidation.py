import mongomock
import pytest
from llm_bench.ops import openrouter_consolidation as consolidation


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _catalog(db, *model_ids):
    for model_id in model_ids:
        db.openrouter_catalog.insert_one({"openrouter_id": model_id})


def _model(db, provider, model_id, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, **extra})


class TestReEnable:
    def test_a_model_disabled_under_the_reversed_policy_comes_back(self, db):
        _catalog(db, "z-ai/glm-5.2")
        _model(
            db,
            "openrouter",
            "z-ai/glm-5.2",
            enabled=False,
            disabled_reason="Phase 1 cleanup: OpenRouter duplicates direct providers",
        )

        assert [m["model_id"] for m in consolidation.plan(db)["reenable"]] == ["z-ai/glm-5.2"]

    def test_a_model_judged_without_ever_being_scheduled_comes_back(self, db):
        _catalog(db, "mistralai/ministral-8b")
        _model(
            db, "openrouter", "mistralai/ministral-8b", enabled=False, disabled_reason="AI Operator: never succeeded"
        )

        assert [m["model_id"] for m in consolidation.plan(db)["reenable"]] == ["mistralai/ministral-8b"]

    def test_a_model_that_was_actually_tried_and_failed_stays_disabled(self, db):
        """The operator's verdict only lacks evidence when there was no attempt."""
        _catalog(db, "some/model")
        _model(db, "openrouter", "some/model", enabled=False, disabled_reason="AI Operator: never succeeded")
        db.bench_jobs.insert_one({"model_id": "some/model", "status": "dead_letter"})

        assert consolidation.plan(db)["reenable"] == []

    def test_a_model_openrouter_no_longer_lists_stays_disabled(self, db):
        _model(db, "openrouter", "gone/model", enabled=False, disabled_reason="Phase 1 cleanup: duplicates")

        assert consolidation.plan(db)["reenable"] == []

    def test_a_definitive_failure_stays_disabled(self, db):
        _catalog(db, "broken/model")
        _model(db, "openrouter", "broken/model", enabled=False, disabled_reason="2 definitive failure(s)")

        assert consolidation.plan(db)["reenable"] == []


class TestRetire:
    def test_non_core_direct_rows_are_retired(self, db):
        _model(db, "deepinfra", "Qwen/Qwen3-Max", enabled=True)
        _model(db, "groq", "llama-3.3-70b-versatile", enabled=True)

        retired = {(r["provider"], r["model_id"]) for r in consolidation.plan(db)["retire"]}
        assert retired == {("deepinfra", "Qwen/Qwen3-Max"), ("groq", "llama-3.3-70b-versatile")}

    def test_the_direct_lanes_are_never_retired(self, db):
        """OpenAI, Vertex and Bedrock measure the owner's real consumption."""
        for provider in consolidation.DIRECT_PROVIDERS:
            _model(db, provider, f"{provider}-model", enabled=True)
        _model(db, "openrouter", "native/model", enabled=True)

        assert consolidation.plan(db)["retire"] == []


class TestApply:
    def test_a_batch_larger_than_the_caps_still_drains(self, db, monkeypatch):
        """Staging everything into one batch would be refused outright."""
        monkeypatch.setenv("BENCHMARK_MAX_CHANGES_PER_BATCH", "40")
        monkeypatch.setenv("BENCHMARK_MAX_CHANGES_PER_PROVIDER", "25")
        for i in range(60):
            _model(db, "deepinfra", f"model-{i}", enabled=True)

        result = consolidation.run(db, apply=True)

        assert result["applied"] is True
        assert db.models.count_documents({"provider": "deepinfra", "enabled": True}) == 0
        assert len(result["batches"]) >= 3

    def test_every_change_is_recorded_for_reversal(self, db):
        _catalog(db, "z-ai/glm-5.2")
        _model(db, "openrouter", "z-ai/glm-5.2", enabled=False, disabled_reason="Phase 1 cleanup: duplicates")
        _model(db, "together", "openai/gpt-oss-20b", enabled=True)

        result = consolidation.run(db, apply=True)

        assert db.models.find_one({"provider": "openrouter"})["enabled"] is True
        assert db.models.find_one({"provider": "together"})["enabled"] is False
        recorded = list(db.bench_mutation_batches.find({}))
        assert {b["_id"] for b in recorded} == set(result["batches"])

    def test_a_dry_run_changes_nothing(self, db):
        _model(db, "together", "openai/gpt-oss-20b", enabled=True)

        consolidation.run(db, apply=False)

        assert db.models.find_one({"provider": "together"})["enabled"] is True


class TestMarketplaceDuplicatesOfDirectLanes:
    def test_openai_rows_on_the_marketplace_lane_are_retired(self, db):
        """The direct key is subsidised; paying a marketplace for it is waste."""
        _model(db, "openrouter", "openai/gpt-5.2-pro", enabled=True)
        _model(db, "openrouter", "openai/o3-mini", enabled=True)

        found = {d["model_id"] for d in consolidation.marketplace_duplicates(db)}
        assert found == {"openai/gpt-5.2-pro", "openai/o3-mini"}

    def test_the_dynamic_alias_form_is_caught_too(self, db):
        _model(db, "openrouter", "~openai/gpt-latest", enabled=True)

        assert [d["model_id"] for d in consolidation.marketplace_duplicates(db)] == ["~openai/gpt-latest"]

    def test_open_weight_models_the_direct_lane_cannot_serve_are_kept(self, db):
        """gpt-oss-* are open weights; the OpenAI API has no endpoint for them.

        Retiring by vendor prefix alone would lose these models rather than
        deduplicate them, which is the opposite of the point.
        """
        for model_id in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"):
            _model(db, "openrouter", model_id, enabled=True)

        assert consolidation.marketplace_duplicates(db) == []

    def test_other_vendors_are_untouched(self, db):
        _model(db, "openrouter", "anthropic/claude-opus-4.8", enabled=True)
        _model(db, "openrouter", "deepseek/deepseek-r1", enabled=True)

        assert consolidation.marketplace_duplicates(db) == []

    def test_applying_disables_them_reversibly(self, db):
        _model(db, "openrouter", "openai/gpt-5.2-pro", enabled=True)

        result = consolidation.run(db, apply=True)

        assert db.models.find_one({"model_id": "openai/gpt-5.2-pro"})["enabled"] is False
        assert {b["_id"] for b in db.bench_mutation_batches.find({})} == set(result["batches"])
