"""Names are derived from the catalogue, and identity is not.

The hand-maintained list is being retired, so every rule here has to hold with
no human in the loop: a stale catalogue must not rewrite the site, a third-party
rename must not become a chart identity, and a model nothing can name must fail
visibly rather than quietly pass as named.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import model_naming

NOW = datetime(2026, 8, 17, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _catalogue(db, model_id, name, *, org=None, seen=NOW, modalities=("text",)):
    db.openrouter_catalog.insert_one(
        {
            "openrouter_id": model_id,
            "name": name,
            "org": org,
            "output_modalities": list(modalities),
            "last_seen_at": seen,
        }
    )


def _model(db, provider, model_id, display_name=None, **extra):
    db.models.insert_one(
        {"provider": provider, "model_id": model_id, "display_name": display_name, "enabled": True, **extra}
    )


def _identity(db, provider, model_id, key):
    db.bench_model_identity.insert_one(
        {"provider": provider, "model_id": model_id, "canonical_key": key, "effective_from": NOW}
    )


class TestParsing:
    def test_the_vendor_prefix_is_split_off(self):
        assert model_naming.parse_catalogue_name("Z.ai: GLM 4.7") == ("Z.ai", "GLM 4.7")

    def test_a_name_without_a_separator_is_the_whole_label(self):
        """42 of 619 catalogue rows have no separator; that is ordinary."""
        assert model_naming.parse_catalogue_name("Codestral 2508") == (None, "Codestral 2508")

    def test_only_the_first_separator_splits(self):
        """Model names carry colons more often than vendors do."""
        assert model_naming.parse_catalogue_name("OpenAI: GPT-4o: Extended") == ("OpenAI", "GPT-4o: Extended")

    def test_missing_and_empty_names_are_not_labels(self):
        assert model_naming.parse_catalogue_name(None) == (None, None)
        assert model_naming.parse_catalogue_name("   ") == (None, None)


class TestCatalogueFreshness:
    def test_a_stale_row_is_not_a_live_answer(self, db):
        """The collection is append-only; April rows are still present."""
        _catalogue(db, "fresh/model", "Vendor: Fresh", seen=NOW)
        _catalogue(db, "old/model", "Vendor: Old", seen=NOW - timedelta(days=120))

        assert set(model_naming.catalogue_labels(db)) == {"fresh/model"}

    def test_freshness_is_relative_to_the_catalogue_not_the_clock(self, db):
        """A discovery outage must degrade to 'no new labels', not 'all stale'.

        Measuring against wall-clock would mean a stalled discovery job silently
        rewrites every name on the site to a fallback.
        """
        old = NOW - timedelta(days=400)
        _catalogue(db, "a/model", "Vendor: A", seen=old)
        _catalogue(db, "b/model", "Vendor: B", seen=old - timedelta(hours=1))

        assert set(model_naming.catalogue_labels(db)) == {"a/model", "b/model"}

    def test_non_text_models_are_not_labelled(self, db):
        _catalogue(db, "img/model", "Vendor: Imager", modalities=("image",))

        assert model_naming.catalogue_labels(db) == {}


class TestPlanning:
    def test_an_openrouter_row_takes_the_catalogue_label(self, db):
        _catalogue(db, "z-ai/glm-4.7", "Z.ai: GLM 4.7", org="z-ai")
        _model(db, "openrouter", "z-ai/glm-4.7", "z-ai/glm-4.7-20251222")

        (proposal,) = model_naming.plan(db)
        assert proposal.label == "GLM 4.7"
        assert proposal.vendor == "z-ai"
        assert proposal.source == model_naming.SOURCE_CATALOGUE

    def test_a_direct_lane_inherits_from_its_identity_sibling(self, db):
        """Bedrock has no presentation feed; identity is how it gets a name."""
        _catalogue(db, "anthropic/claude-opus-4.6", "Anthropic: Claude Opus 4.6", org="anthropic")
        _model(db, "openrouter", "anthropic/claude-opus-4.6")
        _model(db, "bedrock", "us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6")
        _identity(db, "openrouter", "anthropic/claude-opus-4.6", "claude-opus-4.6")
        _identity(db, "bedrock", "us.anthropic.claude-opus-4-6-v1", "claude-opus-4.6")

        labels = {(p.provider, p.label, p.source) for p in model_naming.plan(db)}
        assert ("bedrock", "Claude Opus 4.6", model_naming.SOURCE_IDENTITY_SIBLING) in labels
        assert ("openrouter", "Claude Opus 4.6", model_naming.SOURCE_CATALOGUE) in labels

    def test_a_curated_name_is_not_overwritten_with_a_worse_one(self, db):
        """Declining to replace a readable name is not keeping the hand list."""
        _model(db, "bedrock", "amazon.nova-lite-v1:0", "nova-lite")

        (proposal,) = model_naming.plan(db)
        assert proposal.label == "nova-lite"
        assert proposal.source == model_naming.SOURCE_EXISTING

    def test_a_dated_slug_is_recognised_as_a_raw_id(self, db):
        """This is what discovery wrote, and it must not count as curated."""
        _model(db, "openrouter", "aion-labs/aion-2.0", "aion-labs/aion-2.0-20260223")

        (proposal,) = model_naming.plan(db)
        assert proposal.source == model_naming.SOURCE_FALLBACK
        assert proposal.label == "aion-2.0"

    def test_undefined_is_not_a_name(self, db):
        _model(db, "vertex", "gemini-2.5-flash-lite", "undefined")

        (proposal,) = model_naming.plan(db)
        assert proposal.source == model_naming.SOURCE_FALLBACK
        assert proposal.label == "gemini-2.5-flash-lite"

    def test_the_fallback_is_marked_as_one(self, db):
        """An unnameable model must read as missing data, not as named."""
        _model(db, "bedrock", "amazon.nova-lite-v1:0")

        (proposal,) = model_naming.plan(db)
        assert proposal.source == model_naming.SOURCE_FALLBACK
        assert proposal.label == "amazon.nova-lite-v1"


class TestCollisions:
    def test_two_rows_on_one_provider_cannot_share_a_name(self, db):
        """Publication groups by name within a provider, so a collision would
        average two unrelated deployments into one series."""
        _catalogue(db, "reka/reka-edge", "Reka: Reka Edge", org="reka")
        _catalogue(db, "other/reka-edge", "Other: Reka Edge", org="other")
        _model(db, "openrouter", "reka/reka-edge")
        _model(db, "openrouter", "other/reka-edge")

        labels = [p.label for p in model_naming.plan(db)]
        assert len(set(labels)) == 2, labels
        assert all("Reka Edge" in label for label in labels)

    def test_the_same_name_on_different_providers_is_not_a_collision(self, db):
        """That is the cross-provider case the site exists to show."""
        _catalogue(db, "anthropic/claude-opus-4.6", "Anthropic: Claude Opus 4.6")
        _model(db, "openrouter", "anthropic/claude-opus-4.6")
        _model(db, "bedrock", "us.anthropic.claude-opus-4-6-v1", "Claude Opus 4.6")

        assert [p.label for p in model_naming.plan(db)] == ["Claude Opus 4.6", "Claude Opus 4.6"]


class TestApply:
    def test_labels_are_written_reversibly(self, db):
        _catalogue(db, "z-ai/glm-4.7", "Z.ai: GLM 4.7", org="z-ai")
        _model(db, "openrouter", "z-ai/glm-4.7", "z-ai/glm-4.7-20251222")

        report = model_naming.apply_names(db, apply=True)

        stored = db.models.find_one({"model_id": "z-ai/glm-4.7"})
        assert stored["display_name"] == "GLM 4.7"
        assert stored["display_name_source"] == model_naming.SOURCE_CATALOGUE
        assert {b["_id"] for b in db.bench_mutation_batches.find({})} == set(report["batches"])

    def test_a_report_changes_nothing(self, db):
        _catalogue(db, "z-ai/glm-4.7", "Z.ai: GLM 4.7")
        _model(db, "openrouter", "z-ai/glm-4.7", "z-ai/glm-4.7-20251222")

        model_naming.apply_names(db, apply=False)

        assert db.models.find_one({})["display_name"] == "z-ai/glm-4.7-20251222"

    def test_applying_twice_is_idempotent(self, db):
        """This runs continuously; a no-op pass must write no batch."""
        _catalogue(db, "z-ai/glm-4.7", "Z.ai: GLM 4.7")
        _model(db, "openrouter", "z-ai/glm-4.7", "z-ai/glm-4.7-20251222")

        model_naming.apply_names(db, apply=True)
        second = model_naming.apply_names(db, apply=True)

        assert second["to_change"] == 0
        assert db.bench_mutation_batches.count_documents({}) == 1

    def test_more_models_than_the_batch_cap_still_drain(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MAX_CHANGES_PER_BATCH", "40")
        monkeypatch.setenv("BENCHMARK_MAX_CHANGES_PER_PROVIDER", "25")
        for i in range(60):
            _catalogue(db, f"vendor/model-{i}", f"Vendor: Model {i}")
            _model(db, "openrouter", f"vendor/model-{i}", f"vendor/model-{i}")

        report = model_naming.apply_names(db, apply=True)

        assert report["applied"] is True
        assert db.models.count_documents({"display_name": {"$regex": "^Model "}}) == 60


class TestCollisionsPreferRederivedIds:
    def test_a_duplicate_curated_name_is_split_by_its_id(self, db):
        """OpenAI has two rows both called `gpt-4`.

        `gpt-4` and `gpt-4-turbo` is the right answer; `gpt-4 (gpt-4)` and
        `gpt-4 (gpt-4-turbo)` is what a naive parenthetical produces.
        """
        _model(db, "openai", "gpt-4", "gpt-4")
        _model(db, "openai", "gpt-4-turbo", "gpt-4")

        labels = sorted(p.label for p in model_naming.plan(db))
        assert labels == ["gpt-4", "gpt-4-turbo"]

    def test_a_parenthetical_is_used_when_ids_do_not_separate_them(self, db):
        """`reka/reka-edge` and `other/reka-edge` share a last segment."""
        _catalogue(db, "reka/reka-edge", "Reka: Reka Edge", org="reka")
        _catalogue(db, "other/reka-edge", "Other: Reka Edge", org="other")
        _model(db, "openrouter", "reka/reka-edge")
        _model(db, "openrouter", "other/reka-edge")

        labels = sorted(p.label for p in model_naming.plan(db))
        assert labels == ["Reka Edge (other)", "Reka Edge (reka)"]
