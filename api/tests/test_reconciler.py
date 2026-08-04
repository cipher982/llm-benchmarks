from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import mutations
from llm_bench.ops import reconciler

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"rec-{request.node.name}"]


def enabled(db, provider, model_id):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True})


def catalogue(db, provider, model_id, *, ago):
    db.provider_catalog.insert_one({"provider": provider, "model_id": model_id, "last_seen_at": NOW - ago})


def runs(db, provider, *, count=3, complete=True, status="completed", start_days=1):
    for i in range(count):
        db.bench_discovery_runs.insert_one(
            {
                "provider": provider,
                "status": status,
                "pagination_complete": complete,
                "finished_at": NOW - timedelta(days=start_days + i),
            }
        )


class TestAbsenceRequiresEvidence:
    def test_a_model_absent_from_three_complete_runs_is_retired(self, db):
        enabled(db, "together", "gone")
        catalogue(db, "together", "gone", ago=timedelta(days=10))
        runs(db, "together")

        found = reconciler.find_retirements(db, now=NOW)

        assert [r.subject for r in found] == ["together/gone"]

    def test_a_model_still_listed_is_left_alone(self, db):
        enabled(db, "together", "live")
        catalogue(db, "together", "live", ago=timedelta(hours=2))
        runs(db, "together")

        assert reconciler.find_retirements(db, now=NOW) == []

    def test_failed_runs_are_not_evidence_of_absence(self, db):
        """A run that errored says nothing about what the provider offers."""
        enabled(db, "together", "maybe-gone")
        catalogue(db, "together", "maybe-gone", ago=timedelta(days=10))
        runs(db, "together", status="failed")

        assert reconciler.find_retirements(db, now=NOW) == []

    def test_truncated_runs_are_not_evidence_of_absence(self, db):
        """The Anthropic pagination case: page one is not the catalogue.

        Before the ledger recorded pagination, a truncated read and a genuine
        deletion were indistinguishable.
        """
        enabled(db, "anthropic", "maybe-gone")
        catalogue(db, "anthropic", "maybe-gone", ago=timedelta(days=10))
        runs(db, "anthropic", complete=False)

        assert reconciler.find_retirements(db, now=NOW) == []

    def test_two_complete_runs_are_not_enough(self, db):
        """Daily polling with jitter makes three days as few as two observations."""
        enabled(db, "together", "gone")
        catalogue(db, "together", "gone", ago=timedelta(days=10))
        runs(db, "together", count=2)

        assert reconciler.find_retirements(db, now=NOW) == []

    def test_a_provider_with_no_discovery_authority_is_never_retired(self, db):
        """Bedrock and Vertex are not read at all. Silence there is not absence."""
        enabled(db, "bedrock", "us.meta.llama3-3-70b-instruct-v1:0")
        enabled(db, "vertex", "gemini")
        runs(db, "bedrock")

        assert reconciler.find_retirements(db, now=NOW) == []


class TestApplication:
    def test_dry_run_changes_nothing(self, db):
        enabled(db, "together", "gone")
        catalogue(db, "together", "gone", ago=timedelta(days=10))
        runs(db, "together")

        reconciler.retire(db, now=NOW, dry_run=True)

        assert db.models.find_one({"model_id": "gone"})["enabled"] is True

    def test_retirement_records_a_class_and_is_reversible(self, db):
        enabled(db, "together", "gone")
        catalogue(db, "together", "gone", ago=timedelta(days=10))
        runs(db, "together")

        reconciler.retire(db, now=NOW, dry_run=False)

        doc = db.models.find_one({"model_id": "gone"})
        assert doc["deprecated"] is True
        assert doc["disabled_class"] == "provider_retired"

        batch = db.bench_mutation_batches.find_one()
        mutations.revert(db, batch_id=batch["_id"], now=NOW)
        assert db.models.find_one({"model_id": "gone"})["enabled"] is True

    def test_a_discovery_regression_cannot_empty_a_provider(self, db):
        """The run reports success and every model vanishes — retire nothing."""
        for i in range(40):
            enabled(db, "together", f"m{i}")
        runs(db, "together")

        with pytest.raises(mutations.MutationRefused):
            reconciler.retire(db, now=NOW, dry_run=False)

        assert db.models.count_documents({"enabled": True}) == 40


class TestDisplayNameUnification:
    """One character split a three-provider line in production.

    The pipeline groups by (providerCanonical, display_name), so
    claude-haiku-4.5 and claude-haiku-4-5 were two lines for one model at three
    providers. Derived identity does not have to replace the mapping code to fix
    that — it only has to say which endpoints should share a name.
    """

    def _identify(self, db, provider, model_id, key, display):
        db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True, "display_name": display})
        db.bench_model_identity.insert_one(
            {
                "provider": provider,
                "model_id": model_id,
                "canonical_key": key,
                "effective_from": NOW,
            }
        )

    def test_the_outlier_moves_to_the_majority_name(self, db):
        self._identify(db, "anthropic", "claude-haiku-4-5", "anthropic-claude-haiku-4.5", "claude-haiku-4.5")
        self._identify(db, "bedrock", "us.anthropic.claude-haiku", "anthropic-claude-haiku-4.5", "claude-haiku-4.5")
        self._identify(db, "deepinfra", "anthropic/claude-haiku", "anthropic-claude-haiku-4.5", "claude-haiku-4-5")

        changes = reconciler.unify_display_names(db, now=NOW, dry_run=True)

        assert len(changes) == 1
        assert changes[0]["provider"] == "deepinfra"
        assert changes[0]["to"] == "claude-haiku-4.5"

    def test_an_already_consistent_group_is_untouched(self, db):
        self._identify(db, "groq", "a", "meta-llama-3.3-70b-instruct", "llama-3.3-70b")
        self._identify(db, "together", "b", "meta-llama-3.3-70b-instruct", "llama-3.3-70b")

        assert reconciler.unify_display_names(db, now=NOW, dry_run=True) == []

    def test_endpoints_in_different_groups_are_never_unified(self, db):
        """Haiku must not take Sonnet's name because both are Claude."""
        self._identify(db, "anthropic", "haiku", "anthropic-claude-haiku-4.5", "claude-haiku-4.5")
        self._identify(db, "anthropic", "sonnet", "anthropic-claude-sonnet-4.5", "claude-sonnet-4.5")

        assert reconciler.unify_display_names(db, now=NOW, dry_run=True) == []

    def test_applying_is_reversible(self, db):
        self._identify(db, "anthropic", "a", "k", "claude-haiku-4.5")
        self._identify(db, "bedrock", "b", "k", "claude-haiku-4.5")
        self._identify(db, "deepinfra", "c", "k", "claude-haiku-4-5")

        reconciler.unify_display_names(db, now=NOW, dry_run=False)
        assert db.models.find_one({"model_id": "c"})["display_name"] == "claude-haiku-4.5"

        batch = db.bench_mutation_batches.find_one()
        mutations.revert(db, batch_id=batch["_id"], now=NOW)
        assert db.models.find_one({"model_id": "c"})["display_name"] == "claude-haiku-4-5"

    def test_an_unresolved_endpoint_is_never_renamed(self, db):
        db.models.insert_one({"provider": "groq", "model_id": "x", "enabled": True, "display_name": "mystery"})
        db.bench_model_identity.insert_one(
            {"provider": "groq", "model_id": "x", "canonical_key": None, "effective_from": NOW}
        )
        assert reconciler.unify_display_names(db, now=NOW, dry_run=True) == []

    def test_two_endpoints_at_one_provider_are_never_unified(self, db):
        """DeepSeek-V3.1 and V3.1-Terminus are separate checkpoints at DeepInfra.

        They share a derived key because checkpoint is not part of it. Merging
        them would hide one behind the other, and buys nothing — the chart
        compares providers, and there is only one provider here.
        """
        self._identify(db, "deepinfra", "deepseek-ai/DeepSeek-V3.1", "deepseek-v3.1", "DeepSeek-V3.1")
        self._identify(db, "deepinfra", "deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek-v3.1", "DeepSeek-V3.1-Terminus")

        assert reconciler.unify_display_names(db, now=NOW, dry_run=True) == []

    def test_a_rename_that_collides_at_the_same_provider_is_skipped(self, db):
        """DeepInfra serves both Llama-3.3-70B-Instruct and its Turbo build.

        Renaming the Turbo one onto the other does not add a provider to the
        line — it merges two DeepInfra deployments into one row and averages
        their throughput, hiding one behind the other.
        """
        self._identify(db, "groq", "llama-3.3-70b-versatile", "meta-llama-3.3-70b", "llama-3.3-70b")
        self._identify(db, "deepinfra", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama-3.3-70b", "llama-3.3-70b")
        self._identify(
            db,
            "deepinfra",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama-3.3-70b",
            "Llama-3.3-70B-Instruct-Turbo",
        )

        assert reconciler.unify_display_names(db, now=NOW, dry_run=True) == []

    def test_a_rename_with_no_collision_still_applies(self, db):
        self._identify(db, "anthropic", "claude-haiku", "k", "claude-haiku-4.5")
        self._identify(db, "bedrock", "us.anthropic.claude-haiku", "k", "claude-haiku-4.5")
        self._identify(db, "deepinfra", "anthropic/claude-haiku", "k", "claude-haiku-4-5")

        changes = reconciler.unify_display_names(db, now=NOW, dry_run=True)
        assert [c["provider"] for c in changes] == ["deepinfra"]
