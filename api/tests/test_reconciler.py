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
