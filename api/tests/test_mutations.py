from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import mutations

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"mut-{request.node.name}"]


def model(db, provider, model_id, **fields):
    db.models.insert_one({"provider": provider, "model_id": model_id, **fields})


def batch(db, **kwargs):
    return mutations.MutationBatch(db=db, reason="test", actor="test-agent", **kwargs)


class TestReversibility:
    def test_a_change_records_what_it_overwrote(self, db):
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", enabled=False, disabled_reason="stale")
        b.apply(now=NOW)

        record = db.bench_mutation_batches.find_one({"_id": b.batch_id})
        change = record["changes"][0]
        assert change["before"]["enabled"] is True
        assert change["after"]["enabled"] is False

    def test_revert_restores_the_prior_value(self, db):
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", enabled=False)
        b.apply(now=NOW)
        assert db.models.find_one({"model_id": "m1"})["enabled"] is False

        mutations.revert(db, batch_id=b.batch_id, now=NOW)

        assert db.models.find_one({"model_id": "m1"})["enabled"] is True

    def test_revert_unsets_fields_that_did_not_exist_before(self, db):
        """Reverting should leave the document shaped as it was, not merely equivalent."""
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", disabled_reason="stale")
        b.apply(now=NOW)
        assert "disabled_reason" in db.models.find_one({"model_id": "m1"})

        mutations.revert(db, batch_id=b.batch_id, now=NOW)

        assert "disabled_reason" not in db.models.find_one({"model_id": "m1"})

    def test_a_batch_cannot_be_reverted_twice(self, db):
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", enabled=False)
        b.apply(now=NOW)
        mutations.revert(db, batch_id=b.batch_id, now=NOW)

        with pytest.raises(mutations.MutationRefused):
            mutations.revert(db, batch_id=b.batch_id, now=NOW)

    def test_reverting_a_bulk_demotion_restores_every_model(self, db):
        """The scenario this exists for: an agent demotes a chunk of the catalogue."""
        for i in range(20):
            model(db, "deepinfra", f"m{i}", enabled=True)
        b = batch(db)
        for i in range(20):
            b.set_model_fields(provider="deepinfra", model_id=f"m{i}", enabled=False)
        b.apply(now=NOW)
        assert db.models.count_documents({"enabled": True}) == 0

        mutations.revert(db, batch_id=b.batch_id, now=NOW)

        assert db.models.count_documents({"enabled": True}) == 20


class TestBlastRadius:
    def test_an_over_limit_batch_applies_nothing(self, db):
        """Half a migration is worse than none — it is the state nobody designed for."""
        for i in range(60):
            model(db, "deepinfra", f"m{i}", enabled=True)
        b = batch(db)
        for i in range(60):
            b.set_model_fields(provider="deepinfra", model_id=f"m{i}", enabled=False)

        with pytest.raises(mutations.MutationRefused):
            b.apply(now=NOW)

        assert db.models.count_documents({"enabled": True}) == 60
        assert db.bench_mutation_batches.count_documents({}) == 0

    def test_a_single_provider_cannot_be_emptied_under_the_global_cap(self, db):
        """A whole provider going dark is the shape of the July decay."""
        for i in range(30):
            model(db, "together", f"m{i}", enabled=True)
        b = batch(db)
        for i in range(30):
            b.set_model_fields(provider="together", model_id=f"m{i}", enabled=False)

        with pytest.raises(mutations.MutationRefused, match="per-provider cap"):
            b.apply(now=NOW)

        assert db.models.count_documents({"enabled": True}) == 30

    def test_the_kill_switch_stops_mutations(self, db, monkeypatch):
        monkeypatch.setenv(mutations.KILL_SWITCH_ENV, "1")
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", enabled=False)

        with pytest.raises(mutations.MutationRefused):
            b.apply(now=NOW)

        assert db.models.find_one({"model_id": "m1"})["enabled"] is True

    def test_a_normal_sized_change_is_allowed(self, db):
        for i in range(5):
            model(db, "groq", f"m{i}", enabled=True)
        b = batch(db)
        for i in range(5):
            b.set_model_fields(provider="groq", model_id=f"m{i}", enabled=False)

        assert b.apply(now=NOW)["applied"] == 5


class TestAudit:
    def test_the_batch_records_who_and_why(self, db):
        model(db, "groq", "m1", enabled=True)
        b = mutations.MutationBatch(db=db, reason="stale for 7d", actor="reconciler")
        b.set_model_fields(provider="groq", model_id="m1", enabled=False)
        b.apply(now=NOW)

        record = db.bench_mutation_batches.find_one({"_id": b.batch_id})
        assert record["reason"] == "stale for 7d"
        assert record["actor"] == "reconciler"
        assert record["reverted_at"] is None

    def test_each_touched_model_points_back_at_its_batch(self, db):
        model(db, "groq", "m1", enabled=True)
        b = batch(db)
        b.set_model_fields(provider="groq", model_id="m1", enabled=False)
        b.apply(now=NOW)

        assert db.models.find_one({"model_id": "m1"})["mutation_batch_id"] == b.batch_id
