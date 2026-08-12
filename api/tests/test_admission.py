from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import admission
from llm_bench.scheduler import queue

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(request):
    return mongomock.MongoClient()[f"adm-{request.node.name}"]


def catalogue(db, provider, model_id, name=None):
    db.provider_catalog.insert_one({"provider": provider, "model_id": model_id, "name": name})


def probe_success(db, provider, model_id, *, ago):
    db.metrics_cloud_probe.insert_one(
        {"provider": provider, "model_name": model_id, "run_ts": NOW - ago, "sample_role": "probe"}
    )


def candidate(db, provider, model_id, *, started=timedelta(hours=1)):
    db.models.insert_one(
        {
            "provider": provider,
            "model_id": model_id,
            "enabled": False,
            "status": admission.CANDIDATE_STATUS,
            "admission_started_at": NOW - started,
        }
    )


class TestCandidateSelection:
    def test_models_already_in_the_catalogue_are_not_candidates(self, db):
        catalogue(db, "groq", "known")
        db.models.insert_one({"provider": "groq", "model_id": "known", "enabled": True})
        assert admission.find_candidates(db) == []

    def test_a_rejected_model_is_not_re_probed(self, db):
        """Otherwise every pass pays to re-learn the same no."""
        catalogue(db, "groq", "dead")
        db.models.insert_one(
            {"provider": "groq", "model_id": "dead", "enabled": False, "status": admission.REJECTED_STATUS}
        )
        assert admission.find_candidates(db) == []

    def test_models_many_providers_serve_are_probed_first(self, db):
        """Coverage, not novelty, is what makes a chart line useful.

        A model measured at one provider and available at four is where the site
        is least useful, so it earns the probe budget ahead of something only
        one provider serves.
        """
        for provider in ("groq", "together", "deepinfra", "fireworks"):
            catalogue(db, provider, f"{provider}/popular-model")
        catalogue(db, "deepinfra", "deepinfra/obscure-model")

        first = admission.find_candidates(db, limit=1)[0]
        assert "popular" in first["model_id"]

    def test_new_candidates_are_capped(self, db):
        for i in range(50):
            catalogue(db, "deepinfra", f"m{i}")
        assert len(admission.find_candidates(db, limit=5)) == 5


class TestRegistration:
    def test_a_candidate_is_registered_disabled(self, db):
        """Nothing may reach the site before it has evidence."""
        catalogue(db, "groq", "new-one", name="New One")
        admission.register_candidates(db, now=NOW)
        doc = db.models.find_one({"model_id": "new-one"})
        assert doc["enabled"] is False
        assert doc["status"] == admission.CANDIDATE_STATUS

    def test_registering_twice_does_not_reset_the_clock(self, db):
        catalogue(db, "groq", "new-one")
        admission.register_candidates(db, now=NOW - timedelta(days=2))
        admission.register_candidates(db, now=NOW)
        doc = db.models.find_one({"model_id": "new-one"})
        # mongomock returns naive datetimes, so compare on the instant.
        assert admission._as_utc(doc["admission_started_at"]) == NOW - timedelta(days=2)


class TestProbeEnqueue:
    def test_a_probe_job_is_marked_as_a_probe(self, db):
        """The sample role is what keeps probe results out of the site."""
        candidate(db, "groq", "cand")
        admission.enqueue_probes(db, now=NOW)
        job = db.bench_jobs.find_one({"model_id": "cand"})
        assert job["sample_role"] == "probe"
        assert job["job_kind"] == "probe"

    def test_a_candidate_with_pending_work_is_not_queued_again(self, db):
        candidate(db, "groq", "cand")
        admission.enqueue_probes(db, now=NOW)
        admission.enqueue_probes(db, now=NOW + timedelta(hours=5))
        assert db.bench_jobs.count_documents({"model_id": "cand"}) == 1

    def test_samples_are_spaced_out(self, db):
        """Back-to-back successes mostly prove the endpoint was warm."""
        candidate(db, "groq", "cand")
        probe_success(db, "groq", "cand", ago=timedelta(minutes=10))
        assert admission.enqueue_probes(db, now=NOW) == []

    def test_a_candidate_with_enough_evidence_stops_costing_money(self, db):
        candidate(db, "groq", "cand")
        probe_success(db, "groq", "cand", ago=timedelta(hours=6))
        probe_success(db, "groq", "cand", ago=timedelta(hours=3))
        assert admission.enqueue_probes(db, now=NOW) == []

    def test_probes_per_run_are_capped(self, db):
        for i in range(40):
            candidate(db, "deepinfra", f"c{i}")
        assert len(admission.enqueue_probes(db, now=NOW, limit=7)) == 7


class TestPromotion:
    def test_two_spaced_successes_promote_as_provisional(self, db):
        candidate(db, "groq", "cand")
        probe_success(db, "groq", "cand", ago=timedelta(hours=6))
        probe_success(db, "groq", "cand", ago=timedelta(hours=3))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert promoted == ["groq/cand"]
        doc = db.models.find_one({"model_id": "cand"})
        assert doc["enabled"] is True
        assert doc["status"] == admission.PROMOTED_STATUS
        assert doc["admission_evidence"]["windows"] == 2

    def test_two_successes_in_one_burst_do_not_promote(self, db):
        """One warm moment is one observation, however many calls it contains."""
        candidate(db, "groq", "cand")
        probe_success(db, "groq", "cand", ago=timedelta(minutes=20))
        probe_success(db, "groq", "cand", ago=timedelta(minutes=18))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert promoted == []
        assert db.models.find_one({"model_id": "cand"})["enabled"] is False

    def test_a_candidate_that_never_succeeds_is_rejected_with_a_reason(self, db):
        candidate(db, "groq", "image-model", started=timedelta(days=5))

        _, rejected = admission.evaluate_candidates(db, now=NOW)

        assert [s for s, _ in rejected] == ["groq/image-model"]
        doc = db.models.find_one({"model_id": "image-model"})
        assert doc["status"] == admission.REJECTED_STATUS
        assert doc["disabled_class"] == "hard_model"

    def test_a_candidate_still_within_its_window_is_left_alone(self, db):
        candidate(db, "groq", "slow", started=timedelta(hours=6))
        promoted, rejected = admission.evaluate_candidates(db, now=NOW)
        assert (promoted, rejected) == ([], [])


class TestBenchmarkSurface:
    def test_a_text_returning_model_is_admitted_whatever_kind_it_is(self, db):
        """Guard, moderation and router models are in scope.

        How fast a guard model responds is a real question, and its latency is
        useful. The probe measures whether text comes back at a rate, not what
        the model is for.
        """
        candidate(db, "groq", "meta-llama/Llama-Guard-4-12B")
        probe_success(db, "groq", "meta-llama/Llama-Guard-4-12B", ago=timedelta(hours=6))
        probe_success(db, "groq", "meta-llama/Llama-Guard-4-12B", ago=timedelta(hours=3))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert promoted == ["groq/meta-llama/Llama-Guard-4-12B"]


class TestEligibilitySeam:
    def test_a_probe_may_run_against_a_model_that_is_not_enabled(self, db):
        """Otherwise probe-before-promote is impossible by construction."""
        candidate(db, "groq", "cand")
        assert queue.is_model_eligible(db, provider="groq", model_id="cand", sample_role="probe") is True

    def test_published_work_still_requires_an_enabled_model(self, db):
        candidate(db, "groq", "cand")
        assert queue.is_model_eligible(db, provider="groq", model_id="cand", sample_role="published") is False

    def test_a_rejected_model_gets_no_further_probes(self, db):
        db.models.insert_one(
            {"provider": "groq", "model_id": "dead", "enabled": False, "status": admission.REJECTED_STATUS}
        )
        assert queue.is_model_eligible(db, provider="groq", model_id="dead", sample_role="probe") is False


class TestAdmissionIsReversibleAndBounded:
    def test_a_promotion_can_be_inverted(self, db):
        from llm_bench.ops import mutations

        candidate(db, "groq", "cand")
        probe_success(db, "groq", "cand", ago=timedelta(hours=6))
        probe_success(db, "groq", "cand", ago=timedelta(hours=3))
        admission.evaluate_candidates(db, now=NOW)
        assert db.models.find_one({"model_id": "cand"})["enabled"] is True

        batch = db.bench_mutation_batches.find_one()
        mutations.revert(db, batch_id=batch["_id"], now=NOW)

        doc = db.models.find_one({"model_id": "cand"})
        assert doc["enabled"] is False
        assert doc["status"] == admission.CANDIDATE_STATUS

    def test_a_rejection_carries_its_own_expiry(self, db):
        """A no is evidence about today, not a permanent verdict.

        Providers add capacity and grant entitlements. The ratchet that decayed
        coverage to 11.7% was exactly a no with no way back.
        """
        candidate(db, "groq", "gone", started=timedelta(days=5))
        admission.evaluate_candidates(db, now=NOW)

        doc = db.models.find_one({"model_id": "gone"})
        assert admission._as_utc(doc["recheck_after"]) > NOW

    def test_a_mass_rejection_is_bounded_per_pass(self, db):
        """A probe regression must not empty the candidate pool in one go.

        It must also not jam. This previously raised and applied nothing, which
        looked like safety and was a deadlock: in production 87 due decisions
        against a cap of 40 meant no model was ever promoted again, and every
        new candidate made the batch larger.
        """
        from llm_bench.ops import mutations

        for i in range(60):
            candidate(db, "deepinfra", f"c{i}", started=timedelta(days=5))

        admission.evaluate_candidates(db, now=NOW)

        remaining = db.models.count_documents({"status": admission.CANDIDATE_STATUS})
        decided = 60 - remaining
        assert 0 < decided <= mutations.MAX_CHANGES_PER_PROVIDER
        assert remaining > 0, "one pass must not clear the whole pool"

    def test_the_pool_drains_over_successive_passes(self, db):
        """The cap bounds a pass, so repeated passes finish the backlog."""
        for i in range(60):
            candidate(db, "deepinfra", f"c{i}", started=timedelta(days=5))

        for _ in range(10):
            admission.evaluate_candidates(db, now=NOW)
            if not db.models.count_documents({"status": admission.CANDIDATE_STATUS}):
                break

        assert db.models.count_documents({"status": admission.CANDIDATE_STATUS}) == 0

    def test_a_promotion_is_not_crowded_out_by_pending_rejections(self, db):
        """A model that earned its place must not wait behind dead ones."""
        for i in range(50):
            candidate(db, "deepinfra", f"dead{i}", started=timedelta(days=5))
        candidate(db, "deepinfra", "earned")
        probe_success(db, "deepinfra", "earned", ago=timedelta(hours=6))
        probe_success(db, "deepinfra", "earned", ago=timedelta(hours=3))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert "deepinfra/earned" in promoted
        assert db.models.find_one({"model_id": "earned"})["enabled"] is True


class TestDefinitiveFailures:
    def _dead_probe(self, db, provider, model_id, kind="hard_model", n=1):
        for i in range(n):
            db.bench_jobs.insert_one(
                {
                    "_id": f"probe:{provider}:{model_id}:{i}",
                    "provider": provider,
                    "model_id": model_id,
                    "job_kind": "probe",
                    "status": "dead_letter",
                    "last_attempt_error_kind": kind,
                }
            )

    def test_a_404_rejects_without_waiting_out_the_deadline(self, db):
        """The provider answered. Re-probing for three days re-learns the same no.

        This is the FLUX case: image endpoints 404 on chat completions, and
        without this they would be probed every two hours for three days.
        """
        candidate(db, "deepinfra", "black-forest-labs/FLUX-1-dev", started=timedelta(hours=1))
        self._dead_probe(db, "deepinfra", "black-forest-labs/FLUX-1-dev", n=2)

        _, rejected = admission.evaluate_candidates(db, now=NOW)

        assert [s for s, _ in rejected] == ["deepinfra/black-forest-labs/FLUX-1-dev"]

    def test_one_definitive_failure_is_not_enough(self, db):
        candidate(db, "groq", "maybe", started=timedelta(hours=1))
        self._dead_probe(db, "groq", "maybe", n=1)
        promoted, rejected = admission.evaluate_candidates(db, now=NOW)
        assert (promoted, rejected) == ([], [])

    def test_timeouts_do_not_count_as_an_answer(self, db):
        """A timeout means we did not find out, so the candidate keeps its window."""
        candidate(db, "together", "slow", started=timedelta(hours=1))
        self._dead_probe(db, "together", "slow", kind="timeout", n=4)
        promoted, rejected = admission.evaluate_candidates(db, now=NOW)
        assert (promoted, rejected) == ([], [])


class TestCaseDuplicates:
    def test_a_case_variant_of_an_enabled_model_is_rejected_not_promoted(self, db):
        """Promoting it writes a second enabled row for one endpoint.

        The case-insensitive unique index refuses that, and the refusal aborted
        the entire batch — so one duplicate blocked every other promotion in the
        pass. Providers spell their own IDs inconsistently across surfaces, so
        this is not rare: 18 candidates were in this state in production.
        """
        db.models.insert_one({"provider": "deepinfra", "model_id": "minimaxai/minimax-m3", "enabled": True})
        candidate(db, "deepinfra", "MiniMaxAI/MiniMax-M3")
        probe_success(db, "deepinfra", "MiniMaxAI/MiniMax-M3", ago=timedelta(hours=6))
        probe_success(db, "deepinfra", "MiniMaxAI/MiniMax-M3", ago=timedelta(hours=3))

        promoted, rejected = admission.evaluate_candidates(db, now=NOW)

        assert promoted == []
        assert any("duplicate spelling" in reason for _, reason in rejected)
        doc = db.models.find_one({"model_id": "MiniMaxAI/MiniMax-M3"})
        assert doc["enabled"] is False
        assert doc["disabled_class"] == "duplicate_spelling"

    def test_one_duplicate_does_not_block_other_promotions(self, db):
        db.models.insert_one({"provider": "deepinfra", "model_id": "minimaxai/minimax-m3", "enabled": True})
        candidate(db, "deepinfra", "MiniMaxAI/MiniMax-M3")
        candidate(db, "groq", "genuinely-new")
        probe_success(db, "groq", "genuinely-new", ago=timedelta(hours=6))
        probe_success(db, "groq", "genuinely-new", ago=timedelta(hours=3))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert "groq/genuinely-new" in promoted

    def test_the_exact_same_spelling_is_not_treated_as_a_duplicate(self, db):
        """Only a different spelling of the same endpoint is a duplicate."""
        candidate(db, "groq", "same")
        db.models.update_one({"model_id": "same"}, {"$set": {"enabled": False}})
        probe_success(db, "groq", "same", ago=timedelta(hours=6))
        probe_success(db, "groq", "same", ago=timedelta(hours=3))

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert "groq/same" in promoted

    def test_excluded_provider_candidates_are_not_probed(self, db):
        """Probes for a provider with no worker here would sit queued forever
        (bedrock is measured by its dedicated runner), tripping the queue
        invariants. Admission must not enqueue them."""
        candidate(db, "bedrock", "nvidia.nemotron-nano-12b-v2")
        candidate(db, "groq", "qwen/qwen3.6-27b")
        admission.enqueue_probes(db, now=NOW)
        assert db.bench_jobs.count_documents({"provider": "bedrock"}) == 0
        assert db.bench_jobs.count_documents({"provider": "groq"}) == 1
