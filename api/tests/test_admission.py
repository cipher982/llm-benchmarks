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


def catalogue(db, provider, model_id, name=None, *, last_seen_at=None):
    row = {"provider": provider, "model_id": model_id, "name": name}
    if last_seen_at is not None:
        row["last_seen_at"] = last_seen_at
    db.provider_catalog.insert_one(row)


def probe_success(db, provider, model_id, *, ago):
    db.metrics_cloud_probe.insert_one(
        {"provider": provider, "model_name": model_id, "run_ts": NOW - ago, "sample_role": "probe"}
    )


def candidate(db, provider, model_id, *, started=timedelta(days=7)):
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

    def test_transport_exclusions_do_not_block_direct_provider_candidates(self, db, monkeypatch):
        monkeypatch.setenv("BENCHMARK_OPENROUTER_EXCLUDED_MODEL_PREFIXES", "anthropic/")
        catalogue(db, "openrouter", "anthropic/new-release")
        catalogue(db, "bedrock", "anthropic/new-release")
        catalogue(db, "openrouter", "qwen/new-release")
        candidates = admission.find_candidates(db)
        assert {(row["provider"], row["model_id"]) for row in candidates} == {
            ("bedrock", "anthropic/new-release"),
            ("openrouter", "qwen/new-release"),
        }


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
        assert job["max_attempts"] == 2
        assert "terminal_recovery_active" not in job

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
    def test_shadow_profile_successes_do_not_promote_a_candidate(self, db):
        candidate(db, "openrouter", "qwen/qwen3-thinking")
        for ago in (timedelta(hours=6), timedelta(hours=3)):
            db.metrics_cloud_probe.insert_one(
                {
                    "provider": "openrouter",
                    "model_name": "qwen/qwen3-thinking",
                    "benchmark_profile_id": "cloud-reasoning-v1",
                    "run_ts": NOW - ago,
                    "sample_role": "shadow",
                }
            )

        promoted, _ = admission.evaluate_candidates(db, now=NOW)

        assert promoted == []
        assert db.models.find_one({"model_id": "qwen/qwen3-thinking"})["enabled"] is False

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
                    "started_at": NOW - timedelta(minutes=30),
                    "updated_at": NOW - timedelta(minutes=30),
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


class TestRejectedAdmissionRenewal:
    def _rejected(
        self,
        db,
        provider,
        model_id,
        *,
        recheck_after=NOW - timedelta(hours=1),
        disabled_class="hard_model",
        recovery_of_published_model=True,
    ):
        model = {
            "provider": provider,
            "model_id": model_id,
            "enabled": False,
            "status": admission.REJECTED_STATUS,
            "disabled_class": disabled_class,
            "disabled_reason": "old rejection",
            "disabled_at": NOW - timedelta(days=31),
            "recheck_after": recheck_after,
            "admission_started_at": NOW - timedelta(days=40),
        }
        if recovery_of_published_model is not None:
            model["recovery_of_published_model"] = recovery_of_published_model
        db.models.insert_one(model)

    def _dead_probe(self, db, provider, model_id, n):
        for index in range(n):
            db.bench_jobs.insert_one(
                {
                    "_id": f"old-probe:{provider}:{model_id}:{index}",
                    "provider": provider,
                    "model_id": model_id,
                    "job_kind": "probe",
                    "status": "dead_letter",
                    "last_attempt_error_kind": "hard_model",
                    "started_at": NOW - timedelta(minutes=30),
                    "updated_at": NOW - timedelta(minutes=30),
                }
            )

    def test_due_rejection_with_recent_catalogue_starts_a_new_probe_epoch(self, db):
        self._rejected(db, "groq", "returned")
        catalogue(db, "groq", "returned", last_seen_at=NOW - timedelta(days=1))

        renewed = admission.renew_expired_rejections(db, now=NOW)

        assert renewed == ["groq/returned"]
        doc = db.models.find_one({"model_id": "returned"})
        assert doc["status"] == admission.CANDIDATE_STATUS
        assert admission._as_utc(doc["admission_started_at"]) == NOW
        assert doc["admission_recheck_count"] == 1
        assert doc["recheck_after"] is None
        assert admission.enqueue_probes(db, now=NOW) == ["groq/returned"]
        job = db.bench_jobs.find_one({"model_id": "returned"})
        assert job["max_attempts"] == 1
        assert job["terminal_recovery_active"] is True

    def test_one_fresh_success_restores_a_published_model_without_another_probe(self, db):
        self._rejected(db, "groq", "returned")
        catalogue(db, "groq", "returned", last_seen_at=NOW)
        admission.renew_expired_rejections(db, now=NOW)
        assert admission.enqueue_probes(db, now=NOW) == ["groq/returned"]
        job = db.bench_jobs.find_one({"model_id": "returned"})
        assert job["max_attempts"] == 1
        assert job["terminal_recovery_active"] is True
        success_at = NOW + timedelta(minutes=1)
        db.bench_jobs.update_one({"_id": job["_id"]}, {"$set": {"status": "running"}})
        assert queue.mark_success(db, job_id=job["_id"], now=success_at)

        assert admission.enqueue_probes(db, now=success_at + timedelta(minutes=1)) == []
        promoted, rejected = admission.evaluate_candidates(
            db,
            now=success_at + timedelta(minutes=1),
        )

        assert promoted == ["groq/returned"]
        assert rejected == []
        assert db.models.find_one({"model_id": "returned"})["enabled"] is True

    def test_one_failed_published_recovery_probe_rejects_for_another_30_days(self, db):
        self._rejected(db, "groq", "returned")
        catalogue(db, "groq", "returned", last_seen_at=NOW)
        admission.renew_expired_rejections(db, now=NOW)
        admission.enqueue_probes(db, now=NOW)
        job = db.bench_jobs.find_one({"model_id": "returned"})
        failure_at = NOW + timedelta(minutes=30)
        db.bench_jobs.update_one(
            {"_id": job["_id"]},
            {"$set": {"status": "running", "attempt": 1}},
        )
        job = db.bench_jobs.find_one({"_id": job["_id"]})
        assert (
            queue.mark_failure(
                db,
                job=job,
                error_kind="timeout",
                error_message="recovery probe timed out",
                now=failure_at,
            )
            == "dead_letter"
        )
        assert admission.enqueue_probes(db, now=failure_at) == []
        decision_at = failure_at + timedelta(minutes=1)

        promoted, rejected = admission.evaluate_candidates(db, now=decision_at)

        assert promoted == []
        assert [subject for subject, _ in rejected] == ["groq/returned"]
        model = db.models.find_one({"model_id": "returned"})
        assert model["disabled_class"] == "timeout"
        assert model["recovery_of_published_model"] is True
        assert admission._as_utc(model["recheck_after"]) == decision_at + timedelta(days=30)

    def test_old_successes_and_definitive_failures_do_not_decide_the_new_epoch(self, db):
        self._rejected(db, "groq", "returned")
        catalogue(db, "groq", "returned", last_seen_at=NOW - timedelta(days=1))
        probe_success(db, "groq", "returned", ago=timedelta(hours=6))
        probe_success(db, "groq", "returned", ago=timedelta(hours=3))
        self._dead_probe(db, "groq", "returned", n=2)

        admission.renew_expired_rejections(db, now=NOW)
        enqueued = admission.enqueue_probes(db, now=NOW)
        promoted, rejected = admission.evaluate_candidates(db, now=NOW)

        assert enqueued == ["groq/returned"]
        assert (promoted, rejected) == ([], [])

    def test_only_due_recent_nonduplicate_nonexcluded_rows_are_renewed(self, db):
        self._rejected(db, "groq", "stale")
        catalogue(db, "groq", "stale", last_seen_at=NOW - timedelta(days=8))
        self._rejected(db, "groq", "duplicate", disabled_class="duplicate_spelling")
        catalogue(db, "groq", "duplicate", last_seen_at=NOW)
        self._rejected(db, "bedrock", "excluded")
        catalogue(db, "bedrock", "excluded", last_seen_at=NOW)
        self._rejected(db, "groq", "not-due", recheck_after=NOW + timedelta(seconds=1))
        catalogue(db, "groq", "not-due", last_seen_at=NOW)
        self._rejected(db, "groq", "unmarked", recovery_of_published_model=None)
        catalogue(db, "groq", "unmarked", last_seen_at=NOW)
        self._rejected(db, "groq", "not-published", recovery_of_published_model=False)
        catalogue(db, "groq", "not-published", last_seen_at=NOW)

        assert admission.renew_expired_rejections(db, now=NOW) == []

    def test_hard_capability_and_auth_rejections_can_be_renewed(self, db):
        self._rejected(db, "groq", "capability", disabled_class="hard_capability")
        self._rejected(db, "groq", "auth", disabled_class="auth")
        catalogue(db, "groq", "capability", last_seen_at=NOW)
        catalogue(db, "groq", "auth", last_seen_at=NOW)

        assert set(admission.renew_expired_rejections(db, now=NOW)) == {
            "groq/capability",
            "groq/auth",
        }

    def test_renewals_are_oldest_first_and_mutation_batched(self, db):
        for index in range(4):
            model_id = f"model-{index}"
            self._rejected(
                db,
                "groq",
                model_id,
                recheck_after=NOW - timedelta(days=40 - index),
            )
            catalogue(db, "groq", model_id, last_seen_at=NOW)

        renewed = admission.renew_expired_rejections(db, now=NOW, limit=2)

        assert renewed == ["groq/model-0", "groq/model-1"]
        batch = db.bench_mutation_batches.find_one()
        assert batch["reason"] == "admission rejection recheck"
        assert len(batch["changes"]) == 2

    def test_catalogue_recheck_freshness_is_limited_to_48_hours(self, db):
        self._rejected(db, "groq", "fresh")
        catalogue(db, "groq", "fresh", last_seen_at=NOW - timedelta(hours=47))
        self._rejected(db, "groq", "stale")
        catalogue(db, "groq", "stale", last_seen_at=NOW - timedelta(hours=49))

        assert admission.renew_expired_rejections(db, now=NOW) == ["groq/fresh"]

    def test_admission_pass_shares_registration_cap_with_rejections(self, db, monkeypatch):
        monkeypatch.setattr(admission, "MAX_NEW_CANDIDATES_PER_RUN", 1)
        self._rejected(db, "groq", "renewed")
        catalogue(db, "groq", "renewed", last_seen_at=NOW)
        catalogue(db, "groq", "new")

        report = admission.run_admission_pass(db, now=NOW)

        assert report.registered == []
        assert report.renewed == ["groq/renewed"]
        assert db.models.count_documents({"status": admission.CANDIDATE_STATUS}) == 1
        assert db.bench_jobs.count_documents({"job_kind": "probe"}) == 1
        assert db.models.find_one({"model_id": "renewed"})["status"] == admission.CANDIDATE_STATUS
        assert db.models.find_one({"model_id": "new"}) is None
