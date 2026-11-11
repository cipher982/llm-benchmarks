"""Fixes from the 2026-11-11 model call audit (docs/audits/2026-11-11-model-call-audit.md).

- reasoning ladder: ask each endpoint for the least reasoning it accepts
- budget: short and read to the end when there is nothing to think about
- route renewal: never pay to probe a pinned-endpoint cooldown copy
- classifier: never re-ask a verdict nothing new has happened to
- decode window fields on every OpenRouter row
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace

import mongomock
import pytest
from llm_bench.cloud.providers import openrouter
from llm_bench.ops.route_renewal import renew_pass
from llm_bench.scheduler import reasoning
from llm_bench.scheduler import runner
from llm_bench.scheduler.routing import RouteDecision


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _endpoint(db, model_id="z/m", tag="prov", params=("reasoning", "max_tokens")):
    db.bench_endpoints.insert_one({"model_id": model_id, "endpoint_tag": tag, "supported_parameters": list(params)})


class TestPlan:
    def test_new_endpoint_starts_at_none_and_can_fall_back_to_the_default(self, db):
        _endpoint(db)
        plan = reasoning.plan_for(db, "z/m", "prov")
        assert plan.supported and plan.rungs == ("none", "minimal", "low", None)

    def test_endpoint_without_the_parameter_is_never_sent_one(self, db):
        """`require_parameters` would fail the pinned request outright."""
        _endpoint(db, params=("max_tokens",))
        assert reasoning.plan_for(db, "z/m", "prov").rungs == (None,)

    def test_stored_floor_is_where_the_next_run_starts(self, db):
        _endpoint(db)
        reasoning.record_floor(db, "z/m", "prov", "minimal", rejected=["none"])
        assert reasoning.plan_for(db, "z/m", "prov").rungs == ("minimal", "low", None)

    def test_model_level_rows_read_the_catalogue(self, db):
        db.openrouter_catalog.insert_one({"openrouter_id": "z/m", "supported_parameters": ["reasoning"]})
        assert reasoning.plan_for(db, "z/m", None).first == "none"

    @pytest.mark.parametrize(
        "message,rejected",
        [
            (
                "BadRequestError: Error code: 400 - {'error': {'message': 'Reasoning is mandatory for this endpoint'}}",
                True,
            ),
            ("BadRequestError: Error code: 400 - {'error': {'message': 'Provider returned error'}}", True),
            ("RateLimitError: Error code: 429", False),
            ("NotFoundError: Error code: 404 - No endpoints found", False),
        ],
    )
    def test_only_a_400_counts_as_a_refused_rung(self, message, rejected):
        assert reasoning.is_rejection(message) is rejected


class TestBudget:
    plan = reasoning.ReasoningPlan(supported=True, rungs=("none", None))

    def test_none_is_short_and_read_to_the_end(self):
        assert reasoning.budget_for(2048, "none", self.plan) == (reasoning.SHORT_BUDGET, True)

    def test_short_budget_never_raises_a_price_clamp(self):
        assert reasoning.budget_for(64, "none", self.plan) == (64, True)

    def test_a_reasoning_rung_keeps_headroom_but_is_capped(self):
        assert reasoning.budget_for(2048, "low", self.plan) == (reasoning.REASONING_BUDGET_CAP, False)

    def test_the_provider_default_keeps_the_old_budget(self):
        assert reasoning.budget_for(2048, None, self.plan) == (2048, False)

    def test_headroom_flag_overrides_the_short_budget(self):
        plan = reasoning.ReasoningPlan(supported=True, rungs=("none",), needs_headroom=True)
        assert reasoning.budget_for(2048, "none", plan) == (2048, False)


def _decision():
    return RouteDecision.direct("openrouter", "z/m", reason="test")


class TestLadder:
    @pytest.fixture
    def wired(self, db, monkeypatch):
        db.openrouter_catalog.insert_one({"openrouter_id": "z/m", "supported_parameters": ["reasoning"]})
        client = None

        class Client:
            def __getitem__(self, name):
                return db

            def close(self):
                pass

        monkeypatch.setattr(runner, "mongo_client", lambda: Client())
        monkeypatch.setattr(runner, "mongo_env", lambda: ("mongodb://x", "llm-bench"))
        calls = []
        return db, calls, client

    def _fake(self, calls, behaviour):
        def fake(decision, *, run_ts, run_config, max_tokens, timeout_seconds=None, extra_misc=None):
            effort = (extra_misc or {}).get("reasoning_effort")
            calls.append((effort, max_tokens, (extra_misc or {}).get("read_to_end")))
            outcome = behaviour(effort)
            if isinstance(outcome, runner.AttemptFailure):
                raise outcome
            return SimpleNamespace(), {"visible_output_tokens": 64}

        return fake

    def test_refused_rungs_are_skipped_and_the_working_one_is_stored(self, wired, monkeypatch):
        db, calls, _ = wired

        def behaviour(effort):
            if effort == "none":
                return runner.AttemptFailure("generate", "BadRequestError: Error code: 400 - Reasoning is mandatory")
            return None

        monkeypatch.setattr(runner, "_generate_and_validate", self._fake(calls, behaviour))
        _, metrics, max_tokens, effort = runner._attempt_with_reasoning_ladder(
            _decision(), run_ts="t", run_config={"query": "q"}, base_budget=2048
        )
        assert effort == "minimal" and max_tokens == reasoning.REASONING_BUDGET_CAP
        assert [c[0] for c in calls] == ["none", "minimal"]
        assert metrics["reasoning_effort"] == "minimal" and metrics["reasoning_rungs_rejected"] == ["none"]
        assert reasoning.plan_for(db, "z/m", None).first == "minimal"

    def test_every_rung_refused_falls_back_to_the_provider_default(self, wired, monkeypatch):
        db, calls, _ = wired

        def behaviour(effort):
            if effort is not None:
                return runner.AttemptFailure("generate", "BadRequestError: Error code: 400")
            return None

        monkeypatch.setattr(runner, "_generate_and_validate", self._fake(calls, behaviour))
        *_, effort = runner._attempt_with_reasoning_ladder(
            _decision(), run_ts="t", run_config={"query": "q"}, base_budget=2048
        )
        assert effort is None and [c[0] for c in calls] == ["none", "minimal", "low", None]
        assert reasoning.plan_for(db, "z/m", None).rungs == (None,)

    def test_a_non_rejection_failure_is_raised_without_climbing(self, wired, monkeypatch):
        _, calls, _ = wired
        monkeypatch.setattr(
            runner,
            "_generate_and_validate",
            self._fake(calls, lambda effort: runner.AttemptFailure("generate", "RateLimitError: Error code: 429")),
        )
        with pytest.raises(runner.AttemptFailure):
            runner._attempt_with_reasoning_ladder(_decision(), run_ts="t", run_config={"query": "q"}, base_budget=2048)
        assert len(calls) == 1

    def test_short_budget_exhausted_flags_headroom_for_next_time(self, wired, monkeypatch):
        db, calls, _ = wired
        monkeypatch.setattr(
            runner,
            "_generate_and_validate",
            self._fake(
                calls,
                lambda effort: runner.AttemptFailure(
                    "validate",
                    "visible output empty after token budget was exhausted; retry with a larger output budget",
                ),
            ),
        )
        with pytest.raises(runner.AttemptFailure):
            runner._attempt_with_reasoning_ladder(_decision(), run_ts="t", run_config={"query": "q"}, base_budget=2048)
        assert calls == [("none", reasoning.SHORT_BUDGET, True)]
        assert reasoning.plan_for(db, "z/m", None).needs_headroom is True
        assert reasoning.budget_for(2048, "none", reasoning.plan_for(db, "z/m", None)) == (2048, False)


class TestRequestOptions:
    def _config(self, **misc):
        return SimpleNamespace(misc=misc)

    def test_effort_is_sent_with_the_pin(self):
        options = openrouter._route_options(
            self._config(reasoning_effort="none", route_endpoint_tag="groq", route_policy="pinned-endpoint")
        )
        assert options["reasoning"] == {"effort": "none"}
        assert options["provider"]["only"] == ["groq"]

    def test_effort_is_sent_on_an_unpinned_row(self):
        assert openrouter._route_options(self._config(reasoning_effort="low")) == {"reasoning": {"effort": "low"}}

    def test_no_effort_sends_nothing(self):
        assert openrouter._route_options(self._config()) == {}


def _cooldown(_id, policy):
    return {
        "_id": _id,
        "state": "cooldown",
        "route_policy": policy,
        "source_provider": "openrouter",
        "source_model_id": "z/m",
        "transport_provider": "openrouter",
        "cooldown_until": 0,
        "route_health_failure_count": 1,
    }


class TestRouteRenewal:
    def test_pinned_endpoint_cooldowns_retire_without_a_call(self, db):
        db.bench_route_decisions.insert_many([_cooldown(f"r{i}", "pinned-endpoint") for i in range(20)])
        probes = []
        report = renew_pass(db, now=datetime.now(timezone.utc), probe=lambda d, now: probes.append(d))
        assert probes == []
        assert report["retired_without_probe"] == 20
        assert db.bench_route_decisions.count_documents({"state": "retired"}) == 20

    def test_other_cooldowns_are_still_recovered(self, db):
        db.bench_route_decisions.insert_one(_cooldown("or", "or-served"))
        report = renew_pass(db, now=datetime.now(timezone.utc), probe=lambda d, now: None)
        assert report["considered"] == 1 and report["retired_without_probe"] == 0


class TestWorkerMirror:
    def test_pinned_endpoint_failure_is_not_mirrored(self, db):
        from llm_bench.scheduler import worker

        job = {
            "_id": "j",
            "provider": "openrouter",
            "model_id": "z/m",
            "route_snapshot": {"state": "active", "route_policy": "pinned-endpoint", "route_snapshot_at": "t"},
        }
        result = runner.RunnerResult(status="error", route_attempted=True, error_message="boom")
        worker.persist_route_cooldown(db, job=job, result=result, now=datetime.now(timezone.utc))
        assert db.bench_route_decisions.count_documents({}) == 0


class TestClassifierSkip:
    def test_unchanged_unknown_is_not_re_asked(self, monkeypatch):
        import asyncio
        import os

        from llm_bench.ops import llm_error_classifier as clf

        mdb = mongomock.MongoClient()
        now = datetime(2026, 11, 11)
        coll = mdb[os.getenv("MONGODB_DB", "llm-bench")]["error_rollups"]
        coll.insert_one(
            {"fingerprint": "a", "error_kind": "unknown", "classified_at": now, "last_seen": now - timedelta(hours=1)}
        )
        coll.insert_one(
            {"fingerprint": "b", "error_kind": "unknown", "classified_at": now, "last_seen": now + timedelta(hours=1)}
        )
        coll.insert_one({"fingerprint": "c", "error_kind": None})
        monkeypatch.setenv("MONGODB_URI", "mongodb://x")
        monkeypatch.setattr(clf, "MongoClient", lambda uri: mdb)
        mdb.close = lambda: None
        sent = []

        async def fake_batch(batch):
            sent.extend(d["fingerprint"] for d in batch)
            return [], None

        monkeypatch.setattr(clf, "classify_batch", fake_batch)
        monkeypatch.setattr(
            clf, "update_rollups_with_classifications", lambda *a, **k: {"updated": 0, "skipped": 0, "errors": 0}
        )
        asyncio.run(clf.classify_unclassified_rollups())
        assert sorted(sent) == ["b", "c"]
