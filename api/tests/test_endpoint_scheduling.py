"""Endpoint targets rotate by model popularity, bounded and oldest-first.

The bound applies to jobs created, never to which models are eligible. Within a
model, one oldest endpoint is scheduled per tier opportunity so provider count
does not multiply a popular model's faster cadence into a full sweep.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import cli
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue
from llm_bench.scheduler.mongo import models_collection_name


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def seed_endpoints(db, tags, model_id="openai/gpt-oss-120b"):
    db[models_collection_name()].update_one(
        {"provider": "openrouter", "model_id": model_id},
        {"$set": {"enabled": True, "deprecated": False}},
        upsert=True,
    )
    db[endpoint_discovery.endpoints_collection_name()].insert_many(
        [
            {
                "model_id": model_id,
                "endpoint_tag": tag,
                "provider_canonical": endpoint_discovery.provider_canonical(tag),
                "enabled": True,
            }
            for tag in tags
        ]
    )


class TestGate:
    def test_endpoint_targets_are_off_by_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMARK_ENDPOINT_TARGETS", raising=False)
        assert policies.endpoint_targets_enabled() is False

    def test_the_gate_turns_them_on(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_ENDPOINT_TARGETS", "1")
        assert policies.endpoint_targets_enabled() is True

    def test_default_provider_count_tiers(self, monkeypatch):
        for name in (
            "BENCHMARK_ENDPOINT_HOT_PROVIDER_MIN",
            "BENCHMARK_ENDPOINT_MEDIUM_PROVIDER_MIN",
            "BENCHMARK_ENDPOINT_HOT_HOURS",
            "BENCHMARK_ENDPOINT_MEDIUM_HOURS",
            "BENCHMARK_ENDPOINT_LONG_HOURS",
        ):
            monkeypatch.delenv(name, raising=False)
        assert policies.endpoint_tier_interval_seconds(2) == 96 * 60 * 60
        assert policies.endpoint_tier_interval_seconds(3) == 24 * 60 * 60
        assert policies.endpoint_tier_interval_seconds(8) == 3 * 60 * 60


class TestCandidates:
    def test_each_endpoint_is_its_own_target(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "deepinfra/bf16", "deepinfra/turbo"])
        found = _endpoint_candidates(db, provider="openrouter")

        assert sorted(tag for _, _, tag, _ in found) == ["deepinfra/bf16", "deepinfra/turbo", "groq"]
        docs = list(health.health_collection(db).find({"model_id": "openai/gpt-oss-120b"}))
        assert {doc["cadence_seconds"] for doc in docs} == {3 * 96 * 60 * 60}

    def test_provider_count_change_updates_endpoint_revisit_cadence(self, db):
        seed_endpoints(db, ["p1", "p2"], model_id="m")
        cli._endpoint_candidates(db, provider="openrouter")
        coll = health.health_collection(db)
        assert {doc["cadence_seconds"] for doc in coll.find({"model_id": "m"})} == {2 * 96 * 60 * 60}

        seed_endpoints(db, [f"p{i}" for i in range(3, 9)], model_id="m")
        cli._endpoint_candidates(db, provider="openrouter")
        assert {doc["cadence_seconds"] for doc in coll.find({"model_id": "m"})} == {8 * 3 * 60 * 60}

    def test_one_recent_endpoint_paces_the_whole_model(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "novita/fp8"])
        _endpoint_candidates(db, provider="openrouter")
        health.record_success(
            db,
            provider="openrouter",
            model_id="openai/gpt-oss-120b",
            endpoint_tag="groq",
            cadence_seconds=3600,
        )

        found = _endpoint_candidates(db, provider="openrouter")
        assert found == []

    def test_disabled_endpoints_are_never_scheduled(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq"])
        db[endpoint_discovery.endpoints_collection_name()].insert_one(
            {"model_id": "openai/gpt-oss-120b", "endpoint_tag": "sambanova", "enabled": False}
        )
        found = _endpoint_candidates(db, provider="openrouter")
        assert [tag for _, _, tag, _ in found] == ["groq"]

    def test_the_stalest_endpoint_sorts_first(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "novita/fp8"])
        _endpoint_candidates(db, provider="openrouter")

        now = datetime.now(timezone.utc)
        coll = health.health_collection(db)
        coll.update_one(
            health.health_filter("openrouter", "openai/gpt-oss-120b", "groq"),
            {"$set": {"freshness_status": "stale", "last_success_at": now - timedelta(days=5)}},
        )
        coll.update_one(
            health.health_filter("openrouter", "openai/gpt-oss-120b", "novita/fp8"),
            {
                "$set": {
                    "freshness_status": "critical",
                    "last_success_at": now - timedelta(days=6),
                }
            },
        )

        found = _endpoint_candidates(db, provider="openrouter")
        assert found[0][2] == "novita/fp8", "the endpoint that waited longest must go first"


class TestRotatingSchedulerPass:
    def test_enqueues_one_endpoint_per_model_without_unpinned_duplicates(self, db, monkeypatch):
        seed_endpoints(db, ["alpha", "beta"], model_id="m1")
        seed_endpoints(db, ["alpha"], model_id="m2")
        monkeypatch.setenv("BENCHMARK_ENDPOINT_TARGETS", "1")
        monkeypatch.setenv("BENCHMARK_ENDPOINT_TARGETS_PER_PASS", "25")
        monkeypatch.setattr(cli, "mongo_env", lambda: ("unused", "db"))
        monkeypatch.setattr(cli, "load_provider_models", lambda: {"openrouter": ["m1", "m2", "fallback"]})
        monkeypatch.setattr(cli.health, "refresh_all_model_docs", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli.health, "heartbeat", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli, "route_snapshot", lambda *args, **kwargs: None)

        class Client:
            def __getitem__(self, name):
                return db

            def close(self):
                pass

        monkeypatch.setattr(cli, "mongo_client", Client)
        enqueued = []
        monkeypatch.setattr(
            cli.queue,
            "enqueue_scheduled_job",
            lambda *args, **kwargs: enqueued.append(kwargs) or True,
        )

        assert cli.scheduler_pass(providers="openrouter", limit=100, cadence_seconds=96 * 60 * 60) == 2
        assert [(job["model_id"], job.get("endpoint_tag")) for job in enqueued] == [
            ("m1", "alpha"),
            ("m2", "alpha"),
        ]

    def test_an_active_endpoint_job_paces_its_siblings(self, db):
        seed_endpoints(db, ["alpha", "beta"], model_id="m")
        queue.enqueue_scheduled_job(
            db,
            provider="openrouter",
            model_id="m",
            endpoint_tag="alpha",
            priority=1,
        )

        assert cli._endpoint_candidates(db, provider="openrouter") == []

    def test_an_unpinned_job_also_blocks_endpoint_rotation(self, db):
        seed_endpoints(db, ["alpha"], model_id="m")
        queue.enqueue_scheduled_job(db, provider="openrouter", model_id="m", priority=1)

        assert cli._endpoint_candidates(db, provider="openrouter") == []


class TestEnqueue:
    def test_only_one_endpoint_job_per_model_can_be_active(self, db):
        assert queue.enqueue_scheduled_job(
            db,
            provider="openrouter",
            model_id="openai/gpt-oss-120b",
            priority=1.0,
            endpoint_tag="deepinfra/bf16",
            cadence_seconds=123,
        )
        assert not queue.enqueue_scheduled_job(
            db,
            provider="openrouter",
            model_id="openai/gpt-oss-120b",
            priority=1.0,
            endpoint_tag="deepinfra/turbo",
        )
        jobs = list(db["bench_jobs"].find({}))
        assert len(jobs) == 1
        assert jobs[0]["endpoint_tag"] == "deepinfra/bf16"
        assert jobs[0]["cadence_seconds"] == 123


class TestRowIdentity:
    """A measurement that does not name its endpoint cannot be published as one."""

    def test_a_pinned_decision_carries_tag_and_quantization(self):
        from llm_bench.scheduler.routing import RouteDecision

        d = RouteDecision(
            source_provider="openrouter",
            source_model_id="openai/gpt-oss-120b",
            route_endpoint_tag="coreweave/fp4",
            route_endpoint_quantization="fp4",
        )
        assert d.route_endpoint_tag == "coreweave/fp4"
        assert d.route_endpoint_quantization == "fp4"

    def test_endpoint_fields_survive_the_metric_writer(self):
        """log_mongo copies only registered keys; anything else is dropped."""
        from llm_bench.logging import _optional_metric_fields

        written = _optional_metric_fields(
            {
                "route_endpoint_tag": "coreweave/fp4",
                "quantization": "fp4",
                "unregistered": "dropped",
            }
        )
        assert written == {"route_endpoint_tag": "coreweave/fp4", "quantization": "fp4"}


class TestEndpointPinRoundTrip:
    """The pin has to survive enqueue and come back out as a pinned decision.

    This is the join the whole design rests on: a job that carries a tag but
    resolves to an unpinned route would run against OpenRouter's price-selected
    default and publish the result under the endpoint's name.
    """

    def test_an_endpoint_snapshot_resolves_to_a_pinned_decision(self):
        from llm_bench.scheduler.routing import PINNED_ENDPOINT_POLICY
        from llm_bench.scheduler.routing import RouteDecision
        from llm_bench.scheduler.routing import endpoint_route_snapshot

        snapshot = endpoint_route_snapshot(
            "openrouter",
            "openai/gpt-oss-120b",
            endpoint_tag="cerebras/fp16",
            provider_canonical="cerebras",
            quantization="fp16",
        )
        decision = RouteDecision.from_snapshot("openrouter", "openai/gpt-oss-120b", snapshot)

        assert decision.route_policy == PINNED_ENDPOINT_POLICY
        assert decision.route_endpoint_tag == "cerebras/fp16"
        assert decision.route_endpoint_quantization == "fp16"
        # The family, not the tag: that is all the response can confirm.
        assert decision.route_provider_slug == "cerebras"
        assert decision.transport_provider == "openrouter"

    def test_the_pinned_decision_produces_a_pinned_request(self):
        from llm_bench.cloud.providers.openrouter import _route_options
        from llm_bench.config import CloudConfig
        from llm_bench.scheduler.routing import RouteDecision
        from llm_bench.scheduler.routing import endpoint_route_snapshot

        snapshot = endpoint_route_snapshot(
            "openrouter",
            "openai/gpt-oss-120b",
            endpoint_tag="deepinfra/turbo",
            provider_canonical="deepinfra",
        )
        decision = RouteDecision.from_snapshot("openrouter", "openai/gpt-oss-120b", snapshot)
        config = CloudConfig(
            provider="openrouter",
            model_name="openai/gpt-oss-120b",
            run_ts="2026-08-17 00:00:00",
            temperature=0.1,
            misc={
                "route_policy": decision.route_policy,
                "route_provider_slug": decision.route_provider_slug,
                "route_endpoint_tag": decision.route_endpoint_tag,
            },
        )

        options = _route_options(config)
        assert options["provider"]["only"] == ["deepinfra/turbo"]
        assert options["provider"]["allow_fallbacks"] is False

    def test_a_snapshot_without_a_tag_falls_back_to_direct(self):
        from llm_bench.scheduler.routing import RouteDecision
        from llm_bench.scheduler.routing import endpoint_route_snapshot

        snapshot = endpoint_route_snapshot("openrouter", "m", endpoint_tag="groq", provider_canonical="groq")
        snapshot["route_endpoint_tag"] = ""
        decision = RouteDecision.from_snapshot("openrouter", "m", snapshot)

        assert decision.transport_provider == "direct"
        assert decision.reason == "endpoint-route-has-no-tag"


class TestStalenessIsDerivedNotCached:
    """An endpoint measured once was never scheduled again.

    `record_success` writes `freshness_status: "fresh"` and only a later
    success would change it. Nothing recomputes that label for an endpoint --
    model docs get a reconcile pass, endpoint docs are in no such loop -- so the
    label said "fresh" indefinitely while the measurement aged out. In
    production 705 endpoints carried "fresh" while 618 of them had gone more
    than three cadences unmeasured, and throughput fell to 12 endpoints an hour
    against a catalogue of 690.
    """

    def _catalogued(self, db, model_id="m", tag="novita/fp8"):
        seed_endpoints(db, [tag], model_id=model_id)

    def test_a_stale_endpoint_is_selected_despite_a_fresh_label(self, db):
        now = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        self._catalogued(db)
        health.health_collection(db).insert_one(
            {
                "_id": "openrouter:m:novita/fp8",
                "provider": "openrouter",
                "model_id": "m",
                "endpoint_tag": "novita/fp8",
                "enabled": True,
                "cadence_seconds": 10800,
                # The stored label is the lie; the timestamp is the fact.
                "freshness_status": "fresh",
                "last_success_at": now - timedelta(days=5),
            }
        )
        candidates = cli._endpoint_candidates(db, provider="openrouter", now=now)
        assert [(m, t) for _, m, t, _ in candidates] == [("m", "novita/fp8")]

    def test_a_genuinely_fresh_endpoint_is_not_reselected(self, db):
        now = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        self._catalogued(db)
        health.health_collection(db).insert_one(
            {
                "_id": "openrouter:m:novita/fp8",
                "provider": "openrouter",
                "model_id": "m",
                "endpoint_tag": "novita/fp8",
                "enabled": True,
                "cadence_seconds": 10800,
                "freshness_status": "stale",
                "last_success_at": now - timedelta(minutes=30),
            }
        )
        assert cli._endpoint_candidates(db, provider="openrouter", now=now) == []

    def test_a_never_measured_endpoint_is_still_selected(self, db):
        now = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        self._catalogued(db)
        health.health_collection(db).insert_one(
            {
                "_id": "openrouter:m:novita/fp8",
                "provider": "openrouter",
                "model_id": "m",
                "endpoint_tag": "novita/fp8",
                "enabled": True,
                "cadence_seconds": 10800,
                "last_success_at": None,
            }
        )
        candidates = cli._endpoint_candidates(db, provider="openrouter", now=now)
        assert len(candidates) == 1
