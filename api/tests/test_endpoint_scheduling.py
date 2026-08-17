"""Endpoint targets are scheduled per endpoint, bounded, and stalest-first.

The bound applies to jobs created, never to which endpoints are eligible. A cap
applied to the population is how twelve DeepInfra models went unscheduled for
months with no error and nothing disabled.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import health
from llm_bench.scheduler import policies
from llm_bench.scheduler import queue


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def seed_endpoints(db, tags, model_id="openai/gpt-oss-120b"):
    db[endpoint_discovery.endpoints_collection_name()].insert_many(
        [{"model_id": model_id, "endpoint_tag": t, "enabled": True} for t in tags]
    )


class TestGate:
    def test_endpoint_targets_are_off_by_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMARK_ENDPOINT_TARGETS", raising=False)
        assert policies.endpoint_targets_enabled() is False

    def test_the_gate_turns_them_on(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_ENDPOINT_TARGETS", "1")
        assert policies.endpoint_targets_enabled() is True


class TestCandidates:
    def test_each_endpoint_is_its_own_target(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "deepinfra/bf16", "deepinfra/turbo"])
        found = _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)

        assert sorted(tag for _, _, tag in found) == ["deepinfra/bf16", "deepinfra/turbo", "groq"]

    def test_a_freshly_measured_endpoint_is_not_a_candidate(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "novita/fp8"])
        _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)
        health.record_success(
            db,
            provider="openrouter",
            model_id="openai/gpt-oss-120b",
            endpoint_tag="groq",
            cadence_seconds=3600,
        )

        found = _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)
        assert [tag for _, _, tag in found] == ["novita/fp8"]

    def test_disabled_endpoints_are_never_scheduled(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq"])
        db[endpoint_discovery.endpoints_collection_name()].insert_one(
            {"model_id": "openai/gpt-oss-120b", "endpoint_tag": "sambanova", "enabled": False}
        )
        found = _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)
        assert [tag for _, _, tag in found] == ["groq"]

    def test_the_stalest_endpoint_sorts_first(self, db):
        from llm_bench.scheduler.cli import _endpoint_candidates

        seed_endpoints(db, ["groq", "novita/fp8"])
        _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)

        now = datetime.now(timezone.utc)
        coll = health.health_collection(db)
        coll.update_one(
            health.health_filter("openrouter", "openai/gpt-oss-120b", "groq"),
            {"$set": {"staleness_seconds": 100, "freshness_status": "stale", "last_success_at": now}},
        )
        coll.update_one(
            health.health_filter("openrouter", "openai/gpt-oss-120b", "novita/fp8"),
            {
                "$set": {
                    "staleness_seconds": 99999,
                    "freshness_status": "critical",
                    "last_success_at": now - timedelta(days=3),
                }
            },
        )

        found = _endpoint_candidates(db, provider="openrouter", cadence_seconds=3600)
        assert found[0][2] == "novita/fp8", "the endpoint that waited longest must go first"


class TestEnqueue:
    def test_two_endpoints_of_one_model_produce_two_jobs(self, db):
        for tag in ("deepinfra/bf16", "deepinfra/turbo"):
            queue.enqueue_scheduled_job(
                db,
                provider="openrouter",
                model_id="openai/gpt-oss-120b",
                priority=1.0,
                endpoint_tag=tag,
            )
        jobs = list(db["bench_jobs"].find({}))
        assert len(jobs) == 2, "endpoints collapsed into one job"
        assert sorted(j["endpoint_tag"] for j in jobs) == ["deepinfra/bf16", "deepinfra/turbo"]


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
