from datetime import datetime
from datetime import timezone

import mongomock
from llm_bench import logging as benchmark_logging
from llm_bench.scheduler import queue
from llm_bench.scheduler.routing import DIRECT_TRANSPORT
from llm_bench.scheduler.routing import OPENROUTER_TRANSPORT
from llm_bench.scheduler.routing import RouteDecision
from llm_bench.scheduler.routing import cooldown_route_snapshot
from llm_bench.scheduler.routing import endpoint_route_snapshot
from llm_bench.scheduler.routing import freeze_route_snapshot
from llm_bench.scheduler.routing import resolve_job_route

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def active_snapshot(**overrides):
    value = endpoint_route_snapshot(
        "openrouter",
        "qwen/qwen3-32b",
        endpoint_tag="deepinfra/fp8",
        provider_canonical="deepinfra",
        quantization="fp8",
        provider_display="DeepInfra",
        now=NOW,
    )
    value.update(overrides)
    return value


def test_missing_or_non_active_evidence_is_direct():
    missing = RouteDecision.from_snapshot("openrouter", "m", None)
    pending = RouteDecision.from_snapshot("openrouter", "m", {"state": "canary_pending"})

    assert missing.transport_provider == DIRECT_TRANSPORT
    assert missing.reason == "missing-route-snapshot"
    assert pending.transport_provider == DIRECT_TRANSPORT
    assert pending.reason == "route-decision-version-mismatch"


def test_active_endpoint_route_pins_the_named_endpoint():
    decision = RouteDecision.from_snapshot("openrouter", "qwen/qwen3-32b", active_snapshot())

    assert decision.transport_provider == OPENROUTER_TRANSPORT
    assert decision.route_endpoint_tag == "deepinfra/fp8"
    assert decision.route_provider_slug == "deepinfra"
    assert decision.route_endpoint_quantization == "fp8"
    assert decision.source_model_id == "qwen/qwen3-32b"
    assert decision.metric_fields()["route_policy"] == "pinned-endpoint"


def test_source_mismatch_fails_closed():
    decision = RouteDecision.from_snapshot("openrouter", "other/model", active_snapshot())

    assert decision.transport_provider == DIRECT_TRANSPORT
    assert decision.reason == "route-source-mismatch"


def test_rollback_state_and_unknown_policy_fail_closed():
    rolled_back = RouteDecision.from_snapshot("openrouter", "qwen/qwen3-32b", active_snapshot(state="rollback"))
    retired_policy = RouteDecision.from_snapshot(
        "openrouter", "qwen/qwen3-32b", active_snapshot(route_policy="or-served")
    )
    no_tag = RouteDecision.from_snapshot("openrouter", "qwen/qwen3-32b", active_snapshot(route_endpoint_tag=None))

    assert rolled_back.reason == "route-state-not-active"
    assert retired_policy.reason == "invalid-route-policy"
    assert no_tag.reason == "endpoint-route-has-no-tag"
    assert all(d.transport_provider == DIRECT_TRANSPORT for d in (rolled_back, retired_policy, no_tag))


def test_route_cooldown_resolves_direct():
    cooled = cooldown_route_snapshot(active_snapshot(), failure_reason="route 429", now=NOW, cooldown_seconds=60)
    during = RouteDecision.from_snapshot(cooled["source_provider"], cooled["source_model_id"], cooled, now=NOW)

    assert cooled["route_health_state"] == "cooldown"
    assert during.transport_provider == DIRECT_TRANSPORT
    assert during.reason == "route-state-not-active"


def test_snapshot_is_copied_when_frozen_and_job_resolution_is_fail_closed():
    original = active_snapshot()
    frozen = freeze_route_snapshot("openrouter", "qwen/qwen3-32b", original)
    original["state"] = "direct"
    decision = resolve_job_route({"provider": "openrouter", "model_id": "qwen/qwen3-32b", "route_snapshot": frozen})

    assert frozen["state"] == "active"
    assert decision.transport_provider == OPENROUTER_TRANSPORT


def test_queue_persists_a_frozen_route_snapshot():
    db = mongomock.MongoClient()["llm-bench-test"]
    original = active_snapshot()
    queue.enqueue_manual_job(
        db,
        provider="openrouter",
        model_id="qwen/qwen3-32b",
        now=datetime(2026, 8, 9, tzinfo=timezone.utc),
        route_snapshot=original,
    )
    original["state"] = "direct"

    saved = db.bench_jobs.find_one({"provider": "openrouter"})
    assert saved["route_snapshot"]["state"] == "active"
    assert saved["route_snapshot"]["queued_at"] == "2026-08-09T00:00:00+00:00"


def test_metric_writer_keeps_source_identity_for_routed_config(monkeypatch):
    inserted = {}

    class Collection:
        def insert_one(self, data):
            inserted.update(data)

    monkeypatch.setattr(benchmark_logging, "setup_database", lambda *args: Collection())
    config = benchmark_logging.CloudConfig(
        provider="deepinfra",
        model_name="Qwen/Qwen3-32B",
        run_ts="2026-08-09 00:00:00",
        temperature=0.1,
        transport_provider="openrouter",
        transport_model_id="qwen/qwen3-32b",
    )
    metrics = {
        "gen_ts": "2026-08-09 00:00:01",
        "requested_tokens": 64,
        "output_tokens": 64,
        "generate_time": 1.0,
        "tokens_per_second": 64.0,
        "time_to_first_token": 0.1,
        "times_between_tokens": [],
        "transport_provider": "openrouter",
        "route_model_id": "qwen/qwen3-32b",
    }

    benchmark_logging.log_mongo("cloud", config, metrics, "mongodb://test", "db", "metrics")

    assert inserted["provider"] == "deepinfra"
    assert inserted["model_name"] == "Qwen/Qwen3-32B"
    assert inserted["transport_provider"] == "openrouter"


def test_route_reasoning_exclude_flows_from_snapshot():
    decision = RouteDecision.from_snapshot(
        "openrouter", "qwen/qwen3-32b", active_snapshot(route_reasoning_exclude=True)
    )
    assert decision.route_reasoning_exclude is True
    assert (
        RouteDecision.from_snapshot("openrouter", "qwen/qwen3-32b", active_snapshot()).route_reasoning_exclude is False
    )


def test_retired_route_fields_are_not_written_to_metric_rows(monkeypatch):
    from llm_bench.logging import _optional_metric_fields

    retired = {
        "route_canary_id": "c",
        "route_canary_state": "passed",
        "route_expires_at": "t",
        "route_probe_id": "p",
        "route_revocation_generation": 0,
        "route_decision_version": "or-route-v1",
        "direct_effective_request_hash": "h",
        "routed_effective_request_hash": "h",
        "protocol_version": 2,
        "cached_input_tokens": 0,
    }
    assert _optional_metric_fields({**retired, "route_policy": "pinned-endpoint"}) == {
        "route_policy": "pinned-endpoint"
    }
