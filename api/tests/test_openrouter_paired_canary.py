import time

from llm_bench.scheduler.routing import RouteDecision

from scripts import openrouter_paired_canary
from scripts.openrouter_paired_canary import _planned_orders
from scripts.openrouter_paired_canary import evaluate


def test_planned_orders_balance_two_pair_canary():
    orders = _planned_orders(2, seed=0)

    assert orders == [["direct", "openrouter"], ["openrouter", "direct"]]


def attempt(*, tps: float, ttft: float) -> dict:
    return {
        "status": "success",
        "metrics": {
            "tokens_per_second": tps,
            "time_to_first_token": ttft,
            "provider_metadata_verified": True,
            "observed_provider_slug": "openai",
        },
    }


def test_evaluate_requires_both_transports_and_keeps_cost_unverified():
    result = evaluate(
        [
            {
                "attempts": {
                    "direct": attempt(tps=100, ttft=1),
                    "openrouter": attempt(tps=75, ttft=2),
                }
            },
            {
                "attempts": {
                    "direct": attempt(tps=100, ttft=1),
                    "openrouter": attempt(tps=60, ttft=2),
                }
            },
        ],
        min_route_tps_ratio=0.5,
        max_route_ttft_ratio=3.0,
        required_pairs=2,
        min_success_rate=1.0,
    )

    assert result["output_valid"] is True
    assert result["performance_valid"] is True
    assert result["cost_status"] == "unverified"
    assert result["canary_state"] == "measurement_passed_cost_unverified"


def test_evaluate_fails_when_a_route_attempt_errors():
    result = evaluate(
        [
            {
                "attempts": {
                    "direct": attempt(tps=100, ttft=1),
                    "openrouter": {"status": "error", "error": "429"},
                }
            }
        ],
        min_route_tps_ratio=0.5,
        max_route_ttft_ratio=3.0,
        required_pairs=1,
        min_success_rate=1.0,
    )

    assert result["successful_pairs"] == 0
    assert result["output_valid"] is False
    assert result["canary_state"] == "failed"


def test_evaluate_cost_and_confidence_gates_can_promote():
    result = evaluate(
        [
            {
                "attempts": {
                    "direct": attempt(tps=100, ttft=1)
                    | {
                        "metrics": {
                            **attempt(tps=100, ttft=1)["metrics"],
                            "input_tokens": 10,
                            "output_tokens": 64,
                        }
                    },
                    "openrouter": attempt(tps=95, ttft=1.1)
                    | {
                        "metrics": {
                            **attempt(tps=95, ttft=1.1)["metrics"],
                            "input_tokens": 10,
                            "output_tokens": 64,
                        }
                    },
                }
            }
        ],
        min_route_tps_ratio=0.8,
        max_route_ttft_ratio=1.5,
        required_pairs=1,
        min_success_rate=1.0,
        pricing={
            "direct": {"input_per_token": 1e-6, "output_per_token": 1e-6},
            "openrouter": {"input_per_token": 1e-6, "output_per_token": 1e-6},
        },
    )

    assert result["route_tps_ratio_ci95"] == [0.95, 0.95]
    assert result["route_cost_ratio"] == 1.0
    assert result["cost_status"] == "verified"
    assert result["promotion_valid"] is True
    assert result["canary_state"] == "passed"


def test_attempt_uses_one_deadline_across_retries(monkeypatch):
    decision = RouteDecision.direct("openai", "gpt-4o-mini", reason="test")

    def slow_generate(*args, **kwargs):
        time.sleep(2)

    monkeypatch.setattr(openrouter_paired_canary.runner, "_generate_and_validate", slow_generate)
    started = time.monotonic()
    result = openrouter_paired_canary._attempt(decision, max_tokens=16, deadline_seconds=1)
    elapsed = time.monotonic() - started

    assert result["status"] == "error"
    assert result["retry_count"] == 0
    assert elapsed < 1.5


def _pair(direct_metrics, route_metrics, index=0):
    return {
        "pair_index": index,
        "order": "direct-first",
        "attempts": {
            "direct": {"status": "success", "metrics": direct_metrics},
            "openrouter": {"status": "success", "metrics": route_metrics},
        },
    }


def test_missing_direct_ttft_does_not_erase_metadata_or_cost_evidence():
    from scripts.openrouter_paired_canary import evaluate

    route_metrics = {
        "tokens_per_second": 21.0,
        "time_to_first_token": 0.4,
        "provider_metadata_verified": True,
        "observed_provider_slug": "deepinfra",
        "input_tokens": 20,
        "output_tokens": 64,
    }
    direct_metrics = {
        "tokens_per_second": 20.0,
        "time_to_first_token": None,
        "input_tokens": 20,
        "output_tokens": 64,
    }
    pairs = [_pair(direct_metrics, route_metrics, i) for i in range(30)]
    rates = {"input_per_token": 1e-7, "output_per_token": 2e-7}
    result = evaluate(
        pairs,
        min_route_tps_ratio=0.8,
        max_route_ttft_ratio=1.5,
        pricing={"direct": rates, "openrouter": rates},
        expected_route_provider_slug="deepinfra",
    )
    assert result["ttft_waived_direct_unmeasured"] is True
    assert result["direct_ttft_measured_pairs"] == 0
    assert result["metadata_valid"] is True
    assert result["cost_status"] == "verified"
    assert result["performance_valid"] is True
    assert result["promotion_valid"] is True


def test_measured_direct_ttft_still_enforces_ttft_bound():
    from scripts.openrouter_paired_canary import evaluate

    route_metrics = {
        "tokens_per_second": 21.0,
        "time_to_first_token": 2.0,
        "provider_metadata_verified": True,
        "observed_provider_slug": "deepinfra",
        "input_tokens": 20,
        "output_tokens": 64,
    }
    direct_metrics = {
        "tokens_per_second": 20.0,
        "time_to_first_token": 0.5,
        "input_tokens": 20,
        "output_tokens": 64,
    }
    pairs = [_pair(direct_metrics, route_metrics, i) for i in range(30)]
    result = evaluate(pairs, min_route_tps_ratio=0.8, max_route_ttft_ratio=1.5)
    assert result["ttft_waived_direct_unmeasured"] is False
    assert result["performance_valid"] is False
