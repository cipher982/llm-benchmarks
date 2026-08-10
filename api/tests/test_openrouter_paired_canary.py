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
