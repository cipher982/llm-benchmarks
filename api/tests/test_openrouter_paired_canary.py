from scripts.openrouter_paired_canary import _planned_orders
from scripts.openrouter_paired_canary import evaluate


def test_planned_orders_balance_two_pair_canary():
    orders = _planned_orders(2, seed=0)

    assert orders == [["direct", "openrouter"], ["openrouter", "direct"]]


def attempt(*, tps: float, ttft: float) -> dict:
    return {
        "status": "success",
        "metrics": {"tokens_per_second": tps, "time_to_first_token": ttft},
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
    )

    assert result["successful_pairs"] == 0
    assert result["output_valid"] is False
    assert result["canary_state"] == "failed"
