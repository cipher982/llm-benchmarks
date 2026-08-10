import pytest

from scripts.openrouter_budget import reserve_daily_budget


def test_budget_ledger_reserves_shared_daily_cap(tmp_path):
    ledger = tmp_path / "budget.json"
    reserve_daily_budget(
        ledger,
        amount_usd=2.0,
        batch_max_usd=5.0,
        daily_max_usd=3.0,
        operation="probe",
    )
    with pytest.raises(ValueError, match="daily cost"):
        reserve_daily_budget(
            ledger,
            amount_usd=2.0,
            batch_max_usd=5.0,
            daily_max_usd=3.0,
            operation="canary",
        )
