"""Small shared reservation ledger for bounded OpenRouter evidence runs."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

DEFAULT_BATCH_MAX_USD = 50.0
DEFAULT_DAILY_MAX_USD = 50.0


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def reserve_daily_budget(
    path: Path,
    *,
    amount_usd: float,
    batch_max_usd: float,
    daily_max_usd: float,
    operation: str,
) -> dict[str, Any]:
    """Reserve estimated spend before sending requests.

    The ledger is intentionally conservative. Reservations count against the
    UTC day even if a provider later returns fewer tokens, which prevents
    concurrent evidence commands from overspending a shared cap.
    """

    if amount_usd < 0 or batch_max_usd <= 0 or daily_max_usd <= 0:
        raise ValueError("budget values must be non-negative and caps must be positive")
    if amount_usd > batch_max_usd + 1e-12:
        raise ValueError(f"estimated batch cost ${amount_usd:.4f} exceeds cap ${batch_max_usd:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            ledger = _load(path)
            if ledger.get("date") != today:
                ledger = {"schema_version": 1, "date": today, "reserved_usd": 0.0, "reservations": []}
            reserved = float(ledger.get("reserved_usd", 0.0) or 0.0)
            if reserved + amount_usd > daily_max_usd + 1e-12:
                raise ValueError(f"estimated daily cost ${reserved + amount_usd:.4f} exceeds cap ${daily_max_usd:.4f}")
            entry = {
                "operation": operation,
                "amount_usd": round(amount_usd, 8),
                "reserved_at": datetime.now(timezone.utc).isoformat(),
            }
            ledger.update(
                {
                    "schema_version": 1,
                    "date": today,
                    "daily_max_usd": daily_max_usd,
                    "reserved_usd": round(reserved + amount_usd, 8),
                    "reservations": [*(ledger.get("reservations") or []), entry],
                }
            )
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(ledger, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
            return ledger
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
