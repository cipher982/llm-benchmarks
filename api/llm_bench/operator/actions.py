"""
Action execution for AI Operator decisions.

This module handles executing operator decisions like disabling models,
updating statuses, etc.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from pymongo import MongoClient

from .engine import OperatorDecision
from ..model_lifecycle.collector import LifecycleSnapshot


logger = logging.getLogger(__name__)


def should_auto_execute(
    decision: OperatorDecision,
    snapshot: LifecycleSnapshot
) -> bool:
    """
    Determine if a decision should be auto-executed without human approval.

    NEW (Phase 5): Uses actual signals, not prose parsing.

    Auto-execution rules (from spec):
    - Hard model errors (404, "not found", etc.) for 48+ hours
    - No recent success (7+ days)
    - High confidence (≥ 0.90)

    Args:
        decision: Operator decision to evaluate
        snapshot: Lifecycle snapshot with actual signal data

    Returns:
        True if should auto-execute, False otherwise
    """
    # Only auto-execute disable actions
    if decision.action != "disable":
        return False

    # Must meet confidence threshold
    if decision.confidence < 0.90:
        return False

    # Check for hard failures (404, "not found", etc.)
    if snapshot.errors.hard_failures_7d == 0:
        return False

    # Check error age (must be 48+ hours old)
    error_age_days = snapshot.errors.age_days(datetime.now(timezone.utc))
    if error_age_days is None or error_age_days < 2.0:  # 2 days = 48 hours
        return False

    # Check for recent success (no success in last 7 days)
    success_age_days = snapshot.successes.age_days(datetime.now(timezone.utc))
    if success_age_days is not None and success_age_days <= 7.0:
        return False

    # All conditions met: hard failure, 48+ hours, no recent success, high confidence
    return True


def execute_disable(
    decision: OperatorDecision,
    *,
    client: Optional[MongoClient] = None,
    dry_run: bool = False
) -> bool:
    """
    Execute a disable decision by updating the models collection.

    Args:
        decision: Operator decision to execute
        client: Optional MongoDB client
        dry_run: If True, log what would happen but don't modify DB

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would disable {decision.provider}/{decision.model_id}: "
            f"{decision.reasoning}"
        )
        return True

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")
    models_collection = os.getenv("MONGODB_COLLECTION_MODELS", "models")

    close_client = False
    if client is None:
        client = MongoClient(uri)
        close_client = True

    try:
        collection = client[db_name][models_collection]

        result = collection.update_one(
            {
                "provider": decision.provider,
                "model_id": decision.model_id
            },
            {
                "$set": {
                    "enabled": False,
                    "disabled_reason": f"AI Operator: {decision.reasoning}",
                    "disabled_at": datetime.now(timezone.utc),
                    "disabled_by": decision.suggested_by
                }
            }
        )

        if result.modified_count > 0:
            logger.info(
                f"Disabled {decision.provider}/{decision.model_id}: "
                f"{decision.reasoning}"
            )
            return True
        else:
            logger.warning(
                f"Failed to disable {decision.provider}/{decision.model_id}: "
                f"document not found or already disabled"
            )
            return False

    except Exception as e:
        logger.error(
            f"Error disabling {decision.provider}/{decision.model_id}: {e}"
        )
        return False
    finally:
        if close_client:
            client.close()


def mark_decision_executed(
    decision: OperatorDecision,
    *,
    success: bool = True,
    client: Optional[MongoClient] = None
) -> bool:
    """
    Mark an operator decision as executed in model_status collection.

    Args:
        decision: Operator decision that was executed
        success: Whether execution was successful
        client: Optional MongoDB client

    Returns:
        True if update successful, False otherwise
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")
    status_collection = "model_status"

    close_client = False
    if client is None:
        client = MongoClient(uri)
        close_client = True

    try:
        collection = client[db_name][status_collection]

        update_doc = {
            "operator_decision.status": "auto_executed" if success else "failed",
            "operator_decision.executed_at": datetime.now(timezone.utc)
        }

        result = collection.update_one(
            {
                "provider": decision.provider,
                "model_id": decision.model_id
            },
            {"$set": update_doc}
        )

        return result.modified_count > 0

    except Exception as e:
        logger.error(
            f"Error marking decision executed for "
            f"{decision.provider}/{decision.model_id}: {e}"
        )
        return False
    finally:
        if close_client:
            client.close()


def execute_decisions(
    decisions: List[OperatorDecision],
    snapshots: Optional[List[LifecycleSnapshot]] = None,
    *,
    auto_execute: bool = True,
    dry_run: bool = False,
    client: Optional[MongoClient] = None
) -> dict:
    """
    Execute a list of operator decisions.

    Args:
        decisions: List of decisions to execute
        snapshots: Optional list of snapshots (for signal-based auto-exec checks)
        auto_execute: If True, auto-execute high-confidence decisions
        dry_run: If True, log what would happen but don't modify DB
        client: Optional MongoDB client

    Returns:
        Statistics about execution: {executed: int, skipped: int, failed: int}
    """
    stats = {
        "executed": 0,
        "skipped": 0,
        "failed": 0
    }

    # Build snapshot lookup if provided
    snapshot_map = {}
    if snapshots:
        snapshot_map = {
            (s.provider, s.model_id): s
            for s in snapshots
        }

    for decision in decisions:
        # Check if should auto-execute
        snapshot = snapshot_map.get((decision.provider, decision.model_id))

        # If no snapshot available, skip auto-execution (be conservative)
        if not auto_execute or not snapshot or not should_auto_execute(decision, snapshot):
            stats["skipped"] += 1

            skip_reason = ""
            if not auto_execute:
                skip_reason = "auto_execute=False"
            elif not snapshot:
                skip_reason = "no snapshot data"
            else:
                skip_reason = f"action={decision.action}, confidence={decision.confidence:.2f}"

            logger.info(
                f"Skipping {decision.provider}/{decision.model_id} ({skip_reason})"
            )
            continue

        # Execute the decision
        if decision.action == "disable":
            success = execute_disable(decision, client=client, dry_run=dry_run)

            if success:
                stats["executed"] += 1
                if not dry_run:
                    mark_decision_executed(decision, success=True, client=client)
            else:
                stats["failed"] += 1
                if not dry_run:
                    mark_decision_executed(decision, success=False, client=client)
        else:
            # Monitor and ignore actions don't require execution
            stats["skipped"] += 1

    return stats
