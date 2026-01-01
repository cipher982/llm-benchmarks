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


logger = logging.getLogger(__name__)


def should_auto_execute(decision: OperatorDecision) -> bool:
    """
    Determine if a decision should be auto-executed without human approval.

    Auto-execution rules (from spec):
    1. 404 errors for 48+ hours with high confidence
    2. 401 errors for 48+ hours with high confidence
    3. LLM confidence ≥ 0.95 for disable action

    Args:
        decision: Operator decision to evaluate

    Returns:
        True if should auto-execute, False otherwise
    """
    # Only auto-execute disable actions
    if decision.action != "disable":
        return False

    # High confidence threshold - but must also have 404/401 for 48+ hours
    if decision.confidence >= 0.95:
        # Check if reasoning indicates 404/401 errors for 48+ hours
        reasoning_lower = decision.reasoning.lower()

        # Look for indicators of 404/401 errors
        has_404_401 = any(x in reasoning_lower for x in ["404", "401", "not found", "unauthorized"])

        # Look for indicators of 48+ hours duration
        has_duration = any(x in reasoning_lower for x in [
            "48 hours", "48+ hours", "2 days", "two days",
            "48h", "48hr", "48 hrs",
            "over 48", "more than 48", "exceeds 48"
        ])

        # Only auto-execute if both conditions are met
        if has_404_401 and has_duration:
            return True

    return False


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
    *,
    auto_execute: bool = True,
    dry_run: bool = False,
    client: Optional[MongoClient] = None
) -> dict:
    """
    Execute a list of operator decisions.

    Args:
        decisions: List of decisions to execute
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

    for decision in decisions:
        # Check if should auto-execute
        if not auto_execute or not should_auto_execute(decision):
            stats["skipped"] += 1
            logger.info(
                f"Skipping {decision.provider}/{decision.model_id} "
                f"(action={decision.action}, confidence={decision.confidence:.2f})"
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
