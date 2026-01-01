"""
Core AI Operator engine using LLM reasoning for model lifecycle decisions.

Falls back to classifier.py if LLM calls fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from ..model_lifecycle.classifier import LifecycleStatus, classify_snapshot
from ..model_lifecycle.collector import LifecycleSnapshot


logger = logging.getLogger(__name__)


# Configuration
OPENAI_MODEL = os.getenv("OPERATOR_LLM_MODEL", "gpt-5.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass(slots=True)
class OperatorDecision:
    """Decision made by the AI operator."""
    provider: str
    model_id: str
    action: str  # disable, monitor, ignore
    confidence: float
    reasoning: str
    suggested_at: datetime
    suggested_by: str
    status: str = "pending"  # pending, approved, rejected, auto_executed
    executed_at: Optional[datetime] = None


def format_snapshot_context(snapshot: LifecycleSnapshot) -> str:
    """Format snapshot data for LLM context."""
    lines = [
        f"Provider: {snapshot.provider}",
        f"Model ID: {snapshot.model_id}",
        f"Display Name: {snapshot.display_name}",
        "",
        "=== Success Metrics ===",
    ]

    if snapshot.successes.last_success:
        lines.append(f"Last Success: {snapshot.successes.last_success.isoformat()}")
    else:
        lines.append("Last Success: Never")

    lines.extend([
        f"Successes (7d): {snapshot.successes.successes_7d}",
        f"Successes (30d): {snapshot.successes.successes_30d}",
        f"Successes (120d): {snapshot.successes.successes_120d}",
        "",
        "=== Error Metrics ===",
    ])

    if snapshot.errors.last_error:
        lines.append(f"Last Error: {snapshot.errors.last_error.isoformat()}")
    else:
        lines.append("Last Error: None")

    lines.extend([
        f"Errors (7d): {snapshot.errors.errors_7d}",
        f"Errors (30d): {snapshot.errors.errors_30d}",
        f"Hard Failures (7d): {snapshot.errors.hard_failures_7d}",
        f"Hard Failures (30d): {snapshot.errors.hard_failures_30d}",
        "",
        "=== Recent Error Messages ===",
    ])

    if snapshot.errors.recent_messages:
        for i, msg in enumerate(snapshot.errors.recent_messages[:5], 1):
            ts = msg.timestamp.isoformat() if msg.timestamp else "Unknown"
            lines.append(f"{i}. [{msg.kind}] ({ts}): {msg.message[:200]}")
    else:
        lines.append("No recent errors")

    lines.extend([
        "",
        "=== Metadata ===",
        f"Enabled: {snapshot.metadata.enabled}",
        f"Deprecated: {snapshot.metadata.deprecated}",
    ])

    if snapshot.metadata.deprecation_date:
        lines.append(f"Deprecation Date: {snapshot.metadata.deprecation_date.isoformat()}")

    if snapshot.metadata.successor_model:
        lines.append(f"Successor Model: {snapshot.metadata.successor_model}")

    lines.append(f"Catalog State: {snapshot.catalog_state.value}")

    return "\n".join(lines)


DECISION_SYSTEM_PROMPT = """You are analyzing LLM model health for a benchmarking service.

Based on the provided signals, determine the appropriate action:

- **disable**: Model is permanently broken and should be disabled
  - Examples: 404 "model not found" for 48+ hours, deprecated by provider, authentication permanently fails
  - High confidence required (≥0.90)

- **monitor**: Concerning signals but not conclusive enough to disable
  - Examples: High error rate but might recover, intermittent failures, unclear error messages
  - Medium confidence (0.60-0.89)

- **ignore**: Model is healthy or issue is temporary
  - Examples: Recent successes, errors are rate limits or transient issues, low error rate
  - Any confidence level

Decision criteria:
1. Time matters: 404 errors for 48+ hours → disable; 404 for 6 hours → monitor
2. Error types matter: "model not found" is permanent; "rate limit" is temporary
3. Success pattern matters: Recent successes override old errors
4. Provider signals matter: Already marked deprecated → disable
5. Hard failures count more than soft failures

Return ONLY valid JSON (no markdown, no extra text):
{
  "action": "disable|monitor|ignore",
  "confidence": 0.95,
  "reasoning": "Clear explanation of decision in 1-2 sentences"
}
"""


async def call_llm_for_decision(context: str) -> dict:
    """
    Call OpenAI LLM to reason about model health.

    Returns parsed JSON dict with action, confidence, reasoning.
    Raises exception on API failure.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    prompt = f"""Analyze this model's health signals and determine the appropriate action:

{context}

Respond with JSON only (no markdown):
{{"action": "disable|monitor|ignore", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    request_body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"]["content"]

        # Parse JSON response (handle markdown code fences)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)


def convert_classifier_to_operator_decision(
    snapshot: LifecycleSnapshot,
    now: datetime,
    suggested_by: str = "classifier-fallback"
) -> OperatorDecision:
    """Convert classifier.py decision to OperatorDecision format."""
    decision = classify_snapshot(snapshot, now=now)

    # Map LifecycleStatus to operator actions
    action_map = {
        LifecycleStatus.LIKELY_DEPRECATED: "disable",
        LifecycleStatus.DEPRECATED: "disable",
        LifecycleStatus.DISABLED: "ignore",  # Already disabled
        LifecycleStatus.FAILING: "monitor",
        LifecycleStatus.STALE: "monitor",
        LifecycleStatus.NEVER_SUCCEEDED: "monitor",
        LifecycleStatus.MONITOR: "monitor",
        LifecycleStatus.ACTIVE: "ignore",
    }

    action = action_map.get(decision.status, "monitor")

    # Map confidence
    confidence_map = {"high": 0.90, "medium": 0.70, "low": 0.50}
    confidence = confidence_map.get(decision.confidence, 0.70)

    # Build reasoning from decision reasons
    reasoning = " ".join(decision.reasons) if decision.reasons else "Deterministic classifier decision"

    return OperatorDecision(
        provider=snapshot.provider,
        model_id=snapshot.model_id,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        suggested_at=now,
        suggested_by=suggested_by,
    )


async def generate_decision_for_snapshot(
    snapshot: LifecycleSnapshot,
    now: datetime
) -> OperatorDecision:
    """
    Generate a single decision using LLM reasoning.
    Falls back to classifier.py on failure.
    """
    try:
        context = format_snapshot_context(snapshot)
        result = await call_llm_for_decision(context)

        return OperatorDecision(
            provider=snapshot.provider,
            model_id=snapshot.model_id,
            action=result["action"],
            confidence=float(result["confidence"]),
            reasoning=result["reasoning"],
            suggested_at=now,
            suggested_by=f"ai-operator-{OPENAI_MODEL}",
        )
    except Exception as e:
        logger.warning(
            f"LLM call failed for {snapshot.provider}/{snapshot.model_id}, "
            f"using fallback: {e}"
        )
        return convert_classifier_to_operator_decision(snapshot, now)


async def generate_decisions(
    snapshots: List[LifecycleSnapshot],
    *,
    now: Optional[datetime] = None
) -> List[OperatorDecision]:
    """
    Use LLM to reason about model health signals and suggest actions.

    Falls back to classifier.py if LLM unavailable or fails.

    Args:
        snapshots: List of lifecycle snapshots to analyze
        now: Current time (defaults to utcnow)

    Returns:
        List of operator decisions with reasoning
    """
    now = now or datetime.now(timezone.utc)

    # Process all snapshots concurrently (with reasonable limit)
    decisions = []
    llm_failures = 0

    # Batch processing to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(snapshots), batch_size):
        batch = snapshots[i:i + batch_size]

        # Process batch concurrently
        tasks = [generate_decision_for_snapshot(snap, now) for snap in batch]
        batch_decisions = await asyncio.gather(*tasks, return_exceptions=True)

        for decision in batch_decisions:
            if isinstance(decision, Exception):
                logger.error(f"Unexpected error in batch: {decision}")
                llm_failures += 1
            else:
                decisions.append(decision)
                if decision.suggested_by.startswith("classifier-fallback"):
                    llm_failures += 1

    if llm_failures > 0:
        logger.info(
            f"Used fallback classifier for {llm_failures}/{len(snapshots)} models "
            f"({llm_failures/len(snapshots)*100:.1f}%)"
        )

    return decisions
