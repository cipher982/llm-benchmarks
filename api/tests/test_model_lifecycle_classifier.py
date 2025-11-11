from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_bench.model_lifecycle.classifier import (
    LifecycleDecision,
    LifecycleStatus,
    classify_snapshot,
)
from llm_bench.model_lifecycle.collector import (
    CatalogState,
    ErrorMessage,
    ErrorMetrics,
    LifecycleSnapshot,
    ModelMetadata,
    SuccessMetrics,
)


UTC = timezone.utc


class LifecycleClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2025, 11, 11, tzinfo=UTC)

    def _snapshot(
        self,
        *,
        successes: SuccessMetrics,
        errors: ErrorMetrics,
        catalog_state: CatalogState = CatalogState.UNKNOWN,
        metadata: ModelMetadata | None = None,
    ) -> LifecycleSnapshot:
        metadata = metadata or ModelMetadata(
            provider="anthropic",
            model_id="claude-test",
            display_name="claude-test",
        )
        return LifecycleSnapshot(
            provider=metadata.provider,
            model_id=metadata.model_id,
            metadata=metadata,
            successes=successes,
            errors=errors,
            catalog_state=catalog_state,
        )

    def test_active_when_recent_success_and_no_hard_errors(self) -> None:
        successes = SuccessMetrics(
            last_success=self.now - timedelta(days=3),
            successes_7d=4,
            successes_30d=8,
        )
        errors = ErrorMetrics()

        decision = classify_snapshot(self._snapshot(successes=successes, errors=errors), now=self.now)

        self.assertEqual(decision.status, LifecycleStatus.ACTIVE)
        self.assertEqual(decision.confidence, "high")

    def test_likely_deprecated_with_hard_failures_and_no_success(self) -> None:
        successes = SuccessMetrics()
        errors = ErrorMetrics(
            hard_failures_7d=2,
            hard_failures_30d=3,
            errors_7d=5,
            errors_30d=8,
            recent_messages=[
                ErrorMessage(
                    timestamp=self.now - timedelta(days=1),
                    message="NotFound: model disabled",
                    kind="hard",
                )
            ],
        )

        decision = classify_snapshot(
            self._snapshot(successes=successes, errors=errors, catalog_state=CatalogState.MISSING),
            now=self.now,
        )

        self.assertEqual(decision.status, LifecycleStatus.LIKELY_DEPRECATED)
        self.assertIn("mark_deprecated", decision.recommended_actions)

    def test_stale_after_sixty_days_without_success(self) -> None:
        successes = SuccessMetrics(
            last_success=self.now - timedelta(days=75),
            successes_120d=5,
        )
        errors = ErrorMetrics(errors_30d=1)

        decision = classify_snapshot(self._snapshot(successes=successes, errors=errors), now=self.now)

        self.assertEqual(decision.status, LifecycleStatus.STALE)

    def test_never_succeeded_when_no_history(self) -> None:
        successes = SuccessMetrics()
        errors = ErrorMetrics(errors_30d=2)

        decision = classify_snapshot(self._snapshot(successes=successes, errors=errors), now=self.now)

        self.assertEqual(decision.status, LifecycleStatus.NEVER_SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
