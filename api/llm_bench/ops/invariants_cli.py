"""Run the production invariants and record what they found.

Two subcommands, deliberately separate:

    snapshot  capture the desired set; must run on its own schedule, ahead of
              and independent from evaluation, or the denominator stops being
              something the checks cannot influence
    check     evaluate every invariant and record the run

`check` is read-only. Remediation is a separate concern with its own gates
(batch IDs, before-images, blast-radius limits) and deliberately does not ride
along with the thing that decides whether action is warranted.
"""

from __future__ import annotations

import json
import sys

import typer

from llm_bench.ops import desired_set
from llm_bench.ops import invariants
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env

app = typer.Typer(help="Production invariants over live benchmark state.")


def _db():
    _, db_name = mongo_env()
    return mongo_client()[db_name]


@app.command()
def snapshot() -> None:
    """Capture one immutable desired-set snapshot."""
    captured = desired_set.capture(_db())
    typer.echo(
        f"captured {captured.model_count} models across {len(captured.providers)} providers "
        f"at {captured.captured_at.isoformat()}"
    )


@app.command()
def check(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable results"),
    cadence_seconds: int = typer.Option(1800, "--cadence-seconds"),
    record: bool = typer.Option(True, "--record/--no-record", help="Append a check-run row"),
) -> None:
    """Evaluate every invariant. Exit 1 if anything failed or could not be evaluated."""
    results = invariants.evaluate(_db(), cadence_seconds=cadence_seconds, record=record)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "threshold_version": invariants.THRESHOLD_VERSION,
                    "results": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "evaluated": r.evaluated,
                            "error": r.error,
                            "violations": [{"subject": v.subject, "detail": v.detail} for v in r.violations[:100]],
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        for result in results:
            typer.echo(result.summary)
            for violation in result.violations[:20]:
                typer.echo(f"    {violation.subject}: {violation.detail}")
            if len(result.violations) > 20:
                typer.echo(f"    ... and {len(result.violations) - 20} more")

    # An unevaluated check exits non-zero too. A missing denominator is a state
    # that needs fixing, not a quiet pass.
    if any(not r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    app()
