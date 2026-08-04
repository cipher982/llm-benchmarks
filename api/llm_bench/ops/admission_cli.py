"""Drive model admission: register candidates, probe them, promote what works."""

from __future__ import annotations

import typer

from llm_bench.ops import admission
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env

app = typer.Typer(help="Admit models to the site by measuring them.")


def _db():
    _, db_name = mongo_env()
    return mongo_client()[db_name]


@app.command()
def candidates(limit: int = typer.Option(25, "--limit")) -> None:
    """Show what would be probed next, cheapest possible look."""
    db = _db()
    rows = admission.find_candidates(db, limit=limit)
    typer.echo(f"{len(rows)} candidate(s), most-served first:")
    for row in rows:
        typer.echo(f"  {row['provider']:<11} {row['model_id']}")


@app.command()
def run(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Dry run by default; probing costs money"),
    limit: int = typer.Option(admission.MAX_NEW_CANDIDATES_PER_RUN, "--limit"),
) -> None:
    """One admission pass."""
    db = _db()
    if dry_run:
        rows = admission.find_candidates(db, limit=limit)
        probing = db[admission.models_collection_name()].count_documents({"status": admission.CANDIDATE_STATUS})
        typer.echo(f"DRY RUN — would register {len(rows)} candidate(s); {probing} already probing")
        for row in rows[:20]:
            typer.echo(f"  + {row['provider']:<11} {row['model_id']}")
        promoted, rejected = admission.evaluate_candidates(db)
        typer.echo(f"(evaluation is read-mostly and ran: promoted={promoted}, rejected={len(rejected)})")
        return

    report = admission.run_admission_pass(db)
    typer.echo(report.summary())
    for subject in report.promoted:
        typer.echo(f"  PROMOTED {subject}")
    for subject, reason in report.rejected:
        typer.echo(f"  REJECTED {subject}: {reason}")


@app.command()
def status() -> None:
    """Where every candidate stands."""
    db = _db()
    coll = db[admission.models_collection_name()]
    for state in (admission.CANDIDATE_STATUS, admission.PROMOTED_STATUS, admission.REJECTED_STATUS):
        typer.echo(f"{state:<12} {coll.count_documents({'status': state})}")
    typer.echo(f"probe samples  {db[admission.probe_metrics_collection_name()].estimated_document_count()}")


if __name__ == "__main__":
    app()
