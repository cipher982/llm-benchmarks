"""Every ops module must be reachable from something that runs.

Three times in one session I found working, tested code that nothing called:

  - `llm_error_classifier` — the only invocation in the repository was the
    example in its own docstring, so `unknown` was a terminal state and 946
    error fingerprints accumulated in it.
  - `reconciler` — identity resolution, group consolidation and display-name
    unification, the passes that take split chart lines and rejoin them. Run by
    hand once and never again.
  - the dashboard review queue — a UI fed by a job, consumed by nobody, with 28
    models waiting in it.

Unit tests cannot see this. Each module passed its own tests the whole time; the
defect is the absence of an edge, not a bug inside a node. The symptom is always
silence, which is the failure mode this codebase is worst at noticing.
"""

from __future__ import annotations

import ast
import pathlib

OPS = pathlib.Path(__file__).resolve().parents[1] / "llm_bench" / "ops"
ROOT = pathlib.Path(__file__).resolve().parents[1] / "llm_bench"

# Modules that are entry points themselves — a human or a scheduler invokes
# them directly, so nothing in-process imports them.
ENTRY_POINTS = {
    "admission_cli",
    "invariants_cli",
    "reliability_cli",
    "catalog_quarantine",
    # Run on a schedule from outside the process, by a human or a cron.
    "checkback",
}


def _imported_names(path: pathlib.Path) -> set[str]:
    """Every module name this file imports, however it spells the import."""
    names: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.rsplit(".", 1)[-1])
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return names


def test_every_ops_module_is_reachable_from_something_that_runs():
    """Reachability is transitive.

    `mutations` is imported only by `reconciler` and `admission`, never from
    outside ops, and it is perfectly well wired — because those two are. What
    matters is whether a path exists from something that actually executes, not
    whether the import crosses a directory boundary.
    """
    modules = {p.stem for p in OPS.glob("*.py") if p.stem != "__init__"}

    imports: dict[str, set[str]] = {}
    roots: set[str] = set()
    for path in ROOT.rglob("*.py"):
        names = _imported_names(path)
        if path.parent == OPS:
            imports[path.stem] = names
        else:
            # Anything outside ops is running code — a scheduler loop, a
            # provider adapter, a CLI. What it imports is reachable.
            roots |= names & modules

    reachable = set(roots) | ENTRY_POINTS
    frontier = list(reachable)
    while frontier:
        current = frontier.pop()
        for name in imports.get(current, set()) & modules:
            if name not in reachable:
                reachable.add(name)
                frontier.append(name)

    orphaned = sorted(modules - reachable)
    assert not orphaned, (
        f"ops modules no running code can reach: {orphaned}. "
        "Wire it into a loop or a CLI, or add it to ENTRY_POINTS with a reason. "
        "Code that runs nowhere passes every test it has and does nothing."
    )
