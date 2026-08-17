"""Report or apply derived display names.

python -m llm_bench.ops.model_naming_cli            # audit only
python -m llm_bench.ops.model_naming_cli --apply    # write, reversibly
"""

from __future__ import annotations

import argparse
import json
import os

from pymongo import MongoClient

from llm_bench.ops import model_naming


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the labels (default is a report)")
    parser.add_argument("--show", type=int, default=25, help="how many proposals to print")
    args = parser.parse_args()

    client = MongoClient(os.environ["MONGODB_URI"])
    try:
        db = client[os.getenv("MONGODB_DB", "llm-bench")]
        for proposal in model_naming.plan(db)[: args.show]:
            flag = "*" if proposal.changed else " "
            current = repr(proposal.current)
            label = repr(proposal.label)
            print(f" {flag} {proposal.provider:11s} {current:46s} -> {label:38s} [{proposal.source}]")
        print()
        print(json.dumps(model_naming.apply_names(db, apply=args.apply), indent=2, default=str))
    finally:
        client.close()


if __name__ == "__main__":
    main()
