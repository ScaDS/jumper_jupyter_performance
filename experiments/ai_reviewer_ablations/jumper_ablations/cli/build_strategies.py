"""Render configs/ablation/*.yaml into strategies/strategies.yaml.

Run this after adding an ablation, or let `cli.run` do it - it regenerates the
file at the start of every run so a stale preset can never be measured.
"""
from __future__ import annotations

import argparse
import sys

from jumper_ablations.paths import STRATEGIES_FILE
from jumper_ablations.strategies import (
    available_ablation_ids,
    build_strategies_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        metavar="A,B",
        help="Only these ablations; the default writes every one of them",
    )
    args = parser.parse_args(argv)

    ids = None
    if args.ablations:
        ids = [one.strip() for one in args.ablations.split(",") if one.strip()]
        unknown = sorted(set(ids) - set(available_ablation_ids()))
        if unknown:
            print(
                f"unknown ablation(s): {', '.join(unknown)}\n"
                f"available: {', '.join(available_ablation_ids())}",
                file=sys.stderr,
            )
            return 2

    path = build_strategies_file(ids)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
