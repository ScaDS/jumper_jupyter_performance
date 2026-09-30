"""List the usecases, and make a dropped-in notebook self-sufficient.

    python -m jumper_ablations.cli.usecases
    python -m jumper_ablations.cli.usecases --prepare

A usecase notebook has to run on its own: open it, run it top to bottom, and
it loads the extension, starts the monitor and asks for the review. The
harness then runs the same sequence and only appends the preset to the review
line. `--prepare` writes the two cells a notebook is missing so that it does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

from jumper_ablations.paths import USECASES_DIR
from jumper_ablations.usecases.notebook import (
    REVIEW_MAGIC,
    SETUP_SOURCE,
    needs_setup,
    read_notebook,
)
from jumper_ablations.usecases.registry import (
    NOTEBOOK_SUFFIX,
    discover_usecases,
)

DEFAULT_REVIEW = f"{REVIEW_MAGIC} --benchmark --replay-mode full"


def _missing(notebook) -> list[str]:
    absent = []
    if needs_setup(notebook):
        absent.append("setup")
    sources = [cell.get("source", "") for cell in notebook.cells]
    if not any(REVIEW_MAGIC in source for source in sources):
        absent.append("review")
    return absent


def prepare(path: Path) -> list[str]:
    """Add the cells *path* is missing, and say which were added."""
    notebook = read_notebook(path)
    absent = _missing(notebook)
    if not absent:
        return []

    if "setup" in absent:
        # After the title and before any code: the monitor has to be running
        # before the first cell it is meant to have measured.
        position = 1 if notebook.cells[0]["cell_type"] == "markdown" else 0
        notebook.cells.insert(
            position, nbformat.v4.new_code_cell(SETUP_SOURCE)
        )
    if "review" in absent:
        notebook.cells.append(nbformat.v4.new_code_cell(DEFAULT_REVIEW))

    _, notebook = nbformat.validator.normalize(notebook)
    nbformat.write(notebook, str(path))
    return absent


def _report(root: Path) -> int:
    """List every usecase, and every notebook that is not yet one."""
    usecases = discover_usecases(root)
    for identifier, usecase in usecases.items():
        facts = len(usecase.manifest.reference_facts)
        note = "" if facts else "  (no reference facts: coverage will abstain)"
        target = usecase.layout.payload_index
        print(
            f"  {identifier:48s} payload cell {target:3d}"
            f"  facts {facts}{note}"
        )

    unprepared = []
    for path in sorted(root.rglob(f"*{NOTEBOOK_SUFFIX}")):
        identifier = path.relative_to(root).with_suffix("").as_posix()
        if identifier in usecases:
            continue
        unprepared.append((identifier, _missing(read_notebook(path))))

    if unprepared:
        print("\nnot usable yet:")
        for identifier, absent in unprepared:
            print(
                f"  {identifier:48s} missing: {', '.join(absent) or 'unknown'}"
            )
        print("\n  run with --prepare to add the missing cells")
    return 1 if unprepared else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=USECASES_DIR,
        help="Where usecase notebooks live",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Write the setup and review cells into notebooks missing them",
    )
    args = parser.parse_args(argv)

    if args.prepare:
        for path in sorted(args.root.rglob(f"*{NOTEBOOK_SUFFIX}")):
            added = prepare(path)
            if added:
                identifier = path.relative_to(args.root).with_suffix("")
                print(
                    f"  {identifier.as_posix():48s} added: {', '.join(added)}"
                )

    print(f"usecases under {args.root}:")
    return _report(args.root)


if __name__ == "__main__":
    sys.exit(main())
