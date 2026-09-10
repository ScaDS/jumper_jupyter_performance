"""Reading a usecase notebook: where the prefix ends and the target begins.

Nothing here records cell indices in a file. A notebook says what it is by its
own contents - the one review magic in it marks the target, and the code cell
before that magic is the payload - so reordering or inserting cells cannot
leave a manifest quietly pointing at the wrong thing.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import nbformat

REVIEW_MAGIC = "%perfmonitor_ai_review"

# Only an uncommented magic counts. The usecase notebooks keep commented-out
# review lines around as documentation of alternative targets, and treating one
# of those as the target would benchmark the wrong cell.
_REVIEW_LINE = re.compile(rf"^\s*{re.escape(REVIEW_MAGIC)}\b.*$", re.MULTILINE)

# A cell that is nothing but magics and comments sets the session up; it is not
# a payload, and it is not what the review is pointed at.
_MAGIC_OR_COMMENT = re.compile(r"^\s*(?:[%!]|#|$)")


@dataclasses.dataclass(frozen=True)
class NotebookLayout:
    """What a usecase notebook is made of, in cell positions.

    ``prefix_indices`` is everything the payload needs to have run first, which
    is also what the benchmark replays for every measurement.
    """

    prefix_indices: tuple[int, ...]
    payload_index: int
    review_index: int
    review_line: str

    @property
    def executable_indices(self) -> tuple[int, ...]:
        """Prefix plus payload: the notebook up to and including the target."""
        return (*self.prefix_indices, self.payload_index)


def read_notebook(path: Path) -> nbformat.NotebookNode:
    """Read a usecase notebook, normalised.

    Normalising fills in the cell ids that newer nbformat versions require;
    without it every read of a hand-written notebook warns.
    """
    notebook = nbformat.read(str(path), as_version=4)
    _, notebook = nbformat.validator.normalize(notebook)
    return notebook


def _is_code(cell) -> bool:
    return cell.get("cell_type") == "code"


def _is_payload(cell) -> bool:
    """A code cell with at least one line that is neither magic nor comment."""
    if not _is_code(cell):
        return False
    return any(
        not _MAGIC_OR_COMMENT.match(line)
        for line in cell.get("source", "").splitlines()
    )


def find_review_cell(notebook: nbformat.NotebookNode) -> int:
    """The index of the single cell invoking the review magic.

    One notebook is one experiment, so more than one review cell is a mistake
    in the usecase rather than something to pick between.
    """
    found = [
        index
        for index, cell in enumerate(notebook.cells)
        if _is_code(cell) and _REVIEW_LINE.search(cell.get("source", ""))
    ]
    if not found:
        raise ValueError(
            f"no uncommented '{REVIEW_MAGIC}' cell: a usecase notebook has to "
            "invoke the command it is meant to measure"
        )
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} '{REVIEW_MAGIC}' cells at positions {found}: one "
            "notebook is one experiment, so split them into separate usecases"
        )
    return found[0]


def find_payload_cell(notebook: nbformat.NotebookNode, review_index: int) -> int:
    """The last real code cell before the review - the cell under review."""
    for index in range(review_index - 1, -1, -1):
        if _is_payload(notebook.cells[index]):
            return index
    raise ValueError(
        "no code cell before the review magic: there is nothing to review"
    )


def read_layout(path: Path) -> NotebookLayout:
    """Locate the target, the review invocation and the prefix in *path*."""
    notebook = read_notebook(path)
    review_index = find_review_cell(notebook)
    payload_index = find_payload_cell(notebook, review_index)
    review_source = notebook.cells[review_index].get("source", "")
    match = _REVIEW_LINE.search(review_source)
    return NotebookLayout(
        prefix_indices=tuple(range(payload_index)),
        payload_index=payload_index,
        review_index=review_index,
        review_line=match.group(0).strip(),
    )


def cell_source(notebook: nbformat.NotebookNode, index: int) -> str:
    return notebook.cells[index].get("source", "")
