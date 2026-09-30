"""Reading a usecase notebook: where the prefix ends and the target begins.

Nothing here records cell indices in a file. A notebook says what it is by its
own contents - the code cell at the end is the payload, and the one review
magic in it, if there is one, marks where the measurement happens - so
reordering or inserting cells cannot leave a manifest quietly pointing at the
wrong thing.

A usecase notebook is self-sufficient: it loads the extension, starts the
monitor and invokes the review itself, so a person can open it and run it top
to bottom without the harness. The harness adds nothing to it - it appends the
preset to the review line the notebook already carries, and nothing else. That
is what keeps "what the experiment measured" and "what the notebook does" the
same thing.

`python -m jumper_ablations.cli.usecases --prepare` writes the two missing
cells into a notebook that does not have them yet.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import nbformat

REVIEW_MAGIC = "%perfmonitor_ai_review"

# What the harness prepends to a notebook that does not set JUmPER up itself.
SETUP_SOURCE = "%load_ext jumper_extension\n%perfmonitor_fast_setup"
_SETUP_MARKER = "%load_ext jumper_extension"

# Only an uncommented magic counts. A usecase notebook may keep commented-out
# review lines around as documentation of alternative targets, and treating
# one of those as the target would benchmark the wrong cell.
_REVIEW_LINE = re.compile(
    rf"^\s*{re.escape(REVIEW_MAGIC)}\b.*$",
    re.MULTILINE,
)

# A cell that is nothing but magics and comments sets the session up; it is
# not a payload, and it is not what the review is pointed at.
_MAGIC_OR_COMMENT = re.compile(r"^\s*(?:[%!]|#|$)")


@dataclasses.dataclass(frozen=True)
class NotebookLayout:
    """What a usecase notebook is made of, in cell positions.

    ``prefix_indices`` is everything the payload needs to have run first,
    which is also what the benchmark replays for every measurement.
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

    Normalising fills in the cell ids newer nbformat versions require;
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
    """The single cell invoking the review magic.

    One notebook is one experiment, so both none and several are mistakes in
    the usecase rather than something for the harness to paper over.
    """
    found = [
        index
        for index, cell in enumerate(notebook.cells)
        if _is_code(cell) and _REVIEW_LINE.search(cell.get("source", ""))
    ]
    if not found:
        raise ValueError(
            f"no uncommented '{REVIEW_MAGIC}' cell: a usecase notebook has "
            "to invoke the command it is measured on, so that it can also be "
            "run by hand. Add one, or run "
            "`python -m jumper_ablations.cli.usecases --prepare`"
        )
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} '{REVIEW_MAGIC}' cells at positions {found}: one "
            "notebook is one experiment, so split them into separate usecases"
        )
    return found[0]


def find_payload_cell(
    notebook: nbformat.NotebookNode,
    before: int | None = None,
) -> int:
    """The last real code cell before *before* - the cell under review."""
    limit = len(notebook.cells) if before is None else before
    for index in range(limit - 1, -1, -1):
        if _is_payload(notebook.cells[index]):
            return index
    raise ValueError("no code cell to review: the notebook has no payload")


def needs_setup(notebook: nbformat.NotebookNode) -> bool:
    """Whether the notebook leaves loading the extension to the harness."""
    return not any(
        _SETUP_MARKER in cell.get("source", "")
        for cell in notebook.cells
        if _is_code(cell)
    )


def read_layout(path: Path) -> NotebookLayout:
    """Locate the target, the review invocation and the prefix in *path*."""
    notebook = read_notebook(path)
    if needs_setup(notebook):
        raise ValueError(
            "the notebook never loads the extension, so nothing would be "
            "monitored and the review would have no context to read. Add a "
            f"cell with:\n\n{SETUP_SOURCE}\n\nor run "
            "`python -m jumper_ablations.cli.usecases --prepare`"
        )

    review_index = find_review_cell(notebook)
    payload_index = find_payload_cell(notebook, review_index)
    match = _REVIEW_LINE.search(notebook.cells[review_index]["source"])

    return NotebookLayout(
        prefix_indices=tuple(range(payload_index)),
        payload_index=payload_index,
        review_index=review_index,
        review_line=match.group(0).strip(),
    )


def cell_source(notebook: nbformat.NotebookNode, index: int) -> str:
    return notebook.cells[index].get("source", "")


def payload_type_of(notebook: nbformat.NotebookNode) -> str:
    """The `#### Payload type: ...` line a usecase header carries, if any.

    A convention rather than a requirement: it is what the report groups
    usecases by, and reading it from the notebook saves stating it twice.
    """
    for cell in notebook.cells:
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", "").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("#### payload type:"):
                return stripped.split(":", 1)[1].strip()
    return ""


def title_of(notebook: nbformat.NotebookNode) -> str:
    """The notebook's first heading, used when no manifest names one."""
    for cell in notebook.cells:
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", "").splitlines():
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
    return ""
