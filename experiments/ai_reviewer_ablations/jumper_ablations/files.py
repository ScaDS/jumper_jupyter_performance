"""Writing a file that other processes are reading at the same time."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomically(path: Path, payload: str) -> Path:
    """Write through a neighbouring temporary file and rename over.

    A plain write is visible to a reader while it is still happening, so a
    second process can read a prefix of it and conclude the content is wrong.
    That is not hypothetical here: four shards start within the same second,
    every one of them regenerates the shared strategies file, and every one
    of them then checks the run's snapshot against it.

    The rename is atomic as long as the temporary file is on the same
    filesystem, hence the same directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    with handle:
        handle.write(payload)
    os.replace(handle.name, path)
    return path


def create_atomically(path: Path, payload: str) -> bool:
    """Create *path* with complete content, or report that it existed.

    Returns True when this call is the one that created it.

    The obvious spelling - ``open(path, "x")`` then write - decides the
    winner correctly and still leaves a window: the file exists, and is
    empty, for as long as the write takes. Another process that finds it
    there reads a prefix and concludes the content differs from its own.
    That is not a theoretical window. It killed two shards on one run and
    two more on the next, in two different files, for the same reason.

    Linking a fully written temporary file closes it: the link either fails
    because the target exists, or succeeds with the whole content visible
    from the first instant.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    with handle:
        handle.write(payload)
    try:
        os.link(handle.name, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(handle.name)
