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
