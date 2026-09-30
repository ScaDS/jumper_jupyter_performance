"""Hiding which preset a packet came from - and being exact about the limit.

A judge who can read `no_timing` off a folder name is scoring a label, and the
comparison between presets rests on that not happening. So under blinding a
packet is addressed by a surrogate id, and the mapping back lives only in
`judge/index.csv`, which the protocol tells the session not to open.

What blinding does **not** hide is which context sources the reviewer had.
That is not a policy choice: `sources/analyze.messages.json` is the verbatim
message the model received, and a message that was built without the timing
payload simply does not contain one. The absence is visible in the evidence
itself, and the evidence is the whole point of the packet. Removing
`enabled_sources.json` would hide nothing and would break the coverage
denominator that needs it.

So: blinding hides the preset's **name**, never its **content**. A judge can
always tell that something was withheld; what they cannot tell is which
experimental condition they are looking at, or how it is expected to score.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from jumper_ablations.evaluation.judge.layout import index_path

# Long enough that two units of one run will not collide, short enough to
# quote in a filename and a TASK.md header.
_LENGTH = 12

PREFIX = "unit-"


def packet_id_for(unit_id: str, blind: bool) -> str:
    """The id a packet is addressed by.

    Deterministic, so re-exporting a run keeps the same packet names and the
    verdicts already written against them stay valid.
    """
    if not blind:
        return unit_id
    digest = hashlib.sha256(unit_id.encode("utf-8")).hexdigest()
    return f"{PREFIX}{digest[:_LENGTH]}"


def load_packet_ids(run_directory: Path) -> dict[str, str]:
    """``{unit_id: packet_id}`` for a run, from the index it wrote.

    Read at ingest rather than recomputed, so a run exported blind and a run
    exported in the clear are both resolvable without knowing which it was.
    An absent or unreadable index means the ids were never blinded.
    """
    path = index_path(run_directory)
    if not path.is_file():
        return {}
    mapping = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            unit_id = row.get("unit_id")
            packet_id = row.get("packet_id") or unit_id
            if unit_id:
                mapping[unit_id] = packet_id
    return mapping
