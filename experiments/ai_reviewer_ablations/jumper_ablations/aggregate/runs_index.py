"""One row per run, and the fingerprint that says which rows may be pooled.

Whether two runs can go in one table is a property of what they asked, not of
when they ran: the preset definitions, the protocol, and the payloads. Those
three fields of ``meta.json`` are hashed into a fingerprint, so "which runs
are comparable" becomes a group-by over one small file instead of a walk that
opens every run and compares nested dictionaries.

Equal fingerprint means the runs asked the same question and their units may
be concatenated. Different fingerprint means they may be shown side by side
and labelled, never added together - a number averaged over two definitions
is not a number about either.

The file is derived and disposable. It is rebuilt from the runs by one
command, never appended to by a job, which is why it can be a single CSV on a
filesystem that several hundred shards are writing to.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import logging
from pathlib import Path

from jumper_ablations.files import write_atomically

logger = logging.getLogger("jumper_ablations")

INDEX_NAME = "runs_index.csv"

# The three fields that decide comparability, and nothing else. Adding a
# field here silently splits runs that used to pool; removing one silently
# pools runs that asked different questions.
FINGERPRINT_FIELDS = ("ablations", "protocol", "usecases")

INDEX_COLUMNS = (
    "run_id",
    "fingerprint",
    "created_at",
    "schema_version",
    "suite",
    "usecases",
    "ablations",
    "generations_per_target",
    "repetitions",
    "benchmark_mode",
    "shard_count",
    "records",
    "has_metrics",
    "has_units",
    "has_summary",
    "path",
)


def fingerprint_of(meta: dict) -> str:
    """A short digest of the question a run asked.

    Canonical JSON, so key order in ``meta.json`` cannot make two identical
    runs look different.
    """
    payload = {field: meta.get(field) for field in FINGERPRINT_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def describe(run_directory: Path) -> dict:
    """One index row for one run directory.

    Every field is optional on disk. A run whose first job died before
    writing ``meta.json`` still gets a row, because its records are still
    there to be scored.
    """
    meta = _meta_of(run_directory)
    protocol = meta.get("protocol") or {}
    benchmark = protocol.get("benchmark") or {}
    suite = meta.get("suite") or {}
    return {
        "run_id": str(meta.get("run_id") or run_directory.name),
        "fingerprint": fingerprint_of(meta),
        "created_at": _created_at(run_directory, meta),
        "schema_version": meta.get("schema_version", 1),
        "suite": suite.get("id", ""),
        "usecases": _joined(suite.get("usecases")),
        "ablations": _joined(suite.get("ablations")),
        "generations_per_target": protocol.get("generations_per_target", ""),
        "repetitions": protocol.get("repetitions", ""),
        "benchmark_mode": benchmark.get("mode", ""),
        "shard_count": ((meta.get("config") or {}).get("shard") or {}).get(
            "count", ""
        ),
        "records": _count_records(run_directory),
        "has_metrics": _exists(run_directory, "metrics.csv"),
        "has_units": _exists(run_directory, "metric_units.csv"),
        "has_summary": _exists(run_directory, "summary.csv"),
        "path": str(run_directory),
    }


def build(results_root: Path) -> list[dict]:
    """A row for every run under *results_root*, newest name last.

    Hydra's own output directory lives under the same root and is not a run,
    so anything without records and without a definition is left out.
    """
    rows = []
    for child in sorted(Path(results_root).glob("*")):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "records").is_dir() and not (
            child / "meta.json"
        ).is_file():
            continue
        rows.append(describe(child))
    return rows


def write(results_root: Path, rows: list[dict] | None = None) -> Path:
    """Write the index atomically, so a reader never sees half of it."""
    rows = build(results_root) if rows is None else rows
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(INDEX_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return write_atomically(Path(results_root) / INDEX_NAME, buffer.getvalue())


def groups(rows: list[dict]) -> dict:
    """``{fingerprint: [run_id, ...]}`` - the sets that may be pooled."""
    pooled: dict[str, list[str]] = {}
    for row in rows:
        pooled.setdefault(row["fingerprint"], []).append(row["run_id"])
    return pooled


def _meta_of(run_directory: Path) -> dict:
    path = run_directory / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        logger.warning(f"{path}: {failure}")
        return {}


def _created_at(run_directory: Path, meta: dict) -> str:
    """When the first job recorded itself, else the directory's own time."""
    recorded = (meta.get("machine") or {}).get("recorded_at")
    if recorded:
        return str(recorded)
    stamp = dt.datetime.fromtimestamp(run_directory.stat().st_mtime)
    return stamp.isoformat(timespec="seconds")


def _joined(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(one) for one in value)
    return ""


def _count_records(run_directory: Path) -> int:
    records = run_directory / "records"
    if not records.is_dir():
        return 0
    return sum(1 for _ in records.glob("*.json"))


def _exists(run_directory: Path, name: str) -> bool:
    return (run_directory / name).is_file()
