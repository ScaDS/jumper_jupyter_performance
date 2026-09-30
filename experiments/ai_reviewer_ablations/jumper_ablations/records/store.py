"""Where records live on disk, and how they get back into memory.

One JSON file per record, because a record holds the whole prompt payload and
nesting that into a CSV would destroy it. Beside them, a `runs.jsonl` index and
a flat `runs.csv`. Both are derived from the records and rewritten whole,
never appended to: several shards write this directory at once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from jumper_ablations.files import write_atomically
from jumper_ablations.records.schema import RunRecord

RECORDS_DIRNAME = "records"
INDEX_NAME = "runs.jsonl"
FLAT_NAME = "runs.csv"

# The flat view: one row per record, only the columns that survive being
# scalars. Anything nested stays in the JSON.
_FLAT_COLUMNS = (
    "record_id",
    "usecase",
    "ablation",
    "ablation_family",
    "repetition",
    "generation",
    "phase",
    "reviewer_run_id",
    "requested_replay_mode",
    "actual_replay_mode",
    "degraded",
    "suggestions",
    "baseline_duration_s",
    "best_speedup",
    "llm_latency_s",
    "total_tokens",
    "command_wall_s",
    "created_at",
)


class RecordStore:
    """Append-only store for one run directory."""

    def __init__(self, run_directory: Path):
        self.run_directory = Path(run_directory)
        self.records_directory = self.run_directory / RECORDS_DIRNAME
        self.index_path = self.run_directory / INDEX_NAME

    def ensure(self) -> None:
        self.records_directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, record_id: str) -> Path:
        return self.records_directory / f"{record_id}.json"

    def write(self, record: RunRecord) -> Path:
        """Write one record.

        Atomically, because it is written while a monitor is reading the
        directory and while other shards are writing their own: a reader
        should see a record or not see it, never half of one.

        The flat views are not appended to here. Several shards write this
        directory at once, and an append to one shared index from several
        processes is not atomic - the same mistake that cost this experiment
        two shards on one run. They are regenerated from the records
        instead, which is also the only version that can be trusted after a
        pass has been re-run.
        """
        self.ensure()
        path = self.path_for(record.identity.record_id)
        write_atomically(path, record.model_dump_json(indent=2) + "\n")
        return path

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.load())

    def load(self) -> list[RunRecord]:
        """Every record in the run, in file-name order."""
        return list(load_records(self.run_directory))

    def write_flat_view(self) -> Path:
        """Rewrite the flat views from the records currently on disk.

        Derived, never appended to, so that a shard finishing at the same
        moment as another produces the same file rather than half of two.
        """
        import pandas as pd

        records = self.load()
        rows = [_flat_row(record) for record in records]
        frame = pd.DataFrame(rows, columns=list(_FLAT_COLUMNS))
        path = self.run_directory / FLAT_NAME
        write_atomically(path, frame.to_csv(index=False))
        write_atomically(
            self.index_path,
            "".join(
                json.dumps(_index_entry(record)) + "\n" for record in records
            ),
        )
        return path


def load_records(run_directory: Path) -> Iterable[RunRecord]:
    """Read every record JSON under *run_directory*."""
    directory = Path(run_directory) / RECORDS_DIRNAME
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(RunRecord.model_validate_json(path.read_text("utf-8")))
    return records


def _index_entry(record: RunRecord) -> dict:
    identity = record.identity
    return {
        "record_id": identity.record_id,
        "usecase": identity.usecase,
        "ablation": identity.ablation,
        "generation": identity.generation,
        "repetition": identity.repetition,
        "phase": identity.phase,
        "created_at": identity.created_at,
    }


def _flat_row(record: RunRecord) -> dict:
    identity = record.identity
    baseline = record.outputs.baseline()
    speedups = [
        verdict.speedup
        for verdict in record.outputs.verdicts().values()
        if verdict.speedup is not None
    ]
    return {
        "record_id": identity.record_id,
        "usecase": identity.usecase,
        "ablation": identity.ablation,
        "ablation_family": identity.ablation_family,
        "repetition": identity.repetition,
        "generation": identity.generation,
        "phase": identity.phase,
        "reviewer_run_id": identity.reviewer_run_id,
        "requested_replay_mode": identity.requested_replay_mode,
        "actual_replay_mode": record.environment.actual_replay_mode,
        "degraded": record.environment.degraded,
        "suggestions": len(record.outputs.suggestions),
        "baseline_duration_s": baseline.duration_s if baseline else None,
        "best_speedup": max(speedups) if speedups else None,
        "llm_latency_s": record.cost.llm_latency_s,
        "total_tokens": record.cost.total_tokens,
        "command_wall_s": record.cost.command_wall_s,
        "created_at": identity.created_at,
    }
