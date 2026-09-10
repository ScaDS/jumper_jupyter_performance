"""Where records live on disk, and how they get back into memory.

One JSON file per record, because a record holds the whole prompt payload and
nesting that into a CSV would destroy it. Beside them, a `runs.jsonl` index and
a flat `runs.csv` - the index is what the later phases walk, the CSV is what a
person opens first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

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
        """Write one record and append it to the index."""
        self.ensure()
        path = self.path_for(record.identity.record_id)
        path.write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        with self.index_path.open("a", encoding="utf-8") as index:
            index.write(json.dumps(_index_entry(record)) + "\n")
        return path

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.load())

    def load(self) -> list[RunRecord]:
        """Every record in the run, in file-name order."""
        return list(load_records(self.run_directory))

    def write_flat_view(self) -> Path:
        """Rewrite `runs.csv` from the records currently on disk."""
        import pandas as pd

        rows = [_flat_row(record) for record in self.load()]
        frame = pd.DataFrame(rows, columns=list(_FLAT_COLUMNS))
        path = self.run_directory / FLAT_NAME
        frame.to_csv(path, index=False)
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
