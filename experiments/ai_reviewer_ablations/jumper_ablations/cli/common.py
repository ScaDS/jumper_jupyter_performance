"""What the offline commands all have to do first.

Resolving which run to read, loading its records, and rebuilding the contexts
the metrics see. Shared so `export_judge`, `evaluate` and `report` cannot
disagree about which run they are talking about.
"""
from __future__ import annotations

import logging
from pathlib import Path

from jumper_ablations.config.schema import ExperimentConfig
from jumper_ablations.metrics.context import build_cell_contexts, build_run_contexts
from jumper_ablations.records.schema import PHASE_REVIEW
from jumper_ablations.records.store import load_records
from jumper_ablations.runner.run_directory import RunDirectory
from jumper_ablations.usecases.registry import discover_usecases

logger = logging.getLogger("jumper_ablations")


def resolve_run(config: ExperimentConfig) -> RunDirectory:
    """The run to read: the configured one, or the newest.

    A run id is accepted as well as a path, because the id is what the run
    command prints and what a person remembers.
    """
    results_root = Path(config.results_root)
    if not config.target_run:
        return RunDirectory.latest(results_root)

    candidate = Path(config.target_run)
    if candidate.is_dir():
        return RunDirectory(candidate)
    named = results_root / config.target_run
    if named.is_dir():
        return RunDirectory(named)
    raise FileNotFoundError(
        f"no run '{config.target_run}' - looked at {candidate} and {named}"
    )


def load_contexts(config: ExperimentConfig, run: RunDirectory):
    """Records, plus the two views the metrics are handed.

    Judged metrics are scored per reviewer invocation and only the review
    records carry an analysis, so the run-scope view is restricted to those:
    exporting a packet for a benchmark record would ask a session to judge the
    same analysis twice.
    """
    records = list(load_records(run.path))
    if not records:
        raise FileNotFoundError(f"{run.path} holds no records")

    usecases = discover_usecases(Path(config.usecases_root))
    reviews = [
        record for record in records if record.identity.phase == PHASE_REVIEW
    ]
    return (
        records,
        build_run_contexts(reviews, usecases),
        build_cell_contexts(records, usecases),
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
