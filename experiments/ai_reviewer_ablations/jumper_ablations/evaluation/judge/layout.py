"""Where judge packets and verdicts live inside a run directory.

One place, so `export`, `ingest` and JUDGE_PROTOCOL.md cannot drift apart
about which file an agent session is supposed to write.
"""
from __future__ import annotations

from pathlib import Path

TASKS_DIRNAME = "tasks"
VERDICTS_DIRNAME = "verdicts"
INDEX_NAME = "index.csv"
MISSING_NAME = "missing.csv"


def judge_root(run_directory: Path) -> Path:
    return Path(run_directory) / "judge"


def tasks_root(run_directory: Path) -> Path:
    return judge_root(run_directory) / TASKS_DIRNAME


def verdicts_root(run_directory: Path) -> Path:
    return judge_root(run_directory) / VERDICTS_DIRNAME


def packet_directory(run_directory: Path, rubric: str, unit_id: str) -> Path:
    return tasks_root(run_directory) / rubric / unit_id


def verdict_path(run_directory: Path, rubric: str, unit_id: str) -> Path:
    return verdicts_root(run_directory) / rubric / f"{unit_id}.json"


def index_path(run_directory: Path) -> Path:
    return judge_root(run_directory) / INDEX_NAME


def missing_path(run_directory: Path) -> Path:
    return judge_root(run_directory) / MISSING_NAME
