"""The directory one run writes into, and what else lands beside the records.

Hydra is pointed at this same directory, so ``.hydra/config.yaml`` inside it is
the fully composed setup that produced everything else in it. A result and the
configuration that caused it are therefore never separated, which is the whole
reason the suite is declared in YAML rather than assembled on a command line.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

from jumper_ablations.paths import RESULTS_DIR, STRATEGIES_FILE

META_NAME = "meta.json"
PASSES_NAME = "passes.jsonl"


@dataclasses.dataclass(frozen=True)
class RunDirectory:
    """Every path a run needs, derived from one root."""

    path: Path

    @classmethod
    def create(cls, path: Path) -> "RunDirectory":
        directory = cls(Path(path))
        directory.path.mkdir(parents=True, exist_ok=True)
        directory.records.mkdir(parents=True, exist_ok=True)
        directory.passes.mkdir(parents=True, exist_ok=True)
        return directory

    @classmethod
    def latest(cls, results_root: Path = RESULTS_DIR) -> "RunDirectory":
        """The newest run under *results_root*.

        What the report notebook opens when it is not told otherwise.
        """
        candidates = sorted(
            child for child in Path(results_root).glob("*") if child.is_dir()
        )
        if not candidates:
            raise FileNotFoundError(f"no runs under {results_root}")
        return cls(candidates[-1])

    @property
    def records(self) -> Path:
        return self.path / "records"

    @property
    def passes(self) -> Path:
        return self.path / "passes"

    @property
    def judge(self) -> Path:
        return self.path / "judge"

    @property
    def meta_path(self) -> Path:
        return self.path / META_NAME

    @property
    def passes_index(self) -> Path:
        return self.path / PASSES_NAME

    def pass_directory(
        self,
        usecase: str,
        ablation: str,
        repetition: int,
    ) -> Path:
        name = f"{usecase.replace('/', '-')}__{ablation}__r{repetition:02d}"
        directory = self.passes / name
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        (directory / "tmp").mkdir(parents=True, exist_ok=True)
        return directory

    def snapshot_strategies(self, source: Path = STRATEGIES_FILE) -> Path:
        """Copy the presets in beside the records they produced.

        The file in the experiment folder is regenerated on every run; this
        copy is the one that says what `--strategy base` meant on that day.
        """
        destination = self.path / "strategies.yaml"
        shutil.copyfile(source, destination)
        return destination

    def write_meta(self, meta: dict) -> Path:
        self.meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return self.meta_path

    def append_pass(self, entry: dict) -> None:
        with self.passes_index.open("a", encoding="utf-8") as index:
            index.write(json.dumps(entry) + "\n")


def machine() -> dict:
    """Enough about the host to tell two runs apart after the fact."""
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
