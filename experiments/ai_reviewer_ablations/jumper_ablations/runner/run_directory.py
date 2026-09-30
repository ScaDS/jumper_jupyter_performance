"""The directory one run writes into, and what else lands beside the records.

Hydra is pointed at this same directory, so ``.hydra/config.yaml`` inside it is
the fully composed setup that produced everything else in it. A result and the
configuration that caused it are therefore never separated, which is the whole
reason the suite is declared in YAML rather than assembled on a command line.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from jumper_ablations.files import write_atomically
from jumper_ablations.paths import RESULTS_DIR, STRATEGIES_FILE

META_NAME = "meta.json"
PASSES_NAME = "passes.jsonl"
STRATEGIES_NAME = "strategies.yaml"


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

    @property
    def strategies_snapshot(self) -> Path:
        return self.path / STRATEGIES_NAME

    def snapshot_strategies(self, source: Path = STRATEGIES_FILE) -> Path:
        """Fix the presets this run measures, and hand back the copy.

        The kernels read this copy rather than the file in the experiment
        folder, because that one is regenerated on every run and editable at
        any time: a preset changed while a sweep is running would otherwise
        silently redefine what the remaining passes measure.

        Written once. A later invocation - a resume, or another shard - gets
        the copy that is already there, and is refused if what it would have
        written differs, because half the records would then have been
        produced under presets the other half never saw.
        """
        destination = self.strategies_snapshot
        wanted = Path(source).read_text(encoding="utf-8")
        try:
            with open(destination, "x", encoding="utf-8") as handle:
                handle.write(wanted)
            return destination
        except FileExistsError:
            pass
        if destination.read_text(encoding="utf-8") != wanted:
            raise SystemExit(
                f"{self.path.name} was measured with different strategy "
                f"definitions than {source} now holds. Rebuild is not a "
                "resume: use a new run_id, or restore the presets."
            )
        return destination

    def write_meta(self, meta: dict) -> Path:
        # Through a neighbouring temporary file, so a reader never sees a
        # half-written definition: os.replace is atomic within a filesystem.
        write_atomically(
            self.meta_path,
            json.dumps(meta, indent=2, sort_keys=False) + "\n",
        )
        return self.meta_path

    @property
    def invocations(self) -> Path:
        return self.path / "invocations"

    def record_invocation(self, entry: dict, shard: str = "00-of-01") -> None:
        """Note another invocation against an existing run.

        The definition written by the first invocation stays as it is - it
        describes what the stored records were produced under. Everything
        since goes in a file of its own, named for the shard that wrote it:
        several jobs record themselves at the same moment, and appending to
        one shared file from several nodes is not atomic on a parallel
        filesystem.
        """
        self.invocations.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        write_atomically(
            self.invocations / f"{shard}-{stamp}.json",
            json.dumps(entry, indent=2) + "\n",
        )

    def invocation_entries(self) -> list[dict]:
        """Every invocation that touched this run, oldest first."""
        if not self.invocations.is_dir():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.invocations.glob("*.json"))
        ]

    def meta(self) -> dict:
        """What this run recorded about itself, or {} before it wrote any."""
        if not self.meta_path.is_file():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def claim_meta(self, snapshot: dict) -> dict:
        """Write the run's definition if nobody has, and return what stands.

        Shards start together, so "read it, and write it if it is missing"
        has a window in which all of them find it missing. The exclusive
        create closes that: exactly one shard writes, and every other gets
        back what was written and can check itself against it. The return
        value is therefore the definition of record, never the caller's own.
        """
        payload = json.dumps(snapshot, indent=2, sort_keys=False) + "\n"
        try:
            with open(self.meta_path, "x", encoding="utf-8") as handle:
                handle.write(payload)
            return dict(snapshot)
        except FileExistsError:
            return self.meta()

    def pass_index(self, shard: str = "00-of-01") -> Path:
        """The index this shard appends to.

        One writer per file. Several jobs appending to a single index over a
        network filesystem is not atomic, and a torn line is worse than a
        missing one: it is what decides which passes a resume skips.
        """
        return self.path / f"passes-{shard}.jsonl"

    def append_pass(self, entry: dict, shard: str = "00-of-01") -> None:
        with self.pass_index(shard).open("a", encoding="utf-8") as index:
            index.write(json.dumps(entry) + "\n")

    def pass_entries(self) -> list[dict]:
        """Every recorded pass, from whichever shard recorded it.

        Read together on purpose: a resume must skip what any shard finished,
        so the shards of one attempt need not be the shards of the next.
        """
        entries = []
        indexes = sorted(self.path.glob("passes-*.jsonl"))
        if self.passes_index.is_file():
            # Runs written before the index was split per shard.
            indexes.insert(0, self.passes_index)
        for index in indexes:
            for line in index.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        return entries


def machine() -> dict:
    """Enough about the host to tell two runs apart after the fact.

    The batch job id is recorded because on disk a shard that has not
    started and a shard that died before writing anything are the same
    thing - no index line, no records - and a reader can only tell them
    apart by asking the queue about a job it was told the number of.
    """
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
