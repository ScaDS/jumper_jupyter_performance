"""Fetch the CodeNet submissions every ECCO usecase is ranked against.

    pip install pyarrow
    python scripts/fetch_codenet_submissions.py [--cache DIR]

ECCO places a program "on the spectrum of all user-submitted programs to that
problem". This writes that spectrum for each ``usecases/ecco/*.yaml``: every
accepted CPython 3 submission to the manifest's ``dataset.problem``, from the
Hugging Face mirror of Project CodeNet (iNeil77/CodeNet, Python config), one
file per submission in the directory the manifest's
``verification.submissions`` names. Exact duplicates are dropped, as ECCO
drops repetitive programs; PyPy and Python 2 submissions are left out because
the usecase runs on CPython 3.

Idempotent: a submission already on disk is not written again.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import yaml

EXPERIMENT = Path(__file__).resolve().parent.parent
MIRROR = "https://huggingface.co/datasets/iNeil77/CodeNet/resolve/main/Python"
SHARDS = [f"train-0000{index}-of-00004.parquet" for index in range(4)]
CPYTHON_3 = re.compile(r"^Python ?(\(3|3)")


def _shards(cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SHARDS:
        path = cache / name
        if not path.is_file():
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{MIRROR}/{name}", path)
        paths.append(path)
    return paths


def _targets() -> dict[str, Path]:
    targets = {}
    for manifest_path in sorted((EXPERIMENT / "usecases" / "ecco").glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        problem = manifest["dataset"]["problem"]
        targets[problem] = manifest_path.parent / manifest["verification"]["submissions"]
    return targets


def main() -> None:
    import pyarrow.parquet as parquet

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=EXPERIMENT / "workspace" / "codenet")
    arguments = parser.parse_args()

    targets = _targets()
    seen: dict[str, set[str]] = {problem: set() for problem in targets}
    written = dict.fromkeys(targets, 0)
    for shard in _shards(arguments.cache):
        table = parquet.read_table(
            shard,
            columns=["s_id", "p_id", "original_language", "status", "code"],
            filters=[("p_id", "in", list(targets)), ("status", "=", "Accepted")],
        )
        for row in table.to_pylist():
            if not CPYTHON_3.match(row["original_language"] or ""):
                continue
            code = row["code"] or ""
            normalised = "\n".join(line.rstrip() for line in code.strip().splitlines())
            if normalised in seen[row["p_id"]]:
                continue
            seen[row["p_id"]].add(normalised)
            directory = targets[row["p_id"]]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{row['s_id']}.py"
            if not path.is_file():
                path.write_text(code, encoding="utf-8")
            written[row["p_id"]] += 1

    for problem, count in written.items():
        print(f"{problem}: {count} submissions in {targets[problem]}")


if __name__ == "__main__":
    main()
