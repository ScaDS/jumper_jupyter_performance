"""Rebuild the index of runs.

    python -m jumper_ablations.cli.index

Reads every run under ``results_root`` and writes ``runs_index.csv`` beside
them: one row per run, with the fingerprint that says which runs asked the
same question and may therefore be pooled.

Derived and disposable - losing it costs a second, and it is never written by
a job, so no shard ever contends for it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jumper_ablations.aggregate.runs_index import build, groups, write
from jumper_ablations.cli.common import configure_logging
from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers

logger = logging.getLogger("jumper_ablations")

register_resolvers()


@hydra.main(
    version_base=None, config_path=CLI_CONFIG_PATH, config_name="offline"
)
def main(raw_config: DictConfig) -> int:
    configure_logging()
    config = load_experiment_config(raw_config)
    results_root = Path(config.results_root)
    if not results_root.is_dir():
        logger.error(f"{results_root} does not exist")
        return 2

    rows = build(results_root)
    path = write(results_root, rows)
    logger.info(f"{len(rows)} run(s) in {path}")

    for fingerprint, run_ids in sorted(groups(rows).items()):
        shape = "poolable" if len(run_ids) > 1 else "on its own"
        logger.info(f"{fingerprint} {shape}: {', '.join(sorted(run_ids))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
