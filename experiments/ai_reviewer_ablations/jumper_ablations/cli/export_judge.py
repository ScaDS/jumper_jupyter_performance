"""Write the task packets an agent session judges.

    python -m jumper_ablations.cli.export_judge
    python -m jumper_ablations.cli.export_judge target_run=20260910_143000
    python -m jumper_ablations.cli.export_judge judge.sample_per_cell=2

Each packet is self-contained and, by default, blind: it holds the messages
the reviewer's model actually received, what it answered, the rubric, and the
schema of the answer - and it does not say which preset produced it. See
JUDGE_PROTOCOL.md for what to do with them.
"""
from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig

from jumper_ablations.cli.common import (
    configure_logging,
    load_contexts,
    resolve_run,
)
from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.evaluation.judge.export import export_packets
from jumper_ablations.evaluation.judge.layout import (
    index_path,
    tasks_root,
    verdicts_root,
)
from jumper_ablations.metrics.base import METHOD_JUDGE
from jumper_ablations.metrics.registry import selected_metrics
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers

logger = logging.getLogger("jumper_ablations")

register_resolvers()


@hydra.main(version_base=None, config_path=CLI_CONFIG_PATH, config_name="offline")
def main(raw_config: DictConfig) -> int:
    configure_logging()
    config = load_experiment_config(raw_config)
    run = resolve_run(config)
    _records, run_contexts, cell_contexts = load_contexts(config, run)

    enabled = (
        config.metrics.analysis.enabled_ids()
        + config.metrics.suggestions.enabled_ids()
    )
    metrics = [
        metric
        for metric in selected_metrics(enabled)
        if metric.spec.evaluation_method == METHOD_JUDGE
    ]
    if not metrics:
        logger.info("no judged metrics are enabled; nothing to export")
        return 0

    verdicts_root(run.path).mkdir(parents=True, exist_ok=True)
    packets = export_packets(
        run_directory=run.path,
        metrics=metrics,
        run_contexts=run_contexts,
        cell_contexts=cell_contexts,
        config=config.judge,
    )

    rubrics = sorted({packet.rubric for packet in packets})
    logger.info(
        f"{len(packets)} packet(s) across {len(rubrics)} rubric(s) "
        f"({', '.join(rubrics)}) in {tasks_root(run.path)}"
    )
    logger.info(f"verdicts go to {verdicts_root(run.path)}")
    logger.info(f"unblinding key: {index_path(run.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
