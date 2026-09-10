"""Score the records of a run.

    python -m jumper_ablations.cli.evaluate
    python -m jumper_ablations.cli.evaluate target_run=20260910_143000
    python -m jumper_ablations.cli.evaluate metrics/analysis=none

Cheap and repeatable on purpose: it reads the records the expensive phase
wrote and never runs the reviewer, so adding a metric costs a second rather
than a sweep. Judged metrics are read from the verdict files an agent session
produced; ones nobody has judged are reported as gaps, never as zeros.
"""
from __future__ import annotations

import logging
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig

from jumper_ablations.cli.common import (
    configure_logging,
    load_contexts,
    resolve_run,
)
from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.evaluation.base import EvaluationResult
from jumper_ablations.evaluation.deterministic import DeterministicEvaluation
from jumper_ablations.evaluation.judge import JudgeEvaluation
from jumper_ablations.evaluation.judge.layout import missing_path
from jumper_ablations.metrics.base import METHOD_DETERMINISTIC, METHOD_JUDGE
from jumper_ablations.metrics.registry import selected_metrics
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers

logger = logging.getLogger("jumper_ablations")

METRICS_NAME = "metrics.csv"

register_resolvers()


def _parameters(config) -> dict:
    """Per-metric config parameters, keyed by metric id."""
    parameters = {}
    for group in (config.metrics.analysis, config.metrics.suggestions):
        for item in group.metrics:
            if item.enabled:
                parameters[item.id] = dict(item.parameters)
    return parameters


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
    metrics = selected_metrics(enabled)
    parameters = _parameters(config)

    methods = {
        METHOD_DETERMINISTIC: DeterministicEvaluation(),
        METHOD_JUDGE: JudgeEvaluation(run.path),
    }

    result = EvaluationResult()
    for method_id, method in methods.items():
        group = [
            metric
            for metric in metrics
            if metric.spec.evaluation_method == method_id
        ]
        if not group:
            continue
        logger.info(f"{method_id}: {len(group)} metric(s)")
        result.extend(
            method.evaluate(group, run_contexts, cell_contexts, parameters)
        )

    frame = pd.DataFrame([row.__dict__ for row in result.rows])
    path = run.path / METRICS_NAME
    frame.to_csv(path, index=False)
    logger.info(f"{len(frame)} metric value(s) in {path}")

    if result.gaps:
        gaps = pd.DataFrame(result.gaps)
        gaps_path = missing_path(run.path)
        gaps_path.parent.mkdir(parents=True, exist_ok=True)
        gaps.to_csv(gaps_path, index=False)
        by_reason = gaps["reason"].value_counts().to_dict()
        logger.warning(f"{len(gaps)} gap(s) {by_reason} listed in {gaps_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
