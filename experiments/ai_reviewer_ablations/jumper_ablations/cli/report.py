"""Aggregate the metric values and render the two tables.

    python -m jumper_ablations.cli.report
    python -m jumper_ablations.cli.report target_run=20260910_143000

Writes summary.csv - point estimate, bootstrap interval, and both the plain
and the generation-paired difference from the baseline preset - plus
analysis_metrics.md and suggestions_metrics.md in the shape of
agents/reviewer/ablation_metrics_table_variants.md.
"""
from __future__ import annotations

import logging
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig

from jumper_ablations.aggregate import summarise
from jumper_ablations.cli.common import (
    configure_logging,
    load_contexts,
    resolve_run,
)
from jumper_ablations.cli.evaluate import METRICS_NAME, _parameters
from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.metrics.base import MetricRow
from jumper_ablations.metrics.registry import all_metrics
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers
from jumper_ablations.report.tables import render_tables

logger = logging.getLogger("jumper_ablations")

SUMMARY_NAME = "summary.csv"

register_resolvers()


def _rows_from(frame: pd.DataFrame) -> list[MetricRow]:
    fields = MetricRow.__dataclass_fields__
    rows = []
    for record in frame.to_dict("records"):
        payload = {
            name: record.get(name)
            for name in fields
            if name in record
        }
        value = payload.get("value")
        payload["value"] = None if pd.isna(value) else float(value)
        payload["note"] = "" if pd.isna(payload.get("note")) else payload.get("note")
        rows.append(MetricRow(**payload))
    return rows


@hydra.main(version_base=None, config_path=CLI_CONFIG_PATH, config_name="offline")
def main(raw_config: DictConfig) -> int:
    configure_logging()
    config = load_experiment_config(raw_config)
    run = resolve_run(config)

    metrics_path = run.path / METRICS_NAME
    if not metrics_path.is_file():
        logger.error(
            f"{metrics_path} does not exist - run "
            "`python -m jumper_ablations.cli.evaluate` first"
        )
        return 2

    _records, _run_contexts, cell_contexts = load_contexts(config, run)
    rows = _rows_from(pd.read_csv(metrics_path))

    summary = summarise(
        rows=rows,
        metrics_by_id=all_metrics(),
        cell_contexts=cell_contexts,
        reporting=config.reporting,
        parameters=_parameters(config),
    )
    summary_path = run.path / SUMMARY_NAME
    summary.to_csv(summary_path, index=False)
    logger.info(f"{len(summary)} summary row(s) in {summary_path}")

    for filename, markdown in render_tables(
        summary,
        config.reporting.baseline_ablation,
    ).items():
        path = run.path / filename
        path.write_text(markdown, encoding="utf-8")
        logger.info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
