"""The two tables the experiment exists to fill in.

The shape is fixed by agents/reviewer/ablation_metrics_table_variants.md:
analysis metrics and suggestion metrics, one row per metric, one column per
preset. The report renders that from summary.csv and adds nothing to it -
every number in a table is traceable to a row, and every row to a record.
The figures alongside it answer the same questions in a form that is quicker
to read and no more informative.
"""

from jumper_ablations.report.figures import (
    correctness_rates,
    evidence_coverage,
    paired_deltas,
    speedup_with_coverage,
    write_figures,
)
from jumper_ablations.report.tables import render_tables

__all__ = [
    "correctness_rates",
    "evidence_coverage",
    "paired_deltas",
    "render_tables",
    "speedup_with_coverage",
    "write_figures",
]
