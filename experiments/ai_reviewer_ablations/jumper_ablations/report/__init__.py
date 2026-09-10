"""The two tables the experiment exists to fill in, and the figures beside them.

The shape is fixed by agents/reviewer/ablation_metrics_table_variants.md:
analysis metrics and suggestion metrics, one row per metric, one column per
preset. The report renders that from summary.csv and adds nothing to it -
every number in a table is traceable to a row, and every row to a record.
"""
from jumper_ablations.report.tables import render_tables

__all__ = ["render_tables"]
