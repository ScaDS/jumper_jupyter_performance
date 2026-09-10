"""Rendering summary.csv as the two metric tables.

One table per category, one column per ablation, the baseline preset first so
every other column reads as a difference from it. Values carry their interval
where there is one; a cell with no data says so rather than showing a dash
that could be mistaken for zero.
"""
from __future__ import annotations

import pandas as pd

from jumper_ablations.metrics.base import CATEGORY_ANALYSIS, CATEGORY_SUGGESTIONS

_TITLES = {
    CATEGORY_ANALYSIS: "Analysis Evaluation Metrics",
    CATEGORY_SUGGESTIONS: "Suggestions Evaluation Metrics",
}

NO_DATA = "not measured"


def _format(estimate, low, high) -> str:
    if estimate is None or pd.isna(estimate):
        return NO_DATA
    text = f"{float(estimate):.3g}"
    if low is not None and high is not None and pd.notna(low) and pd.notna(high):
        text += f" [{float(low):.3g}, {float(high):.3g}]"
    return text


def _ordered_ablations(frame: pd.DataFrame, baseline: str) -> list[str]:
    present = list(dict.fromkeys(frame["ablation"]))
    if baseline in present:
        present.remove(baseline)
        return [baseline, *sorted(present)]
    return sorted(present)


def render_category(
    summary: pd.DataFrame,
    category: str,
    baseline_ablation: str,
) -> str:
    """One markdown table for one category, across every usecase in the run."""
    frame = summary[summary["category"] == category]
    if frame.empty:
        return f"#### {_TITLES[category]}\n\n_No metrics of this category ran._\n"

    sections = [f"#### {_TITLES[category]}", ""]
    for usecase, rows in frame.groupby("usecase", sort=True):
        ablations = _ordered_ablations(rows, baseline_ablation)
        sections.append(f"**Usecase:** `{usecase}`")
        sections.append("")
        sections.append(
            "| Metric | Reported Value | Method | "
            + " | ".join(f"`{one}`" for one in ablations)
            + " |"
        )
        sections.append(
            "|---|---|---|" + "|".join("---" for _ in ablations) + "|"
        )

        keys = rows[["metric", "reported_value", "evaluation_method"]]
        for (metric, reported_value, method) in (
            keys.drop_duplicates().itertuples(index=False)
        ):
            cells = []
            for ablation in ablations:
                match = rows[
                    (rows["metric"] == metric)
                    & (rows["reported_value"] == reported_value)
                    & (rows["ablation"] == ablation)
                ]
                if match.empty:
                    cells.append(NO_DATA)
                    continue
                row = match.iloc[0]
                cells.append(_format(row["estimate"], row["ci_low"], row["ci_high"]))
            sections.append(
                f"| `{metric}` | `{reported_value}` | {method} | "
                + " | ".join(cells)
                + " |"
            )
        sections.append("")

    sections.append(
        "Values are point estimates with a bootstrap interval where one could "
        "be computed. An interval needs at least two generations; "
        f"`{NO_DATA}` means the metric produced no value for that preset, "
        "which is not the same as a value of zero."
    )
    sections.append("")
    return "\n".join(sections)


def render_tables(summary: pd.DataFrame, baseline_ablation: str) -> dict:
    """``{filename: markdown}`` for the two category tables."""
    return {
        "analysis_metrics.md": render_category(
            summary,
            CATEGORY_ANALYSIS,
            baseline_ablation,
        ),
        "suggestions_metrics.md": render_category(
            summary,
            CATEGORY_SUGGESTIONS,
            baseline_ablation,
        ),
    }
