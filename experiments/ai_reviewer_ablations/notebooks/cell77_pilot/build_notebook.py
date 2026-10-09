"""Generate pilot_report.ipynb: headings and figures, nothing else.

The prose lives in report.md under the same headings. This script exists so the
notebook's cell structure is reviewable as code rather than as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = '''\
"""Sources, palette and the shared visual vocabulary."""
import csv
import json
import os
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

STORAGE = Path(
    os.environ.get(
        "JUMPER_ABLATION_STORAGE", "/data/horse/ws/albu670g-jumper/ablations"
    )
)
RUN = STORAGE / "results" / "cell77_pilot"

BASELINE = "base"
ABLATIONS = ["code_only", "no_timing", "no_perf"]
PRESETS = [BASELINE] + ABLATIONS

# Validated in both modes with the dataviz palette validator. The baseline is
# not a series - it is the reference every series is measured against - so it
# wears neutral ink rather than a categorical hue.
COLOUR = {"code_only": "#2a78d6", "no_timing": "#c2571f", "no_perf": "#1d8a5e"}
INK = "#1a1a19"
MUTED = "#6b7078"
LINE = "#d9dce1"
SURFACE = "#fcfcfb"

LAYOUT = dict(
    template="simple_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Helvetica, Arial, sans-serif", size=13, color=INK),
    margin=dict(l=90, r=40, t=70, b=60),
    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read(name):
    with (RUN / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


summary = read("summary.csv")
units = read("metric_units.csv")
meta = json.loads((RUN / "meta.json").read_text(encoding="utf-8"))
SOURCES = ["code", "timing", "tags", "perf", "raw_perf", "hardware", "packages"]


def cell(preset, metric, reported):
    for row in summary:
        if (
            row["ablation"] == preset
            and row["metric"] == metric
            and row["reported_value"] == reported
        ):
            return row
    return {}


def clears_zero(row):
    low, high = number(row.get("paired_delta_ci_low")), number(
        row.get("paired_delta_ci_high")
    )
    if low is None or high is None:
        return False
    return (low > 0 and high > 0) or (low < 0 and high < 0)


print(f"{RUN}: {len(summary)} summary rows, {len(units)} per-unit values")
'''

FIGURE_1 = '''\
"""The entire difference between the four conditions."""
grid = [
    [
        1.0
        if ((meta["ablations"][preset].get("effect") or {}).get("context") or {}).get(
            source, False
        )
        else 0.0
        for source in SOURCES
    ]
    for preset in PRESETS
]
labels = [
    ["given" if value else "withheld" for value in row] for row in grid
]

figure = go.Figure(
    go.Heatmap(
        z=grid,
        x=SOURCES,
        y=PRESETS,
        text=labels,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[[0, SURFACE], [1, "#dce7f7"]],
        showscale=False,
        xgap=2,
        ygap=2,
    )
)
figure.update_layout(
    **LAYOUT,
    title="What each preset was shown",
    height=300,
    xaxis=dict(side="top", tickfont=dict(size=12)),
    yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
)
figure.show()
'''

FIGURE_2 = '''\
"""Every comparison whose paired interval stays clear of zero."""
rows = []
for row in summary:
    base = number(row.get("baseline_estimate"))
    delta = number(row.get("paired_delta"))
    if not clears_zero(row) or not base or delta is None:
        continue
    rows.append(
        dict(
            preset=row["ablation"],
            key=f"{row['metric']}.{row['reported_value']}",
            kind=row["value_kind"],
            relative=delta / abs(base),
            low=number(row["paired_delta_ci_low"]) / abs(base),
            high=number(row["paired_delta_ci_high"]) / abs(base),
            delta=delta,
            paired_n=row["paired_n"],
        )
    )
rows.sort(key=lambda one: (ABLATIONS.index(one["preset"]), one["relative"]))

figure = go.Figure()
for preset in ABLATIONS:
    mine = [one for one in rows if one["preset"] == preset]
    if not mine:
        continue
    figure.add_trace(
        go.Scatter(
            x=[one["relative"] for one in mine],
            y=[f"{one['key']}  " for one in mine],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[one["high"] - one["relative"] for one in mine],
                arrayminus=[one["relative"] - one["low"] for one in mine],
                color=COLOUR[preset],
                thickness=2,
                width=0,
            ),
            mode="markers+text",
            marker=dict(size=10, color=COLOUR[preset], line=dict(width=0)),
            text=[
                f"  {one['delta']:+,.3g} (n={one['paired_n']})" for one in mine
            ],
            textposition="middle right",
            textfont=dict(size=10, color=MUTED),
            name=preset,
        )
    )
figure.add_vline(x=0, line=dict(color=INK, width=1))
figure.update_layout(
    **LAYOUT,
    title=f"{len(rows)} of {len(summary)} comparisons move the number",
    height=60 + 22 * len(rows),
    xaxis=dict(
        title="paired difference from base, as a fraction of base",
        tickformat="+.0%",
        range=[-1.45, 1.1],
        gridcolor=LINE,
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(tickfont=dict(size=10, family="monospace"), autorange="reversed"),
)
figure.show()
'''

FIGURE_3 = '''\
"""Naming the limiting resource, beside how well-founded the analysis reads."""
panels = [
    ("bottleneck_identification", "resource_score", "Resource named (0-2)", [0, 2.2]),
    ("precision", "supported_claim_precision", "Claims the sources support", [0, 1.0]),
    ("precision", "hallucination_rate", "Claims about nothing shown", [0, 0.26]),
]
figure = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[title for _, _, title, _ in panels],
    horizontal_spacing=0.09,
)
for index, (metric, reported, _title, bounds) in enumerate(panels, start=1):
    for preset in PRESETS:
        row = cell(preset, metric, reported)
        value = number(row.get("estimate"))
        if value is None:
            continue
        marked = clears_zero(row)
        figure.add_trace(
            go.Bar(
                x=[preset],
                y=[value],
                marker=dict(
                    color=COLOUR.get(preset, "#c9cbcf"),
                    line=dict(
                        color=INK if marked else SURFACE,
                        width=1.5 if marked else 2,
                    ),
                ),
                text=[f"{value:.3g}"],
                textposition="outside",
                textfont=dict(size=11, color=INK),
                showlegend=False,
                width=0.62,
            ),
            row=1,
            col=index,
        )
    figure.update_yaxes(
        range=bounds, gridcolor=LINE, showgrid=True, row=1, col=index
    )
    figure.update_xaxes(tickfont=dict(size=11), row=1, col=index)

figure.update_layout(
    **LAYOUT,
    title="Without telemetry the analysis reads better and names the wrong resource",
    height=430,
    bargap=0.3,
)
figure.show()
'''

FIGURE_4 = '''\
"""Coverage against the reachable denominator it is computed over."""
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Weighted recall of the reference facts",
        "Facts the preset could reach at all",
    ],
    horizontal_spacing=0.12,
)
for scope, dash, name in (
    ("conditional_evidence_coverage", "solid", "conditional"),
    ("global_evidence_coverage", "dot", "global"),
):
    values, presets = [], []
    for preset in PRESETS:
        value = number(cell(preset, scope, "weighted_recall").get("estimate"))
        if value is not None:
            values.append(value)
            presets.append(preset)
    figure.add_trace(
        go.Scatter(
            x=presets,
            y=values,
            mode="lines+markers+text",
            line=dict(color=INK, width=2, dash=dash),
            marker=dict(size=11, color=INK),
            text=[f"{value:.2f}" for value in values],
            textposition="top center",
            textfont=dict(size=11, color=INK),
            name=name,
        ),
        row=1,
        col=1,
    )

reachable = [
    number(cell(preset, "conditional_evidence_coverage", "facts").get("estimate"))
    for preset in PRESETS
]
figure.add_trace(
    go.Bar(
        x=PRESETS,
        y=reachable,
        marker=dict(color=[COLOUR.get(preset, "#c9cbcf") for preset in PRESETS]),
        text=[f"{int(value)}" if value else "" for value in reachable],
        textposition="outside",
        textfont=dict(size=11, color=INK),
        showlegend=False,
        width=0.62,
    ),
    row=1,
    col=2,
)
figure.update_yaxes(
    range=[0, 1.05], gridcolor=LINE, showgrid=True, row=1, col=1
)
figure.update_yaxes(range=[0, 7], gridcolor=LINE, showgrid=True, row=1, col=2)
figure.update_layout(
    **LAYOUT,
    title="High conditional coverage over a denominator that collapsed",
    height=430,
    bargap=0.3,
)
figure.show()
'''

FIGURE_5 = '''\
"""Speedup of the suggestions, per unit and in aggregate."""
per_unit = [
    row
    for row in units
    if row["metric"] == "mean_median_speedup"
    and row["reported_value"] == "geometric_mean_speedup"
    and number(row["value"]) is not None
]
figure = go.Figure()
for preset in PRESETS:
    mine = [number(row["value"]) for row in per_unit if row["ablation"] == preset]
    if not mine:
        continue
    figure.add_trace(
        go.Box(
            y=mine,
            name=preset,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker=dict(
                size=8, color=COLOUR.get(preset, "#9aa0aa"), opacity=0.85
            ),
            line=dict(color=MUTED, width=1.5),
            fillcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
    )
figure.add_hline(
    y=1.0,
    line=dict(color=INK, width=1, dash="dot"),
    annotation_text="no change",
    annotation_position="right",
    annotation_font=dict(size=11, color=MUTED),
)
figure.update_layout(
    **LAYOUT,
    title="What the suggestions actually did to the runtime",
    height=430,
    yaxis=dict(
        title="geometric mean speedup, per unit",
        gridcolor=LINE,
        showgrid=True,
        range=[0, 1.6],
    ),
    xaxis=dict(tickfont=dict(size=12)),
)
figure.show()
'''

FIGURE_6 = '''\
"""What each preset cost, beside the speedup it bought."""
figure = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Input tokens per review",
        "Review latency (s)",
        "Geometric mean speedup",
    ],
    horizontal_spacing=0.09,
)
panels = [
    ("end_to_end_cost", "mean_input_tokens", 1),
    ("end_to_end_cost", "mean_review_latency_s", 2),
    ("mean_median_speedup", "geometric_mean_speedup", 3),
]
for metric, reported, column in panels:
    for preset in PRESETS:
        row = cell(preset, metric, reported)
        value = number(row.get("estimate"))
        if value is None:
            continue
        figure.add_trace(
            go.Bar(
                x=[preset],
                y=[value],
                marker=dict(
                    color=COLOUR.get(preset, "#c9cbcf"),
                    line=dict(
                        color=INK if clears_zero(row) else SURFACE,
                        width=1.5 if clears_zero(row) else 2,
                    ),
                ),
                text=[f"{value:,.3g}"],
                textposition="outside",
                textfont=dict(size=11, color=INK),
                showlegend=False,
                width=0.62,
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(gridcolor=LINE, showgrid=True, row=1, col=column)
    figure.update_xaxes(tickfont=dict(size=11), row=1, col=column)
figure.update_layout(
    **LAYOUT,
    title="Cheaper, faster, and worse at the thing it is for",
    height=430,
    bargap=0.3,
)
figure.show()
'''

FIGURE_7 = '''\
"""How many paired units each kind of finding rests on."""
counts = {}
for row in summary:
    if row["ablation"] == BASELINE:
        continue
    method = row["evaluation_method"]
    paired = number(row.get("paired_n")) or 0
    counts.setdefault(method, []).append(paired)

gaps = read("judge/missing.csv") if (RUN / "judge" / "missing.csv").is_file() else []
verdicts = len(list((RUN / "judge" / "verdicts").rglob("*.json")))

figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Paired units behind a comparison",
        "Judge verdicts written, and gaps left",
    ],
    horizontal_spacing=0.14,
)
for method, colour in (("deterministic", "#1d8a5e"), ("judge", "#c2571f")):
    values = counts.get(method, [])
    if not values:
        continue
    figure.add_trace(
        go.Box(
            y=values,
            name=method,
            boxpoints=False,
            line=dict(color=colour, width=2),
            fillcolor="rgba(0,0,0,0)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
figure.add_trace(
    go.Bar(
        x=["verdicts", "gaps"],
        y=[verdicts, len(gaps)],
        marker=dict(color=["#1d8a5e", "#c2571f"]),
        text=[str(verdicts), str(len(gaps))],
        textposition="outside",
        textfont=dict(size=12, color=INK),
        showlegend=False,
        width=0.5,
    ),
    row=1,
    col=2,
)
figure.update_yaxes(
    title="units", range=[0, 6], gridcolor=LINE, showgrid=True, row=1, col=1
)
figure.update_yaxes(gridcolor=LINE, showgrid=True, row=1, col=2)
figure.update_layout(
    **LAYOUT,
    title="What the judged half of the report rests on",
    height=400,
    bargap=0.4,
)
figure.show()
'''

CELLS = [
    ("markdown", "# Context-source ablations on minian cell 77"),
    ("code", SETUP),
    ("markdown", "## 1. What each preset was shown"),
    ("code", FIGURE_1),
    ("markdown", "## 2. Which comparisons moved"),
    ("code", FIGURE_2),
    ("markdown", "## 3. The resource, and how the analysis reads"),
    ("code", FIGURE_3),
    ("markdown", "## 4. Conditional against global coverage"),
    ("code", FIGURE_4),
    ("markdown", "## 5. What the suggestions did to the runtime"),
    ("code", FIGURE_5),
    ("markdown", "## 6. What it cost"),
    ("code", FIGURE_6),
    ("markdown", "## 7. What the judged half rests on"),
    ("code", FIGURE_7),
]


def build() -> dict:
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        lines = source.rstrip("\n").split("\n")
        body = [f"{line}\n" for line in lines[:-1]] + [lines[-1]]
        identity = f"cell-{index:02d}"
        if kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": identity,
                    "metadata": {},
                    "source": body,
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "id": identity,
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body,
                }
            )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    target = HERE / "pilot_report.ipynb"
    target.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    notebook = build()
    headings = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    print(f"wrote {target} with {len(notebook['cells'])} cells")
    for heading in headings:
        print(f"  {heading}")
