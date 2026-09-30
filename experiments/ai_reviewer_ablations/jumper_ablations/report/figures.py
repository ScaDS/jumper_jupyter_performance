"""The four figures worth drawing, and nothing decorative.

Each answers one question the tables answer less directly:

1. what the best correct rewrite of a preset is worth, and over how much of
   its output - the speedup and its coverage in one mark, because either
   alone is misleading;
2. what happened to the suggestions - one composition per preset, in the
   reserved status colours;
3. how much of the reference evidence each preset recovered, conditional
   against global - the pair that separates "used its context badly" from
   "was given too little";
4. how far each preset sits from the full-context baseline, paired by
   generation, with the interval that says whether the gap is real.

All four are horizontal: the category axis holds preset names, which are words
and belong on the axis that has room for them.
"""

from __future__ import annotations

import pandas as pd

from jumper_ablations.report.theme import STATUS, layout, tokens

# A bar segment narrower than this has no room for a label inside it.
_LABEL_FLOOR = 0.08

# Outside labels are drawn past the end of the bar, so the axis needs room
# for them or the longest one is silently clipped at the plot edge.
_LABEL_HEADROOM = 1.55


def _ordered(frame: pd.DataFrame, baseline: str) -> list[str]:
    """Ablations with the baseline first, so the rest read as differences."""
    present = list(dict.fromkeys(frame["ablation"]))
    if baseline in present:
        present.remove(baseline)
        return [baseline, *sorted(present)]
    return sorted(present)


def _pick(frame: pd.DataFrame, metric: str, reported_value: str) -> dict:
    rows = frame[
        (frame["metric"] == metric)
        & (frame["reported_value"] == reported_value)
    ]
    return dict(zip(rows["ablation"], rows["estimate"]))


def _figure(
    traces: list,
    mode: str,
    title: str,
    height: int,
    barmode: str | None = None,
    xaxes: dict | None = None,
    yaxes: dict | None = None,
):
    import plotly.graph_objects as graph_objects

    figure = graph_objects.Figure(data=traces)
    options = layout(mode, title, height)
    if barmode:
        options["barmode"] = barmode
    figure.update_layout(**options)
    figure.update_xaxes(**(xaxes or {}))
    figure.update_yaxes(**(yaxes or {}))
    return figure


def speedup_with_coverage(
    summary: pd.DataFrame,
    usecase: str,
    baseline: str = "base",
    mode: str = "light",
):
    """Correctness-conditioned speedup, labelled with its coverage.

    One series, so no legend: the title names it. The coverage rides on the
    bar's own label rather than in a second chart, because a speedup read
    without its coverage is the single easiest way to misread this
    experiment.
    """
    import plotly.graph_objects as graph_objects

    palette = tokens(mode)
    frame = summary[summary["usecase"] == usecase]
    speedups = _pick(
        frame, "correctness_conditioned_speedup", "geometric_mean_speedup"
    )
    coverage = _pick(frame, "correctness_conditioned_speedup", "coverage")
    ablations = [one for one in _ordered(frame, baseline) if one in speedups]

    values = [speedups.get(one) for one in ablations]
    shares = [coverage.get(one) for one in ablations]
    labels = [
        (
            f"  {value:.2f}x  ·  {share:.0%} coverage"
            if value is not None and share is not None
            else "  not measured"
        )
        for value, share in zip(values, shares)
    ]

    trace = graph_objects.Bar(
        x=values,
        y=ablations,
        orientation="h",
        marker={"color": palette["series_1"]},
        text=labels,
        textposition="outside",
        textfont={"color": palette["text_secondary"]},
        hovertemplate=(
            "<b>%{y}</b><br>geometric mean speedup %{x:.2f}x"
            "<br>coverage %{customdata:.0%}<extra></extra>"
        ),
        customdata=shares,
    )
    measured = [value for value in values if value is not None]
    return _figure(
        [trace],
        mode,
        f"Speedup of correct suggestions - {usecase}",
        max(240, 46 * len(ablations) + 120),
        xaxes={
            "title": "geometric mean speedup (x)",
            "range": [0, max(measured or [1.0]) * _LABEL_HEADROOM],
        },
        yaxes={"autorange": "reversed"},
    )


def correctness_rates(
    summary: pd.DataFrame,
    usecase: str,
    baseline: str = "base",
    mode: str = "light",
):
    """What became of the suggestions, as one composition per preset.

    Status colours, and therefore a legend and a direct label on every
    segment wide enough to hold one: two of these steps are below 3:1 on the
    light surface, so colour is never the only thing saying which is which.
    """
    import plotly.graph_objects as graph_objects

    palette = tokens(mode)
    frame = summary[summary["usecase"] == usecase]
    ablations = _ordered(frame, baseline)
    series = (
        ("verified", "verified_rate", STATUS["good"]),
        ("unverified", "unverified_rate", STATUS["warning"]),
        ("differs", "differs_rate", STATUS["critical"]),
        ("never ran", "failed_rate", palette["muted"]),
    )

    traces = []
    for name, reported_value, colour in series:
        values = _pick(frame, "correctness_rates", reported_value)
        shares = [values.get(one) for one in ablations]
        traces.append(
            graph_objects.Bar(
                x=shares,
                y=ablations,
                orientation="h",
                name=name,
                marker={
                    "color": colour,
                    # The 2px surface ring the mark spec asks for between
                    # touching fills.
                    "line": {"color": palette["surface"], "width": 2},
                },
                text=[
                    (
                        f"{share:.0%}"
                        if share is not None and share >= _LABEL_FLOOR
                        else ""
                    )
                    for share in shares
                ],
                textposition="inside",
                insidetextfont={"color": palette["surface"]},
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{name} %{{x:.0%}}<extra></extra>"
                ),
            )
        )

    return _figure(
        traces,
        mode,
        f"Correctness of suggestions - {usecase}",
        max(240, 46 * len(ablations) + 140),
        barmode="stack",
        xaxes={"title": "share of suggestions", "tickformat": ".0%"},
        yaxes={"autorange": "reversed"},
    )


def evidence_coverage(
    summary: pd.DataFrame,
    usecase: str,
    baseline: str = "base",
    mode: str = "light",
):
    """Conditional against global recall, side by side.

    The gap between the two bars is the finding: a preset can use everything
    it was given and still be missing what it was not.
    """
    import plotly.graph_objects as graph_objects

    palette = tokens(mode)
    frame = summary[summary["usecase"] == usecase]
    ablations = _ordered(frame, baseline)
    series = (
        ("conditional", "conditional_evidence_coverage", palette["series_1"]),
        ("global", "global_evidence_coverage", palette["series_2"]),
    )

    traces = []
    for name, metric, colour in series:
        values = _pick(frame, metric, "weighted_recall")
        shares = [values.get(one) for one in ablations]
        traces.append(
            graph_objects.Bar(
                x=shares,
                y=ablations,
                orientation="h",
                name=name,
                marker={"color": colour},
                text=[
                    f"  {share:.0%}" if share is not None else ""
                    for share in shares
                ],
                textposition="outside",
                textfont={"color": palette["text_secondary"]},
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{name} weighted recall "
                    "%{x:.0%}<extra></extra>"
                ),
            )
        )

    return _figure(
        traces,
        mode,
        f"Evidence coverage - {usecase}",
        max(260, 52 * len(ablations) + 140),
        barmode="group",
        xaxes={
            "title": "weighted recall",
            "tickformat": ".0%",
            "range": [0, 1.15],
        },
        yaxes={"autorange": "reversed"},
    )


def paired_deltas(
    summary: pd.DataFrame,
    usecase: str,
    metric: str,
    reported_value: str,
    baseline: str = "base",
    mode: str = "light",
):
    """Each preset's distance from the baseline, with the interval of that
    distance.

    Paired by unit, so the comparison is between runs asked for under the
    same conditions rather than between two piles of numbers. A dot whose
    interval crosses zero is not a finding, and the zero line is drawn so
    that is visible without arithmetic.

    The bar is the interval of the *difference*, resampled from the paired
    differences themselves. A preset's own interval must never be drawn here
    however tempting its availability: it describes the scatter of one
    preset's values, carries nothing about the baseline's, and reading it as
    a difference invents and hides effects in equal measure. Where the
    difference has no interval - nothing paired, or a single pair - the dot
    is drawn bare, which is the honest rendering of "not enough to say".
    """
    import plotly.graph_objects as graph_objects

    palette = tokens(mode)
    frame = summary[
        (summary["usecase"] == usecase)
        & (summary["metric"] == metric)
        & (summary["reported_value"] == reported_value)
        & (summary["ablation"] != baseline)
    ]
    if frame.empty:
        return None

    frame = frame.sort_values("ablation")
    deltas = [
        (
            row.paired_delta
            if pd.notna(row.paired_delta)
            else row.delta_vs_baseline
        )
        for row in frame.itertuples()
    ]
    paired = [
        pd.notna(row.paired_delta)
        and pd.notna(row.paired_delta_ci_low)
        and pd.notna(row.paired_delta_ci_high)
        for row in frame.itertuples()
    ]
    colours = [
        palette["worse"] if (delta or 0) < 0 else palette["better"]
        for delta in deltas
    ]
    spread = [
        (
            (
                row.paired_delta - row.paired_delta_ci_low,
                row.paired_delta_ci_high - row.paired_delta,
            )
            if is_paired
            else (0.0, 0.0)
        )
        for row, is_paired in zip(frame.itertuples(), paired)
    ]

    trace = graph_objects.Scatter(
        x=deltas,
        y=list(frame["ablation"]),
        mode="markers",
        marker={
            "size": 11,
            "color": colours,
            "line": {"color": palette["surface"], "width": 2},
        },
        error_x={
            "type": "data",
            "symmetric": False,
            "array": [high for _, high in spread],
            "arrayminus": [low for low, _ in spread],
            "color": palette["axis"],
            "thickness": 2,
            "width": 0,
        },
        hovertemplate=(
            "<b>%{y}</b><br>difference from " + baseline + ": %{x:.3g}"
            "<extra></extra>"
        ),
        showlegend=False,
    )

    figure = _figure(
        [trace],
        mode,
        f"{metric}.{reported_value} against `{baseline}` - {usecase}",
        max(240, 42 * len(frame) + 130),
        xaxes={"title": f"difference from {baseline}", "zeroline": True},
        yaxes={"autorange": "reversed"},
    )
    figure.add_vline(x=0, line_color=palette["axis"], line_width=2)
    return figure


def write_figures(figures: dict, directory) -> list:
    """Write each figure to PNG, skipping the ones kaleido cannot render."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures.items():
        if figure is None:
            continue
        path = directory / f"{name}.png"
        try:
            figure.write_image(str(path), scale=2)
        except Exception as failure:
            print(f"skipped {name}.png: {failure}")
            continue
        written.append(path)
    return written
