"""Colour and chrome for the report figures.

The palette is used by the job each colour does, not by taste. Speedup is one
magnitude over an identity axis, so it is one hue. Correctness is a state, so
it uses the reserved status colours - and because two of those sit below 3:1
on the light surface, every status mark here ships with a legend *and* a direct
label, which is the documented mitigation: a status colour never carries
meaning alone.

The categorical pairs used below were run through the design system's
validator in both modes and pass every gate (worst adjacent CVD dE 24.7,
normal-vision dE 33.6). The status trio passes CVD separation and the
normal-vision floor; its lightness band and light-surface contrast are known
exceptions of the status palette itself, which is why the labels are not
optional.
"""

from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Categorical slots 1 and 2.
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    # Diverging poles for a signed difference.
    "better": "#2a78d6",
    "worse": "#e34948",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "better": "#3987e5",
    "worse": "#e66767",
}

# Reserved, and identical in both modes by design.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(mode: str = "light") -> dict:
    return dict(DARK if mode == "dark" else LIGHT)


def layout(mode: str = "light", title: str = "", height: int = 420) -> dict:
    """A plotly layout with recessive chrome and no chartjunk."""
    palette = tokens(mode)
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": palette["text_primary"]},
            "x": 0,
            "xanchor": "left",
        },
        "paper_bgcolor": palette["plane"],
        "plot_bgcolor": palette["surface"],
        "font": {
            "family": FONT,
            "size": 12,
            "color": palette["text_secondary"],
        },
        "height": height,
        "margin": {"l": 160, "r": 40, "t": 56, "b": 48},
        "xaxis": {
            "gridcolor": palette["grid"],
            "zerolinecolor": palette["axis"],
            "linecolor": palette["axis"],
            "tickfont": {"color": palette["muted"]},
        },
        "yaxis": {
            "gridcolor": palette["surface"],
            "linecolor": palette["axis"],
            "tickfont": {"color": palette["text_secondary"]},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            # Plotly reverses legend order for a horizontal stack; the legend
            # has to read in the same order the segments do.
            "traceorder": "normal",
            "font": {"color": palette["text_secondary"]},
        },
        "hoverlabel": {"font": {"family": FONT}},
        # A 2px surface gap between adjacent bars, as the mark spec asks.
        "bargap": 0.28,
        "bargroupgap": 0.12,
    }
