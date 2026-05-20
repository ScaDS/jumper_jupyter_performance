"""Renderer registry infrastructure.

Defines PlotResult/SeriesItem and the RENDERERS dict.
Actual renderer implementations live in renderers.py.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class SeriesItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    data: pd.Series
    color: str = "blue"
    width: float = 2.0
    opacity: float = 1.0
    linestyle: Literal["solid", "dashed", "dotted"] = "solid"


class PlotResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    series: list[SeriesItem]
    title: str
    ylim: Optional[tuple[float, float]] = None


RENDERERS: dict[str, Callable] = {}


def register(plot_type: str) -> Callable:
    """Register a renderer function for a given plot type.

    Usage::

        from jumper_extension.adapters.visualizer.render import register, PlotResult, SeriesItem

        @register("my_type")
        def render_my_type(df, config, level, hardware, io_window):
            series = df[config.column]
            return PlotResult(
                series=[SeriesItem(label=config.label, data=series)],
                title=config.title,
                ylim=config.ylim,
            )
    """
    def decorator(fn: Callable) -> Callable:
        RENDERERS[plot_type] = fn
        return fn
    return decorator
