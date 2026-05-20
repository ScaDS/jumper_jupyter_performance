"""Pydantic v2 models for plot-subset descriptors.

``MetricConfig`` is a discriminated union on the ``type`` field; Pydantic
selects the correct concrete model automatically during ``model_validate()``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class SeriesStyle(BaseModel):
    column: str
    label: str
    color: str = "steelblue"
    width: float = 2.0
    y_axis: Literal["left", "right"] = "left"


class SingleSeriesConfig(BaseModel):
    type: Literal["single_series"]
    column: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class SummarySeriesConfig(BaseModel):
    type: Literal["summary_series"]
    columns: list[str]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class MultiSeriesConfig(BaseModel):
    type: Literal["multi_series"]
    prefix: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class CompositeSeriesConfig(BaseModel):
    type: Literal["composite_series"]
    series: list[SeriesStyle]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


MetricConfig = Annotated[
    Union[
        SingleSeriesConfig,
        SummarySeriesConfig,
        MultiSeriesConfig,
        CompositeSeriesConfig,
    ],
    Field(discriminator="type"),
]

def validate_metric_config(data: dict):
    from pydantic import TypeAdapter
    return TypeAdapter(MetricConfig).validate_python(data)
