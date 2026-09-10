"""Did the analysis identify the bottleneck - resource, place, and why?

Three separate judgements rather than one, because they fail separately and
the ablations are expected to break them separately. Removing the metric
summary should hurt the resource call; removing the code should hurt the
place; removing the telemetry should hurt the causal argument while leaving
the other two intact. A single score would hide exactly the structure the
experiment is looking for.

Each is scored 0 / 1 / 2: wrong, partly right, right.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from jumper_ablations.metrics.base import (
    CATEGORY_ANALYSIS,
    METHOD_JUDGE,
    SCOPE_RUN,
    JudgeMetric,
    MetricSpec,
)

MAXIMUM_SCORE = 2


class BottleneckVerdict(BaseModel):
    """What the judge writes for one analysis."""

    resource_score: int = Field(ge=0, le=MAXIMUM_SCORE)
    code_region_score: int = Field(ge=0, le=MAXIMUM_SCORE)
    causal_explanation_score: int = Field(ge=0, le=MAXIMUM_SCORE)
    # Which resource and which lines the analysis actually named, quoted, so a
    # score can be checked against the analysis rather than trusted.
    named_resource: str = ""
    named_code_region: str = ""
    justification: str = ""


class BottleneckIdentification(JudgeMetric):
    spec = MetricSpec(
        id="bottleneck_identification",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "resource_score",
            "code_region_score",
            "causal_explanation_score",
            "total_score",
            "normalised_score",
        ),
        description=(
            "Whether the analysis named the right resource, the responsible "
            "code region, and a causal explanation, each scored 0/1/2 against "
            "the usecase's reference facts and the sources the model was "
            "given."
        ),
    )
    verdict_model = BottleneckVerdict

    def from_verdict(self, verdict: BottleneckVerdict, context) -> dict:
        total = (
            verdict.resource_score
            + verdict.code_region_score
            + verdict.causal_explanation_score
        )
        return {
            "resource_score": verdict.resource_score,
            "code_region_score": verdict.code_region_score,
            "causal_explanation_score": verdict.causal_explanation_score,
            "total_score": total,
            "normalised_score": total / (3 * MAXIMUM_SCORE),
        }
