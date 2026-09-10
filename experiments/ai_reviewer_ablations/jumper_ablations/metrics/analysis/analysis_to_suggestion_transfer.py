"""Did the suggestions follow from the analysis, or ignore it?

The reviewer's two model calls are chained: the analysis is handed to the
suggest step as the diagnosis to act on. A preset can therefore produce a good
analysis and waste it, and the measured speedups alone would not say which of
the two halves failed.

Two questions. Do the suggestions address the bottleneck the analysis named,
and do they respect the constraints it stated - "the results must stay
identical", "the GPU is unavailable", whatever the analysis committed to.
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
from jumper_ablations.statistics import rate


class TransferVerdict(BaseModel):
    suggestions_total: int = Field(ge=0)
    bottleneck_aligned: int = Field(ge=0)
    # Constraints the analysis stated, and how many suggestions kept them.
    constraints_stated: int = Field(ge=0)
    constraint_observations: int = Field(ge=0)
    constraints_preserved: int = Field(ge=0)
    notes: str = ""


class AnalysisToSuggestionTransfer(JudgeMetric):
    spec = MetricSpec(
        id="analysis_to_suggestion_transfer",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "bottleneck_aligned_rate",
            "preserved_constraint_rate",
            "constraints_stated",
            "suggestions_total",
        ),
        description=(
            "Share of suggestions that address the diagnosed bottleneck, and "
            "share of the analysis's stated constraints that the suggestions "
            "preserve."
        ),
    )
    verdict_model = TransferVerdict

    def from_verdict(self, verdict: TransferVerdict, context) -> dict:
        return {
            "bottleneck_aligned_rate": rate(
                verdict.bottleneck_aligned,
                verdict.suggestions_total,
            ),
            "preserved_constraint_rate": rate(
                verdict.constraints_preserved,
                verdict.constraint_observations,
            ),
            "constraints_stated": verdict.constraints_stated,
            "suggestions_total": verdict.suggestions_total,
        }
