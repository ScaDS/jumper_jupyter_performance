"""Is the analysis about the optimization decision, or about everything?

The analyze step is asked for a bottleneck in two to four sentences and told
explicitly not to propose code changes. Two things go wrong when context is
removed or added: with too little the model pads with generalities, and with
too much it narrates the telemetry. Both show up here, and neither shows up in
a correctness score.

Premature suggestions are counted separately because they are an instruction
failure rather than a relevance failure, and the prompt ablations are expected
to move exactly that number.
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


class RelevanceVerdict(BaseModel):
    total_claims: int = Field(ge=0)
    # Claims that bear on deciding what to optimize.
    relevant_claims: int = Field(ge=0)
    # Observations that are true and beside the point.
    irrelevant_observations: int = Field(ge=0)
    # Code changes proposed in a step that was told not to propose any.
    premature_suggestions: int = Field(ge=0)
    notes: str = ""


class RelevanceConciseness(JudgeMetric):
    spec = MetricSpec(
        id="relevance_conciseness",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "relevant_claim_fraction",
            "irrelevant_observation_count",
            "premature_suggestion_count",
            "total_claims",
        ),
        description=(
            "Share of claims that bear on the optimization decision, plus the "
            "count of irrelevant observations and of code changes proposed in "
            "the analysis step."
        ),
    )
    verdict_model = RelevanceVerdict

    def from_verdict(self, verdict: RelevanceVerdict, context) -> dict:
        return {
            "relevant_claim_fraction": rate(
                verdict.relevant_claims,
                verdict.total_claims,
            ),
            "irrelevant_observation_count": verdict.irrelevant_observations,
            "premature_suggestion_count": verdict.premature_suggestions,
            "total_claims": verdict.total_claims,
        }
