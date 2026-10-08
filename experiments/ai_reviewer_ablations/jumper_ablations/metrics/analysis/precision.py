"""Is every claim in the analysis supported by what the model was shown?

The failure this catches is the one the harness itself walked into: a model
handed thin context still answers, fluently and specifically, about things
that were never in front of it. Precision over claims is the direct measure of
that, and it is the number most likely to move when a source is removed.

The judge counts claims against the packet's sources only. A claim that is
true of the world but unsupported by the sources is unsupported here - the
question is whether the model reasoned from its evidence, not whether it
guessed well.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from jumper_ablations.metrics.base import (
    CATEGORY_ANALYSIS,
    METHOD_JUDGE,
    SCOPE_RUN,
    JudgeMetric,
    MetricSpec,
)
from jumper_ablations.statistics import rate


class PrecisionVerdict(BaseModel):
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    # Claims the sources actively disagree with.
    contradictions: int = Field(ge=0)
    # Claims about things that are not in the sources at all.
    hallucinations: int = Field(ge=0)
    examples: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parts_fit_the_whole(self) -> "PrecisionVerdict":
        # Every claim is supported, contradicted or unfounded, so the three
        # counts partition the total. Without this a verdict that parses can
        # still produce a precision above one.
        counted = (
            self.supported_claims + self.contradictions + self.hallucinations
        )
        if counted > self.total_claims:
            raise ValueError(
                f"{counted} classified claims exceed "
                f"{self.total_claims} total_claims"
            )
        return self


class Precision(JudgeMetric):
    spec = MetricSpec(
        id="precision",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "supported_claim_precision",
            "contradiction_rate",
            "hallucination_rate",
            "contradiction_count",
            "hallucination_count",
            "total_claims",
        ),
        description=(
            "Share of the analysis's claims that the given sources support, "
            "and how many contradict them or refer to nothing in them."
        ),
    )
    verdict_model = PrecisionVerdict

    def from_verdict(self, verdict: PrecisionVerdict, context) -> dict:
        return {
            "supported_claim_precision": rate(
                verdict.supported_claims,
                verdict.total_claims,
            ),
            "contradiction_rate": rate(
                verdict.contradictions,
                verdict.total_claims,
            ),
            "hallucination_rate": rate(
                verdict.hallucinations,
                verdict.total_claims,
            ),
            "contradiction_count": verdict.contradictions,
            "hallucination_count": verdict.hallucinations,
            "total_claims": verdict.total_claims,
        }
