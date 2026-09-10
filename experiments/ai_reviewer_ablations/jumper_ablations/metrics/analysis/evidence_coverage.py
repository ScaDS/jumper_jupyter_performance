"""Two coverage numbers from one judgement, and why they are two.

The judge answers one question - which of the usecase's reference facts does
this analysis actually state? - and the harness divides that answer by two
different denominators.

*Conditional* coverage divides by the facts this ablation could still reach:
a preset with `timing` removed cannot be blamed for missing a fact that only
the timing payload carried. It measures whether the model used what it got.

*Global* coverage divides by every fact, for every preset alike. It measures
what was lost by removing the source. A preset can score perfectly on the
first and badly on the second, and that combination is precisely the finding
the experiment is after: the model used its context well, and the context was
not enough.

Weights come from the usecase manifest, so "missed the dominant bottleneck"
and "missed a detail" are not the same miss.
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

RUBRIC = "evidence_coverage"

# A fact the analysis gestures at without stating counts for half its weight.
PARTIAL_CREDIT = 0.5


class EvidenceCoverageVerdict(BaseModel):
    """Which reference facts the analysis states, by id.

    The judge only identifies facts. Weighting and the two denominators are
    arithmetic, and arithmetic belongs here rather than in a session that
    would have to be trusted to do it the same way twice.
    """

    covered_fact_ids: list[str] = Field(default_factory=list)
    partially_covered_fact_ids: list[str] = Field(default_factory=list)
    notes: str = ""


def _coverage(verdict: EvidenceCoverageVerdict, facts) -> dict:
    """Weighted recall over *facts*, plus the counts behind it."""
    if not facts:
        return {
            "weighted_recall": None,
            "covered_weight": 0.0,
            "total_weight": 0.0,
            "facts": 0,
        }

    covered = set(verdict.covered_fact_ids)
    partial = set(verdict.partially_covered_fact_ids) - covered
    earned = 0.0
    for fact in facts:
        if fact.id in covered:
            earned += fact.weight
        elif fact.id in partial:
            earned += fact.weight * PARTIAL_CREDIT

    total = sum(fact.weight for fact in facts)
    return {
        "weighted_recall": earned / total if total else None,
        "covered_weight": earned,
        "total_weight": total,
        "facts": len(facts),
    }


def _all_facts(context):
    usecase = context.usecase
    return list(usecase.manifest.reference_facts) if usecase else []


def _reachable_facts(context):
    """Facts whose source this ablation still had switched on."""
    enabled = context.record.inputs.enabled_sources
    return [fact for fact in _all_facts(context) if enabled.get(fact.source, False)]


class ConditionalEvidenceCoverage(JudgeMetric):
    spec = MetricSpec(
        id="conditional_evidence_coverage",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "weighted_recall",
            "covered_weight",
            "total_weight",
            "facts",
        ),
        description=(
            "Weighted recall over the reference facts this ablation could "
            "still reach - how well the model used the context it received."
        ),
    )
    verdict_model = EvidenceCoverageVerdict

    def rubric_id(self) -> str:
        return RUBRIC

    def from_verdict(self, verdict: EvidenceCoverageVerdict, context) -> dict:
        return _coverage(verdict, _reachable_facts(context))


class GlobalEvidenceCoverage(JudgeMetric):
    spec = MetricSpec(
        id="global_evidence_coverage",
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "weighted_recall",
            "covered_weight",
            "total_weight",
            "facts",
        ),
        description=(
            "Weighted recall over every reference fact, with the same "
            "denominator for every preset - what was lost by removing a "
            "source. Read next to the conditional number, never instead of it."
        ),
    )
    verdict_model = EvidenceCoverageVerdict

    def rubric_id(self) -> str:
        return RUBRIC

    def from_verdict(self, verdict: EvidenceCoverageVerdict, context) -> dict:
        return _coverage(verdict, _all_facts(context))
